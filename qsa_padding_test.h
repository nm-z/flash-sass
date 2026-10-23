/* Direct masked-padding boundary test. This constructs a bounded finite compact
 * fixture and runs the production capped attention/merge operations at a chosen
 * visible position. It does not claim to regenerate the historical indexer,
 * gather, or a whole-model prompt. The sequential kv test covers append/gather. */
static int test_padding(unsigned visible, unsigned budget, unsigned iters) {
	unsigned n = qsa_nkeys(visible - 1, budget);
	unsigned cpos = (4 * (budget + 1) + 2048 + 31) / 32 * 32;
	size_t kcf = ATTN_KC_FLOATS(cpos), vcf = (size_t)cpos * 512;
	int host = getenv("QSA_HOSTKV") != NULL, shown = 0, bad = 0;
	const char *km = getenv("QSA_POISON_K"), *vm = getenv("QSA_POISON_V");
	float pk = 0.f, pv = 0.f;
	int use_k = padding_poison_value_(km, &pk), use_v = padding_poison_value_(vm, &pv);
	float *q = malloc(6144 * sizeof(float)), *gate = malloc(12288 * sizeof(float));
	float *kc = calloc(kcf, sizeof(float)), *vc = calloc(vcf, sizeof(float));
	float *pm = malloc(24 * 8 * sizeof(float)), *pd = malloc(24 * 8 * sizeof(float));
	float *po = malloc(24 * 8 * 256 * sizeof(float)), *out = malloc(6144 * sizeof(float));
	float *got = malloc(24 * 8 * 256 * sizeof(float));
	if (!q || !gate || !kc || !vc || !pm || !pd || !po || !out || !got) {
		fprintf(stderr, "padding fixture allocation failed\n"); return 1;
	}
	for (int i = 0; i < 6144; i++) q[i] = 2.f * frand();
	for (int i = 0; i < 12288; i++) gate[i] = 4.f * frand();
	for (unsigned j = 0; j < n; j++)
		for (int h = 0; h < 2; h++) for (int d = 0; d < 256; d++) {
			kc[attn_kidx(j, h, d)] = 2.f * frand();
			vc[(size_t)j * 512 + h * 256 + d] = frand();
		}
	attn_partials_gpu(q, kc, vc, n, pm, pd, po);
	attn_merge_gpu(pm, pd, po, gate, out);
	for (unsigned j = n; j < cpos; j++)
		for (int h = 0; h < 2; h++) for (int d = 0; d < 256; d++) {
			float sign = (j + (unsigned)h + (unsigned)d) & 1 ? -1.f : 1.f;
			if (use_k) kc[attn_kidx(j, h, d)] = !strcmp(km, "finite") ? sign * pk : pk;
			if (use_v) vc[(size_t)j * 512 + h * 256 + d] = !strcmp(vm, "finite") ? sign * pv : pv;
		}
	CUdeviceptr dq = dalloc(6144 * 4), dg = dalloc(12288 * 4);
	CUdeviceptr dk = host ? halloc(kcf * 4) : dalloc(kcf * 4);
	CUdeviceptr dv = host ? halloc(vcf * 4) : dalloc(vcf * 4);
	CUdeviceptr dm = dalloc(24 * 8 * 4), dd = dalloc(24 * 8 * 4);
	CUdeviceptr dp = dalloc(24 * 8 * 256 * 4), dout = dalloc(6144 * 4);
	CK(cuMemcpyHtoD(dq, q, 6144 * 4)); CK(cuMemcpyHtoD(dg, gate, 12288 * 4));
	CK(cuMemcpyHtoD(dk, kc, kcf * 4)); CK(cuMemcpyHtoD(dv, vc, vcf * 4));
	CK(cuMemsetD32(dm, 0x7fc00000u, 24 * 8));
	CK(cuMemsetD32(dd, 0x7fc00000u, 24 * 8));
	CK(cuMemsetD32(dp, 0x7fc00000u, 24 * 8 * 256));
	CK(cuMemsetD32(dout, 0x7fc00000u, 6144));
	desc_t prog[2];
	d_init(&prog[0], OP_ATTNPARTN, 32);
	d_ptr(&prog[0], 0x140, dq); d_ptr(&prog[0], 0x148, dk); d_ptr(&prog[0], 0x150, dv);
	d_ptr(&prog[0], 0x158, dm); d_ptr(&prog[0], 0x160, dd); d_ptr(&prog[0], 0x168, dp);
	/* Fixed visible count keeps repeated timing iterations on exactly the same
	 * logical fixture. The persistent kernel itself starts at visible-1. */
	d_u32(&prog[0], 0x170, visible); d_u32(&prog[0], 0x174, budget);
	d_init(&prog[1], OP_ATTNMERGE, 24);
	d_ptr(&prog[1], 0x140, dm); d_ptr(&prog[1], 0x148, dd); d_ptr(&prog[1], 0x150, dp);
	d_ptr(&prog[1], 0x158, dg); d_ptr(&prog[1], 0x160, dout);
	run_token(prog, 2, visible - 1);
	CK(cuMemcpyDtoH(got, dm, 24 * 8 * 4)); bad |= cmpb("pm", pm, got, 24 * 8 * 4, &shown);
	CK(cuMemcpyDtoH(got, dd, 24 * 8 * 4)); bad |= cmpb("pd", pd, got, 24 * 8 * 4, &shown);
	CK(cuMemcpyDtoH(got, dp, 24 * 8 * 256 * 4)); bad |= cmpb("po", po, got, 24 * 8 * 256 * 4, &shown);
	CK(cuMemcpyDtoH(got, dout, 6144 * 4)); bad |= cmpb("out", out, got, 6144 * 4, &shown);
	for (int i = 0; i < 6144; i++) if (!isfinite(got[i])) {
		fprintf(stderr, "nonfinite valid fixture output at %d\n", i); bad = 1; break;
	}
	if (iters) {
		unsigned nph = 2, start = 0, end = iters;
		CK(cuMemcpyHtoD(dCTL + 8, &start, 4));
		void *params[] = { &dPROG, &nph, &dCTL, &end };
		CUevent e0, e1; float ms;
		CK(cuEventCreate(&e0, 0)); CK(cuEventCreate(&e1, 0));
		CK(cuEventRecord(e0, 0));
		CK(cuLaunchKernel(fn, 16, 1, 1, 256, 1, 1, 0, 0, params, NULL));
		CK(cuEventRecord(e1, 0)); CK(cuCtxSynchronize());
		CK(cuEventElapsedTime(&ms, e0, e1));
		printf("capped component timing: %.3f us per iteration (%u iterations; attention+merge only)\n", 1000.0 * ms / iters, iters);
		CK(cuEventDestroy(e0)); CK(cuEventDestroy(e1));
	}
	/* Attention treats both input allocations as read-only, padding included. */
	float *readback = malloc((kcf > vcf ? kcf : vcf) * sizeof(float));
	if (!readback) { fprintf(stderr, "readback allocation failed\n"); bad = 1; }
	else {
		CK(cuMemcpyDtoH(readback, dk, kcf * 4)); bad |= cmpb("kc unchanged", kc, readback, kcf * 4, &shown);
		CK(cuMemcpyDtoH(readback, dv, vcf * 4)); bad |= cmpb("vc unchanged", vc, readback, vcf * 4, &shown);
		free(readback);
	}
	bad |= test_guard_check();
	printf("capped padding: visible=%u position=%u NBLK=%u keys=%u K=%s V=%s compact_cache=%s pm/pd/po/out=%s valid-data-seed=4242\n",
		visible, visible - 1, budget, n, km ? km : "off", vm ? vm : "off",
		host ? "host" : "device", bad ? "MISMATCH" : "BIT-EXACT");
	test_guard_free();
	free(q); free(gate); free(kc); free(vc); free(pm); free(pd); free(po); free(out); free(got);
	return bad;
}
