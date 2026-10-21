#define _POSIX_C_SOURCE 200809L
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#define QSA_BUDGET 512u
#define CAPTURE_POSITION 2052u
#define CANDIDATE_BLOCKS 513u

static int wait_for(pid_t pid) {
	int status;
	while (waitpid(pid, &status, 0) < 0) {
		if (errno == EINTR) continue;
		perror("waitpid");
		return 125;
	}
	if (WIFEXITED(status)) return WEXITSTATUS(status);
	if (WIFSIGNALED(status)) return 128 + WTERMSIG(status);
	return 125;
}

static int run_to_files(char *const argv[], const char *out_path,
			const char *err_path) {
	pid_t pid = fork();
	if (pid < 0) { perror("fork"); return 125; }
	if (pid == 0) {
		int out = open(out_path, O_WRONLY | O_CREAT | O_TRUNC, 0600);
		int err = open(err_path, O_WRONLY | O_CREAT | O_TRUNC, 0600);
		if (out < 0 || err < 0 || dup2(out, STDOUT_FILENO) < 0 ||
		    dup2(err, STDERR_FILENO) < 0) {
			perror("redirect child output"); _exit(125);
		}
		close(out); close(err);
		execvp(argv[0], argv);
		perror(argv[0]); _exit(127);
	}
	return wait_for(pid);
}

static int copy_file(const char *from, const char *to) {
	int in = open(from, O_RDONLY);
	if (in < 0) { perror(from); return 1; }
	int out = open(to, O_WRONLY | O_CREAT | O_TRUNC, 0600);
	if (out < 0) { perror(to); close(in); return 1; }
	char buf[65536]; int bad = 0;
	for (;;) {
		ssize_t n = read(in, buf, sizeof buf);
		if (n < 0) { if (errno == EINTR) continue; perror(from); bad = 1; break; }
		if (!n) break;
		ssize_t off = 0;
		while (off < n) {
			ssize_t w = write(out, buf + off, (size_t)(n - off));
			if (w < 0) { if (errno == EINTR) continue; perror(to); bad = 1; break; }
			off += w;
		}
		if (bad) break;
	}
	if (close(in)) { perror(from); bad = 1; }
	if (close(out)) { perror(to); bad = 1; }
	return bad;
}

static int contains_text(const char *path, const char *needle) {
	FILE *f = fopen(path, "rb");
	if (!f) { perror(path); return 0; }
	char buf[4096]; int found = 0;
	while (fgets(buf, sizeof buf, f)) if (strstr(buf, needle)) { found = 1; break; }
	if (ferror(f)) { perror(path); found = 0; }
	fclose(f); return found;
}

static int count_token_lines(const char *path, unsigned *count) {
	FILE *f = fopen(path, "rb");
	if (!f) { perror(path); return 1; }
	unsigned n = 0, id; char tail;
	int bad = 0;
	char line[64];
	while (fgets(line, sizeof line, f)) {
		if (sscanf(line, "%u %c", &id, &tail) != 1 || id > 248319u || !strchr(line, '\n')) {
			fprintf(stderr, "invalid token ID line in %s\n", path); bad = 1; break;
		}
		if (n == UINT_MAX) { fprintf(stderr, "too many prompt tokens\n"); bad = 1; break; }
		n++;
	}
	if (ferror(f)) { perror(path); bad = 1; }
	if (fclose(f)) { perror(path); bad = 1; }
	if (!bad) *count = n;
	return bad;
}

static int validate_capture(const char *path, uint64_t *hash) {
	FILE *f = fopen(path, "rb");
	if (!f) { perror(path); return 1; }
	uint32_t ids[QSA_BUDGET];
	int ok = fread(ids, sizeof ids[0], QSA_BUDGET, f) == QSA_BUDGET && fgetc(f) == EOF;
	if (fclose(f)) ok = 0;
	if (!ok) { fprintf(stderr, "production capture must contain exactly 512 uint32 values\n"); return 1; }
	uint64_t h = 0xcbf29ce484222325ull;
	for (unsigned i = 0; i < QSA_BUDGET; ++i) {
		if (ids[i] >= CANDIDATE_BLOCKS || (i && ids[i - 1] >= ids[i])) {
			fprintf(stderr, "production selection is not sorted, unique, and below %u candidates\n", CANDIDATE_BLOCKS);
			return 1;
		}
		const unsigned char *p = (const unsigned char *)&ids[i];
		for (unsigned j = 0; j < sizeof ids[i]; ++j) { h ^= p[j]; h *= 0x100000001b3ull; }
	}
	*hash = h; return 0;
}

static int csv_pair_identity(const char *left_path, const char *right_path) {
	FILE *left = fopen(left_path, "rb"), *right = fopen(right_path, "rb");
	if (!left || !right) {
		perror("open paired CSV"); if (left) fclose(left); if (right) fclose(right); return 1;
	}
	char a[2048], b[2048]; unsigned line = 0; int failed = 0;
	while (fgets(a, sizeof a, left) && fgets(b, sizeof b, right)) {
		line++;
		if (line == 1) { if (strcmp(a, b)) failed = 1; continue; }
		char *af[10], *bf[10], *sa = NULL, *sb = NULL; size_t na = 0, nb = 0;
		for (char *v = strtok_r(a, ",\r\n", &sa); v && na < 10; v = strtok_r(NULL, ",\r\n", &sa)) af[na++] = v;
		for (char *v = strtok_r(b, ",\r\n", &sb); v && nb < 10; v = strtok_r(NULL, ",\r\n", &sb)) bf[nb++] = v;
		if (na != 10 || nb != 10) { failed = 1; continue; }
		for (size_t i = 0; i < 10; ++i) if (i != 7 && strcmp(af[i], bf[i])) failed = 1;
	}
	if (fgets(a, sizeof a, left) || fgets(b, sizeof b, right)) failed = 1;
	if (ferror(left) || ferror(right)) failed = 1;
	fclose(left); fclose(right);
	if (failed) fprintf(stderr, "host/VRAM CSVs differ outside the measured GPU timing field\n");
	return failed;
}

static int put_directory(const char *directory, const char *label) {
	int pipefd[2];
	if (pipe(pipefd) < 0) { perror("pipe"); return 125; }
	pid_t put_pid = fork();
	if (put_pid < 0) { perror("fork gpuq put"); close(pipefd[0]); close(pipefd[1]); return 125; }
	if (put_pid == 0) {
		close(pipefd[1]);
		if (dup2(pipefd[0], STDIN_FILENO) < 0) { perror("dup2 gpuq put"); _exit(125); }
		close(pipefd[0]); execlp("gpuq", "gpuq", "put", label, (char *)NULL);
		perror("gpuq put"); _exit(127);
	}
	pid_t tar_pid = fork();
	if (tar_pid < 0) { perror("fork tar"); close(pipefd[0]); close(pipefd[1]); (void)wait_for(put_pid); return 125; }
	if (tar_pid == 0) {
		close(pipefd[0]);
		if (dup2(pipefd[1], STDOUT_FILENO) < 0) { perror("dup2 tar"); _exit(125); }
		close(pipefd[1]); execlp("tar", "tar", "-czf", "-", "-C", directory, ".", (char *)NULL);
		perror("tar"); _exit(127);
	}
	close(pipefd[0]); close(pipefd[1]);
	int tar_rc = wait_for(tar_pid), put_rc = wait_for(put_pid);
	if (tar_rc) fprintf(stderr, "tar exit status %d\n", tar_rc);
	if (put_rc) fprintf(stderr, "gpuq put exit status %d\n", put_rc);
	return tar_rc ? tar_rc : put_rc;
}

static int remove_tree(const char *path) {
	pid_t pid = fork();
	if (pid < 0) { perror("fork rm"); return 125; }
	if (pid == 0) { execlp("rm", "rm", "-rf", path, (char *)NULL); perror("rm"); _exit(127); }
	return wait_for(pid);
}

static int run_gather_pair(const char *work, const char *selection_path,
			   const char *qsa_revision) {
	char csv[2][PATH_MAX], json[2][PATH_MAX], err[2][PATH_MAX];
	if (setenv("DIE_CUBIN", "./die.cubin", 1) < 0 ||
	    setenv("GATHER_IDS", selection_path, 1) < 0 ||
	    setenv("CUDA_VISIBLE_DEVICES", "GPU-431371bd-b482-96b2-4b70-04f83c6d075b", 1) < 0) {
		perror("configure gather pair"); return 2;
	}
	for (int host = 0; host <= 1; ++host) {
		char host_text[2] = {(char)('0' + host), '\0'};
		if (snprintf(csv[host], sizeof csv[host], "%s/gather-host%s.csv", work, host_text) >= (int)sizeof csv[host] ||
		    snprintf(json[host], sizeof json[host], "%s/gather-host%s.json", work, host_text) >= (int)sizeof json[host] ||
		    snprintf(err[host], sizeof err[host], "%s/gather-host%s.stderr", work, host_text) >= (int)sizeof err[host]) {
			fprintf(stderr, "gather output path is too long\n"); return 2;
		}
		char *const argv[] = {"./qsa_gather_bench", "513", "1", "8", "captured", host_text, csv[host], "4242", NULL};
		int rc = run_to_files(argv, json[host], err[host]);
		if (rc) { fprintf(stderr, "captured host=%s gather failed with status %d\n", host_text, rc); return rc; }
	}
	char pair_json[PATH_MAX], pair_err[PATH_MAX];
	if (snprintf(pair_json, sizeof pair_json, "%s/gather-pair-check.json", work) >= (int)sizeof pair_json ||
	    snprintf(pair_err, sizeof pair_err, "%s/gather-pair-check.stderr", work) >= (int)sizeof pair_err) {
		fprintf(stderr, "pair check path is too long\n"); return 2;
	}
	char filter[2048];
	int filter_n = snprintf(filter, sizeof filter,
		"($h0[0].preflight_verified_all == true) and ($h1[0].preflight_verified_all == true) and "
		"($h0[0].source_revision == $h1[0].source_revision) and ($h0[0].source_revision == \"%s\") and "
		"($h0[0].pattern == \"captured\") and ($h1[0].pattern == \"captured\") and "
		"($h0[0].host_kv == 0) and ($h1[0].host_kv == 1) and "
		"($h0[0].position_start == 2052) and ($h1[0].position_start == 2052) and "
		"($h0[0].position_end == 2052) and ($h1[0].position_end == 2052) and "
		"($h0[0].tail_residue_start == $h1[0].tail_residue_start) and "
		"($h0[0].cache_fnv1a64 == $h1[0].cache_fnv1a64) and "
		"($h0[0].selections_fnv1a64 == $h1[0].selections_fnv1a64) and "
		"($h0[0].logical_keys == $h1[0].logical_keys) and ($h0[0].physical_gathered_slots == $h1[0].physical_gathered_slots) and "
		"($h0[0].selection_changes == 0) and ($h1[0].selection_changes == 0) and "
		"($h0[0].verified_output_signature_changes == 0) and ($h1[0].verified_output_signature_changes == 0)",
		qsa_revision);
	if (filter_n < 0 || filter_n >= (int)sizeof filter) {
		fprintf(stderr, "pair filter is too long\n"); return 2;
	}
	char *const jq[] = {"jq", "-n", "-e", "--slurpfile", "h0", json[0], "--slurpfile", "h1", json[1], filter, NULL};
	int jq_rc = run_to_files(jq, pair_json, pair_err);
	int csv_rc = csv_pair_identity(csv[0], csv[1]);
	if (jq_rc || csv_rc) { fprintf(stderr, "real-selection placement pair failed validation\n"); return 1; }
	return 0;
}

int main(int argc, char **argv) {
	if (argc != 10) {
		fprintf(stderr, "usage: issue16_production_launcher <tokenizer.gguf> <ple.gguf> <hold0> <hold1> <hold2> <hold3> <hold4> <hold5> <source-revision>\n");
		return 2;
	}
	const char *job = getenv("GPUQ_JOB_ID");
	if (!job || !*job) { fprintf(stderr, "GPUQ_JOB_ID is required\n"); return 2; }
	char work[PATH_MAX], label[256], prompt_src[PATH_MAX], rendered[PATH_MAX], ids_txt[PATH_MAX];
	char model_out[PATH_MAX], model_err[PATH_MAX], selection[PATH_MAX];
	if (snprintf(work, sizeof work, "/dev/shm/fs16-production-%s", job) >= (int)sizeof work ||
	    snprintf(label, sizeof label, "issue16.production-selection.%s", job) >= (int)sizeof label ||
	    snprintf(prompt_src, sizeof prompt_src, "%s/prompt-source.txt", work) >= (int)sizeof prompt_src ||
	    snprintf(rendered, sizeof rendered, "%s/rendered-prompt.txt", work) >= (int)sizeof rendered ||
	    snprintf(ids_txt, sizeof ids_txt, "%s/prompt.ids.txt", work) >= (int)sizeof ids_txt ||
	    snprintf(model_out, sizeof model_out, "%s/model.stdout", work) >= (int)sizeof model_out ||
	    snprintf(model_err, sizeof model_err, "%s/model.stderr", work) >= (int)sizeof model_err ||
	    snprintf(selection, sizeof selection, "%s/production-selection.ids.u32", work) >= (int)sizeof selection) {
		fprintf(stderr, "GPUQ path is too long\n"); return 2;
	}
	if (mkdir(work, 0700) < 0) { perror(work); return 2; }
	int run_rc = 0; unsigned prompt_tokens = 0; uint64_t selection_hash = 0;
	int model_ok = 0;
	fprintf(stderr, "issue16 production capture: staging prompt and model inputs\n"); fflush(stderr);
	if (chdir("/dev/shm/fs16-issue16-production") < 0) {
		perror("chdir staged inputs"); run_rc = 1;
	}
	char launcher_source[PATH_MAX], tokenizer_source[PATH_MAX];
	if (snprintf(launcher_source, sizeof launcher_source, "%s/launcher-source.c", work) >= (int)sizeof launcher_source ||
	    snprintf(tokenizer_source, sizeof tokenizer_source, "%s/tokenizer-source.c", work) >= (int)sizeof tokenizer_source) {
		fprintf(stderr, "source bundle path is too long\n"); run_rc = 2;
	}
	char model_copy[PATH_MAX], gather_copy[PATH_MAX], helper_copy[PATH_MAX], cubin_copy[PATH_MAX], launcher_bin_copy[PATH_MAX];
	if (!run_rc && (snprintf(model_copy, sizeof model_copy, "%s/model_test", work) >= (int)sizeof model_copy ||
			snprintf(gather_copy, sizeof gather_copy, "%s/qsa_gather_bench", work) >= (int)sizeof gather_copy ||
			snprintf(helper_copy, sizeof helper_copy, "%s/production_prompt_ids", work) >= (int)sizeof helper_copy ||
			snprintf(cubin_copy, sizeof cubin_copy, "%s/die.cubin", work) >= (int)sizeof cubin_copy ||
			snprintf(launcher_bin_copy, sizeof launcher_bin_copy, "%s/issue16_production_launcher", work) >= (int)sizeof launcher_bin_copy)) {
		fprintf(stderr, "binary bundle path is too long\n"); run_rc = 2;
	}
	if (!run_rc && (copy_file("./issue16_production_launcher.c", launcher_source) ||
			copy_file("./production_prompt_ids.c", tokenizer_source) ||
			copy_file("./production-prompt.txt", prompt_src) ||
			copy_file("./model_test", model_copy) ||
			copy_file("./issue16_production_launcher", launcher_bin_copy))) {
		fprintf(stderr, "failed copying production input or source into result bundle\n"); run_rc = 1;
	}
	if (!run_rc && (copy_file("./qsa_gather_bench", gather_copy) ||
			copy_file("./production_prompt_ids", helper_copy) ||
			copy_file("./die.cubin", cubin_copy))) {
		fprintf(stderr, "failed copying binary inputs into result bundle\n"); run_rc = 1;
	}
	for (int i = 0; !run_rc && i < 6; ++i) {
		char dest[PATH_MAX];
		if (snprintf(dest, sizeof dest, "%s/holder-die%d.manifest", work, i) >= (int)sizeof dest) {
			fprintf(stderr, "holder manifest path is too long\n"); run_rc = 2;
		} else if (copy_file(argv[3 + i], dest)) run_rc = 1;
	}
	if (!run_rc) {
		fprintf(stderr, "issue16 production capture: render chat template and tokenize full user prompt\n"); fflush(stderr);
		char render_out[PATH_MAX], render_err[PATH_MAX];
		if (snprintf(render_out, sizeof render_out, "%s/tokenizer.stdout", work) >= (int)sizeof render_out ||
		    snprintf(render_err, sizeof render_err, "%s/tokenizer.stderr", work) >= (int)sizeof render_err) {
			fprintf(stderr, "tokenizer log path is too long\n"); run_rc = 2;
		} else {
			char *const token_argv[] = {"./production_prompt_ids", (char *)argv[1], (char *)"./production-prompt.txt", rendered, ids_txt, NULL};
			run_rc = run_to_files(token_argv, render_out, render_err);
			if (!run_rc && count_token_lines(ids_txt, &prompt_tokens)) run_rc = 1;
			if (!run_rc && prompt_tokens <= CAPTURE_POSITION) {
				fprintf(stderr, "rendered prompt has %u tokens and cannot reach position %u\n", prompt_tokens, CAPTURE_POSITION);
				run_rc = 3;
			}
			if (!run_rc) { fprintf(stderr, "issue16 production capture: prompt has %u tokens; running the prompt plus one generated token through all 48 blocks\n", prompt_tokens); fflush(stderr); }
		}
	}
	if (!run_rc) {
		const char *envs[][2] = {
			{"CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5"}, {"CUDA_DEVICE_ORDER", "PCI_BUS_ID"},
			{"SPLIT_SHARES", "112,97,81,116,97,97"}, {"ATTN_MAP", "1,1,1,1,4,4,4,4,5,5,5,5"},
			{"ATTN_DIE", "1"}, {"MODEL_ATTN", "qsa"}, {"FLASH_NBLK", "512"},
			{"FLASH_KV_HOST", "0"}, {"MODEL_NOTWIN", "1"}, {"MODEL_IDS", ids_txt},
			{"MODEL_PLE", argv[2]}, {"MODEL_PLE_OFF", "192"}, {"QSA_CAPTURE_IDS", selection},
			{"QSA_CAPTURE_POSITION", "2052"}, {"QSA_CAPTURE_LAYER", "47"}, {"DIE_CUBIN", "./die.cubin"}
		};
		for (size_t i = 0; i < sizeof envs / sizeof envs[0]; ++i)
			if (setenv(envs[i][0], envs[i][1], 1) < 0) { perror(envs[i][0]); run_rc = 2; break; }
		char token_count[32]; snprintf(token_count, sizeof token_count, "%u", prompt_tokens + 1);
		char *model_argv[16] = {"./model_test", (char *)"/dev/null", (char *)"0", (char *)"48", token_count, (char *)"eh",
			(char *)argv[3], (char *)argv[4], (char *)argv[5], (char *)argv[6], (char *)argv[7], (char *)argv[8], NULL};
		if (!run_rc) run_rc = run_to_files(model_argv, model_out, model_err);
		if (!run_rc && (!contains_text(model_out, "OK:") ||
				!contains_text(model_err, "captured 512 production QSA IDs from MODEL_ATTN=qsa block 47 die 5 at position 2052 of 513 candidate blocks"))) {
			fprintf(stderr, "full-model run did not report success and the requested production selection\n"); run_rc = 1;
		}
		if (!run_rc) run_rc = validate_capture(selection, &selection_hash);
		if (!run_rc) {
			model_ok = 1;
			fprintf(stderr, "issue16 production capture: accepted sorted 512-ID selection, fnv1a64=%016llx\n", (unsigned long long)selection_hash);
			fflush(stderr);
		}
	}
	if (!run_rc) {
		fprintf(stderr, "issue16 production capture: paired gather preflight on VRAM and pinned host memory\n"); fflush(stderr);
		if (setenv("CUDA_VISIBLE_DEVICES", "GPU-431371bd-b482-96b2-4b70-04f83c6d075b", 1) < 0) { perror("CUDA_VISIBLE_DEVICES for gather"); run_rc = 2; }
		char qsa_revision[8]; memcpy(qsa_revision, argv[9], 7); qsa_revision[7] = '\0';
		if (!run_rc) run_rc = run_gather_pair(work, selection, qsa_revision);
		if (!run_rc) { fprintf(stderr, "issue16 production capture: paired gather output checks passed\n"); fflush(stderr); }
	}
	char report_path[PATH_MAX];
	if (snprintf(report_path, sizeof report_path, "%s/production-manifest.json", work) >= (int)sizeof report_path) {
		fprintf(stderr, "manifest path is too long\n"); run_rc = run_rc ? run_rc : 2;
	} else {
		FILE *report = fopen(report_path, "wb");
		if (!report) { perror(report_path); run_rc = run_rc ? run_rc : 1; }
		else {
			fprintf(report,
			"{\"source_revision\":\"%s\",\"gpuq_job_id\":\"%s\",\"model_test_status\":%s,\"model_test_scalar_twin\":false,\"prompt_tokens\":%u,\"model_tokens_run\":%u,\"generated_tokens\":1,\"tokenizer_gguf\":\"%s\",\"ple_gguf\":\"%s\",\"ple_data_offset_bytes\":192,\"selection_source\":\"full-model MODEL_ATTN=qsa OP_IDXSEL output\",\"selection_layer\":47,\"selection_position\":%u,\"candidate_blocks\":%u,\"selected_blocks\":%u,\"logical_tail_tokens\":1,\"selection_fnv1a64\":\"%016llx\",\"model_kv_placement\":\"device-vram (FLASH_KV_HOST=0)\",\"gather_pair\":\"captured selection; distinct paired arms host_kv=0 and host_kv=1; nv=513, tail=1, position=2052, 8 iterations each\",\"gather_fixture\":\"finite position/channel-signature INT8 K/V and FP16 scales, exact C reference preflight\",\"prompt_rendering\":\"model tokenizer.chat_template, add_generation_prompt=true, parse_special=true\"}\n",
				argv[9], job, model_ok ? "true" : "false", prompt_tokens, prompt_tokens + 1, argv[1], argv[2], CAPTURE_POSITION,
				CANDIDATE_BLOCKS, QSA_BUDGET, (unsigned long long)selection_hash);
			if (fclose(report)) { perror("close manifest"); run_rc = run_rc ? run_rc : 1; }
		}
	}
	if (argc == 10 && !run_rc) {
		char identity[PATH_MAX], identity_err[PATH_MAX];
		if (snprintf(identity, sizeof identity, "%s/binary-identities.sha256", work) >= (int)sizeof identity ||
		    snprintf(identity_err, sizeof identity_err, "%s/binary-identities.stderr", work) >= (int)sizeof identity_err) {
			fprintf(stderr, "identity path is too long\n"); run_rc = 2;
		} else {
			char *const sha[] = {"sha256sum", model_copy, gather_copy, helper_copy, cubin_copy, prompt_src, rendered, ids_txt, selection, NULL};
			int sha_rc = run_to_files(sha, identity, identity_err);
			if (sha_rc) run_rc = sha_rc;
		}
	}
	if (!run_rc) printf("{\"gpuq_job_id\":\"%s\",\"source_revision\":\"%s\",\"prompt_tokens\":%u,\"production_selection\":true,\"candidate_blocks\":%u,\"selected_blocks\":%u,\"selection_fnv1a64\":\"%016llx\",\"paired_gather_preflight\":true}\n",
			job, argv[9], prompt_tokens, CANDIDATE_BLOCKS, QSA_BUDGET, (unsigned long long)selection_hash);
	if (run_rc) fprintf(stderr, "production selection validation failed with status %d; uploading available diagnostics\n", run_rc);
	int put_rc = put_directory(work, label);
	if (put_rc == 0) {
		fprintf(stderr, "uploaded production selection evidence as %s\n", label);
		int rm_rc = remove_tree(work);
		if (rm_rc) fprintf(stderr, "could not remove %s (status %d)\n", work, rm_rc);
	} else fprintf(stderr, "production evidence remains at %s after upload failure\n", work);
	return run_rc ? run_rc : put_rc;
}
