/* Experimental scalar CUDA graph executor. The graph owns the cross-die
 * dependencies that the persistent interpreter represents with flag polling.
 * Use the CUDA 10/11 entry-point ABI explicitly with newer installed headers. */
extern CUresult graph_add_kernel(CUgraphNode *, CUgraph, const CUgraphNode *, size_t,
    const CUDA_KERNEL_NODE_PARAMS_v1 *) __asm__("cuGraphAddKernelNode");
extern CUresult graph_add_edges(CUgraph, const CUgraphNode *, const CUgraphNode *, size_t)
    __asm__("cuGraphAddDependencies");
extern CUresult graph_instantiate(CUgraphExec *, CUgraph, CUgraphNode *, char *, size_t)
    __asm__("cuGraphInstantiate");
static int graph_mode;
static CUmodule graph_modules[MAXD];
static CUfunction graph_functions[MAXD][OP_COUNT], graph_advance;
static CUstream graph_stream;
static CUcontext graph_context;
static int graph_owns_context;
typedef struct { CUgraph graph; CUgraphExec exec; uint64_t hash; } graph_plan;
static graph_plan graph_plans[3];
static unsigned graph_next;
static graph_plan *graph_active;

static void graph_release(void) {
    if (!core) return;
    cuCtxSetCurrent(graph_context ? graph_context : ctx[0]);
    for (unsigned i=0;i<3;i++) {
        if (graph_plans[i].exec) cuGraphExecDestroy(graph_plans[i].exec);
        if (graph_plans[i].graph) cuGraphDestroy(graph_plans[i].graph);
    }
    memset(graph_plans,0,sizeof graph_plans); graph_active=NULL; graph_next=0;
    if (graph_stream) cuStreamDestroy(graph_stream);
    graph_stream=NULL;
    for (int d=0;d<nd;d++) if (graph_modules[d]) {
        cuCtxSetCurrent(ctx[d]);cuModuleUnload(graph_modules[d]);graph_modules[d]=NULL;
    }
    if (graph_owns_context && graph_context) cuCtxDestroy(graph_context);
    graph_context=NULL;graph_owns_context=0;
    memset(graph_functions,0,sizeof graph_functions);graph_mode=0;
}
static void graph_initialize(void) {
    const char *path=getenv("FLASH_GRAPH_CUBIN");
    graph_mode=path && *path;
    if (!graph_mode) return;
    if (two_position || kv_trace) {
        fprintf(stderr,"graph executor requires scalar positions and no KV instrumentation\n");
        longjmp(init_jb,1);
    }
    for (int d=0;d<nd;d++) {
        CK(cuCtxSetCurrent(ctx[d]));CK(cuModuleLoad(&graph_modules[d],path));
        for (int op=0;op<OP_COUNT;op++) {
            if (op==OP_XWAIT || op>=OP_BF16BIASP) continue;
            char name[80];snprintf(name,sizeof name,"graph_%s",die_opname[op]);
            CK(cuModuleGetFunction(&graph_functions[d][op],graph_modules[d],name));
        }
    }
    CK(cuCtxSetCurrent(ctx[0]));
    CK(cuModuleGetFunction(&graph_advance,graph_modules[0],"graph_advance"));
    graph_context=ctx[0];
    const char *anchor=getenv("FLASH_GRAPH_ANCHOR");
    if (anchor && *anchor) {
        char *end;long die=strtol(anchor,&end,10);
        int ordinal=(*end || die<0 || die>7)?-1:hold_ordinal((int)die);
        if (ordinal<0) {fprintf(stderr,"invalid FLASH_GRAPH_ANCHOR\n");longjmp(init_jb,1);}
        CUdevice device;CK(cuDeviceGet(&device,ordinal));
        CK(cuCtxCreate_v2(&graph_context,CU_CTX_SCHED_SPIN,device));
        graph_owns_context=1;
    }
    CK(cuStreamCreate(&graph_stream,CU_STREAM_NON_BLOCKING));
}
static void graph_prepare(const prog_t *programs,unsigned positions) {
    if (!graph_mode) return;
    if (positions!=1) {fprintf(stderr,"graph executor received paired program\n");longjmp(run_jb,1);}
    uint64_t hash=0xcbf29ce484222325ull;
    for (int d=0;d<nd;d++) {
        hash=(hash^(unsigned)programs[d].n)*0x100000001b3ull;
        const unsigned char *b=(const unsigned char *)programs[d].p;
        for (size_t i=0;i<(size_t)programs[d].n*sizeof(phase_t);i++) hash=(hash^b[i])*0x100000001b3ull;
    }
    for (unsigned i=0;i<3;i++) if (graph_plans[i].exec && graph_plans[i].hash==hash) {
        graph_active=&graph_plans[i];return;
    }
    CK(cuCtxSetCurrent(graph_context));
    graph_plan *plan=&graph_plans[graph_next++%3];
    if (plan->exec) CK(cuGraphExecDestroy(plan->exec));
    if (plan->graph) CK(cuGraphDestroy(plan->graph));
    memset(plan,0,sizeof *plan);
    CK(cuGraphCreate(&plan->graph,0));
    CUgraphNode (*nodes)[4096]=calloc(MAXD,sizeof *nodes);
    CUgraphNode (*sends)[4096]=calloc(MAXD,sizeof *sends);
    if (!nodes || !sends) {free(nodes);free(sends);longjmp(run_jb,1);}
    unsigned exchanges[MAXD]={0};
    for (int d=0;d<nd;d++) {
        CK(cuCtxSetCurrent(ctx[d]));
        for (int i=0;i<programs[d].n;i++) {
            const phase_t *p=&programs[d].p[i];unsigned op=p->w[0];
            if (op>=OP_COUNT || p->w[30]&24u) {fprintf(stderr,"invalid graph opcode/flags\n");longjmp(run_jb,1);}
            CUgraphNode *deps=i?&nodes[d][i-1]:NULL;
            if (!p->w[1] || op==OP_XWAIT) CK(cuGraphAddEmptyNode(&nodes[d][i],plan->graph,deps,i?1:0));
            else {
                if (!graph_functions[d][op]) {fprintf(stderr,"missing graph op %s\n",die_opname[op]);longjmp(run_jb,1);}
                struct { uint32_t words[30]; CUdeviceptr ctl; } args;
                memcpy(args.words,p->w+2,sizeof args.words);args.ctl=dCTL[d];
                size_t size=sizeof args;
                void *extra[]={CU_LAUNCH_PARAM_BUFFER_POINTER,&args,CU_LAUNCH_PARAM_BUFFER_SIZE,&size,CU_LAUNCH_PARAM_END};
                CUDA_KERNEL_NODE_PARAMS_v1 k={.func=graph_functions[d][op],
                    .gridDimX=p->w[1],.gridDimY=1,.gridDimZ=1,
                    .blockDimX=die_opthreads[op],.blockDimY=1,.blockDimZ=1,.extra=extra};
                CK(graph_add_kernel(&nodes[d][i],plan->graph,deps,i?1:0,&k));
            }
            if (p->w[30]&1u) sends[d][exchanges[d]++]=nodes[d][i];
        }
        if (exchanges[d]!=exchanges[0]) {fprintf(stderr,"graph exchange counts disagree\n");longjmp(run_jb,1);}
    }
    CUgraphNode joins[4096];
    CK(cuCtxSetCurrent(graph_context));
    for (unsigned exchange=0;exchange<exchanges[0];exchange++) {
        CUgraphNode sources[MAXD];
        for (int d=0;d<nd;d++) sources[d]=sends[d][exchange];
        CK(cuGraphAddEmptyNode(&joins[exchange],plan->graph,sources,(size_t)nd));
    }
    for (int d=0;d<nd;d++) {
        unsigned exchange=0;
        for (int i=0;i<programs[d].n;i++) {
            unsigned flags=programs[d].p[i].w[30];
            if (flags&2u) {
                if (!exchange) {fprintf(stderr,"graph waits before first send\n");longjmp(run_jb,1);}
                CK(graph_add_edges(plan->graph,&joins[exchange-1],&nodes[d][i],1));
            }
            if (flags&1u) exchange++;
        }
    }
    CK(cuCtxSetCurrent(ctx[0]));
    CUgraphNode terminals[MAXD],advance;
    CUdeviceptr controls[6];
    for (int d=0;d<nd;d++) terminals[d]=nodes[d][programs[d].n-1];
    for (int d=0;d<6;d++) controls[d]=dCTL[d<nd?d:0];
    size_t size=sizeof controls;
    void *extra[]={CU_LAUNCH_PARAM_BUFFER_POINTER,controls,CU_LAUNCH_PARAM_BUFFER_SIZE,&size,CU_LAUNCH_PARAM_END};
    CUDA_KERNEL_NODE_PARAMS_v1 finish={.func=graph_advance,.gridDimX=1,.gridDimY=1,.gridDimZ=1,
        .blockDimX=1,.blockDimY=1,.blockDimZ=1,.extra=extra};
    CK(graph_add_kernel(&advance,plan->graph,terminals,(size_t)nd,&finish));
    free(nodes);free(sends);
    char error[1024]={0};CUgraphNode error_node=NULL;
    CK(cuCtxSetCurrent(graph_context));
    size_t available,total;CK(cuMemGetInfo(&available,&total));
    fprintf(stderr,"graph: instantiate %u exchanges with %.1f MiB free on control context\n",exchanges[0],available/1048576.);
    CUresult result=graph_instantiate(&plan->exec,plan->graph,&error_node,error,sizeof error);
    if (result) fprintf(stderr,"graph instantiation: %s\n",error);
    CK(result);plan->hash=hash;graph_active=plan;
}
