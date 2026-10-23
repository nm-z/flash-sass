/* One stream/graph per die; native kernel prologues and epilogues own the
 * peer handshake. The driver sees only local dependencies. */
static int graph_local;
static graph_plan graph_local_plans[MAXD];
static CUstream graph_local_streams[MAXD];
static CUfunction graph_local_advance[MAXD];
static unsigned graph_local_positions;
static double graph_local_launch_s,graph_local_wait_s,graph_local_status_s;
static void graph_local_release(void) {
    for (int d=0;d<nd;d++) {
        cuCtxSetCurrent(ctx[d]);
        if (graph_local_plans[d].exec) cuGraphExecDestroy(graph_local_plans[d].exec);
        if (graph_local_plans[d].graph) cuGraphDestroy(graph_local_plans[d].graph);
        if (graph_local_streams[d]) cuStreamDestroy(graph_local_streams[d]);
    }
    memset(graph_local_plans,0,sizeof graph_local_plans);
    memset(graph_local_streams,0,sizeof graph_local_streams);
    memset(graph_local_advance,0,sizeof graph_local_advance);graph_local=0;
}
static void graph_local_initialize(void) {
    for (int d=0;d<nd;d++) {
        CK(cuCtxSetCurrent(ctx[d]));
        CK(cuModuleGetFunction(&graph_local_advance[d],graph_modules[d],"graph_advance_local"));
        CK(cuStreamCreate(&graph_local_streams[d],CU_STREAM_NON_BLOCKING));
    }
}
static void graph_local_prepare(const prog_t *programs) {
    graph_build_error=1;
    for (int d=0;d<nd;d++) {
        const prog_t *p=&programs[d];
        uint64_t hash=0xcbf29ce484222325ull;
        const unsigned char *bytes=(const unsigned char *)p->p;
        for (size_t i=0;i<(size_t)p->n*sizeof(phase_t);i++) hash=(hash^bytes[i])*0x100000001b3ull;
        graph_plan *plan=&graph_local_plans[d];
        if (plan->exec && plan->hash==hash) continue;
        CK(cuCtxSetCurrent(ctx[d]));
        if (plan->exec) CK(cuGraphExecDestroy(plan->exec));
        if (plan->graph) CK(cuGraphDestroy(plan->graph));
        memset(plan,0,sizeof *plan);CK(cuGraphCreate(&plan->graph,0));
        CUgraphNode prior=NULL;
        for (int i=0;i<p->n;i++) {
            const phase_t *phase=&p->p[i];unsigned op=phase->w[0];
            if (op>=OP_COUNT || op==OP_XWAIT || phase->w[30]&24u || !phase->w[1] || !graph_functions[d][op]) {
                fprintf(stderr,"unsupported local graph phase die=%d phase=%d op=%u\n",d,i,op);longjmp(run_jb,1);
            }
            struct {uint32_t words[30];CUdeviceptr ctl;} args;
            memcpy(args.words,phase->w+2,sizeof args.words);args.words[29]=(uint32_t)i;args.ctl=dCTL[d];
            size_t size=sizeof args;
            void *extra[]={CU_LAUNCH_PARAM_BUFFER_POINTER,&args,CU_LAUNCH_PARAM_BUFFER_SIZE,&size,CU_LAUNCH_PARAM_END};
            CUDA_KERNEL_NODE_PARAMS_v1 k={.func=graph_functions[d][op],.gridDimX=phase->w[1],.gridDimY=1,.gridDimZ=1,
                .blockDimX=die_opthreads[op],.blockDimY=1,.blockDimZ=1,.extra=extra};
            CUgraphNode next;CK(graph_add_kernel(&next,plan->graph,i?&prior:NULL,i?1:0,&k));prior=next;
        }
        CUdeviceptr control=dCTL[d];size_t size=sizeof control;
        void *extra[]={CU_LAUNCH_PARAM_BUFFER_POINTER,&control,CU_LAUNCH_PARAM_BUFFER_SIZE,&size,CU_LAUNCH_PARAM_END};
        CUDA_KERNEL_NODE_PARAMS_v1 finish={.func=graph_local_advance[d],.gridDimX=1,.gridDimY=1,.gridDimZ=1,
            .blockDimX=1,.blockDimY=1,.blockDimZ=1,.extra=extra};
        CUgraphNode final;CK(graph_add_kernel(&final,plan->graph,&prior,1,&finish));
        char log[1024]={0};CUgraphNode bad=NULL;
        CUresult result=graph_instantiate(&plan->exec,plan->graph,&bad,log,sizeof log);
        if (result) fprintf(stderr,"local graph die %d: %s\n",d,log);
        CK(result);plan->hash=hash;
    }
    graph_build_error=0;
}
static void graph_local_abort(void) {
    *h_abort=1;
    for (int d=0;d<nd;d++) if (cuCtxSetCurrent(ctx[d])==CUDA_SUCCESS)
        if (cuMemcpyHtoDAsync(dCTL[d]+0x2c,h_abort,4,pollst[d])==CUDA_SUCCESS) cuStreamSynchronize(pollst[d]);
}
static void graph_local_execute(void) {
    CUresult error=CUDA_SUCCESS;
    double begin=now();
    for (int d=0;d<nd;d++) {
        CUresult r=cuCtxSetCurrent(ctx[d]);
        if (!r) r=cuGraphLaunch(graph_local_plans[d].exec,graph_local_streams[d]);
        if (r && !error) error=r;
    }
    double started=now();
    graph_local_launch_s+=started-begin;
    double timeout=getenv("FLASH_GRAPH_TOKEN_TIMEOUT")?atof(getenv("FLASH_GRAPH_TOKEN_TIMEOUT")):300.;
    if (timeout<=0 || timeout>300) timeout=300;
    struct timespec pause={0,1000000};
    while (!error) {
        int complete=1;
        for (int d=0;d<nd;d++) {
            CUresult r=cuCtxSetCurrent(ctx[d]);
            if (!r) r=cuStreamQuery(graph_local_streams[d]);
            if (r==CUDA_ERROR_NOT_READY) complete=0;
            else if (r) {error=r;break;}
        }
        if (complete || error) break;
        if (now()-started>timeout) {error=CUDA_ERROR_LAUNCH_TIMEOUT;break;}
        nanosleep(&pause,NULL);
    }
    if (error) graph_local_abort();
    for (int d=0;d<nd;d++) {
        CUresult r=cuCtxSetCurrent(ctx[d]);if (!r) r=cuStreamSynchronize(graph_local_streams[d]);
        if (r && !error) {error=r;graph_local_abort();}
    }
    CK(error);
    double finished=now();graph_local_wait_s+=finished-started;
    for (int d=0;d<nd;d++) {
        uint32_t expired;CK(cuCtxSetCurrent(ctx[d]));CK(cuMemcpyDtoH(&expired,dCTL[d]+0x2c,4));
        if (expired) {fprintf(stderr,"local graph peer wait expired on die %d\n",d);longjmp(run_jb,1);}
    }
    graph_local_status_s+=now()-finished;graph_local_positions++;
}
