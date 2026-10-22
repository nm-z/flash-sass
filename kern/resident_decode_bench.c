/* Reuse the GPUQ goal-request-fixtures 64-token prompt with private cores and
 * the existing IPC weight holders. No weight loader or serving reload. */
#define _GNU_SOURCE
#define _POSIX_C_SOURCE 200809L
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <sys/time.h>
#include <signal.h>
#include <unistd.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <ctype.h>

static pid_t child=-1;
static pid_t debugger_child=-1,server_pid=-1;
static char socket_path[108];
static FILE *active;
static char restore_command[4096],before_config[4096];
static int restore_pending;
static FILE *connect_core(void) {
    int fd=socket(AF_UNIX,SOCK_STREAM,0); if(fd<0) return NULL;
    struct sockaddr_un sa={.sun_family=AF_UNIX}; strcpy(sa.sun_path,socket_path);
    if(connect(fd,(struct sockaddr *)&sa,sizeof sa)) { close(fd); return NULL; }
    struct ucred peer;socklen_t peer_size=sizeof peer;
    if(!getsockopt(fd,SOL_SOCKET,SO_PEERCRED,&peer,&peer_size))server_pid=peer.pid;
    long timeout=120;
    const char *option=getenv("BENCH_SOCKET_TIMEOUT");
    if(option) { char *end;long v=strtol(option,&end,10);if(*option && !*end && v>=30 && v<=900)timeout=v; }
    struct timeval tv={.tv_sec=timeout};
    setsockopt(fd,SOL_SOCKET,SO_RCVTIMEO,&tv,sizeof tv);
    setsockopt(fd,SOL_SOCKET,SO_SNDTIMEO,&tv,sizeof tv);
    FILE *f=fdopen(fd,"r+"); if(!f) { close(fd); return NULL; }
    setvbuf(f,NULL,_IOLBF,0); return f;
}
static int restore(void) {
    if(!restore_pending) return 0;
    restore_pending=0;
    if(debugger_child>0) { int status;waitpid(debugger_child,&status,0);debugger_child=-1; }
    if(active) { fclose(active); active=NULL; }
    FILE *f=connect_core(); char line[4096];
    if(!f) { fputs("RESTORE FAILED: connection\n",stderr); return -1; }
    fprintf(f,"%s\n",restore_command); fflush(f);
    int ok=fgets(line,sizeof line,f) && !strncmp(line,"ok",2);
    if(ok) printf("restore: %s",line);
    fclose(f);
    if(!ok) { fputs("RESTORE FAILED: reload\n",stderr); return -1; }
    f=connect_core(); if(!f) return -1;
    fputs("config\n",f); fflush(f);
    ok=fgets(line,sizeof line,f) && !strcmp(line,before_config);
    printf("restore: config %s\n",ok?"identical":"DIFFERS");
    if(!ok) fprintf(stderr,"before: %safter: %s",before_config,line);
    fclose(f); return ok?0:-1;
}
static void cleanup(void) {
    if(restore_pending) { restore(); return; }
    if(child>0) { kill(child,SIGTERM); waitpid(child,NULL,0); child=-1; }
    if(*socket_path && strncmp(socket_path,"/tmp/flash-60-",14)==0) unlink(socket_path);
}
static void fail(const char *why) { fprintf(stderr,"decode benchmark: %s\n",why); cleanup(); exit(1); }
static const char prompt[] = "248045 8678 198 24342 286 4879 369 716 310 830 11553 13 5044 1683 15060 1472 279 3274 11 9307 1328 30800 11 2814 47675 25605 11 321 60445 55404 11 27224 11 321 30246 303 279 1534 4087 13 248046 198 248045 846 198 9930 1503 11679 28052 9561 30 79091 25899 303 17248 3992 13 248046 198 248045 74455 198 248068 198";
static void start_cuda_debugger(void) {
    if(server_pid<=0)fail("no verified socket peer for CUDA debugger");
    printf("cuda_debug_server_pid=%ld diagnostic_only=1\n",(long)server_pid);fflush(stdout);
    debugger_child=fork();if(debugger_child<0)fail("debugger fork");
    if(debugger_child)return;
    struct timespec delay={.tv_sec=2};nanosleep(&delay,NULL);
    char pid[32],focus[6][64];snprintf(pid,sizeof pid,"%ld",(long)server_pid);
    char *args[96];int n=0;
#define ARG(s) args[n++]=(char *)(s)
    ARG("sudo");ARG("-n");ARG("/usr/bin/timeout");ARG("--signal=TERM");ARG("--kill-after=5");ARG("30");
    ARG("/home/nate/qwen3-next/toolchains/cuda-11.8/bin/cuda-gdb");ARG("-q");ARG("-batch");ARG("-p");ARG(pid);
    ARG("-ex");ARG("set pagination off");ARG("-ex");ARG("bt 8");
    ARG("-ex");ARG("p *(unsigned int *)&mtp_steps");ARG("-ex");ARG("info cuda kernels");
    for(int d=0;d<6;d++) {
        snprintf(focus[d],sizeof focus[d],"cuda device %d sm 0 warp 0 lane 0",d);
        ARG("-ex");ARG(focus[d]);ARG("-ex");ARG("info registers R0 R3 R7 R8 R13 R18 R19 R20 R21 R22 R24 R25");
        ARG("-ex");ARG("x/8i $pc");
    }
    args[n]=NULL;
#undef ARG
    execvp(args[0],args);perror("cuda debugger exec");_exit(127);
}
static FILE *start(const char *core,const char *lib,const char *cubin,int fused) {
    snprintf(socket_path,sizeof socket_path,"/tmp/flash-60-%ld.sock",(long)getpid());
    child=fork(); if(child<0) fail("fork");
    if(!child) {
        setenv("FLASH_SOCK",socket_path,1);
        setenv("CUDA_VISIBLE_DEVICES","0,1,2,3,4,5,6",1);
        setenv("FLASH_ANCHOR_DIE","6",1);
        setenv("FLASH_MAXT","16384",1); setenv("FLASH_ENGINE_MAXT","16384",1);
        setenv("FLASH_MAXIMG","1",1); setenv("FLASH_PLE_DISK","1",1);
        setenv("FLASH_PLE_OFF","192",1); setenv("FLASH_PLE_READERS","1",1);
        setenv("FLASH_PLE_FILE","",1); setenv("FLASH_TWO_POSITION","0",1);
        setenv("FLASH_PREFILL_CHUNK","0",1); setenv("FLASH_TRACE","0",1);
        setenv("FLASH_KV_TRACE","0",1); setenv("FLASH_KV_HOST","0",1);
        setenv("FLASH_ROPE_OWNER","-1",1); setenv("FLASH_ATTN_Q_SPLIT","1",1);
        setenv("FLASH_FUSED_MOE",fused?"1":"0",1);
        setenv("FLASH_FAST_EXCHANGE",fused?"1":"0",1);
        setenv("SPLIT_SHARES","112,97,81,116,97,97",1);
        setenv("ATTN_MAP","1,1,1,1,4,4,4,4,5,5,5,5",1);
        setenv("FLASH_CUBIN",cubin,1);
        execl(core,core,"/home/nate/codex/flash-astra-perf-20261018/ple-direct.gguf",lib,
              "/tmp/hold-h0.manifest","/tmp/hold-h1.manifest","/tmp/hold-h2.manifest",
              "/tmp/hold-h3.manifest","/tmp/hold-h4.manifest","/tmp/hold-h5.manifest",(char *)0);
        perror("exec flashcore"); _exit(127);
    }
    for(int i=0;i<600;i++) {
        int status; pid_t p=waitpid(child,&status,WNOHANG);
        if(p==child) { child=-1; fail("private core exited during initialization"); }
        int fd=socket(AF_UNIX,SOCK_STREAM,0); if(fd<0) fail("socket");
        struct sockaddr_un sa={.sun_family=AF_UNIX}; strcpy(sa.sun_path,socket_path);
        if(connect(fd,(struct sockaddr *)&sa,sizeof sa)==0) {
            struct timeval tv={.tv_sec=120};
            setsockopt(fd,SOL_SOCKET,SO_RCVTIMEO,&tv,sizeof tv);
            setsockopt(fd,SOL_SOCKET,SO_SNDTIMEO,&tv,sizeof tv);
            FILE *f=fdopen(fd,"r+"); if(!f) fail("fdopen"); setvbuf(f,NULL,_IOLBF,0); return f;
        }
        close(fd); struct timespec ts={.tv_nsec=100000000}; nanosleep(&ts,NULL);
    }
    fail("private core startup deadline"); return NULL;
}
static void command(FILE *f,const char *s) {
    char *line=NULL; size_t cap=0;
    fprintf(f,"%s\n",s); fflush(f);
    if(getline(&line,&cap,f)<=0) fail("command EOF");
    fputs(line,stdout); fflush(stdout);
    if(!strncmp(line,"error",5)) { free(line); fail("command error"); }
    free(line);
}
static void exchange_dump(FILE *f,const char *name,int run) {
    for(int d=0;d<6;d++) {
        printf("xtrace arm=%s die=%d run=%d\n",name,d,run);fprintf(f,"xtrace %d\n",d);fflush(f);
        char *row=NULL;size_t capacity=0;int complete=0;
        while(getline(&row,&capacity,f)>0) {
            fputs(row,stdout);
            if(!strncmp(row,"error:",6))fail("exchange trace unavailable");
            if(!strncmp(row,"xtrace_end",10)) {
                int td;unsigned records,invalid;
                if(sscanf(row,"xtrace_end die=%d records=%u invalid=%u",&td,&records,&invalid)!=3 || td!=d || !records || invalid)fail("exchange trace incomplete");
                complete=1;break;
            }
        }
        free(row);if(!complete)fail("exchange trace EOF");
    }
}
static void value(const char *cfg,const char *key,const char *fallback,char *dst,size_t size) {
    char token[80]; snprintf(token,sizeof token," %s=",key);
    const char *p=strstr(cfg,token); if(p) p+=strlen(token); else p=fallback;
    size_t n=strcspn(p," \r\n"); if(n>=size) fail("config field too long");
    memcpy(dst,p,n); dst[n]=0;
}
static void check_scope(FILE *f,int mtp_expected,int verify_expected) {
    char line[4096],layers[32],dies[32],blocks[32];
    fputs("config\n",f);fflush(f);
    if(!fgets(line,sizeof line,f)) fail("config EOF");
    fputs(line,stdout);
    value(line,"layers","",layers,sizeof layers);
    value(line,"dies","",dies,sizeof dies);
    value(line,"nblk","",blocks,sizeof blocks);
    if(strcmp(layers,"48/48") || strcmp(dies,"6") || strcmp(blocks,"512"))
        fail("benchmark requires full 48 layers, six dies, and 512 attention blocks");
    if(mtp_expected>=0) {
        char actual[16];value(line,"mtp_shadow","missing",actual,sizeof actual);
        if(strcmp(actual,mtp_expected?"1":"0"))fail("MTP shadow mode could not be enabled; no request submitted");
    }
    if(verify_expected>=0) {
        char actual[16],expected[16];value(line,"verify_audit","missing",actual,sizeof actual);
        snprintf(expected,sizeof expected,"%d",verify_expected);
        if(strcmp(actual,expected))fail("rollback audit mode does not match request");
    }
}
int main(int argc,char **argv) {
    setvbuf(stdout,NULL,_IONBF,0);
    int resident=(argc==8 || argc==10) && !strcmp(argv[1],"--resident");
    if(argc!=5 && !resident) { fprintf(stderr,"usage: resident_decode_bench CORE LIB BASE_CUBIN CANDIDATE_CUBIN\n       resident_decode_bench --resident SOCKET LIB BASE_CUBIN CANDIDATE_CUBIN RESTORE_LIB RESTORE_CUBIN [--reference IDS_FILE]\n"); return 2; }
    if(argc==10 && strcmp(argv[8],"--reference")) fail("expected --reference IDS_FILE");
    const char *reference=argc==10?argv[9]:NULL;
    if(!getenv("GPUQ_JOB_ID")) fail("run inside gpuq");
    int pair_cost=getenv("BENCH_PAIR_COST") && atoi(getenv("BENCH_PAIR_COST"));
    int collect_mismatches=getenv("BENCH_COLLECT_MISMATCHES") && atoi(getenv("BENCH_COLLECT_MISMATCHES"));
    int repeatability=getenv("BENCH_REPEATABILITY") && atoi(getenv("BENCH_REPEATABILITY"));
    int verify_stream=getenv("BENCH_VERIFY_STREAM") && atoi(getenv("BENCH_VERIFY_STREAM"));
    int prefix_hash=getenv("BENCH_PREFIX_HASH") && atoi(getenv("BENCH_PREFIX_HASH"));
    int candidate_only=getenv("BENCH_CANDIDATE_ONLY") && atoi(getenv("BENCH_CANDIDATE_ONLY"));
    int generate=getenv("BENCH_GENERATE")?atoi(getenv("BENCH_GENERATE")):128;
    if(generate<1 || generate>128) fail("generation count must be 1..128");
    unsigned mismatches=0;
    unsigned prefix_mismatches=0;
    int native_chunk=getenv("BENCH_NATIVE_CHUNK")?atoi(getenv("BENCH_NATIVE_CHUNK")):0;
    int exact_batch=getenv("BENCH_EXACT_BATCH")?atoi(getenv("BENCH_EXACT_BATCH")):0;
    int mtp_audit=getenv("BENCH_MTP_AUDIT")?atoi(getenv("BENCH_MTP_AUDIT")):0;
    int repetitions=getenv("BENCH_RUNS")?atoi(getenv("BENCH_RUNS")):4;
    if(repetitions<1 || repetitions>4)fail("BENCH_RUNS must be1..4");
    int verify_rollback=getenv("BENCH_VERIFY_ROLLBACK")?atoi(getenv("BENCH_VERIFY_ROLLBACK")):0;
    int verify_states=getenv("BENCH_VERIFY_STATES")?atoi(getenv("BENCH_VERIFY_STATES")):1;
    int verify_matrix=getenv("BENCH_VERIFY_MATRIX")?atoi(getenv("BENCH_VERIFY_MATRIX")):0;
    int kernel_only=getenv("BENCH_KERNEL_ONLY")?atoi(getenv("BENCH_KERNEL_ONLY")):0;
    int plain_decode=getenv("BENCH_NO_MTP") && atoi(getenv("BENCH_NO_MTP"));
    int separate_chunk=getenv("BENCH_SEPARATE_CHUNK") && atoi(getenv("BENCH_SEPARATE_CHUNK"));
    if(exact_batch && (exact_batch!=8 || !resident || native_chunk || pair_cost || candidate_only || prefix_hash))
        fail("exact batch requires resident matched A/B mode with width 8");
    if(mtp_audit && ((mtp_audit<1 || mtp_audit>4) || !resident || exact_batch || native_chunk || pair_cost || (candidate_only && mtp_audit<2) || prefix_hash))
        fail("MTP audit requires resident matched A/B mode");
    if(mtp_audit>=2 && (reference || verify_stream))fail("unchecked rollout does not use a reference or stream verification");
    if(verify_rollback && (verify_rollback<1 || verify_rollback>8 || !resident || exact_batch || mtp_audit || native_chunk || pair_cost || candidate_only || prefix_hash || (verify_states!=0 && verify_states!=1)))
        fail("rollback audit requires resident matched mode, kept1..8 and state check0/1");
    if(verify_matrix && (verify_matrix!=1 || verify_rollback!=1))fail("rollback matrix starts with BENCH_VERIFY_ROLLBACK=1");
    if(kernel_only && (kernel_only!=1 || !resident || native_chunk || exact_batch || mtp_audit || verify_rollback || pair_cost || (candidate_only && !plain_decode)))fail("kernel-only comparison requires ordinary resident matched mode");
    if(native_chunk && (pair_cost || !resident || native_chunk<128 || native_chunk>1024 || (native_chunk&(native_chunk-1)))) fail("native chunk requires resident mode and power-of-two size 128..1024");
    if(candidate_only && !native_chunk && mtp_audit<2 && !plain_decode) fail("candidate-only requires native chunk, plain decode or unchecked MTP");
    if(separate_chunk && !native_chunk) fail("separate chunk module requires native chunk mode");
    if(pair_cost && !resident) fail("paired cost mode requires the resident path");
    int copies=getenv("BENCH_PAIR_PROMPT_COPIES")?atoi(getenv("BENCH_PAIR_PROMPT_COPIES")):4;
    if(copies<1||copies>4) fail("prompt copies must be 1..4");
    int prompt_copies=plain_decode?(getenv("BENCH_PROMPT_COPIES")?atoi(getenv("BENCH_PROMPT_COPIES")):1):(native_chunk || exact_batch || mtp_audit || verify_rollback)?(getenv("BENCH_PROMPT_COPIES")?atoi(getenv("BENCH_PROMPT_COPIES")):4):1;
    if(prompt_copies<1 || prompt_copies>256) fail("prompt copies must be 1..256");
    int positions=pair_cost?64*copies:generate;
    unsigned expected[256]; int have=0;
    if(reference) {
        FILE *rf=fopen(reference,"r");if(!rf) fail("reference file open");
        for(int i=0;i<positions;i++) if(fscanf(rf,"%u",&expected[i])!=1 || expected[i]>=248320) { fclose(rf);fail("invalid reference token/count"); }
        int c;while((c=fgetc(rf))!=EOF) if(!isspace((unsigned char)c)) { fclose(rf);fail("extra reference data"); }
        fclose(rf);have=1;
        printf("reference_file=%s expected_tokens=%d validation_only=1\n",reference,positions);
    }
    char maxt[32]="16384",img[32]="4096",rope[32]="5",pairs[32]="0",chunk[32]="0",qs[32]="1",trace[32]="0",ple[1024]="";
    char kv[32]="0",layers[32]="48",blocks[32]="512",readers[32]="1";
    if(resident) {
        if(strlen(argv[2])>=sizeof socket_path) fail("socket path too long");
        strcpy(socket_path,argv[2]); active=connect_core(); if(!active) fail("resident connection");
        fputs("config\n",active); fflush(active);
        if(!fgets(before_config,sizeof before_config,active)) fail("initial config EOF");
        printf("before: %s",before_config);
        const char *expected_rev=getenv("BENCH_EXPECT_RESTORE_REV");
        if(expected_rev) {
            char actual_rev[160];value(before_config,"rev","",actual_rev,sizeof actual_rev);
            if(strcmp(actual_rev,expected_rev)) fail("restore revision changed; no reload performed");
        }
        char cubin[1024]; value(before_config,"cubin","",cubin,sizeof cubin);
        if(strcmp(cubin,argv[7])) fail("resident cubin changed before admission");
        value(before_config,"maxt","230000",maxt,sizeof maxt);
        value(before_config,"img_rows","4096",img,sizeof img);
        value(before_config,"rope_owner","-1",rope,sizeof rope);
        value(before_config,"two_position","0",pairs,sizeof pairs);
        value(before_config,"prefill_chunk","0",chunk,sizeof chunk);
        value(before_config,"attn_q_split","0",qs,sizeof qs);
        value(before_config,"trace_tokens","0",trace,sizeof trace);
        value(before_config,"ple_file","",ple,sizeof ple);
        value(before_config,"kv_host","0",kv,sizeof kv);
        value(before_config,"layers","48",layers,sizeof layers);
        value(before_config,"nblk","512",blocks,sizeof blocks);
        value(before_config,"ple_readers","1",readers,sizeof readers);
        if(exact_batch && (strcmp(layers,"48/48") || strcmp(blocks,"512"))) fail("unexpected production scope");
        if(!strcmp(ple,"core")) *ple=0;
        snprintf(restore_command,sizeof restore_command,"reload %s FLASH_CUBIN=%s FLASH_ENGINE_MAXT=%s FLASH_MAXIMG=%s FLASH_ROPE_OWNER=%s FLASH_TWO_POSITION=%s FLASH_PREFILL_CHUNK=%s FLASH_ATTN_Q_SPLIT=%s FLASH_TRACE=%s FLASH_PLE_FILE=%s FLASH_FUSED_MOE=0 FLASH_FAST_EXCHANGE=0 FLASH_CHUNK_PREFILL=0 FLASH_VERIFY_STREAM=0",argv[6],argv[7],maxt,img,rope,pairs,chunk,qs,trace,ple);
        size_t rused=strlen(restore_command);
        snprintf(restore_command+rused,sizeof restore_command-rused," FLASH_PREFIX_HASH=0 FLASH_CHUNK_CUBIN= FLASH_PHASE_BATCH=0 FLASH_EXACT_CUBIN= FLASH_MTP_AUDIT=0 FLASH_MTP_MODEL= FLASH_MTP_CUBIN= FLASH_VERIFY_AUDIT=0 FLASH_VERIFY_STATE_CHECK=1 FLASH_LAYERS=%d FLASH_NBLK=%s FLASH_KV_HOST=%s FLASH_PLE_READERS=%s",atoi(layers),blocks,kv,readers);
        rused=strlen(restore_command);snprintf(restore_command+rused,sizeof restore_command-rused," FLASH_XTRACE=0 FLASH_GR_REPLICATE_FFN=0 FLASH_LOCKSTEP_PHASES=0 FLASH_STRICT_PHASE_JOINS=0 FLASH_GR_WRITE_TRACE=0");
        char old_gr[16];value(before_config,"fused_gr","0",old_gr,sizeof old_gr);
        rused=strlen(restore_command);snprintf(restore_command+rused,sizeof restore_command-rused," FLASH_FUSED_GR=%s",old_gr);
        char old_residual[64];value(before_config,"residual_storage","fp32",old_residual,sizeof old_residual);
        rused=strlen(restore_command);snprintf(restore_command+rused,sizeof restore_command-rused," FLASH_RESIDUAL_FP8=%d",!strncmp(old_residual,"fp8",3));
        char old_producer[16],old_group[16],old_balance[16];
        value(before_config,"producer_quant","0",old_producer,sizeof old_producer);
        value(before_config,"matrix_group","0",old_group,sizeof old_group);
        value(before_config,"balanced","0",old_balance,sizeof old_balance);
        rused=strlen(restore_command);snprintf(restore_command+rused,sizeof restore_command-rused," FLASH_PRODUCER_QUANT=%s FLASH_MATRIX_GROUP=%s FLASH_ATTN_BALANCE=%s FLASH_PHASE_WATCH=0 FLASH_WATCHDOG_S=600",old_producer,old_group,old_balance);
        char old_mode[16],old_model[1024],old_cubin[1024];
        value(before_config,"mtp_mode","0",old_mode,sizeof old_mode);
        if(atoi(old_mode)) {
            value(before_config,"mtp_model","",old_model,sizeof old_model);value(before_config,"mtp_cubin","",old_cubin,sizeof old_cubin);
            size_t used=strlen(restore_command);
            snprintf(restore_command+used,sizeof restore_command-used," FLASH_MTP_AUDIT=%s FLASH_MTP_MODEL=%s FLASH_MTP_CUBIN=%s",old_mode,old_model,old_cubin);
            char old_drafts[16],old_width[16],old_exact[1024];
            value(before_config,"mtp_drafts","1",old_drafts,sizeof old_drafts);value(before_config,"phase_batch","0",old_width,sizeof old_width);
            value(before_config,"exact_cubin","",old_exact,sizeof old_exact);if(!strcmp(old_exact,"none"))old_exact[0]=0;
            used=strlen(restore_command);
            snprintf(restore_command+used,sizeof restore_command-used," FLASH_MTP_DRAFTS=%s FLASH_PHASE_BATCH=%s FLASH_EXACT_CUBIN=%s",old_drafts,old_width,old_exact);
            char old_tile[16],old_cache[16],old_chunk_module[1024];
            value(before_config,"native_chunk","0",old_tile,sizeof old_tile);value(before_config,"chunk_maxt","16384",old_cache,sizeof old_cache);
            value(before_config,"chunk_cubin","",old_chunk_module,sizeof old_chunk_module);if(!strcmp(old_chunk_module,"none"))old_chunk_module[0]=0;
            used=strlen(restore_command);
            snprintf(restore_command+used,sizeof restore_command-used," FLASH_CHUNK_PREFILL=%s FLASH_CHUNK_MAXT=%s FLASH_CHUNK_CUBIN=%s",old_tile,old_cache,old_chunk_module);
        }
        fclose(active); active=NULL; restore_pending=1;
    }
    unsigned completed_runs=0;
    for(int arm=candidate_only?1:0;arm<2;arm++) {
        if(repeatability)have=0;
        unsigned long long state_reference[6][48]={{0}};
        unsigned char state_seen[6][48]={{0}};
        FILE *f;
        if(resident) {
            active=connect_core(); if(!active) fail("reload connection");
            char cmd[4096];
            snprintf(cmd,sizeof cmd,"reload %s FLASH_CUBIN=%s FLASH_ENGINE_MAXT=%s FLASH_MAXIMG=%s FLASH_ROPE_OWNER=%s FLASH_TWO_POSITION=%d FLASH_PREFILL_CHUNK=0 FLASH_ATTN_Q_SPLIT=%s FLASH_TRACE=0 FLASH_PLE_FILE=/home/nate/codex/flash-astra-perf-20261018/ple-direct.gguf FLASH_FUSED_MOE=%d FLASH_FAST_EXCHANGE=%d FLASH_CHUNK_PREFILL=%d FLASH_CHUNK_MAXT=8448 FLASH_CHUNK_XH=0 FLASH_CHUNK_RELAY=0 FLASH_VERIFY_STREAM=%d",argv[3],argv[separate_chunk?4:4+arm],native_chunk?"16384":maxt,native_chunk?"1":img,rope,pair_cost?arm:0,qs,(pair_cost||native_chunk||kernel_only)?0:arm,(pair_cost||native_chunk||kernel_only)?0:arm,arm?native_chunk:0,verify_stream);
            size_t used=strlen(cmd);snprintf(cmd+used,sizeof cmd-used," FLASH_PREFIX_HASH=%d FLASH_CHUNK_CUBIN=%s",prefix_hash,separate_chunk && arm?argv[5]:"");
            if(exact_batch) {
                snprintf(cmd,sizeof cmd,"reload %s FLASH_CUBIN=%s FLASH_EXACT_CUBIN=%s FLASH_PHASE_BATCH=%d FLASH_ENGINE_MAXT=16384 FLASH_MAXIMG=1 FLASH_ROPE_OWNER=%s FLASH_TWO_POSITION=1 FLASH_PREFILL_CHUNK=64 FLASH_ATTN_Q_SPLIT=%s FLASH_TRACE=0 FLASH_KV_TRACE=0 FLASH_KV_HOST=0 FLASH_LAYERS=48 FLASH_NBLK=512 FLASH_PLE_READERS=1 FLASH_PLE_FILE=/home/nate/codex/flash-astra-perf-20261018/ple-direct.gguf FLASH_FUSED_MOE=0 FLASH_FAST_EXCHANGE=0 FLASH_CHUNK_PREFILL=0 FLASH_CHUNK_CUBIN= FLASH_PREFIX_HASH=0 FLASH_VERIFY_STREAM=%d",argv[3],argv[4],arm?argv[5]:"",arm?exact_batch:0,rope,qs,verify_stream);
            }
            if(mtp_audit) {
                snprintf(cmd,sizeof cmd,"reload %s FLASH_CUBIN=%s FLASH_MTP_CUBIN=%s FLASH_MTP_MODEL=/home/nate/models-cache/flash-sass/mtp-Qwen3.8-Flash-Next-Q8_0.gguf FLASH_MTP_AUDIT=%d FLASH_PHASE_BATCH=0 FLASH_EXACT_CUBIN= FLASH_ENGINE_MAXT=16384 FLASH_MAXIMG=1 FLASH_ROPE_OWNER=%s FLASH_TWO_POSITION=0 FLASH_PREFILL_CHUNK=64 FLASH_ATTN_Q_SPLIT=%s FLASH_TRACE=0 FLASH_KV_TRACE=0 FLASH_KV_HOST=0 FLASH_LAYERS=48 FLASH_NBLK=512 FLASH_PLE_READERS=1 FLASH_PLE_FILE=/home/nate/codex/flash-astra-perf-20261018/ple-direct.gguf FLASH_FUSED_MOE=0 FLASH_FAST_EXCHANGE=0 FLASH_CHUNK_PREFILL=0 FLASH_CHUNK_CUBIN= FLASH_PREFIX_HASH=0 FLASH_VERIFY_STREAM=%d",argv[3],argv[4],arm?argv[5]:"",arm?mtp_audit:0,rope,qs,verify_stream);
            }
            if(verify_rollback) {
                snprintf(cmd,sizeof cmd,"reload %s FLASH_CUBIN=%s FLASH_EXACT_CUBIN=%s FLASH_PHASE_BATCH=%d FLASH_VERIFY_AUDIT=%d FLASH_VERIFY_STATE_CHECK=%d FLASH_MTP_AUDIT=0 FLASH_ENGINE_MAXT=16384 FLASH_MAXIMG=1 FLASH_ROPE_OWNER=%s FLASH_TWO_POSITION=0 FLASH_PREFILL_CHUNK=64 FLASH_ATTN_Q_SPLIT=%s FLASH_TRACE=0 FLASH_KV_TRACE=0 FLASH_KV_HOST=0 FLASH_LAYERS=48 FLASH_NBLK=512 FLASH_PLE_READERS=1 FLASH_PLE_FILE=/home/nate/codex/flash-astra-perf-20261018/ple-direct.gguf FLASH_FUSED_MOE=0 FLASH_FAST_EXCHANGE=0 FLASH_CHUNK_PREFILL=0 FLASH_PREFIX_HASH=0 FLASH_VERIFY_STREAM=%d",argv[3],argv[4],arm?argv[5]:"",arm?8:0,arm?verify_rollback:0,verify_states,rope,qs,verify_stream);
            }
            { size_t used=strlen(cmd);snprintf(cmd+used,sizeof cmd-used," FLASH_FUSED_GR=0"); }
            if(mtp_audit==4) {
                const char *model=getenv("FLASH_DFLASH_MODEL"),*module=getenv("FLASH_DFLASH_CUBIN");
                if(!model || !module)fail("DFlash2 requires checkpoint and SASS module paths");
                size_t used=strlen(cmd);
                snprintf(cmd+used,sizeof cmd-used," FLASH_DFLASH_MODEL=%s FLASH_DFLASH_CUBIN=%s FLASH_DFLASH_DEVICE=6",model,module);
            }
            if(mtp_audit==3 && getenv("BENCH_MTP_DRAFTS") && atoi(getenv("BENCH_MTP_DRAFTS"))==4) {
                const char *module=getenv("FLASH_EXACT_CUBIN");if(!module)fail("four-draft MTP requires target batch module");
                size_t used=strlen(cmd);snprintf(cmd+used,sizeof cmd-used," FLASH_MTP_DRAFTS=4 FLASH_PHASE_BATCH=5 FLASH_EXACT_CUBIN=%s",module);
            } else if(mtp_audit) {
                size_t used=strlen(cmd);snprintf(cmd+used,sizeof cmd-used," FLASH_MTP_DRAFTS=1");
            }
            if(mtp_audit==3 && getenv("BENCH_MTP_PREFILL") && atoi(getenv("BENCH_MTP_PREFILL"))) {
                const char *module=getenv("FLASH_CHUNK_CUBIN");if(!module)fail("MTP chunk prefill requires chunk module");
                unsigned tile=(unsigned)atoi(getenv("BENCH_MTP_PREFILL"));
                size_t used=strlen(cmd);
                snprintf(cmd+used,sizeof cmd-used," FLASH_CHUNK_PREFILL=%u FLASH_CHUNK_CUBIN=%s FLASH_CHUNK_MAXT=16384 FLASH_ENGINE_MAXT=32768 FLASH_PREFILL_CHUNK=16384",tile,module);
            }
            if(mtp_audit==3 && getenv("BENCH_FUSED_GR") && atoi(getenv("BENCH_FUSED_GR"))) {
                size_t used=strlen(cmd);snprintf(cmd+used,sizeof cmd-used," FLASH_FUSED_GR=1");
            }
            { size_t used=strlen(cmd);snprintf(cmd+used,sizeof cmd-used," FLASH_RESIDUAL_FP8=%d",getenv("BENCH_RESIDUAL_FP8") && atoi(getenv("BENCH_RESIDUAL_FP8"))); }
            if(plain_decode) {
                int phase_arm=arm || (getenv("BENCH_BASE_PHASE") && atoi(getenv("BENCH_BASE_PHASE")));
                size_t used=strlen(cmd);
                snprintf(cmd+used,sizeof cmd-used," FLASH_MTP_AUDIT=0 FLASH_MTP_DRAFTS=1 FLASH_PHASE_BATCH=0 FLASH_EXACT_CUBIN= FLASH_TWO_POSITION=1 FLASH_PREFILL_CHUNK=64 FLASH_CHUNK_PREFILL=0 FLASH_CHUNK_CUBIN= FLASH_RESIDUAL_FP8=0");
                used=strlen(cmd);snprintf(cmd+used,sizeof cmd-used," FLASH_FUSED_GR=%d FLASH_PRODUCER_QUANT=%d FLASH_MATRIX_GROUP=%d",
                    phase_arm && getenv("BENCH_FUSED_GR") && atoi(getenv("BENCH_FUSED_GR")),
                    phase_arm && getenv("BENCH_PRODUCER_QUANT") && atoi(getenv("BENCH_PRODUCER_QUANT")),
                    phase_arm && getenv("BENCH_MATRIX_GROUP") && atoi(getenv("BENCH_MATRIX_GROUP")));
                used=strlen(cmd);snprintf(cmd+used,sizeof cmd-used," FLASH_ATTN_BALANCE=%d",phase_arm && getenv("BENCH_PRODUCER_QUANT") && atoi(getenv("BENCH_PRODUCER_QUANT")));
                if(getenv("BENCH_SCALAR_PREFIX") && atoi(getenv("BENCH_SCALAR_PREFIX"))) {
                    used=strlen(cmd);snprintf(cmd+used,sizeof cmd-used," FLASH_TWO_POSITION=0");
                }
            }
            if(getenv("BENCH_CONTEXT")) {
                unsigned context=(unsigned)atoi(getenv("BENCH_CONTEXT"));
                if(!context || context>262144)fail("invalid BENCH_CONTEXT");
                size_t used=strlen(cmd);snprintf(cmd+used,sizeof cmd-used," FLASH_ENGINE_MAXT=%u FLASH_MAXIMG=1",context);
            }
            if(getenv("BENCH_PHASE_WATCH") && atoi(getenv("BENCH_PHASE_WATCH"))) {
                size_t used=strlen(cmd);snprintf(cmd+used,sizeof cmd-used," FLASH_PHASE_WATCH=1 FLASH_WATCHDOG_S=30");
            }
            {size_t used=strlen(cmd);snprintf(cmd+used,sizeof cmd-used," FLASH_XTRACE=%d",getenv("BENCH_XTRACE")?atoi(getenv("BENCH_XTRACE")):0);}
            {size_t used=strlen(cmd);snprintf(cmd+used,sizeof cmd-used," FLASH_XTRACE_FIRST=%s FLASH_XTRACE_HASH_ONLY=%d",getenv("BENCH_XTRACE_FIRST")?getenv("BENCH_XTRACE_FIRST"):"",getenv("BENCH_XTRACE_HASH_ONLY")?atoi(getenv("BENCH_XTRACE_HASH_ONLY")):0);}
            {size_t used=strlen(cmd);snprintf(cmd+used,sizeof cmd-used," FLASH_GR_REPLICATE_FFN=%d",arm&&getenv("BENCH_GR_REPLICA_FFN")?atoi(getenv("BENCH_GR_REPLICA_FFN")):0);}
            {size_t used=strlen(cmd);snprintf(cmd+used,sizeof cmd-used," FLASH_STRICT_PHASE_JOINS=%d",arm&&getenv("BENCH_STRICT_PHASE_JOINS")?atoi(getenv("BENCH_STRICT_PHASE_JOINS")):0);}
            {size_t used=strlen(cmd);snprintf(cmd+used,sizeof cmd-used," FLASH_LOCKSTEP_PHASES=%d",arm&&getenv("BENCH_LOCKSTEP_PHASES")?atoi(getenv("BENCH_LOCKSTEP_PHASES")):0);}
            {size_t used=strlen(cmd);snprintf(cmd+used,sizeof cmd-used," FLASH_GR_WRITE_TRACE=%d",getenv("BENCH_GR_WRITE_TRACE")?atoi(getenv("BENCH_GR_WRITE_TRACE")):0);}
            command(active,cmd); fclose(active); active=NULL;
            f=connect_core(); if(!f) fail("post-reload connection"); active=f;
        } else { f=start(argv[1],argv[2],argv[3+arm],arm); active=f; }
        const char *name=kernel_only?(arm?"kernel-candidate":"baseline"):verify_rollback?(arm?"rollback-audit":"baseline"):mtp_audit?(arm?(mtp_audit==4?"dflash2-full-model":mtp_audit==3?"mtp-full-model":mtp_audit==2?"mtp-rollout":"mtp-shadow"):"baseline"):pair_cost?(arm?"paired-full-head":"scalar-full-head"):exact_batch?(arm?"exact-b8":"production-b2"):(arm?(native_chunk?"native-chunk":"fused"):"baseline");
        const char *workload=mtp_audit==4?"dflash2-full-model-unchecked":mtp_audit==3?"mtp-full-model-unchecked":mtp_audit==2?"mtp-unchecked-rollout":verify_rollback?"rollback-state-diagnostic":mtp_audit?"draft-shadow-diagnostic":pair_cost?"teacher_forced":candidate_only?(generate==1&&prefix_hash?"native-prefix-repeatability":"decode-repeatability"):"decode";
        if(getenv("BENCH_CUDA_DEBUG") && atoi(getenv("BENCH_CUDA_DEBUG")))workload="cuda-debug-diagnostic";
        if(pair_cost && plain_decode)name=arm?"teacher-forced-candidate":"teacher-forced-baseline";
        if(plain_decode && getenv("BENCH_BASE_PHASE") && atoi(getenv("BENCH_BASE_PHASE")))name=arm?"split-candidate":"phase-baseline";
        printf("{\"arm\":\"%s\",\"workload\":\"%s\",\"prompt_tokens\":%d,\"generated_tokens\":%d,\"warmup_runs\":%d,\"baseline_compared\":%s,\"reference_kind\":\"%s\"}\n",name,repeatability?"same_kernel_repeatability":workload,pair_cost?positions:64*prompt_copies,pair_cost?0:generate,arm&&verify_matrix?8:1,(candidate_only||repeatability)?"false":"true",repeatability?"first_request_same_kernel":mtp_audit>=2?"none":reference?"frozen_scalar":"first_run");
        if(mtp_audit>=2)command(f,"config");
        else check_scope(f,mtp_audit?arm:-1,verify_rollback?(arm?verify_rollback:0):-1);
        command(f,"stop");
        if(getenv("BENCH_SAMPLING") && atoi(getenv("BENCH_SAMPLING")))command(f,"sample 7 1 20 0.95");
        else command(f,"sample 7 0 20 0.95");
        command(f,"timings 1");
        for(int round=0;round<(arm&&verify_matrix?8:1);round++) {
        int current_keep=arm&&verify_matrix?round+1:verify_rollback;
        if(arm&&verify_matrix) { char selection[48];snprintf(selection,sizeof selection,"verifykeep %d",current_keep);command(f,selection); }
        for(int run=0;run<repetitions;run++) {
            int run_id=4*round+run;
            printf("{\"arm\":\"%s\",\"run\":%d,\"kept\":%d,\"warmup\":%s}\n",name,run_id,arm?current_keep:0,run?"false":"true"); fflush(stdout);
            if(pair_cost) { fputs("gen 0",f); for(int i=0;i<copies;i++) fprintf(f," %s",prompt); fputc('\n',f); }
            else { fprintf(f,"chat %d",generate-1); for(int i=0;i<prompt_copies;i++) fprintf(f," %s",prompt); fputc('\n',f); }
            fflush(f);
            if(arm && !round && !run && getenv("BENCH_CUDA_DEBUG") && atoi(getenv("BENCH_CUDA_DEBUG")))start_cuda_debugger();
            char *line=NULL; size_t cap=0; int n=0,done=0; unsigned run_mismatches=0;
            while(getline(&line,&cap,f)>0) {
                fputs(line,stdout);
                unsigned token;
                if(sscanf(line,"tok %u",&token)==1) {
                    if(n>=positions) fail("too many generated tokens");
                    if(mtp_audit<2 && have && token!=expected[n]) {
                        if(run_mismatches<3) fprintf(stderr,"arm=%d run=%d token=%d got=%u expected=%u\n",arm,run_id,n,token,expected[n]);
                        run_mismatches++; mismatches++;
                        if(!collect_mismatches) fail("output token mismatch");
                    }
                    if(!have) expected[n]=token;
                    n++;
                }
                if(pair_cost && !strncmp(line,"tokens:",7)) {
                    char *p=line+7,*end;
                    while(1) {
                        unsigned long v=strtoul(p,&end,10); if(p==end) break; p=end;
                        if(n>=positions || v>=248320) fail("invalid teacher-forced output");
                        if(have && v!=expected[n]) {
                            if(run_mismatches<3)fprintf(stderr,"arm=%d run=%d fixed_position=%d got=%lu expected=%u\n",arm,run_id,n,v,expected[n]);
                            run_mismatches++;mismatches++;
                            if(!collect_mismatches)fail("paired/scalar full-head output mismatch");
                        }
                        if(!have) expected[n]=(unsigned)v;
                        n++;
                    }
                }
                if(!strncmp(line,"error",5)) fail("engine error");
                if((!pair_cost&&!strncmp(line,"done ",5)) || (pair_cost&&!strncmp(line,"time: ",6))) { done=1; break; }
            }
            free(line); fflush(stdout);
            if(!done||n!=positions) fail("incomplete generation");
            if(getenv("BENCH_STATE_HASH") && atoi(getenv("BENCH_STATE_HASH"))) {
                fputs("statehash\n",f);fflush(f);char *hl=NULL;size_t hc=0;int hd=0;
                unsigned state_rows=0,state_errors=0;
                while(getline(&hl,&hc,f)>0){
                    fputs(hl,stdout);if(!strcmp(hl,"statehash done\n")){hd=1;break;}
                    int sd,sl;unsigned long long sh;
                    /* GDN state has no unused positional padding. QSA's combined
                     * hash includes stale padding, so it is not an equality oracle. */
                    if(sscanf(hl,"statehash die=%d layer=%d state=%llx",&sd,&sl,&sh)==3 && sd>=0 && sd<6 && sl>=0 && sl<48 && sl%4!=3){
                        state_rows++;
                        if(state_seen[sd][sl] && state_reference[sd][sl]!=sh){state_errors++;mismatches++;}
                        if(!state_seen[sd][sl]){state_reference[sd][sl]=sh;state_seen[sd][sl]=1;}
                    }
                }
                free(hl);if(!hd)fail("incomplete state hash readback");
                if(state_rows!=216)fail("incomplete full-model GDN state coverage");
                printf("stateaudit arm=%s run=%d rows=%u mismatches=%u reference=first_request_same_kernel\n",name,run_id,state_rows,state_errors);
            }
            if(mtp_audit>=2)printf("{\"arm\":\"%s\",\"run\":%d,\"correctness_checked\":false,\"generated_tokens\":%d}\n",name,run_id,n);
            else printf("{\"arm\":\"%s\",\"run\":%d,\"output_mismatches\":%u,\"correct\":%s}\n",name,run_id,run_mismatches,run_mismatches?"false":"true");
            if(exact_batch) {
                char stats[256];unsigned width;int batched;
                fputs("batchstats\n",f);fflush(f);
                if(!fgets(stats,sizeof stats,f) || sscanf(stats,"batchstats width=%u positions=%d",&width,&batched)!=2)
                    fail("batch statistics unavailable");
                fputs(stats,stdout);
                if(width!=(unsigned)(arm?exact_batch:0) || (arm && batched<=0) || (!arm && batched))
                    fail("exact batch execution count does not match requested mode");
            }
            if(mtp_audit>=2) {
                char stats[512];fputs("mtpstats\n",f);fflush(f);
                if(fgets(stats,sizeof stats,f))fputs(stats,stdout);
            } else if(mtp_audit) {
                char stats[512];unsigned steps,proposals,matches;int enabled;
                fputs("mtpstats\n",f);fflush(f);
                if(!fgets(stats,sizeof stats,f) || sscanf(stats,"mtpstats shadow=1 enabled=%d steps=%u one_step_proposals=%u one_step_matches=%u",&enabled,&steps,&proposals,&matches)!=4)
                    fail("MTP shadow statistics unavailable");
                fputs(stats,stdout);
                if(enabled!=arm || matches>proposals || (!arm && (steps||proposals)) ||
                   (arm && (steps!=(unsigned)(64*prompt_copies+generate-1) || proposals!=(unsigned)generate)))
                    fail("MTP shadow coverage does not match the target request");
            }
            if(verify_rollback) {
                char stats[512];unsigned kept,executed,replayed,changed;int state_check;
                fputs("verifystats\n",f);fflush(f);
                if(!fgets(stats,sizeof stats,f) || sscanf(stats,"verifystats kept=%u window_start=16 width=8 verified_positions=%u replayed_positions=%u state_check=%d changed_buffers=%u",&kept,&executed,&replayed,&state_check,&changed)!=5)
                    fail("rollback statistics unavailable");
                fputs(stats,stdout);
                if(kept!=(unsigned)(arm?current_keep:0) || executed!=(unsigned)(arm?8:0) || replayed!=(unsigned)(arm?8-current_keep:0) || (arm && state_check!=verify_states))
                    fail("rollback coverage does not match request");
                prefix_mismatches+=changed;
            }
            if(prefix_hash && native_chunk && arm) {
                char hashline[512];unsigned changed;
                fputs("chunkhash\n",f);fflush(f);
                if(!fgets(hashline,sizeof hashline,f)) fail("prefix fingerprint EOF");
                fputs(hashline,stdout);
                char *p=strstr(hashline,"changed_buffers=");
                if(!p || sscanf(p,"changed_buffers=%u",&changed)!=1) fail("prefix fingerprint unavailable");
                prefix_mismatches+=changed;
            }
            have=1;completed_runs++;
            if(repeatability && !pair_cost) {
                fputs("tokenaudit\n",f);fflush(f);char *audit_line=NULL;size_t audit_cap=0;int complete=0;
                while(getline(&audit_line,&audit_cap,f)>0) {
                    fputs(audit_line,stdout);
                    if(!strncmp(audit_line,"error:",6))fail("tokenaudit unavailable");
                    if(!strncmp(audit_line,"tokenaudit_end",14)){
                        unsigned audit_count,stream_bad,peer_bad,input_bad;
                        if(sscanf(audit_line,"tokenaudit_end count=%u stream_mismatches=%u peer_mismatches=%u input_mismatches=%u",&audit_count,&stream_bad,&peer_bad,&input_bad)!=4 || audit_count!=(unsigned)generate)fail("tokenaudit coverage");
                        mismatches+=stream_bad+peer_bad+input_bad;complete=1;break;
                    }
                }
                free(audit_line);if(!complete)fail("tokenaudit incomplete");
            }
            if(getenv("BENCH_XTRACE_EACH") && atoi(getenv("BENCH_XTRACE_EACH")))exchange_dump(f,name,run_id);
        }
        }
        if(getenv("BENCH_PROFILE") && atoi(getenv("BENCH_PROFILE")))for(int d=0;d<6;d++) {
            printf("profile arm=%s die=%d\n",name,d);fprintf(f,"prof %d all\n",d);fflush(f);
            char *row=NULL;size_t capacity=0;
            while(getline(&row,&capacity,f)>0) { fputs(row,stdout);if(!strncmp(row,"  longest:",10))break; }
            free(row);
        }
        if(getenv("BENCH_XTRACE") && atoi(getenv("BENCH_XTRACE")) && !(getenv("BENCH_XTRACE_EACH") && atoi(getenv("BENCH_XTRACE_EACH"))))exchange_dump(f,name,-1);
        if(!resident) { fputs("quit\n",f); fflush(f); }
        fclose(f); active=NULL;
        if(!resident) {
            int status; waitpid(child,&status,0); child=-1; unlink(socket_path);
            if(!WIFEXITED(status)||WEXITSTATUS(status)) fail("private core exit");
        }
    }
    if(getenv("BENCH_KEEP_LOADED") && atoi(getenv("BENCH_KEEP_LOADED")) && (mtp_audit>=3 || plain_decode) && candidate_only) { restore_pending=0;puts("deployment: measured full-model engine remains loaded"); }
    if(restore()) return 1;
    if(prefix_mismatches) { fprintf(stderr,"resident_decode_bench: FAILED, %u prefix-state mismatches\n",prefix_mismatches);return 1; }
    if(mismatches) { fprintf(stderr,"resident_decode_bench: FAILED, %u output/state/consensus mismatches; timings are diagnostic only\n",mismatches); return 1; }
    if(repeatability && pair_cost)printf("resident_decode_bench: same-kernel teacher-forced output repeatability PASS; no cross-kernel oracle or stream comparison\n");
    else if(repeatability)printf("resident_decode_bench: same-kernel repeatability and GPU/stream consensus PASS; no cross-kernel oracle comparison\n");
    else if(mtp_audit>=2)printf("resident_decode_bench: unchecked MTP execution completed; no target verification, acceptance, or output comparison\n");
    else if(candidate_only && reference) printf("resident_decode_bench: %u %d-position output(s) match frozen scalar reference; no matched live A/B\n",completed_runs,positions);
    else if(candidate_only) printf("resident_decode_bench: candidate repeatability PASS; no independent output reference or matched live A/B\n");
    else printf("resident_decode_bench: all %u %d-position outputs match\n",completed_runs,positions);
    return 0;
}
