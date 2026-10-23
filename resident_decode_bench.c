/* Reuse the GPUQ goal-request-fixtures 64-token prompt with private cores and
 * the existing IPC weight holders. No weight loader or serving reload. */
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

static pid_t child=-1;
static char socket_path[108];
static FILE *active;
static char restore_command[4096],before_config[4096];
static int restore_pending;
static FILE *connect_core(void) {
    int fd=socket(AF_UNIX,SOCK_STREAM,0); if(fd<0) return NULL;
    struct sockaddr_un sa={.sun_family=AF_UNIX}; strcpy(sa.sun_path,socket_path);
    if(connect(fd,(struct sockaddr *)&sa,sizeof sa)) { close(fd); return NULL; }
    struct timeval tv={.tv_sec=120};
    setsockopt(fd,SOL_SOCKET,SO_RCVTIMEO,&tv,sizeof tv);
    setsockopt(fd,SOL_SOCKET,SO_SNDTIMEO,&tv,sizeof tv);
    FILE *f=fdopen(fd,"r+"); if(!f) { close(fd); return NULL; }
    setvbuf(f,NULL,_IOLBF,0); return f;
}
static int restore(void) {
    if(!restore_pending) return 0;
    restore_pending=0;
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
static void value(const char *cfg,const char *key,const char *fallback,char *dst,size_t size) {
    char token[80]; snprintf(token,sizeof token," %s=",key);
    const char *p=strstr(cfg,token); if(p) p+=strlen(token); else p=fallback;
    size_t n=strcspn(p," \r\n"); if(n>=size) fail("config field too long");
    memcpy(dst,p,n); dst[n]=0;
}
int main(int argc,char **argv) {
    setvbuf(stdout,NULL,_IONBF,0);
    int resident=argc==8 && !strcmp(argv[1],"--resident");
    if(argc!=5 && !resident) { fprintf(stderr,"usage: resident_decode_bench CORE LIB BASE_CUBIN CANDIDATE_CUBIN\n       resident_decode_bench --resident SOCKET LIB BASE_CUBIN CANDIDATE_CUBIN RESTORE_LIB RESTORE_CUBIN\n"); return 2; }
    if(!getenv("GPUQ_JOB_ID")) fail("run inside gpuq");
    char maxt[32]="16384",img[32]="4096",rope[32]="5",pairs[32]="0",chunk[32]="0",qs[32]="1",trace[32]="0",ple[1024]="";
    if(resident) {
        if(strlen(argv[2])>=sizeof socket_path) fail("socket path too long");
        strcpy(socket_path,argv[2]); active=connect_core(); if(!active) fail("resident connection");
        fputs("config\n",active); fflush(active);
        if(!fgets(before_config,sizeof before_config,active)) fail("initial config EOF");
        printf("before: %s",before_config);
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
        if(!strcmp(ple,"core")) *ple=0;
        snprintf(restore_command,sizeof restore_command,"reload %s FLASH_CUBIN=%s FLASH_ENGINE_MAXT=%s FLASH_MAXIMG=%s FLASH_ROPE_OWNER=%s FLASH_TWO_POSITION=%s FLASH_PREFILL_CHUNK=%s FLASH_ATTN_Q_SPLIT=%s FLASH_TRACE=%s FLASH_PLE_FILE=%s FLASH_FUSED_MOE=0 FLASH_FAST_EXCHANGE=0",argv[6],argv[7],maxt,img,rope,pairs,chunk,qs,trace,ple);
        fclose(active); active=NULL; restore_pending=1;
    }
    unsigned expected[128]; int have=0;
    for(int arm=0;arm<2;arm++) {
        FILE *f;
        if(resident) {
            active=connect_core(); if(!active) fail("reload connection");
            char cmd[4096];
            snprintf(cmd,sizeof cmd,"reload %s FLASH_CUBIN=%s FLASH_ENGINE_MAXT=%s FLASH_MAXIMG=%s FLASH_ROPE_OWNER=%s FLASH_TWO_POSITION=0 FLASH_PREFILL_CHUNK=0 FLASH_ATTN_Q_SPLIT=%s FLASH_TRACE=0 FLASH_PLE_FILE=/home/nate/codex/flash-astra-perf-20261018/ple-direct.gguf FLASH_FUSED_MOE=%d FLASH_FAST_EXCHANGE=%d",argv[3],argv[4+arm],maxt,img,rope,qs,arm,arm);
            command(active,cmd); fclose(active); active=NULL;
            f=connect_core(); if(!f) fail("post-reload connection"); active=f;
        } else { f=start(argv[1],argv[2],argv[3+arm],arm); active=f; }
        printf("{\"arm\":\"%s\",\"prompt_tokens\":64,\"generated_tokens\":128,\"warmup_runs\":1}\n",arm?"fused":"baseline");
        command(f,"config"); command(f,"stop"); command(f,"sample 7 0 20 0.95"); command(f,"timings 1");
        for(int run=0;run<4;run++) {
            printf("{\"arm\":\"%s\",\"run\":%d,\"warmup\":%s}\n",arm?"fused":"baseline",run,run?"false":"true"); fflush(stdout);
            fprintf(f,"chat 127 %s\n",prompt); fflush(f);
            char *line=NULL; size_t cap=0; int n=0,done=0;
            while(getline(&line,&cap,f)>0) {
                fputs(line,stdout);
                unsigned token;
                if(sscanf(line,"tok %u",&token)==1) {
                    if(n>=128) fail("too many generated tokens");
                    if(have && token!=expected[n]) { fprintf(stderr,"arm=%d run=%d token=%d got=%u expected=%u\n",arm,run,n,token,expected[n]); fail("output token mismatch"); }
                    if(!have) expected[n]=token;
                    n++;
                }
                if(!strncmp(line,"error",5)) fail("engine error");
                if(!strncmp(line,"done ",5)) { done=1; break; }
            }
            free(line); fflush(stdout);
            if(!done||n!=128) fail("incomplete generation");
            have=1;
        }
        if(!resident) { fputs("quit\n",f); fflush(f); }
        fclose(f); active=NULL;
        if(!resident) {
            int status; waitpid(child,&status,0); child=-1; unlink(socket_path);
            if(!WIFEXITED(status)||WEXITSTATUS(status)) fail("private core exit");
        }
    }
    if(restore()) return 1;
    puts("resident_decode_bench: all eight 128-token outputs match"); return 0;
}
