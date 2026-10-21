#define _GNU_SOURCE
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <signal.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <time.h>

int main(int argc,char **argv) {
    setvbuf(stdout,NULL,_IONBF,0);
    if(argc!=5||!getenv("GPUQ_JOB_ID")) { fprintf(stderr,"usage inside gpuq: core_recover OLD_PID CORE LIB CUBIN\n"); return 2; }
    pid_t old=(pid_t)strtol(argv[1],NULL,10);
    if(old<=1) return 2;
    char path[128],cmd[4096]; snprintf(path,sizeof path,"/proc/%ld/cmdline",(long)old);
    struct stat st; if(stat(path,&st)||st.st_uid!=geteuid()) { perror("old core identity"); return 1; }
    FILE *f=fopen(path,"r"); if(!f) return 1;
    size_t n=fread(cmd,1,sizeof cmd-1,f); fclose(f); cmd[n]=0;
    const char *base=strrchr(cmd,'/'); base=base?base+1:cmd;
    if(strcmp(base,"flashcore")) { fprintf(stderr,"refusing non-core PID\n"); return 1; }
    printf("stopping stalled core PID %ld; holders untouched\n",(long)old);
    if(kill(old,SIGKILL)) { perror("kill core"); return 1; }
    for(int i=0;i<300 && !kill(old,0);i++) { struct timespec t={.tv_nsec=100000000}; nanosleep(&t,0); }
    if(!kill(old,0)) { fprintf(stderr,"old core still present\n"); return 1; }
    char unit[128],cubin[1200];
    snprintf(unit,sizeof unit,"--unit=flash-sass-recovered-%ld",(long)getpid());
    snprintf(cubin,sizeof cubin,"FLASH_CUBIN=%s",argv[4]);
    pid_t p=fork(); if(p<0) return 1;
    if(!p) {
        execlp("systemd-run","systemd-run","--user",unit,"--collect",
               "--property=Restart=on-failure","--property=RestartSec=5",
               "env","CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6","FLASH_ANCHOR_DIE=6",
               "FLASH_SOCK=/tmp/flash.sock","FLASH_MAXT=262144","FLASH_ENGINE_MAXT=230000",
               "FLASH_MAXIMG=4096","FLASH_PLE_DISK=1","FLASH_PLE_OFF=192","FLASH_PLE_READERS=1",
               "FLASH_PLE_FILE=/home/nate/codex/flash-astra-perf-20261018/ple-direct.gguf",
               "FLASH_TWO_POSITION=1","FLASH_PREFILL_CHUNK=64","FLASH_KV_HOST=0",
               "FLASH_KV_TRACE=0","FLASH_TRACE=0","FLASH_ROPE_OWNER=5","FLASH_ATTN_Q_SPLIT=1",
               "FLASH_FUSED_MOE=0","FLASH_FAST_EXCHANGE=0",
               "SPLIT_SHARES=112,97,81,116,97,97","ATTN_MAP=1,1,1,1,4,4,4,4,5,5,5,5",cubin,
               argv[2],"/home/nate/codex/flash-astra-perf-20261018/ple-direct.gguf",argv[3],
               "/tmp/hold-h0.manifest","/tmp/hold-h1.manifest","/tmp/hold-h2.manifest",
               "/tmp/hold-h3.manifest","/tmp/hold-h4.manifest","/tmp/hold-h5.manifest",(char *)0);
        perror("systemd-run"); _exit(127);
    }
    int status; waitpid(p,&status,0); if(!WIFEXITED(status)||WEXITSTATUS(status)) return 1;
    for(int i=0;i<600;i++) {
        int fd=socket(AF_UNIX,SOCK_STREAM,0);
        struct sockaddr_un sa={.sun_family=AF_UNIX}; strcpy(sa.sun_path,"/tmp/flash.sock");
        if(fd>=0 && !connect(fd,(struct sockaddr *)&sa,sizeof sa)) {
            struct ucred peer; socklen_t len=sizeof peer;
            if(getsockopt(fd,SOL_SOCKET,SO_PEERCRED,&peer,&len)||peer.pid==old||peer.uid!=geteuid()) { close(fd); return 1; }
            struct timeval tv={.tv_sec=10}; setsockopt(fd,SOL_SOCKET,SO_RCVTIMEO,&tv,sizeof tv);
            FILE *s=fdopen(fd,"r+"); if(!s) { close(fd); return 1; }
            fputs("config\n",s); fflush(s);
            char line[4096]; int ok=fgets(line,sizeof line,s)&&!strncmp(line,"config ",7)&&strstr(line,argv[4]);
            if(ok) printf("recovered core PID %ld via %s\n%s",(long)peer.pid,unit,line);
            fclose(s); return ok?0:1;
        }
        if(fd>=0) close(fd);
        struct timespec t={.tv_nsec=100000000}; nanosleep(&t,0);
    }
    fprintf(stderr,"recovered core did not become ready; inspect %s\n",unit); return 1;
}
