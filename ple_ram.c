/* A read-only RAM file for PLE preads. Fill and lock chunks incrementally so
 * source cache pages can be reclaimed without creating disk writeback. */
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
static volatile sig_atomic_t stopping;
static void stop(int sig) { (void)sig;stopping=1; }
static uint64_t available(void) {
    FILE *f=fopen("/proc/meminfo","r");if(!f)return 0;
    char line[256];unsigned long long kb=0;
    while(fgets(line,sizeof line,f))if(sscanf(line,"MemAvailable: %llu kB",&kb)==1)break;
    fclose(f);return kb*1024;
}
int main(int argc,char **argv) {
    if(argc!=3){fprintf(stderr,"usage: %s PLE_FILE READY_FILE\n",argv[0]);return 2;}
    int source=-1,cache=-1,ready_written=0,result=1;
    unsigned char *data=MAP_FAILED;size_t bytes=0;
    struct stat st;
    source=open(argv[1],O_RDONLY|O_CLOEXEC);
    if(source<0 || fstat(source,&st) || !S_ISREG(st.st_mode) || st.st_size<=0){perror("PLE source");goto out;}
    bytes=(size_t)st.st_size;
    const uint64_t reserve=2ull<<30;
    if(available()<(uint64_t)bytes+reserve){fprintf(stderr,"insufficient available RAM for PLE plus 2 GiB reserve\n");goto out;}
    cache=memfd_create("flash-ple-ram",MFD_CLOEXEC|MFD_ALLOW_SEALING);
    if(cache<0 || ftruncate(cache,(off_t)bytes)){perror("PLE RAM file");goto out;}
    data=mmap(NULL,bytes,PROT_READ|PROT_WRITE,MAP_SHARED,cache,0);
    if(data==MAP_FAILED){perror("map PLE RAM file");goto out;}
    struct sigaction action={.sa_handler=stop};sigemptyset(&action.sa_mask);
    sigaction(SIGTERM,&action,NULL);sigaction(SIGINT,&action,NULL);
    const size_t chunk=64u<<20;
    for(size_t offset=0;offset<bytes && !stopping;offset+=chunk) {
        size_t n=bytes-offset<chunk?bytes-offset:chunk,done=0;
        if(available()<reserve+n){fprintf(stderr,"RAM reserve reached while loading PLE\n");goto out;}
        while(done<n && !stopping) {
            ssize_t got=pread(source,data+offset+done,n-done,(off_t)(offset+done));
            if(got<0 && errno==EINTR)continue;
            if(got<=0){fprintf(stderr,"PLE read at %zu: %s\n",offset+done,got<0?strerror(errno):"unexpected EOF");goto out;}
            done+=(size_t)got;
        }
        if(stopping)goto out;
        if(mlock(data+offset,n)){perror("lock PLE RAM chunk");goto out;}
        if((offset/chunk)%16==0){fprintf(stderr,"PLE RAM: %zu / %zu bytes filled and locked\n",offset+n,bytes);fflush(stderr);}
    }
    if(stopping)goto out;
    size_t pages=(bytes+(size_t)getpagesize()-1)/(size_t)getpagesize();
    unsigned char *resident=malloc(pages);
    if(!resident || mincore(data,bytes,resident)){free(resident);perror("PLE residency check");goto out;}
    size_t missing=0;for(size_t i=0;i<pages;i++)missing+=!(resident[i]&1);free(resident);
    if(missing){fprintf(stderr,"PLE RAM: %zu pages missing\n",missing);goto out;}
    if(mprotect(data,bytes,PROT_READ) || fcntl(cache,F_ADD_SEALS,F_SEAL_GROW|F_SEAL_SHRINK|F_SEAL_FUTURE_WRITE|F_SEAL_SEAL)) {
        perror("seal PLE RAM file");goto out;
    }
    close(source);source=-1;
    FILE *ready=fopen(argv[2],"wx");
    if(!ready){perror("create PLE ready file");goto out;}
    ready_written=1;
    int failed=fprintf(ready,"/proc/%ld/fd/%d\nbytes=%zu pages=%zu source_device=%llu source_inode=%llu\n",
        (long)getpid(),cache,bytes,pages,(unsigned long long)st.st_dev,(unsigned long long)st.st_ino)<0;
    if(fclose(ready))failed=1;
    if(failed){perror("write PLE ready file");goto out;}
    fprintf(stderr,"PLE RAM: ready, %zu bytes and %zu resident pages, fd /proc/%ld/fd/%d\n",bytes,pages,(long)getpid(),cache);
    sigset_t blocked,prior;sigemptyset(&blocked);sigaddset(&blocked,SIGTERM);sigaddset(&blocked,SIGINT);
    sigprocmask(SIG_BLOCK,&blocked,&prior);
    while(!stopping)sigsuspend(&prior);
    sigprocmask(SIG_SETMASK,&prior,NULL);result=0;
out:
    if(ready_written)unlink(argv[2]);
    if(data!=MAP_FAILED){munlock(data,bytes);munmap(data,bytes);}
    if(cache>=0)close(cache);if(source>=0)close(source);
    return result;
}
