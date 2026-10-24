/* Keep an existing PLE file in the host page cache. The serving engine keeps
 * its fd and compact CUDA staging allocation; its preads hit these same pages.
 * No CUDA context, device allocation, model copy, or data modification. */
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

static volatile sig_atomic_t stopping;
static void stop(int signal_number) { (void)signal_number;stopping=1; }
int main(int argc,char **argv) {
    if (argc!=3) {fprintf(stderr,"usage: %s PLE_FILE READY_FILE\n",argv[0]);return 2;}
    int fd=open(argv[1],O_RDONLY|O_CLOEXEC);
    if (fd<0) {perror("open PLE");return 1;}
    struct stat st;
    if (fstat(fd,&st) || !S_ISREG(st.st_mode) || st.st_size<=0) {
        fprintf(stderr,"PLE must be a nonempty regular file\n");close(fd);return 1;
    }
    size_t bytes=(size_t)st.st_size;
    void *mapping=mmap(NULL,bytes,PROT_READ,MAP_SHARED,fd,0);
    if (mapping==MAP_FAILED) {perror("mmap PLE");close(fd);return 1;}
    struct sigaction action={.sa_handler=stop};sigemptyset(&action.sa_mask);
    sigaction(SIGTERM,&action,NULL);sigaction(SIGINT,&action,NULL);
    fprintf(stderr,"PLE residency: locking %zu bytes from %s\n",bytes,argv[1]);
    if (mlock(mapping,bytes)) {perror("mlock PLE");munmap(mapping,bytes);close(fd);return 1;}
    if (stopping) {munlock(mapping,bytes);munmap(mapping,bytes);close(fd);return 1;}
    FILE *ready=fopen(argv[2],"wx");
    if (!ready) {perror("create ready file");munlock(mapping,bytes);munmap(mapping,bytes);close(fd);return 1;}
    int failed=fprintf(ready,"pid=%ld bytes=%zu device=%llu inode=%llu\n",(long)getpid(),bytes,
        (unsigned long long)st.st_dev,(unsigned long long)st.st_ino)<0;
    if (fclose(ready)) failed=1;
    if (failed) {perror("write ready file");munlock(mapping,bytes);munmap(mapping,bytes);close(fd);return 1;}
    fprintf(stderr,"PLE residency: ready; all %zu bytes locked until this process exits\n",bytes);
    sigset_t blocked,previous;sigemptyset(&blocked);sigaddset(&blocked,SIGTERM);sigaddset(&blocked,SIGINT);
    sigprocmask(SIG_BLOCK,&blocked,&previous);
    while (!stopping) sigsuspend(&previous);
    sigprocmask(SIG_SETMASK,&previous,NULL);
    unlink(argv[2]);munlock(mapping,bytes);munmap(mapping,bytes);close(fd);
    return 0;
}
