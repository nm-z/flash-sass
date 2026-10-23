#ifndef FLASH_PLE_IO_H
#define FLASH_PLE_IO_H
#include <errno.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>
typedef struct { int position, head, errnum, eof; off_t offset; size_t bytes, expected; } ple_io_error;
#ifdef FLASH_PLE_IO_TEST
enum { PLE_TEST_NONE, PLE_TEST_PARTIAL, PLE_TEST_EINTR, PLE_TEST_EOF, PLE_TEST_EIO };
typedef struct { int mode, position, head; size_t after; } ple_io_test_plan;
static ple_io_test_plan ple_io_plan;
static inline int ple_io_test_set(const char *mode,int position,int head,size_t after) {
 int m=PLE_TEST_NONE;
 if(!mode||!strcmp(mode,"none"))m=PLE_TEST_NONE;
 else if(!strcmp(mode,"partial"))m=PLE_TEST_PARTIAL;
 else if(!strcmp(mode,"eintr"))m=PLE_TEST_EINTR;
 else if(!strcmp(mode,"eof"))m=PLE_TEST_EOF;
 else if(!strcmp(mode,"eio"))m=PLE_TEST_EIO;
 else return -1;
 if(position<0||head<0||head>=16||after>=90)return -1;
 ple_io_plan=(ple_io_test_plan){m,position,head,after};return 0;
}
static inline int ple_io_test_configure(void) {
 const char *mode=getenv("FLASH_PLE_IO_TEST_FAULT"),*pos=getenv("FLASH_PLE_IO_TEST_POS"),*head=getenv("FLASH_PLE_IO_TEST_HEAD"),*after=getenv("FLASH_PLE_IO_TEST_AFTER");
 return ple_io_test_set(mode,pos?atoi(pos):0,head?atoi(head):0,after?(size_t)strtoul(after,NULL,10):0);
}
static inline ssize_t ple_io_pread(int fd,void *dst,size_t count,off_t offset,int position,int head,size_t got,int *injected) {
 if(ple_io_plan.mode&&ple_io_plan.position==position&&ple_io_plan.head==head) {
  if(got<ple_io_plan.after){if(count>ple_io_plan.after-got)count=ple_io_plan.after-got;}
  else if(ple_io_plan.mode==PLE_TEST_PARTIAL){if(count>7)count=7;}
  else if(ple_io_plan.mode==PLE_TEST_EINTR&&!*injected){*injected=1;errno=EINTR;return -1;}
  else if(ple_io_plan.mode==PLE_TEST_EOF)return 0;
  else if(ple_io_plan.mode==PLE_TEST_EIO){errno=EIO;return -1;}
 }
 return pread(fd,dst,count,offset);
}
#endif
/* A row becomes ready only after all heads succeed; failed tails are not filled. */
static inline int ple_read_exact(int fd,void *dst,size_t expected,off_t offset,int position,int head,ple_io_error *error) {
 size_t got=0;
#ifdef FLASH_PLE_IO_TEST
 int injected=0;
#endif
 while(got<expected) {
  ssize_t n;
#ifdef FLASH_PLE_IO_TEST
  n=ple_io_pread(fd,(unsigned char*)dst+got,expected-got,offset+(off_t)got,position,head,got,&injected);
#else
  n=pread(fd,(unsigned char*)dst+got,expected-got,offset+(off_t)got);
#endif
  if(n>0){got+=(size_t)n;continue;}
  if(n<0&&errno==EINTR)continue;
  if(error)*error=(ple_io_error){.position=position,.head=head,.offset=offset,.bytes=got,.expected=expected,.eof=n==0,.errnum=n==0?0:(errno?errno:EIO)};
  return -1;
 }
 return 0;
}
#endif
