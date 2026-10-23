#define _GNU_SOURCE
#define FLASH_PLE_IO_TEST 1
#include "ple_io.h"
#include <assert.h>
#include <fcntl.h>
#include <stdio.h>
int main(void){
 char name[]="/tmp/flash-ple-io-XXXXXX";int fd=mkstemp(name);assert(fd>=0);assert(!unlink(name));
 unsigned char expected[90],actual[90];for(int i=0;i<90;i++)expected[i]=(unsigned char)(3*i+1);
 assert(write(fd,expected,90)==90);ple_io_error e={0};
 const char*ok[]={"none","partial","eintr"};
 for(int k=0;k<3;k++){assert(!ple_io_test_set(ok[k],7,3,17));memset(actual,0,90);assert(!ple_read_exact(fd,actual,90,0,7,3,&e));assert(!memcmp(actual,expected,90));}
 const char*bad[]={"eof","eio"};
 for(int k=0;k<2;k++){assert(!ple_io_test_set(bad[k],7,3,17));memset(actual,0xaa,90);assert(ple_read_exact(fd,actual,90,0,7,3,&e)==-1);assert(e.position==7&&e.head==3&&e.offset==0&&e.bytes==17&&e.expected==90);assert(e.eof==(k==0)&&e.errnum==(k==0?0:EIO));assert(!memcmp(actual,expected,17)&&actual[17]==0xaa);assert(!ple_read_exact(fd,actual,90,0,8,3,&e));assert(!memcmp(actual,expected,90));}
 assert(ple_io_test_set("unknown",0,0,0)==-1);assert(ple_io_test_set("eio",0,16,0)==-1);assert(ple_io_test_set("eio",0,0,90)==-1);assert(!ple_io_test_set("none",0,0,0));
 assert(!ftruncate(fd,17));assert(ple_read_exact(fd,actual,90,10,2,4,&e)==-1);assert(e.eof&&e.bytes==7&&e.offset==10);assert(ple_read_exact(-1,actual,90,0,2,4,&e)==-1);assert(!e.eof&&e.errnum==EBADF&&e.bytes==0);assert(!close(fd));
 puts("ple_io: partial reads, EINTR, EOF, EIO and failure metadata passed");return 0;
}
