#include <nvvm.h>
#include <stdio.h>
#include <stdlib.h>
int main(int c,char**v){FILE*f=fopen(v[1],"rb");fseek(f,0,SEEK_END);long n=ftell(f);rewind(f);char*s=malloc(n+1);fread(s,1,n,f);s[n]=0;
nvvmProgram p;nvvmCreateProgram(&p);nvvmAddModuleToProgram(p,s,n,v[1]);const char*o[]={"-arch=compute_52"};if(nvvmCompileProgram(p,1,o))return 1;
size_t m;nvvmGetCompiledResultSize(p,&m);char*x=malloc(m);nvvmGetCompiledResult(p,x);fputs(x,stdout);return 0;}
