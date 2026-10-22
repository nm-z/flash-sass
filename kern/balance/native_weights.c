/* Preserve resident native weights, then redistribute their exact packed rows.
 * No model kernels or arithmetic are executed. Original cache arenas also give
 * the holder loader an NFS-independent restoration path. All GPU invocations
 * of save/load belong in gpuq; plan/balance are CPU-only operations. */
#define _POSIX_C_SOURCE 200809L
#include "../hold.h"
#include "../prog.h"
#include "split.h"
#include "gguf.h"
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>
static const char *suffix(const char *name) {
    if(strncmp(name,"blk.",4))return NULL;
    const char *p=strchr(name+4,'.');return p?p+1:NULL;
}
#include "../hold_selection.h"

CUresult cuCtxCreate_v2(CUcontext *,unsigned,CUdevice);
static void die(const char *s) { fprintf(stderr,"native_weights: %s: %s\n",s,strerror(errno));exit(1); }
static void require(int yes,const char *s) { if(!yes){fprintf(stderr,"native_weights: %s\n",s);exit(1);} }
#define CK(x) do { CUresult r=(x);if(r!=CUDA_SUCCESS){const char *s="?";cuGetErrorString(r,&s);fprintf(stderr,"native_weights: %s: %s\n",#x,s);exit(1);} }while(0)
static void path(char p[1024],const char *dir,const char *file) {
    require(snprintf(p,1024,"%s/%s",dir,file)<1024,"path too long");
}
static int fresh(const char *p) { int fd=open(p,O_WRONLY|O_CREAT|O_EXCL,0600);if(fd<0)die(p);return fd; }
static void newdir(const char *p) { if(mkdir(p,0700))die(p); }
static void write_at(int fd,const void *v,size_t n,size_t off) {
    const unsigned char *p=v;
    while(n){ssize_t z=pwrite(fd,p,n,(off_t)off);if(z<0 && errno==EINTR)continue;if(z<=0)die("pwrite");p+=z;n-=(size_t)z;off+=(size_t)z;}
}
static void finish(int fd) { if(fsync(fd)||close(fd))die("finish cache"); }
static void metadata_read(hold_arena *a,const char *p,int *logical) {
    FILE *f=fopen(p,"r");if(!f)die(p);
    char *line=NULL;size_t cap=0;memset(a,0,sizeof *a);*logical=-1;
    require(getline(&line,&cap,f)>0,"empty metadata");
    unsigned version=0;
    if(sscanf(line,"native_weights %u bytes %zu physical %d logical %d",&version,&a->bytes,&a->die,logical)==4)
        require(version==1,"unsupported native cache version");
    else require(sscanf(line,"arena %zu die %d",&a->bytes,&a->die)==2,"invalid source manifest");
    while(getline(&line,&cap,f)>0) {
        hold_tensor t;require(sscanf(line,"%95s %zu %zu %u %d %d %d %d",t.name,&t.off,&t.bytes,&t.type,&t.rows,&t.K,&t.stride,&t.experts)==8,"invalid tensor metadata");
        require(t.off<=a->bytes && t.bytes<=a->bytes-t.off,"tensor outside arena");
        require(t.rows>=0 && t.K>=0 && t.stride>=0 && t.experts>=0,"negative tensor shape");
        require(!hold_find(a,t.name),"duplicate tensor metadata");
        hold_tensor *q=realloc(a->t,(size_t)(a->n+1)*sizeof *q);if(!q)die("metadata allocation");a->t=q;a->t[a->n++]=t;
    }
    require(!ferror(f),"metadata read failed");free(line);fclose(f);
}
static void tensor_line(FILE *f,const hold_tensor *t) {
    fprintf(f,"%s %zu %zu %u %d %d %d %d\n",t->name,t->off,t->bytes,t->type,t->rows,t->K,t->stride,t->experts);
}
static void metadata_write(const char *dir,const hold_arena *a,int logical) {
    char p[1024],done[1024];path(p,dir,"metadata.pending");path(done,dir,"metadata");
    FILE *f=fdopen(fresh(p),"w");if(!f)die(p);
    fprintf(f,"native_weights 1 bytes %zu physical %d logical %d\n",a->bytes,a->die,logical);
    for(int i=0;i<a->n;i++)tensor_line(f,a->t+i);
    if(fflush(f)||fsync(fileno(f))||fclose(f))die("metadata write");
    if(rename(p,done))die("publish metadata");
}
static void policy_save(const char *dir) {
    const char *shares=getenv("SPLIT_SHARES"),*attn=getenv("ATTN_MAP");
    require(shares && *shares && attn && *attn,"explicit source SPLIT_SHARES and ATTN_MAP required");
    char p[1024];path(p,dir,"policy");FILE *f=fdopen(fresh(p),"w");if(!f)die(p);
    fprintf(f,"SPLIT_SHARES %s\nATTN_MAP %s\n",shares,attn);if(fclose(f))die("policy write");
}
static void policy_read(const char *dir) {
    char p[1024],shares[512],attn[512];path(p,dir,"policy");FILE *f=fopen(p,"r");if(!f)die(p);
    require(fscanf(f,"SPLIT_SHARES %511s\nATTN_MAP %511s",shares,attn)==2,"invalid source policy");fclose(f);
    require(!setenv("SPLIT_SHARES",shares,1) && !setenv("ATTN_MAP",attn,1),"set source policy");
}
static void save(const char *out,char **manifest) {
    newdir(out);policy_save(out);CK(cuInit(0));
    const size_t chunk=8u<<20;unsigned char *buf=malloc(chunk);if(!buf)die("copy buffer");
    for(int d=0;d<6;d++) {
        hold_arena meta;int logical;metadata_read(&meta,manifest[d],&logical);
        require(logical<0 && !hold_find(&meta,FLASH_BALANCE_LAYOUT_MARKER),"save expects original whole-expert holders");
        CUdevice dev;CUcontext ctx;int ordinal=hold_ordinal(meta.die);require(ordinal>=0,"source GPU not visible");
        CK(cuDeviceGet(&dev,ordinal));CK(cuCtxCreate_v2(&ctx,CU_CTX_SCHED_BLOCKING_SYNC,dev));
        hold_arena source;require(!hold_open(&source,manifest[d]),"open resident holder");
        require(source.bytes==meta.bytes && source.die==meta.die && source.n==meta.n,"source manifest changed");
        for(int i=0;i<meta.n;i++) {
            const hold_tensor *a=source.t+i,*b=meta.t+i;
            require(!strcmp(a->name,b->name) && a->off==b->off && a->bytes==b->bytes &&
                a->type==b->type && a->rows==b->rows && a->K==b->K && a->stride==b->stride &&
                a->experts==b->experts,"source tensor metadata changed");
        }
        char sub[64],dir[1024],file[1024];snprintf(sub,sizeof sub,"die%d",d);path(dir,out,sub);newdir(dir);path(file,dir,"arena.bin");
        int fd=fresh(file);
        for(size_t off=0;off<meta.bytes;off+=chunk) {
            size_t n=meta.bytes-off;if(n>chunk)n=chunk;
            CK(cuMemcpyDtoH(buf,source.base+off,n));write_at(fd,buf,n,off);
        }
        finish(fd);metadata_write(dir,&meta,d);
        printf("native_saved logical=%d physical=%d bytes=%zu\n",d,meta.die,meta.bytes);fflush(stdout);
        hold_close(&source);CK(cuCtxDestroy(ctx));free(meta.t);
    }
    free(buf);
}

typedef struct { hold_arena a;const unsigned char *image; } backing;
typedef struct { int rows,experts,K,slices;size_t rowbytes;sel_t source[6];const hold_tensor *tensor[6]; } shape;
static unsigned selected(const sel_t *s) { unsigned n=0;for(int i=0;i<s->nr;i++)n+=(unsigned)s->r[i].n;return n; }
static int block_of(const char *name) { return !strncmp(name,"blk.",4)?atoi(name+4):0; }
static shape infer(backing src[6],const char *name) {
    shape g={0};const hold_tensor *sample=NULL;
    for(int d=0;d<6;d++) {
        const hold_tensor *t=hold_find(&src[d].a,name);g.tensor[d]=t;if(!t || !t->bytes)continue;
        if(!sample)sample=t;
        require(t->type==sample->type && t->K==sample->K && t->stride==sample->stride,"source format mismatch");
        if(t->rows>g.rows)g.rows=t->rows;
    }
    require(sample!=NULL,"no resident source for tensor");
    g.slices=sample->type==GG_Q6_K && sample->K==1536?4:1;
    require(g.rows%g.slices==0,"invalid Q6 slice count");g.rows/=g.slices;
    g.K=g.slices==4?6144:sample->K;
    const char *suf=suffix(name);g.experts=suf && strstr(suf,"_exps.weight")?512:1;
    g.rowbytes=sample->bytes/(size_t)sample->rows/(unsigned)sample->experts;
    require(g.rowbytes*(size_t)sample->rows*(unsigned)sample->experts==sample->bytes,"nonintegral native rows");
    split_on=1;balance_on=0;
    int full=g.rows;
    if(g.experts==1)for(int d=0;d<6;d++) {
        sel_t x;holder_split(d,6,block_of(name));select_rows(name,g.rows,1,&x);
        for(int i=0;i<x.nr;i++)if(x.r[i].r0+x.r[i].n>full)full=x.r[i].r0+x.r[i].n;
    }
    g.rows=full;
    for(int d=0;d<6;d++) {
        const hold_tensor *t=hold_find(&src[d].a,name);holder_split(d,6,block_of(name));
        select_rows(name,g.rows,g.experts,g.source+d);
        unsigned n=selected(g.source+d);
        if(!t){require(!n || !strcmp(name,"token_embd.weight"),"missing source shard");continue;}
        if(t->rows!=(g.experts>1?g.rows:(int)n*g.slices) ||
           t->experts!=(g.experts>1?(int)n:1)) {
            fprintf(stderr,"native_weights: source policy mismatch tensor=%s logical=%d rows=%d experts=%d expected_rows=%d expected_experts=%d\n",
                name,d,t->rows,t->experts,g.experts>1?g.rows:(int)n*g.slices,g.experts>1?(int)n:1);exit(1);
        }
        require(t->bytes==g.rowbytes*(size_t)t->rows*(unsigned)t->experts,"inconsistent source byte count");
    }
    return g;
}
static int local_index(const sel_t *s,unsigned global) {
    unsigned base=0;
    for(int i=0;i<s->nr;i++) {
        if(global>=(unsigned)s->r[i].r0 && global-(unsigned)s->r[i].r0<(unsigned)s->r[i].n)
            return (int)(base+global-(unsigned)s->r[i].r0);
        base+=(unsigned)s->r[i].n;
    }
    return -1;
}
static unsigned global_index(const sel_t *s,unsigned local) {
    for(int i=0;i<s->nr;i++){if(local<(unsigned)s->r[i].n)return (unsigned)s->r[i].r0+local;local-=(unsigned)s->r[i].n;}
    require(0,"destination row outside selection");return 0;
}
static const unsigned char *source_row(backing src[6],const shape *g,unsigned row,unsigned expert,unsigned slice) {
    for(int d=0;d<6;d++) {
        const hold_tensor *t=g->tensor[d];if(!t || !t->bytes)continue;
        int local=local_index(g->source+d,g->experts>1?expert:row);if(local<0)continue;
        size_t r=g->experts>1?(size_t)local*(unsigned)g->rows+row:(size_t)slice*(unsigned)(t->rows/g->slices)+(unsigned)local;
        require(row<(unsigned)g->rows && slice<(unsigned)g->slices && (r+1)*g->rowbytes<=t->bytes,"native row address outside source");
        require(src[d].image!=NULL,"source image not mapped");
        return src[d].image+t->off+r*g->rowbytes;
    }
    require(0,"global row/expert has no resident owner");return NULL;
}
static void add_tensor(hold_arena *a,hold_tensor t) {
    t.off=(a->bytes+255)&~(size_t)255;a->bytes=t.off+t.bytes;
    hold_tensor *q=realloc(a->t,(size_t)(a->n+1)*sizeof *q);if(!q)die("destination metadata");a->t=q;a->t[a->n++]=t;
}
static void balance(backing src[6],const char *out) {
    hold_arena dst[6]={{0}};int fd[6]={-1,-1,-1,-1,-1,-1};char dirs[6][1024];
    if(out){newdir(out);policy_save(out);}
    for(int d=0;d<6;d++) {
        dst[d].die=(int[]){0,1,6,3,4,5}[d];
        hold_tensor marker={0};snprintf(marker.name,sizeof marker.name,"%s",FLASH_BALANCE_LAYOUT_MARKER);
        marker.rows=d;marker.K=6;marker.stride=8;marker.experts=1;add_tensor(dst+d,marker);
        if(out){char sub[64],p[1024];snprintf(sub,sizeof sub,"die%d",d);path(dirs[d],out,sub);newdir(dirs[d]);path(p,dirs[d],"arena.bin");fd[d]=fresh(p);}
    }
    /* The first holder has zero-byte metadata for unowned projections, too.
     * Reject any name omitted by that manifest rather than silently losing it. */
    for(int d=1;d<6;d++)for(int i=0;i<src[d].a.n;i++)require(hold_find(&src[0].a,src[d].a.t[i].name)!=NULL,"first manifest lacks a tensor name");
    for(int i=0;i<src[0].a.n;i++) {
        const char *name=src[0].a.t[i].name;const char *suf=suffix(name);
        if(suf && !strcmp(suf,"emap")) {
            uint32_t ids[512];for(unsigned j=0;j<512;j++)ids[j]=j;
            for(int d=0;d<6;d++){hold_tensor t=src[0].a.t[i];t.bytes=sizeof ids;t.rows=512;t.K=1;t.stride=0;t.experts=1;add_tensor(dst+d,t);if(out)write_at(fd[d],ids,sizeof ids,dst[d].t[dst[d].n-1].off);}continue;
        }
        shape g=infer(src,name);const hold_tensor *sample=NULL;
        for(int d=0;d<6;d++){const hold_tensor *t=hold_find(&src[d].a,name);if(t && t->bytes){sample=t;break;}}
        require(sample!=NULL,"missing sample tensor");
        for(int d=0;d<6;d++) {
            split_on=balance_on=1;holder_split(d,6,block_of(name));sel_t sel;select_rows(name,g.rows,g.experts,&sel);
            unsigned rows=sel.row_grain?balance_row_count((unsigned)g.rows,d,6,sel.row_grain):selected(&sel);
            hold_tensor t=*sample;t.rows=(int)rows*g.slices;t.experts=g.experts;
            t.bytes=(size_t)t.rows*(unsigned)t.experts*g.rowbytes;
            if(sel.embedding_groups){require(sample->type==GG_Q3_K && g.K==2560 && g.rowbytes==1100,"embedding source must be original raw Q3_K");t.rows=g.rows;t.K=(int)sel.embedding_groups*32;t.stride=(int)sel.embedding_groups*16;t.bytes=(size_t)t.rows*(unsigned)t.stride;}
            add_tensor(dst+d,t);t=dst[d].t[dst[d].n-1];
            printf("native_plan logical=%d tensor=%s rows=%d K=%d experts=%d bytes=%zu offset=%zu\n",d,name,t.rows,t.K,t.experts,t.bytes,t.off);
            if(!out)continue;
            unsigned char *image=malloc(t.bytes?t.bytes:1);if(!image)die("destination tensor scratch");
            if(sel.embedding_groups) {
                for(unsigned r=0;r<(unsigned)g.rows;r++) {
                    const unsigned char *source=source_row(src,&g,r,0,0);
                    for(unsigned j=0;j<sel.embedding_groups;j++)
                        balance_pack_embedding_group(source,32*(sel.embedding_first+j),image+((size_t)r*sel.embedding_groups+j)*16);
                }
            } else for(unsigned e=0;e<(unsigned)g.experts;e++)for(unsigned s=0;s<(unsigned)g.slices;s++) {
                const unsigned char *expert=g.experts>1?source_row(src,&g,0,e,s):NULL;
                for(unsigned r=0;r<rows;r++) {
                    unsigned global=sel.row_grain?balance_global_row(r,d,6,sel.row_grain):global_index(&sel,r);
                    const unsigned char *source=expert?expert+(size_t)global*g.rowbytes:source_row(src,&g,global,e,s);
                    memcpy(image+(((size_t)e*g.slices+s)*rows+r)*g.rowbytes,source,g.rowbytes);
                }
            }
            write_at(fd[d],image,t.bytes,t.off);free(image);
        }
    }
    for(int d=0;d<6;d++) {
        dst[d].bytes=((dst[d].bytes+255)&~(size_t)255)+(4u<<20);
        printf("native_total logical=%d physical=%d tensors=%d bytes=%zu\n",d,dst[d].die,dst[d].n,dst[d].bytes);
        if(out){if(ftruncate(fd[d],(off_t)dst[d].bytes))die("cache size");finish(fd[d]);metadata_write(dirs[d],dst+d,d);}
        free(dst[d].t);
    }
}
static volatile sig_atomic_t stopped;
static void on_signal(int sig) { (void)sig;stopped=1; }
static void load(const char *dir,int physical,const char *manifest,int notify) {
    char p[1024];path(p,dir,"metadata");hold_arena a;int logical;metadata_read(&a,p,&logical);
    require(logical>=0 && logical<6,"invalid cached logical rank");
    path(p,dir,"arena.bin");int fd=open(p,O_RDONLY);if(fd<0)die(p);struct stat st;if(fstat(fd,&st))die(p);
    require(st.st_size>=0 && (size_t)st.st_size==a.bytes,"cached arena has wrong size");
    unsigned char *data=mmap(NULL,a.bytes,PROT_READ,MAP_PRIVATE,fd,0);if(data==MAP_FAILED)die("map cache");close(fd);
    CK(cuInit(0));CUdevice dev;CUcontext ctx;int ordinal=hold_ordinal(physical);require(ordinal>=0,"destination GPU not visible");
    CK(cuDeviceGet(&dev,ordinal));CK(cuCtxCreate_v2(&ctx,CU_CTX_SCHED_BLOCKING_SYNC,dev));CUdeviceptr arena;CK(cuMemAlloc(&arena,a.bytes));
    const size_t chunk=8u<<20;for(size_t off=0;off<a.bytes;off+=chunk){size_t n=a.bytes-off;if(n>chunk)n=chunk;CK(cuMemcpyHtoD(arena+off,data+off,n));}
    munmap(data,a.bytes);CUipcMemHandle h;CK(cuIpcGetMemHandle(&h,arena));
    char pending[1100],ready[1100];require(snprintf(pending,sizeof pending,"%s.pending",manifest)<(int)sizeof pending,"manifest path too long");require(snprintf(ready,sizeof ready,"%s.ready",manifest)<(int)sizeof ready,"ready path too long");
    require(access(manifest,F_OK)!=0 && errno==ENOENT && access(ready,F_OK)!=0 && errno==ENOENT,"manifest or ready path already exists");
    FILE *f=fdopen(fresh(pending),"w");if(!f)die(pending);
    fprintf(f,"arena %zu die %d handle ",a.bytes,physical);for(size_t i=0;i<sizeof h;i++)fprintf(f,"%02x",((unsigned char *)&h)[i]);fputc('\n',f);
    for(int i=0;i<a.n;i++)tensor_line(f,a.t+i);
    if(fflush(f)||fsync(fileno(f))||fclose(f)||rename(pending,manifest))die("publish holder manifest");
    int rfd=fresh(ready);write_at(rfd,"ready\n",6,0);finish(rfd);
    printf("native_holder_ready logical=%d physical=%d bytes=%zu manifest=%s\n",logical,physical,a.bytes,manifest);fflush(stdout);
    if(notify>=0) {
        /* All GPU loading finishes before the queue runner returns. The
         * detached process subsequently retains immutable allocation only. */
        int null=open("/dev/null",O_RDWR);if(null<0)die("open /dev/null");
        if(dup2(null,0)<0 || dup2(null,1)<0 || dup2(null,2)<0)die("detach holder streams");
        if(null>2)close(null);
        require(write(notify,"R",1)==1,"notify holder readiness");close(notify);
    }
    signal(SIGTERM,on_signal);signal(SIGINT,on_signal);while(!stopped)pause();
    unlink(ready);CK(cuMemFree(arena));CK(cuCtxDestroy(ctx));free(a.t);
}
int main(int argc,char **argv) {
    if(argc==9 && !strcmp(argv[1],"save")){save(argv[2],argv+3);return 0;}
    if(argc==5 && (!strcmp(argv[1],"load") || !strcmp(argv[1],"launch"))) {
        char *end;long d=strtol(argv[3],&end,10);require(*argv[3] && !*end && d>=0 && d<32,"invalid physical GPU");
        if(!strcmp(argv[1],"load")){load(argv[2],(int)d,argv[4],-1);return 0;}
        int ready[2];if(pipe(ready))die("holder readiness pipe");pid_t pid=fork();if(pid<0)die("fork holder");
        if(!pid){close(ready[0]);if(setsid()<0)die("holder session");load(argv[2],(int)d,argv[4],ready[1]);_exit(0);}
        close(ready[1]);char response=0;ssize_t n;do n=read(ready[0],&response,1);while(n<0 && errno==EINTR);close(ready[0]);
        if(n!=1 || response!='R'){int status;waitpid(pid,&status,0);fprintf(stderr,"native_weights: holder did not become ready\n");return 1;}
        printf("native_holder_detached pid=%ld physical=%ld manifest=%s\n",(long)pid,d,argv[4]);return 0;
    }
    backing src[6]={0};
    if(argc==8 && !strcmp(argv[1],"plan")) {
        require(getenv("SPLIT_SHARES") && getenv("ATTN_MAP"),"source split policy required");
        for(int d=0;d<6;d++){int logical;metadata_read(&src[d].a,argv[d+2],&logical);require(logical<0,"plan expects live original manifests");}
        balance(src,NULL);
    } else if(argc==4 && !strcmp(argv[1],"balance")) {
        policy_read(argv[2]);
        for(int d=0;d<6;d++) {
            char sub[64],dir[1024],p[1024];snprintf(sub,sizeof sub,"die%d",d);path(dir,argv[2],sub);path(p,dir,"metadata");int logical;metadata_read(&src[d].a,p,&logical);require(logical==d,"cached source rank mismatch");
            path(p,dir,"arena.bin");int fd=open(p,O_RDONLY);if(fd<0)die(p);struct stat st;if(fstat(fd,&st))die(p);require(st.st_size>=0 && (size_t)st.st_size==src[d].a.bytes,"source cache size mismatch");
            src[d].image=mmap(NULL,src[d].a.bytes,PROT_READ,MAP_PRIVATE,fd,0);if(src[d].image==MAP_FAILED)die("map source cache");close(fd);
        }
        balance(src,argv[3]);
    } else {fprintf(stderr,"usage: native_weights save NEW_DIR MANIFEST0..5 | plan MANIFEST0..5 | balance SOURCE_CACHE NEW_CACHE | load|launch DIE_CACHE PHYSICAL_GPU NEW_MANIFEST\n");return 2;}
    for(int d=0;d<6;d++){if(src[d].image)munmap((void *)src[d].image,src[d].a.bytes);free(src[d].a.t);}return 0;
}
