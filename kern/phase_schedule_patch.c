/* Preserve production math, changing three Maxwell scheduling fields and,
 * optionally, the cache policy of three mutable controller reads.
 * The input identity and opcode words bind this patch to
 * the measured attention-q-release image; it is not a general binary editor. */
#define _POSIX_C_SOURCE 200809L
#include <elf.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static void fail(const char *s) { fprintf(stderr,"phase schedule patch: %s\n",s);exit(1); }
static uint64_t fnv(const uint8_t *p,size_t n) { uint64_t h=UINT64_C(0xcbf29ce484222325);for(size_t i=0;i<n;i++){h^=p[i];h*=UINT64_C(0x100000001b3);}return h; }
static int range(size_t n,uint64_t at,uint64_t len) { return at<=n && len<=n-at; }
static uint64_t word(const uint8_t *p) { uint64_t x;memcpy(&x,p,8);return x; }
static void control(uint8_t *text,size_t size,size_t pc,uint64_t opcode,unsigned mask,unsigned expected,unsigned replacement) {
    if(!range(size,pc,8) || !(pc&31) || word(text+pc)!=opcode)fail("instruction identity mismatch");
    size_t bundle=pc&~(size_t)31;unsigned shift=(unsigned)(((pc&31)/8-1)*21);
    uint64_t packed=word(text+bundle);unsigned before=(unsigned)((packed>>shift)&0x1fffff);
    if((before&mask)!=expected)fail("unexpected original scheduling field");
    unsigned after=(before&~mask)|replacement;
    packed=(packed&~((uint64_t)mask<<shift))|((uint64_t)replacement<<shift);
    memcpy(text+bundle,&packed,8);
    printf("instruction=0x%zx control_before=0x%05x control_after=0x%05x\n",pc,before,after);
}
int main(int argc,char **argv) {
    int coherent=0,prefetch=0;
    if(argc<3) { fprintf(stderr,"usage: phase_schedule_patch ORIGINAL.cubin OUTPUT.cubin [--coherent-start] [--separate-prefetch]\n");return 2; }
    for(int i=3;i<argc;i++) {
        if(!strcmp(argv[i],"--coherent-start"))coherent=1;
        else if(!strcmp(argv[i],"--separate-prefetch"))prefetch=1;
        else fail("unknown option");
    }
    FILE *f=fopen(argv[1],"rb");if(!f)fail("open input");
    if(fseek(f,0,SEEK_END))fail("input seek");
    long length=ftell(f);
    if(length<(long)sizeof(Elf64_Ehdr)||length>16*1024*1024)fail("input size");
    rewind(f);size_t n=(size_t)length;uint8_t *data=malloc(n);if(!data||fread(data,1,n,f)!=n)fail("read input");fclose(f);
    uint64_t before=fnv(data,n);
    if(before!=UINT64_C(0x9abf8aafb9134d59))fail("requires original production cubin fnv9abf8aafb9134d59");
    Elf64_Ehdr header;memcpy(&header,data,sizeof header);
    if(memcmp(header.e_ident,ELFMAG,SELFMAG)||header.e_ident[EI_CLASS]!=ELFCLASS64||header.e_ident[EI_DATA]!=ELFDATA2LSB||header.e_machine!=190||header.e_shentsize!=sizeof(Elf64_Shdr)||header.e_shstrndx>=header.e_shnum||!range(n,header.e_shoff,(uint64_t)header.e_shnum*sizeof(Elf64_Shdr)))fail("ELF header");
    Elf64_Shdr names;memcpy(&names,data+header.e_shoff+(size_t)header.e_shstrndx*sizeof names,sizeof names);
    if(!range(n,names.sh_offset,names.sh_size))fail("ELF names");
    Elf64_Shdr kernel={0};unsigned matches=0;
    for(unsigned i=0;i<header.e_shnum;i++) {
        Elf64_Shdr section;memcpy(&section,data+header.e_shoff+(size_t)i*sizeof section,sizeof section);
        if(section.sh_name>=names.sh_size)fail("ELF section name");
        const char *name=(const char *)data+names.sh_offset+section.sh_name;
        if(!memchr(name,0,(size_t)names.sh_size-section.sh_name))fail("unterminated section name");
        if(!strcmp(name,".text.die")){kernel=section;matches++;}
    }
    if(matches!=1||!range(n,kernel.sh_offset,kernel.sh_size))fail("kernel section");
    uint8_t *code=data+kernel.sh_offset;size_t size=(size_t)kernel.sh_size;
    /* The P2 mirror consumer followed only seven one-cycle stores. Allow the
     * producer to settle before ctl[8] consumes it, without moving any opcode. */
    /* Long stalls require the yield bit; without it values12..15 behave as6. */
    control(code,size,0x70dd0,UINT64_C(0x5b6402000ff70317),0x1f,0x16,0x0f);
    /* Protect the timestamp store's address/data until read barrier2 retires.
     * Barrier2 is free after BDONE; barrier3 may still guard a nobar prefetch. */
    control(code,size,0x70fd0,UINT64_C(0xeedc200010041813),0x700,0x700,0x100);
    control(code,size,0x70fe8,UINT64_C(0x5c10000001570707),0x1f800,0x800,0x1800);
    if(coherent) {
        const size_t pc[]={0x88,0x98,0xb0};
        const uint64_t old[]={UINT64_C(0xeed4200000870e0d),UINT64_C(0xeed4200002070e0d),UINT64_C(0xeed4200000470e0d)};
        for(unsigned i=0;i<3;i++) {
            if(!range(size,pc[i],8)||word(code+pc[i])!=old[i])fail("controller load identity mismatch");
            uint64_t replacement=old[i]|UINT64_C(0x0000400000000000);
            memcpy(code+pc[i],&replacement,8);
            printf("instruction=0x%zx controller_load_cache=CG\n",pc[i]);
        }
    }
    if(prefetch) {
        /* Descriptor payload and its broadcast token-mask load can complete
         * independently. Give the mask its own write barrier, then wait for
         * both values and for its address use before reusing pointer registers. */
        control(code,size,0x1a8,UINT64_C(0xeed4200007ce1004),0xe0,0x60,0x80);
        control(code,size,0x428,UINT64_C(0xeed4200007c61004),0xe0,0x60,0x80);
        control(code,size,0x70bb8,UINT64_C(0xeed4200007ce1004),0x7e0,0x260,0x780);
        control(code,size,0x268,UINT64_C(0x5c28000000170404),0x1f800,0x4800,0xc800);
        control(code,size,0x438,UINT64_C(0x5c28000000170404),0x1f800,0x4800,0xc800);
        control(code,size,0x70c58,UINT64_C(0x5c1080000157180e),0x1f800,0x2000,0xa000);
    }
    uint64_t after=fnv(data,n);
    f=fopen(argv[2],"wx");if(!f)fail("output exists or cannot open");
    if(fwrite(data,1,n,f)!=n||fclose(f))fail("write output");
    free(data);
    printf("bytes=%zu fnv_before=%016" PRIx64 " fnv_after=%016" PRIx64 " math_opcodes_unchanged=1 coherent_controller_reads=%d separate_prefetch=%d\n",n,before,after,coherent,prefetch);return 0;
}
