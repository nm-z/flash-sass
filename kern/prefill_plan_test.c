#include "prefill.h"
#include <assert.h>
#include <limits.h>
#include <stdio.h>

static void parse_tests(void) {
    struct { const char *text; int expected; } ok[] = {
        {NULL,0},{"0",0},{"1",1},{"2",2},{"64",64},{"00064",64},{"4096",4096}
    };
    for (unsigned i=0;i<sizeof ok/sizeof ok[0];i++) {
        int size=-1;
        assert(prefill_parse_chunk(ok[i].text,&size)==0);
        assert(size==ok[i].expected);
    }
    const char *bad[]={"","-1","+1"," 2","2 ","2x","1.5","4097","2147483647","999999999999999999999999"};
    for(unsigned i=0;i<sizeof bad/sizeof bad[0];i++) {
        int size=77;
        assert(prefill_parse_chunk(bad[i],&size)==-1 && size==77);
    }
    assert(prefill_parse_chunk("64",NULL)==-1);
    int batch=-1;
    assert(!prefill_parse_batch(NULL,&batch) && batch==0);
    assert(!prefill_parse_batch("0",&batch) && batch==0);
    assert(!prefill_parse_batch("8",&batch) && batch==8);
    const char *bad_batch[]={"","1","2","4","08","32","-8","+8"," 8","8 ","8x"};
    for(unsigned i=0;i<sizeof bad_batch/sizeof bad_batch[0];i++) {
        batch=77;assert(prefill_parse_batch(bad_batch[i],&batch)==-1 && batch==77);
    }
    assert(prefill_parse_batch("8",NULL)==-1);
}
static void batch_tests(void) {
    for(int nprompt=1;nprompt<=96;nprompt++)
    for(int resume=0;resume<nprompt;resume++)
    for(int until=resume;until<nprompt;until++) {
        int pos=resume,seen[96]={0};
        const int widths[]={8,2,1};
        for(unsigned k=0;k<3;k++) {
            int end=prefill_batch_end(pos,until,widths[k]);
            assert(end>=pos && end<=until && (end-pos)%widths[k]==0);
            for(;pos<end;pos+=widths[k])for(int j=0;j<widths[k];j++)seen[pos+j]++;
        }
        assert(pos==until);
        for(int p=0;p<nprompt;p++)assert(seen[p]==(p>=resume && p<until));
    }
    assert(prefill_batch_end(INT_MAX-9,INT_MAX-1,8)==INT_MAX-1);
    assert(prefill_batch_end(-1,8,8)==-1);
    assert(prefill_batch_end(9,8,8)==-1);
    assert(prefill_batch_end(0,8,0)==-1);
    assert(prefill_batch_end(0,8,4)==-1);
}
static int collect(int pos,int nprompt,int until,int size,int paired,
                   int *seen,int *chunks,int cancel_after) {
    int last=pos-1;
    prefill_chunk_t c;
    while(prefill_plan_chunk(pos,nprompt,until,size,paired,&c)==1) {
        assert(c.begin==pos && c.end>pos && c.end-pos<=size);
        assert(c.end<=until && c.end<nprompt);
        assert(c.pair_end>=pos && c.pair_end<=c.end);
        assert((c.pair_end-pos)%2==0);
        if(paired)assert(c.end-c.pair_end<=1);
        else assert(c.pair_end==pos);
        for(int p=pos;p<c.pair_end;p+=2) {
            assert(p==last+1 && p+1<nprompt-1);
            ++seen[p];++seen[p+1];last=p+1;
        }
        for(int p=c.pair_end;p<c.end;p++) {
            assert(p==last+1);++seen[p];last=p;
        }
        pos=c.end;
        ++*chunks;
        if(cancel_after && *chunks==cancel_after)return pos;
    }
    return pos;
}
static void range_tests(void) {
    for(int n=1;n<=48;n++)
    for(int resume=0;resume<=n;resume++)
    for(int until=resume;until<=n;until++)
    for(int size=1;size<=17;size++)
    for(int paired=0;paired<=1;paired++) {
        int seen[49]={0},chunks=0;
        int pos=collect(resume,n,until,size,paired,seen,&chunks,0);
        int stop=until<n-1?until:n-1;
        if(stop<resume)stop=resume;
        assert(pos==stop);
        for(int p=0;p<n;p++)assert(seen[p]==(p>=resume && p<stop));
    }
}
static void snapshot_cancel_tests(void) {
    for(int size=1;size<=17;size++)
    for(int snap=3;snap<=30;snap++) {
        int seen[33]={0},chunks=0;
        int pos=collect(3,33,snap,size,1,seen,&chunks,0);
        assert(pos==snap);
        for(int p=snap;p<33;p++)assert(seen[p]==0);
        pos=collect(pos,33,33,size,1,seen,&chunks,0);
        assert(pos==32);
        for(int p=0;p<33;p++)assert(seen[p]==(p>=3 && p<32));
        assert(seen[32]==0); /* final prompt belongs to the full-head B1 pass */
    }
    int seen[33]={0},chunks=0;
    int pos=collect(3,33,33,7,1,seen,&chunks,1);
    assert(pos==10 && chunks==1);
    for(int p=10;p<33;p++)assert(seen[p]==0);
    pos=collect(pos,33,33,7,1,seen,&chunks,0);
    assert(pos==32);
    for(int p=3;p<32;p++)assert(seen[p]==1);
}
static void edge_tests(void) {
    prefill_chunk_t c;
    assert(prefill_plan_chunk(INT_MAX-9,INT_MAX,INT_MAX,4096,1,&c)==1);
    assert(c.end==INT_MAX-1 && c.pair_end==INT_MAX-1);
    assert(prefill_plan_chunk(0,1,1,64,1,&c)==0);
    assert(prefill_plan_chunk(1,1,1,64,1,&c)==0);
    assert(prefill_plan_chunk(0,0,0,64,1,&c)==-1);
    assert(prefill_plan_chunk(-1,2,2,64,1,&c)==-1);
    assert(prefill_plan_chunk(3,2,2,64,1,&c)==-1);
    assert(prefill_plan_chunk(1,2,0,64,1,&c)==-1);
    assert(prefill_plan_chunk(0,2,3,64,1,&c)==-1);
    assert(prefill_plan_chunk(0,2,2,0,1,&c)==-1);
    assert(prefill_plan_chunk(0,2,2,4097,1,&c)==-1);
    assert(prefill_plan_chunk(0,2,2,64,1,NULL)==-1);
}
int main(void) {
    parse_tests();batch_tests();range_tests();snapshot_cancel_tests();edge_tests();
    puts("prefill planner: configuration, exhaustive boundaries, snapshot, cancellation, overflow PASS");
    return 0;
}
