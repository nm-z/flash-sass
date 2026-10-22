/* Consume actual balanced_body JSON records; never compare device clock epochs.
 * A phase's die duration is max(CTA body-minus-explicit-poll duration). Every
 * expected phase, die and CTA is required. This is a timing gate, not a model
 * correctness oracle or a claim that remaining body time is pure arithmetic. */
#define _POSIX_C_SOURCE 200809L
#include "../json.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
typedef struct { unsigned mask,resident,cycle_mask;int invalid;uint64_t body,poll,elapsed,cycles; } die_t;
typedef struct { char op[64];int layer,have,invalid;die_t die[6]; } phase_t;
typedef struct token { uint64_t request,position;phase_t *phase;struct token *next; } token_t;
static void fail(const char *s) { fprintf(stderr,"balance_report: %s\n",s);exit(2); }
static int integer(const jv *v,uint64_t *n) { if(!v || v->k!=J_INT || v->i<0)return 0;*n=(uint64_t)v->i;return 1; }
static int field(const jv *v,const char *key,uint64_t *n) { return integer(jv_getz(v,key),n); }
static unsigned count(unsigned n) { unsigned c=0;for(;n;n&=n-1)c++;return c; }
static unsigned argument(const char *s) { char *end;unsigned long v=strtoul(s,&end,10);if(!*s || *end || v>65536)fail("invalid argument");return (unsigned)v; }
int main(int argc,char **argv) {
    if(argc!=3)fail("usage: balance_report EXPECTED_PHASES EXPECTED_TRACE_POSITIONS < job-stdout");
    unsigned expected=argument(argv[1]),positions=argument(argv[2]);
    if(!expected || expected>=500 || !positions || positions>256)fail("require 1..499 phases and 1..256 trace positions");
    token_t *tokens=NULL,**tail=&tokens;unsigned ntokens=0,records=0,bad_records=0;
    int physical[6]={-1,-1,-1,-1,-1,-1};char *line=NULL;size_t capacity=0;ssize_t length;
    while((length=getline(&line,&capacity,stdin))>0) {
        if(!strstr(line,"\"kind\":\"balanced_body\""))continue;
        arena *a=arena_new();const char *error=NULL;jv *v=json_parse(a,line,(size_t)length,&error);
        if(!v || v->k!=J_DICT)fail("malformed body record");
        uint64_t request,position,phase,die,cta,sm,gpu,body,poll,elapsed,abi;
        if(!field(v,"trace_abi",&abi) || (abi!=2 && abi!=3) || !field(v,"request",&request) || !field(v,"position",&position) || !field(v,"phase",&phase) ||
           !field(v,"die",&die) || !field(v,"cta",&cta) || !field(v,"resident_sm",&sm) ||
           !field(v,"physical_gpu",&gpu) || !field(v,"body_minus_poll_ns",&body) ||
           !field(v,"explicit_poll_ns",&poll) || !field(v,"elapsed_ns",&elapsed) ||
           phase>=expected || die>=6 || cta>=16 || sm>=32 || gpu>=32)fail("invalid body record dimensions");
        jv *valid=jv_getz(v,"valid"),*op=jv_getz(v,"op"),*layer=jv_getz(v,"layer");
        if(!op || op->k!=J_STR || !op->s.n || op->s.n>=64 || !layer || layer->k!=J_INT || layer->i< -1 || layer->i>48)
            fail("invalid operation/layer metadata");
        token_t *t=tokens;for(;t;t=t->next)if(t->request==request && t->position==position)break;
        if(!t) {
            if(ntokens>=positions)fail("more traced positions than the declared workload");
            t=calloc(1,sizeof *t);if(!t)fail("token allocation");t->phase=calloc(expected,sizeof *t->phase);if(!t->phase)fail("phase allocation");
            t->request=request;t->position=position;*tail=t;tail=&t->next;ntokens++;
        }
        phase_t *p=t->phase+phase;die_t *d=p->die+die;
        if(!p->have){memcpy(p->op,op->s.s,op->s.n);p->layer=(int)layer->i;p->have=1;}
        else if(strlen(p->op)!=op->s.n || memcmp(p->op,op->s.s,op->s.n) || p->layer!=layer->i)p->invalid=1;
        if(physical[die]<0)physical[die]=(int)gpu;else if(physical[die]!=(int)gpu)d->invalid=1;
        if(d->mask&(1u<<cta))d->invalid=1;
        d->mask|=1u<<cta;d->resident|=1u<<sm;
        if(!valid || valid->k!=J_BOOL || !valid->b || poll>elapsed || body!=elapsed-poll)d->invalid=1;
        if(body>d->body)d->body=body;
        if(poll>d->poll)d->poll=poll;
        if(elapsed>d->elapsed)d->elapsed=elapsed;
        if(abi==3) {
            uint64_t cycles,cycle_poll,cycle_elapsed;
            jv *cv=jv_getz(v,"cycles_valid");
            if(!field(v,"body_minus_poll_cycles",&cycles) || !field(v,"explicit_poll_cycles",&cycle_poll) ||
               !field(v,"elapsed_cycles",&cycle_elapsed) || !cv || cv->k!=J_BOOL)fail("missing cycle fields");
            if(cv->b && cycle_poll<=cycle_elapsed && cycles==cycle_elapsed-cycle_poll){
                d->cycle_mask|=1u<<cta;if(cycles>d->cycles)d->cycles=cycles;
            }
        }
        records++;bad_records+=d->invalid!=0;arena_free(a);
    }
    free(line);if(ferror(stdin))fail("input read failure");
    unsigned bad_phases=0,unbalanced=0,cycle_complete=0,cycle_unbalanced=0;double worst=0,cycle_worst=0;
    puts("request\tposition\tphase\tlayer\top\tdie0_body_us\tdie1_body_us\tdie2_body_us\tdie3_body_us\tdie4_body_us\tdie5_body_us\tmean_us\tmax_deviation_pct\tcoverage_valid\twithin_10pct\tdie0_resident_sms\tdie1_resident_sms\tdie2_resident_sms\tdie3_resident_sms\tdie4_resident_sms\tdie5_resident_sms\tdie0_body_cycles\tdie1_body_cycles\tdie2_body_cycles\tdie3_body_cycles\tdie4_body_cycles\tdie5_body_cycles\tcycle_max_deviation_pct\tcycle_coverage_valid\tcycles_within_10pct");
    for(token_t *t=tokens;t;t=t->next)for(unsigned i=0;i<expected;i++) {
        phase_t *p=t->phase+i;int valid=p->have && !p->invalid;double mean=0,deviation=0;unsigned __int128 sum=0;
        for(unsigned d=0;d<6;d++){valid&=p->die[d].mask==65535u && !p->die[d].invalid;mean+=(double)p->die[d].body/6;sum+=p->die[d].body;}
        if(mean<=0)valid=0;
        if(mean>0)for(unsigned d=0;d<6;d++){double delta=fabs((double)p->die[d].body/mean-1);if(delta>deviation)deviation=delta;}
        int pass=valid;
        for(unsigned d=0;d<6;d++)if((unsigned __int128)p->die[d].body*60<sum*9 || (unsigned __int128)p->die[d].body*60>sum*11)pass=0;
        bad_phases+=!valid;unbalanced+=!pass;if(deviation>worst)worst=deviation;
        printf("%llu\t%llu\t%u\t%d\t%s",(unsigned long long)t->request,(unsigned long long)t->position,i,p->have?p->layer:-1,p->have?p->op:"MISSING");
        for(unsigned d=0;d<6;d++)printf("\t%.6f",p->die[d].body/1000.0);
        printf("\t%.6f\t%.6f\t%d\t%d",mean/1000.0,deviation*100,valid,pass);
        for(unsigned d=0;d<6;d++)printf("\t%u",count(p->die[d].resident));
        uint64_t cycle_sum=0;int cycle_valid=valid;
        for(unsigned d=0;d<6;d++){cycle_sum+=p->die[d].cycles;cycle_valid&=p->die[d].cycle_mask==65535u;}
        if(!cycle_sum)cycle_valid=0;
        double cycle_deviation=0;int cycle_pass=cycle_valid;
        for(unsigned d=0;d<6;d++){
            uint64_t cycles=p->die[d].cycles;
            printf("\t%llu",(unsigned long long)cycles);
            if(cycle_sum){double delta=fabs((double)cycles*6/cycle_sum-1);if(delta>cycle_deviation)cycle_deviation=delta;}
            if((unsigned __int128)cycles*60<(unsigned __int128)cycle_sum*9 ||
               (unsigned __int128)cycles*60>(unsigned __int128)cycle_sum*11)cycle_pass=0;
        }
        if(cycle_valid){cycle_complete++;cycle_unbalanced+=!cycle_pass;if(cycle_deviation>cycle_worst)cycle_worst=cycle_deviation;}
        printf("\t%.6f\t%d\t%d",cycle_deviation*100,cycle_valid,cycle_pass);
        putchar('\n');
    }
    int map_valid=1;for(unsigned d=0;d<6;d++){if(physical[d]<0)map_valid=0;for(unsigned e=0;e<d;e++)if(physical[d]==physical[e])map_valid=0;}
    int complete=ntokens==positions && records==positions*expected*6*16 && !bad_records && !bad_phases && map_valid;
    fprintf(stderr,"balance_report positions=%u/%u phases_per_position=%u cta_records=%u invalid_records=%u invalid_phases=%u failing_phases=%u max_deviation_pct=%.6f coverage_complete=%d body_timing_within_10pct=%d trace_overhead_included=1 resident_sm_is_not_useful_utilization=1\n",
        ntokens,positions,expected,records,bad_records,bad_phases,unbalanced,worst*100,complete,complete && !unbalanced);
    fprintf(stderr,"balance_cycles complete_phases=%u/%u failing_phases=%u max_deviation_pct=%.6f same_sm_differences_only=1 frequency_normalized=0 production_gate_unchanged=1\n",
        cycle_complete,positions*expected,cycle_unbalanced,cycle_worst*100);
    while(tokens){token_t *next=tokens->next;free(tokens->phase);free(tokens);tokens=next;}
    return complete && !unbalanced?0:1;
}
