#include "timings.h"
#include <assert.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>
static void near(double a,double b){assert(fabs(a-b)<1e-9);}
static void check(flash_timing t,int g,int executed,int pp,int intervals){
    flash_timing_values v=flash_timing_derive(&t);
    assert(v.generated==g && v.executed==executed);
    assert(v.prompt_processed==pp && v.decode_intervals==intervals);
    assert(v.executed==v.prompt_processed+v.decode_intervals);
}
/* Deterministic event source for the CLI/actual-HTTP forwarding test. */
static int fixture(const char *mode,int p,int g,int resumed){
    flash_timing t={.request_id=42,.prompt=p,.completed=p-1+g,.resumed=resumed,
        .counts_known=1,.counts_final=1,.stop_known=1,.first_valid=g>0,
        .build_ms=2.123456789,.first_ms=12.3456789,.run_ms=56.7890123};
    if(!strcmp(mode,"one")||!strcmp(mode,"stop"))t.run_ms=t.first_ms;
    if(!strcmp(mode,"stop"))t.ended=1;
    if(!strcmp(mode,"partial")){t.error_code=-2;t.counts_final=0;t.stop_known=0;}
    if(!strcmp(mode,"abort")){t.aborted=1;t.ended=0;}
    if(!strcmp(mode,"prefillabort")){t.completed=7;t.resumed=0;t.aborted=1;t.first_valid=0;}
    if(!strcmp(mode,"unknown")){t.counts_known=0;t.counts_final=0;t.stop_known=0;t.first_valid=0;t.build_ms=NAN;t.run_ms=NAN;t.error_code=-1;}
    char json[4096];assert(!flash_timing_json(json,sizeof json,&t));puts(json);return 0;
}
int main(int argc,char **argv){
    if(argc==5)return fixture(argv[1],atoi(argv[2]),atoi(argv[3]),atoi(argv[4]));
    assert(argc==1);
    char json[4096],small[8];
    flash_timing one={.prompt=1,.completed=1,.counts_known=1,.counts_final=1,.stop_known=1,.first_valid=1,.build_ms=2,.run_ms=8,.first_ms=8};
    check(one,1,1,1,0);assert(!flash_timing_json(json,sizeof json,&one));
    assert(strstr(json,"\"decode_tokens_per_second\":null"));
    assert(strstr(json,"\"predicted_per_second\":null"));
    assert(strstr(json,"\"device_first_publish_ms\":null"));
    assert(flash_timing_json(small,sizeof small,&one)==-1);
    flash_timing resume={.prompt=10,.resumed=7,.completed=13,.counts_known=1,.counts_final=1,.stop_known=1,.first_valid=1,.build_ms=1,.run_ms=50,.first_ms=20};
    check(resume,4,6,3,3);flash_timing_values v=flash_timing_derive(&resume);
    near(v.post_ms,30);near(v.prompt_rate,150);near(v.decode_rate,100);
    assert(v.prompt_rate_valid && v.decode_rate_valid);
    flash_timing prompt_only=resume;prompt_only.completed=9;prompt_only.first_valid=0;
    check(prompt_only,0,2,2,0);assert(!flash_timing_derive(&prompt_only).decode_rate_valid);
    flash_timing stop={.prompt=6,.completed=6,.counts_known=1,.counts_final=1,.stop_known=1,.ended=1,.first_valid=1,.run_ms=6,.first_ms=6};
    check(stop,1,6,6,0);assert(!flash_timing_json(json,sizeof json,&stop));
    assert(strstr(json,"\"stop_token_observed\":true"));assert(strstr(json,"\"decode_tokens_per_second\":null"));
    flash_timing abort={.prompt=6,.completed=3,.counts_known=1,.counts_final=1,.stop_known=1,.aborted=1,.run_ms=9};
    check(abort,0,3,3,0);assert(!flash_timing_json(json,sizeof json,&abort));
    assert(strstr(json,"\"aborted\":true"));assert(strstr(json,"\"first_observed_ms\":null"));assert(strstr(json,"\"post_first_ms\":null"));
    assert(strstr(json,"\"stop_token_observed\":false"));
    abort.completed=7;abort.first_valid=1;abort.first_ms=10;abort.run_ms=20;
    check(abort,2,7,6,1);v=flash_timing_derive(&abort);near(v.decode_rate,100);
    flash_timing partial=abort;partial.counts_final=0;partial.stop_known=0;partial.error_code=-2;
    v=flash_timing_derive(&partial);assert(v.post_valid && !v.decode_rate_valid && !v.prompt_rate_valid);
    assert(!flash_timing_json(json,sizeof json,&partial));assert(strstr(json,"\"counts_final\":false"));
    assert(strstr(json,"\"positions_completed\":7"));assert(strstr(json,"\"stop_token_observed\":null"));
    flash_timing unknown={.prompt=10,.build_ms=NAN,.run_ms=NAN,.error_code=-1};
    assert(!flash_timing_json(json,sizeof json,&unknown));assert(strstr(json,"\"positions_completed\":null"));
    assert(strstr(json,"\"generated_tokens\":null"));assert(strstr(json,"\"run_ms\":null"));
    one.completed=2;one.first_ms=0;one.run_ms=10;v=flash_timing_derive(&one);
    assert(v.post_valid && !v.prompt_rate_valid && v.decode_rate_valid);near(v.decode_rate,100);
    one.run_ms=NAN;assert(!flash_timing_json(json,sizeof json,&one));assert(strstr(json,"\"run_ms\":null"));
    assert(!flash_timing_request_fits(INT_MAX,INT_MAX,262144));
    assert(!flash_timing_request_fits(0,0,262144));assert(!flash_timing_request_fits(1,-1,262144));
    assert(flash_timing_request_fits(262144,0,262144));assert(!flash_timing_request_fits(262144,1,262144));
    assert(flash_timing_request_fits(1,262143,262144));
    flash_timing_device(&resume,1000000000000000000ull,1000000000020000000ull,1000000000050000000ull,4);
    v=flash_timing_derive(&resume);assert(v.device_rate_valid);near(v.device_decode_ms,30);near(v.device_rate,100);
    assert(!flash_timing_json(json,sizeof json,&resume));assert(strstr(json,"\"device_first_publish_ns\":\"1000000000020000000\""));
    flash_timing_device(&resume,100,200,300,2);assert(!flash_timing_derive(&resume).device_rate_valid);
    flash_timing_device(&resume,200,100,300,4);assert(!resume.device_first_valid);
    flash_timing_device(&resume,100,200,150,4);assert(!resume.device_first_valid);
    flash_timing_device(&resume,0,200,300,4);assert(!resume.device_first_valid);
    puts("timings_test: count, cancellation, failure, overflow, aliases and device-clock fixtures PASS");return 0;
}
