#include "timings.h"
#include <assert.h>
#include <string.h>
static void near(double a,double b){assert(fabs(a-b)<1e-9);}
static void check(flash_timing t,int g,int executed,int pp,int intervals){
    flash_timing_values v=flash_timing_derive(&t);
    assert(v.generated==g && v.executed==executed);
    assert(v.prompt_processed==pp && v.decode_intervals==intervals);
    assert(v.executed==v.prompt_processed+v.decode_intervals);
}
int main(void){
    char json[2048],small[8];
    flash_timing one={.prompt=1,.completed=1,.first_valid=1,.build_ms=2,.run_ms=8,.first_ms=8};
    check(one,1,1,1,0);assert(!flash_timing_json(json,sizeof json,&one));
    assert(strstr(json,"\"decode_tokens_per_second\":null"));
    assert(strstr(json,"\"device_first_publish_ms\":null"));
    assert(flash_timing_json(small,sizeof small,&one)==-1);
    flash_timing resume={.prompt=10,.resumed=7,.completed=13,.first_valid=1,.build_ms=1,.run_ms=50,.first_ms=20};
    check(resume,4,6,3,3);flash_timing_values v=flash_timing_derive(&resume);
    near(v.post_ms,30);near(v.prompt_rate,150);near(v.decode_rate,100);
    assert(v.prompt_rate_valid && v.decode_rate_valid);
    flash_timing stop={.prompt=6,.completed=6,.ended=1,.first_valid=1,.run_ms=6,.first_ms=6};
    check(stop,1,6,6,0);assert(!flash_timing_json(json,sizeof json,&stop));
    assert(strstr(json,"\"stop_token_observed\":true"));assert(strstr(json,"\"decode_tokens_per_second\":null"));
    flash_timing abort={.prompt=6,.completed=3,.aborted=1,.run_ms=9};
    check(abort,0,3,3,0);assert(!flash_timing_json(json,sizeof json,&abort));
    assert(strstr(json,"\"aborted\":true"));assert(strstr(json,"\"first_observed_ms\":null"));assert(strstr(json,"\"post_first_ms\":null"));
    abort.completed=7;abort.first_valid=1;abort.first_ms=10;abort.run_ms=20;
    check(abort,2,7,6,1);v=flash_timing_derive(&abort);near(v.decode_rate,100);
    one.completed=2;one.first_ms=0;one.run_ms=10;v=flash_timing_derive(&one);
    assert(v.post_valid && !v.prompt_rate_valid && v.decode_rate_valid);near(v.decode_rate,100);
    one.run_ms=NAN;assert(!flash_timing_json(json,sizeof json,&one));assert(strstr(json,"\"run_ms\":null"));
    puts("timings_test: ok");return 0;
}
