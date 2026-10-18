#define main flashd_program_main
#include "flashd.c"
#undef main
#include <assert.h>
static char *read_pair(int fd) {
    sbuf b={0};char buf[1024];ssize_t n;
    while((n=read(fd,buf,sizeof buf))>0)sb_put(&b,buf,(size_t)n);
    assert(n==0);if(!b.s)sb_puts(&b,"");return b.s;
}
static int helper_tests(void) {
    int p[2];char *wire;assert(socketpair(AF_UNIX,SOCK_STREAM,0,p)==0);
    char expected_pid[32]; snprintf(expected_pid,sizeof expected_pid,"%ld",(long)getpid());
    assert(setenv("FLASHD_EXPECT_CORE_PID",expected_pid,1)==0);
    assert(engine_peer_matches(p[0]));
    snprintf(expected_pid,sizeof expected_pid,"%ld",(long)getpid()+1);
    assert(setenv("FLASHD_EXPECT_CORE_PID",expected_pid,1)==0);
    assert(!engine_peer_matches(p[0]));
    assert(setenv("FLASHD_EXPECT_CORE_PID","invalid",1)==0);
    assert(!engine_peer_matches(p[0]));
    assert(setenv("FLASHD_EXPECT_CORE_PID","99999999999999999999999999",1)==0);
    assert(!engine_peer_matches(p[0]));
    assert(unsetenv("FLASHD_EXPECT_CORE_PID")==0);
    assert(engine_peer_matches(p[0]));
    gen g={.fd=p[0],.stream=1,.created=1,.request_start_ms=http_clock_ms()};
    const char *role="data: {\"choices\":[{\"delta\":{\"role\":\"assistant\",\"content\":null}}]}\n\n";
    assert(chunk(p[0],role,strlen(role))==0);
    assert(chunk(p[0],": keepalive\n\n",13)==0);assert(g.first_text_send_ms==0);
    emit_delta(&g,"content","",0);assert(g.first_text_send_ms==0);
    emit_delta(&g,"content","a",1);assert(!g.gone);
    assert(g.first_text_send_ms>=g.request_start_ms && g.first_text_send_ms>0);
    double first=g.first_text_send_ms;emit_delta(&g,"content","b",1);assert(g.first_text_send_ms==first);
    assert(shutdown(p[0],SHUT_WR)==0);wire=read_pair(p[1]);
    assert(strstr(wire,": keepalive\n\n") && strstr(wire,"\"content\":\"a\"") && strstr(wire,"\"content\":\"b\""));
    free(wire);sb_free(&g.content);sb_free(&g.reasoning);close(p[0]);close(p[1]);
    assert(socketpair(AF_UNIX,SOCK_STREAM,0,p)==0);
    admitted_error(p[0],1,400,"bad admitted prompt");
    assert(shutdown(p[0],SHUT_WR)==0);wire=read_pair(p[1]);
    assert(!strstr(wire,"HTTP/1.1") && strstr(wire,"data: {\"error\":") && strstr(wire,"\"code\":400"));
    assert(strstr(wire,"bad admitted prompt") && strstr(wire,"data: [DONE]\n\n"));
    size_t n=strlen(wire);assert(n>=5 && !memcmp(wire+n-5,"0\r\n\r\n",5));
    free(wire);close(p[0]);close(p[1]);
    assert(socketpair(AF_UNIX,SOCK_STREAM,0,p)==0);
    admitted_error(p[0],0,400,"bad ordinary prompt");
    assert(shutdown(p[0],SHUT_WR)==0);wire=read_pair(p[1]);
    assert(strstr(wire,"HTTP/1.1 400 Bad Request\r\n")==wire);
    assert(strstr(wire,"\"code\":400") && strstr(wire,"bad ordinary prompt") && !strstr(wire,"data: [DONE]"));
    free(wire);close(p[0]);close(p[1]);
    puts("flashd_timing_test: actual role, keepalive, delivery and admitted-error helpers PASS");return 0;
}
int main(int argc,char **argv) {
    if(argc==2 && !strcmp(argv[1],"--unit"))return helper_tests();
    const char *cap=getenv("FLASHD_TEST_ANSWER_MAX");if(cap)ANSWER_MAX=strtol(cap,NULL,10);
    const char *drain=getenv("FLASHD_TEST_CANCEL_DRAIN_MS");if(drain)CANCEL_DRAIN_MS=strtod(drain,NULL);
    return flashd_program_main(argc,argv);
}
