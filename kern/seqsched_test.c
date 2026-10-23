#define _GNU_SOURCE
#include "seqsched.h"
#include <assert.h>
#include <stdio.h>
#include <sched.h>
static seqsched s;
static int log_order[3], log_n;
static int cancelled, claim_result;
static int is_cancelled(void *p) { (void)p; return __atomic_load_n(&cancelled, __ATOMIC_ACQUIRE); }
static void deadline_yield(double *start) {
 struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts);
 double now=ts.tv_sec+ts.tv_nsec*1e-9;
 if (!*start) *start=now;
 assert(now-*start < 5.0); sched_yield();
}
static void wait_issued(unsigned long long n) {
 double start=0;
 for (;;) {
  pthread_mutex_lock(&s.mu); int ready=s.issued >= n; pthread_mutex_unlock(&s.mu);
  if (ready) return; deadline_yield(&start);
 }
}
static void wait_admission(void) {
 double start=0;
 for (;;) {
  pthread_mutex_lock(&s.mu); int ready=s.head != NULL; pthread_mutex_unlock(&s.mu);
  if (ready) return; deadline_yield(&start);
 }
}
static void *ordered(void *p) {
 int id=(int)(long)p;
 int slot=seqsched_claim(&s,id,NULL,NULL); assert(slot==id);
 assert(seqsched_enter(&s)==0);
 log_order[log_n++]=id;
 seqsched_leave(&s); seqsched_release(&s,slot); return NULL;
}
static void *waiting(void *p) {
 (void)p; claim_result=seqsched_claim(&s,0,is_cancelled,NULL); return NULL;
}
int main(void) {
 seqsched_init(&s,3);
 assert(seqsched_claim(&s,3,NULL,NULL)==-1);
 assert(seqsched_enter(&s)==0);
 pthread_t t[3];
 for (int i=0;i<3;i++) {
  assert(!pthread_create(&t[i],NULL,ordered,(void *)(long)i));
  wait_issued((unsigned long long)i+2);
 }
 seqsched_leave(&s);
 for(int i=0;i<3;i++) assert(!pthread_join(t[i],NULL));
 assert(log_n==3);
 for(int i=0;i<3;i++) assert(log_order[i]==i);
 int slot=seqsched_claim(&s,0,NULL,NULL); assert(slot==0);
 pthread_t waiter; assert(!pthread_create(&waiter,NULL,waiting,NULL));
 wait_admission();
 __atomic_store_n(&cancelled,1,__ATOMIC_RELEASE);
 pthread_mutex_lock(&s.mu); pthread_cond_broadcast(&s.cv); pthread_mutex_unlock(&s.mu);
 assert(!pthread_join(waiter,NULL)); assert(claim_result==-1);
 seqsched_release(&s,slot);
 assert(seqsched_claim(&s,0,NULL,NULL)==0); seqsched_release(&s,0);
 __atomic_store_n(&s.broken,1,__ATOMIC_RELEASE);
 assert(seqsched_claim(&s,-1,NULL,NULL)==-1);
 assert(seqsched_enter(&s)==-1); seqsched_leave(&s);
 seqsched_destroy(&s); puts("seqsched: FIFO turns, bounded slots, cancellation, reuse and fault propagation passed");
 return 0;
}
