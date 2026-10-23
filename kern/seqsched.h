/* Bounded sequence admission and FIFO GPU turns. No CUDA dependency. */
#ifndef FLASH_SEQSCHED_H
#define FLASH_SEQSCHED_H
#include <pthread.h>
#include <time.h>
#include <string.h>
#define SEQ_MAX 16
typedef struct seq_wait { struct seq_wait *next; int wanted; } seq_wait;
typedef struct {
    pthread_mutex_t mu;
    pthread_cond_t cv;
    int n, busy[SEQ_MAX], broken;
    unsigned long long issued, serving;
    seq_wait *head, *tail;
} seqsched;
static void seqsched_init(seqsched *s, int n) {
    memset(s, 0, sizeof *s); s->n = n;
    pthread_mutex_init(&s->mu, NULL); pthread_cond_init(&s->cv, NULL);
}
static int seqsched_claim(seqsched *s, int wanted, int (*alive)(void *), void *arg) {
    if (wanted < -1 || wanted >= s->n) return -1;
    seq_wait w = { .wanted = wanted }; int found = -1;
    pthread_mutex_lock(&s->mu);
    if (s->tail) s->tail->next = &w; else s->head = &w;
    s->tail = &w;
    for (;;) {
        if (__atomic_load_n(&s->broken, __ATOMIC_ACQUIRE) || (alive && alive(arg))) break;
        /* Admit the oldest eligible waiter. A caller waiting for occupied slot 0
         * must not prevent another caller from using an available slot 1. */
        seq_wait *eligible = NULL;
        int candidate = -1;
        for (seq_wait *q = s->head; q && !eligible; q = q->next)
            for (int i = 0; i < s->n; i++)
                if (!s->busy[i] && (q->wanted < 0 || q->wanted == i)) {
                    eligible = q; candidate = i; break;
                }
        if (eligible == &w) { found = candidate; s->busy[found] = 1; break; }
        struct timespec ts; clock_gettime(CLOCK_REALTIME, &ts);
        ts.tv_nsec += 100000000;
        if (ts.tv_nsec >= 1000000000) { ts.tv_sec++; ts.tv_nsec -= 1000000000; }
        pthread_cond_timedwait(&s->cv, &s->mu, &ts);
    }
    seq_wait **p = &s->head;
    while (*p != &w) p = &(*p)->next;
    *p = w.next;
    if (s->tail == &w) {
        s->tail = NULL;
        for (seq_wait *q = s->head; q; q = q->next) s->tail = q;
    }
    pthread_cond_broadcast(&s->cv); pthread_mutex_unlock(&s->mu);
    return found;
}
static int seqsched_enter(seqsched *s) {
    pthread_mutex_lock(&s->mu);
    unsigned long long mine = s->issued++;
    while (mine != s->serving) pthread_cond_wait(&s->cv, &s->mu);
    int broken = __atomic_load_n(&s->broken, __ATOMIC_ACQUIRE);
    pthread_mutex_unlock(&s->mu); return broken ? -1 : 0;
}
static void seqsched_leave(seqsched *s) {
    pthread_mutex_lock(&s->mu); s->serving++;
    pthread_cond_broadcast(&s->cv); pthread_mutex_unlock(&s->mu);
}
static void seqsched_release(seqsched *s, int slot) {
    pthread_mutex_lock(&s->mu); s->busy[slot] = 0;
    pthread_cond_broadcast(&s->cv); pthread_mutex_unlock(&s->mu);
}
static void seqsched_destroy(seqsched *s) {
    pthread_cond_destroy(&s->cv); pthread_mutex_destroy(&s->mu);
}
#endif
