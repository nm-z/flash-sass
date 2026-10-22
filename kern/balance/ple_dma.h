#ifndef FLASH_BALANCE_PLE_DMA_H
#define FLASH_BALANCE_PLE_DMA_H
#include <cuda.h>
#include <pthread.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <errno.h>
#include <sched.h>

/* One serialized caller prepares and finishes each immutable position batch.
 * Workers bind their compute context
 * once, then enqueue payload and epoch in the same nonblocking stream. The
 * caller waits for enqueue completion only; pinned source slots remain valid
 * until the request's GPU work completes. No worker longjmps into the caller. */
struct ple_dma_pool;
struct ple_dma_worker {
    struct ple_dma_pool *pool;unsigned die;
    CUcontext context;CUstream stream;CUdeviceptr rows,ready;
    CUresult status;uint64_t begin_ns,end_ns;
};
struct ple_dma_pool {
    pthread_mutex_t mutex;pthread_cond_t work,done;
    pthread_t threads[6];struct ple_dma_worker workers[6];
    unsigned count,started,bound,finished;int initialized,stopping;
    uint64_t generation;unsigned position;uint32_t expected_epoch;
    const void *payload;const uint32_t *epoch;
};
struct ple_dma_span {uint64_t first_begin,last_begin,first_end,last_end;};
static uint64_t ple_dma_ns(void) {
    struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);
    return (uint64_t)t.tv_sec*1000000000ull+(uint64_t)t.tv_nsec;
}
/* Wake during host preparation rather than after it. The bounded active wait
 * overlaps existing I/O; it never admits a copy before the release store.
 * Back off if preparation takes unusually long, and remain stoppable. */
static int ple_dma_wait_ready(struct ple_dma_pool *p,const uint32_t *epoch,uint32_t expected) {
    uint64_t until=ple_dma_ns()+10000000ull;unsigned spins=0;
    while(__atomic_load_n(epoch,__ATOMIC_ACQUIRE)!=expected) {
        if(__atomic_load_n(&p->stopping,__ATOMIC_ACQUIRE))return 0;
#if defined(__x86_64__) || defined(__i386__)
        __asm__ __volatile__("pause");
#else
        sched_yield();
#endif
        if(++spins==1024){spins=0;if(ple_dma_ns()>=until){struct timespec nap={0,20000};nanosleep(&nap,NULL);}}
    }
    return !__atomic_load_n(&p->stopping,__ATOMIC_ACQUIRE);
}
static void *ple_dma_worker_main(void *arg) {
    struct ple_dma_worker *w=arg;struct ple_dma_pool *p=w->pool;uint64_t seen=0;
    CUresult bound=cuCtxSetCurrent(w->context);
    pthread_mutex_lock(&p->mutex);w->status=bound;p->bound++;pthread_cond_broadcast(&p->done);
    while(!p->stopping && bound==CUDA_SUCCESS) {
        while(!p->stopping && seen==p->generation)pthread_cond_wait(&p->work,&p->mutex);
        if(p->stopping)break;
        unsigned q=p->position;const void *src=p->payload;const uint32_t *epoch=p->epoch;uint32_t expected=p->expected_epoch;
        seen=p->generation;pthread_mutex_unlock(&p->mutex);
        if(!ple_dma_wait_ready(p,epoch,expected)){pthread_mutex_lock(&p->mutex);break;}
        uint64_t begin=ple_dma_ns();
        CUresult status=cuMemcpyHtoDAsync(w->rows+(size_t)q*1440,src,1440,w->stream);
        if(status==CUDA_SUCCESS)status=cuMemcpyHtoDAsync(w->ready+(size_t)q*4,epoch,4,w->stream);
        uint64_t end=ple_dma_ns();
        pthread_mutex_lock(&p->mutex);
        w->status=status;w->begin_ns=begin;w->end_ns=end;p->finished++;
        if(p->finished==p->count)pthread_cond_broadcast(&p->done);
    }
    pthread_mutex_unlock(&p->mutex);
    if(bound==CUDA_SUCCESS)cuCtxSetCurrent(NULL);
    return NULL;
}
static void ple_dma_destroy(struct ple_dma_pool *p) {
    if(!p->initialized)return;
    pthread_mutex_lock(&p->mutex);__atomic_store_n(&p->stopping,1,__ATOMIC_RELEASE);
    pthread_cond_broadcast(&p->work);pthread_cond_broadcast(&p->done);pthread_mutex_unlock(&p->mutex);
    for(unsigned d=0;d<p->started;d++)pthread_join(p->threads[d],NULL);
    pthread_cond_destroy(&p->done);pthread_cond_destroy(&p->work);pthread_mutex_destroy(&p->mutex);
    memset(p,0,sizeof *p);
}
static int ple_dma_init(struct ple_dma_pool *p,unsigned n,const CUcontext *contexts,
                        const CUstream *streams,const CUdeviceptr *rows,const CUdeviceptr *ready,
                        CUresult *bind_status) {
    if(n!=6 || p->initialized)return EINVAL;
    memset(p,0,sizeof *p);*bind_status=CUDA_SUCCESS;p->count=n;
    int e=pthread_mutex_init(&p->mutex,NULL);if(e)return e;
    e=pthread_cond_init(&p->work,NULL);if(e){pthread_mutex_destroy(&p->mutex);return e;}
    e=pthread_cond_init(&p->done,NULL);if(e){pthread_cond_destroy(&p->work);pthread_mutex_destroy(&p->mutex);return e;}
    p->initialized=1;
    for(unsigned d=0;d<n;d++) {
        p->workers[d]=(struct ple_dma_worker){.pool=p,.die=d,.context=contexts[d],.stream=streams[d],.rows=rows[d],.ready=ready[d]};
        e=pthread_create(p->threads+d,NULL,ple_dma_worker_main,p->workers+d);
        if(e){ple_dma_destroy(p);return e;}p->started++;
    }
    pthread_mutex_lock(&p->mutex);
    while(p->bound<n)pthread_cond_wait(&p->done,&p->mutex);
    for(unsigned d=0;d<n;d++)if(p->workers[d].status!=CUDA_SUCCESS)*bind_status=p->workers[d].status;
    pthread_mutex_unlock(&p->mutex);return 0;
}
static CUresult ple_dma_prepare(struct ple_dma_pool *p,unsigned q,const void *payload,
                                const uint32_t *epoch,uint32_t expected) {
    if(!p->initialized || p->stopping)return CUDA_ERROR_INVALID_CONTEXT;
    pthread_mutex_lock(&p->mutex);
    if(p->generation && p->finished!=p->count){pthread_mutex_unlock(&p->mutex);return CUDA_ERROR_NOT_READY;}
    p->position=q;p->payload=payload;p->epoch=epoch;p->expected_epoch=expected;p->finished=0;p->generation++;
    pthread_cond_broadcast(&p->work);
    pthread_mutex_unlock(&p->mutex);return CUDA_SUCCESS;
}
static CUresult ple_dma_finish(struct ple_dma_pool *p,struct ple_dma_span *span) {
    pthread_mutex_lock(&p->mutex);
    while(p->finished<p->count && !p->stopping)pthread_cond_wait(&p->done,&p->mutex);
    if(p->stopping){pthread_mutex_unlock(&p->mutex);return CUDA_ERROR_INVALID_CONTEXT;}
    CUresult status=CUDA_SUCCESS;
    *span=(struct ple_dma_span){.first_begin=UINT64_MAX,.first_end=UINT64_MAX};
    for(unsigned d=0;d<p->count;d++) {
        const struct ple_dma_worker *w=p->workers+d;
        if(w->status!=CUDA_SUCCESS)status=w->status;
        if(w->begin_ns<span->first_begin)span->first_begin=w->begin_ns;
        if(w->begin_ns>span->last_begin)span->last_begin=w->begin_ns;
        if(w->end_ns<span->first_end)span->first_end=w->end_ns;
        if(w->end_ns>span->last_end)span->last_end=w->end_ns;
    }
    pthread_mutex_unlock(&p->mutex);return status;
}
#endif
