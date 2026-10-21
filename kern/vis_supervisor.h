#ifndef VIS_SUPERVISOR_H
#define VIS_SUPERVISOR_H

/* Call before CUDA/backend initialization. Returns -1 for the parent, or
 * the private connected AF_UNIX stream fd for the GPU worker. The worker's
 * final "--worker-fd N" pair is removed from argc/argv; other args retain
 * their values/order. Invalid reserved arguments return -1 with errno set
 * and are rejected by vis_supervisor_main with exit status 2.
 * Worker: initialize encoder; write exactly "ready\n"; serve enc requests
 * on this fd. Initialization failures exit normally with their status.
 */
int vis_supervisor_worker_fd(int *argc, char **argv);

/* Parent entry: argv[3] is the external Unix socket path. Fork+exec the
 * same binary with the original args plus "--worker-fd N". Initial worker
 * startup waits up to 120s for ready/exit before binding externally, and
 * preserves nonzero worker exit statuses (including invalid/no-GPU exit 2).
 * Serve serial clients and persistent enc requests; no request deadline.
 * A complete successful header/body is buffered before publication.
 * Worker failure: "err GPU encoder worker terminated\n", reap worker,
 * start a fresh worker for the NEXT request. No current-image retry.
 * In-band err lines are forwarded and their worker is retired as well, so
 * the next request does not race a poisoned-context worker's shutdown.
 * SIGTERM/SIGINT/SIGHUP shut down and reap the worker and remove the socket.
 * Must run in the parent before CUDA or other worker threads initialize.
 */
int vis_supervisor_main(int argc, char **argv);

#endif
