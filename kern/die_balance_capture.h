#ifndef DIE_BALANCE_CAPTURE_H
#define DIE_BALANCE_CAPTURE_H

#include <json-c/json.h>
#include <sys/types.h>

typedef struct dc_relay dc_relay;

int dc_validation_lock(const char *path, const char *job_id, const char *descriptor,
                      int *lease_fd, char **error);
int dc_engine_mapping(pid_t pid, const char *maps, char ***fields, char **error);
int dc_nvidia_rows(const char *raw, int mode, json_object **rows, char **error);
int dc_stop_owned_child(pid_t pid);
dc_relay *dc_relay_open(const char *path, const char *upstream, pid_t core_pid,
                        const char *log_path, char **error);
int dc_relay_check(dc_relay *relay, char **error);
int dc_relay_pop(dc_relay *relay, json_object **evidence, char **error);
int dc_relay_close(dc_relay *relay, char **error);
int dc_capture_run(int argc, char **argv);

#endif
