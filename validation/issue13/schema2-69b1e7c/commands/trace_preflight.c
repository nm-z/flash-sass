#define _POSIX_C_SOURCE 200809L
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int main(int argc, char **argv) {
	if (argc != 2 || !argv[1][0]) {
		fprintf(stderr, "usage: trace_preflight OUTPUT_DIRECTORY\n");
		return 2;
	}
	const char *runner = getenv("FLASH_TRACE_VALIDATION_RUNNER");
	const char *config = getenv("FLASH_TRACE_VALIDATION_CONFIG");
	if (!runner || !runner[0] || !config || !config[0]) {
		fprintf(stderr, "trace_preflight: missing runner/config environment\n");
		return 2;
	}
	char *const args[] = {
		"ruby", (char *)runner, "--config", (char *)config,
		"--preflight", argv[1], NULL
	};
	execvp(args[0], args);
	fprintf(stderr, "trace_preflight: cannot exec ruby: %s\n", strerror(errno));
	return 127;
}
