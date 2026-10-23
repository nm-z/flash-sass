#ifndef FLASH_PLE_READ_RELOAD_H
#define FLASH_PLE_READ_RELOAD_H
#include "ple_read_pool.h"
#include <stdlib.h>

/* The core keeps this allocation across library reloads, but a freshly loaded
 * engine restarts its request IDs at 1. Call after all old kernels and row
 * readers have stopped, before launching or staging for the new engine. Use the
 * core's capacity, including its two guard positions, rather than an engine's
 * possibly smaller FLASH_ENGINE_MAXT. The row bytes themselves remain intact. */
static uint32_t *ple_read_reset_ready(void *host, size_t ready_offset, size_t maxt) {
	uint32_t *ready = (uint32_t *)((uint8_t *)host + ready_offset);
	memset(ready, 0, (maxt + 2) * sizeof *ready);
	return ready;
}

/* Inspect every override before changing the environment or unloading.
 * Delimiters match engine_session's reload parser. */
static int ple_read_reload_preflight(const char *arguments) {
	size_t bytes = strlen(arguments) + 1;
	char *copy = malloc(bytes);
	if (!copy) return ENOMEM;
	memcpy(copy, arguments, bytes);
	const char name[] = "FLASH_PLE_READERS=";
	char *next = NULL;
	int error = 0;
	for (char *token = strtok_r(copy, " \t\n", &next); token;
	     token = strtok_r(NULL, " \t\n", &next)) {
		if (!strncmp(token, name, sizeof name - 1)) {
			unsigned readers;
			error = ple_read_parse_readers(token + sizeof name - 1, &readers);
			if (error) break;
		}
	}
	free(copy);
	return error;
}
#endif
