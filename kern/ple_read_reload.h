#ifndef FLASH_PLE_READ_RELOAD_H
#define FLASH_PLE_READ_RELOAD_H
#include "ple_read_pool.h"
#include <stdlib.h>

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
