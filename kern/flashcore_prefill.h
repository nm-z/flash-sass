#ifndef FLASHCORE_PREFILL_H
#define FLASHCORE_PREFILL_H

#include <stdlib.h>

/* Socket clients such as flashd should use paired prompt chunks by default.
 * An explicit FLASH_PREFILL_CHUNK value, including 0, remains authoritative. */
static inline int flashcore_http_prefill_default(void) {
    if (!getenv("FLASH_SOCK") || getenv("FLASH_PREFILL_CHUNK")) return 0;
    return setenv("FLASH_PREFILL_CHUNK", "64", 0) == 0 ? 1 : -1;
}

#endif
