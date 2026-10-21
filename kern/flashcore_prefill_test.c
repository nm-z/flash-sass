#include "flashcore_prefill.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>

static void clear_settings(void) {
    assert(unsetenv("FLASH_SOCK") == 0);
    assert(unsetenv("FLASH_PREFILL_CHUNK") == 0);
}

int main(void) {
    clear_settings();
    assert(flashcore_http_prefill_default() == 0);
    assert(getenv("FLASH_PREFILL_CHUNK") == NULL);

    assert(setenv("FLASH_SOCK", "/tmp/flash.sock", 1) == 0);
    assert(flashcore_http_prefill_default() == 1);
    assert(strcmp(getenv("FLASH_PREFILL_CHUNK"), "64") == 0);
    assert(flashcore_http_prefill_default() == 0);

    assert(setenv("FLASH_PREFILL_CHUNK", "0", 1) == 0);
    assert(flashcore_http_prefill_default() == 0);
    assert(strcmp(getenv("FLASH_PREFILL_CHUNK"), "0") == 0);
    assert(setenv("FLASH_PREFILL_CHUNK", "128", 1) == 0);
    assert(flashcore_http_prefill_default() == 0);
    assert(strcmp(getenv("FLASH_PREFILL_CHUNK"), "128") == 0);

    clear_settings();
    assert(setenv("FLASH_PREFILL_CHUNK", "32", 1) == 0);
    assert(flashcore_http_prefill_default() == 0);
    assert(strcmp(getenv("FLASH_PREFILL_CHUNK"), "32") == 0);
    clear_settings();
    puts("flashcore HTTP prefill default: socket scope and explicit overrides PASS");
    return 0;
}
