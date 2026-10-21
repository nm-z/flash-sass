#include "test_fixture.h"
#include <errno.h>
#include <string.h>

int main(int argc, char **argv) {
    int acceptance = argc == 3 && !strcmp(argv[1], "acceptance");
    if (argc != 2 && !acceptance) {
        fprintf(stderr, "usage: %s OUTPUT | acceptance OUTPUT\n", argv[0]);
        return 1;
    }
    const char *output = argv[acceptance ? 2 : 1];
    FILE *file = fopen(output, "wb");
    if (!file) {
        fprintf(stderr, "%s: %s\n", output, strerror(errno));
        return 1;
    }
    int result = acceptance ? issue80_write_acceptance_fixture(file) : issue80_write_fixture(file);
    if (fclose(file) != 0) result = -1;
    if (result < 0) {
        fprintf(stderr, "write %s: %s\n", output, strerror(errno));
        return 1;
    }
    return 0;
}
