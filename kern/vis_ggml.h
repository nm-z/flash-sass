#ifndef VIS_GGML_H
#define VIS_GGML_H
#include "ggml-backend.h"

void vis_ggml_begin(ggml_backend_dev_t device);
bool vis_ggml_observe(struct ggml_tensor *tensor, bool ask, void *unused);
const char *vis_ggml_result(void);

#endif
