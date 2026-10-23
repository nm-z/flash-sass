import halide as hl, numpy as np
x, xo, xi = hl.Var("x"), hl.Var("xo"), hl.Var("xi")
f = hl.Func("f")
f[x] = hl.f32(x) * 2.0
f.gpu_tile(x, xo, xi, 256)
t = hl.get_host_target().with_feature(hl.TargetFeature.CUDA).with_feature(hl.TargetFeature.CUDACapability50)
out = f.realize([1 << 20], t)
print("halide", np.asarray(out)[1000], t)
