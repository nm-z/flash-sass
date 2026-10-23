import taichi as ti
ti.init(arch=ti.cuda, offline_cache=False, device_memory_GB=1.0)
arch = ti.lang.impl.current_cfg().arch
x = ti.field(ti.f32, 1 << 20)
@ti.kernel
def k():
    for i in x:
        x[i] = i * 2.0
k()
print("taichi", ti.__version__, "arch", arch, "x[1000]", x[1000])
assert arch == ti.cuda, "fell back off CUDA"
