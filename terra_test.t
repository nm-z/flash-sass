local C = terralib.includec("stdio.h")
local tid = cudalib.nvvm_read_ptx_sreg_tid_x
terra k(a : &float) a[tid()] = 2.0f * tid() end
local M = terralib.cudacompile({ k = k })
terra main() : int
  var d : &float
  C.printf("cudaMalloc %d\n", cudalib.cuMemAlloc_v2([&uint64](&d), 4 * 256))
  var launch = terralib.CUDAParams { 1,1,1, 256,1,1, 0, nil }
  M.k(&launch, d)
  var h : float[256]
  cudalib.cuMemcpyDtoH_v2(&h, [uint64](d), 4 * 256)
  C.printf("terra k[100]=%f\n", h[100])
  return 0
end
main()
