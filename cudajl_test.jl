using CUDA
CUDA.versioninfo()
a = CUDA.zeros(Float32, 1 << 20)
function k(a)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= length(a)
        a[i] = 2f0 * (i - 1)
    end
    return
end
@cuda threads=256 blocks=cld(length(a), 256) k(a)
println("cudajl a[1001]=", Array(a)[1001], " capability=", CUDA.capability(device()))
