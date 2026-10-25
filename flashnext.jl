# flashnext.jl: Qwen3.8-Flash-Next (GSQ-RCO Q2_0) inference with DFlash2 speculation on Tesla M60 dies (sm_52).
# One source file. Run inside the juliac runner (jlrun flashnext.jl ARGS...) or build it with juliac.
using CUDA, Printf
lmod(n) = first(m for (k, m) in Base.loaded_modules if k.name == n)
const GPUC = lmod("GPUCompiler")
const CUC = lmod("CUDACore")       # the raw driver API (cuMemAlloc_v2, ...)
const UA = lmod("UnsafeAtomics")   # relaxed loads/stores at system scope (the LL words and flags)

# ---------------------------------------------------------------------------------------------
# 1. Observation: every measurement is one JSON line that gpuq ingests into the job's database.
# ---------------------------------------------------------------------------------------------
jstr(s::AbstractString) = "\"" * escape_string(s) * "\""
jval(v::AbstractString) = jstr(v)
jval(v::Bool) = v ? "true" : "false"
jval(v::Integer) = string(v)
jval(v::AbstractFloat) = isfinite(v) ? @sprintf("%.6g", v) : "null"
jval(v::Symbol) = jstr(String(v))
function metric(name, unit, value; dims...)
    d = join((jstr(String(k)) * ":" * jval(v) for (k, v) in dims), ",")
    println("{\"gpuq_metric\":", jstr(name), ",\"unit\":", jstr(unit), ",\"value\":", jval(value), ",\"dimensions\":{", d, "}}")
    flush(stdout)
end
note(s...) = (println(stderr, s...); flush(stderr))

# ---------------------------------------------------------------------------------------------
# 2. GGUF reader: mmap, v3 metadata, the tensor table, split shards (every offset from the file)
# ---------------------------------------------------------------------------------------------
# type id => (name, elements per block, bytes per block) (ggml type ids)
const GGT = Dict{Int,Tuple{String,Int,Int}}(
    0 => ("F32", 1, 4), 1 => ("F16", 1, 2), 2 => ("Q4_0", 32, 18), 6 => ("Q5_0", 32, 22), 8 => ("Q8_0", 32, 34),
    11 => ("Q3_K", 256, 110), 12 => ("Q4_K", 256, 144), 13 => ("Q5_K", 256, 176), 14 => ("Q6_K", 256, 210),
    20 => ("IQ4_NL", 32, 18), 23 => ("IQ4_XS", 256, 136), 30 => ("BF16", 1, 2), 42 => ("Q2_0", 64, 18))
const T_F32, T_F16, T_Q4_0, T_Q5_0, T_Q8_0, T_Q3_K, T_Q4_K, T_Q5_K, T_Q6_K, T_IQ4_NL, T_IQ4_XS, T_BF16, T_Q2_0 =
    0, 1, 2, 6, 8, 11, 12, 13, 14, 20, 23, 30, 42
tname(t) = haskey(GGT, t) ? GGT[t][1] : "type$t"

struct GTensor
    name::String
    typ::Int
    dims::Vector{Int}      # ggml order: dims[1] is the contiguous (row) length
    shard::Int
    off::Int               # absolute byte offset in the shard file
    nbytes::Int
end
nrows(t::GTensor) = prod(t.dims) ÷ t.dims[1]
rowbytes(t::GTensor) = (b = GGT[t.typ]; t.dims[1] ÷ b[2] * b[3])

struct GModel
    paths::Vector{String}
    maps::Vector{Vector{UInt8}}
    kv::Dict{String,Any}
    tensors::Dict{String,GTensor}
    order::Vector{String}
end

function mmap_file(path)
    fd = ccall(:open, Cint, (Cstring, Cint), path, 0)
    fd < 0 && error("open $path failed")
    len = filesize(path)
    p = ccall(:mmap, Ptr{UInt8}, (Ptr{Cvoid}, Csize_t, Cint, Cint, Cint, Int64), C_NULL, len, 1, 1, fd, 0)
    p == Ptr{UInt8}(typemax(UInt)) && error("mmap $path failed")
    ccall(:close, Cint, (Cint,), fd)
    unsafe_wrap(Array, p, len)
end

mutable struct Cur
    m::Vector{UInt8}
    p::Int                 # 0-based offset
end
function rd(c::Cur, ::Type{T}) where {T}
    c.p + sizeof(T) > length(c.m) && error("gguf: read past end")
    v = GC.@preserve c unsafe_load(Ptr{T}(pointer(c.m) + c.p))
    c.p += sizeof(T)
    v
end
function rds(c::Cur)
    n = Int(rd(c, UInt64))
    s = String(c.m[c.p+1:c.p+n])
    c.p += n
    s
end
const GSCALAR = Dict(0 => UInt8, 1 => Int8, 2 => UInt16, 3 => Int16, 4 => UInt32, 5 => Int32, 6 => Float32, 7 => Bool, 10 => UInt64, 11 => Int64, 12 => Float64)
function rdval(c::Cur, t)
    t == 8 && return rds(c)
    if t == 9
        et = Int(rd(c, UInt32)); n = Int(rd(c, UInt64))
        et == 8 && return [rds(c) for _ in 1:n]
        T = GSCALAR[et]
        v = Vector{T}(undef, n)
        GC.@preserve c v unsafe_copyto!(Ptr{UInt8}(pointer(v)), pointer(c.m) + c.p, n * sizeof(T))
        c.p += n * sizeof(T)
        return v
    end
    rd(c, GSCALAR[t])
end

function gguf_open(paths::Vector{String}; header_only = false)
    maps = Vector{UInt8}[]
    kv = Dict{String,Any}()
    tens = Dict{String,GTensor}()
    order = String[]
    for (si, path) in enumerate(paths)
        m = mmap_file(path)
        push!(maps, m)
        c = Cur(m, 0)
        String(m[1:4]) == "GGUF" || error("$path: not GGUF")
        c.p = 4
        ver = rd(c, UInt32); ver == 3 || error("$path: gguf v$ver")
        nt = Int(rd(c, UInt64)); nkv = Int(rd(c, UInt64))
        align = 32
        for _ in 1:nkv
            k = rds(c); t = Int(rd(c, UInt32)); v = rdval(c, t)
            k == "general.alignment" && (align = Int(v))
            haskey(kv, k) || (kv[k] = v)
        end
        infos = Tuple{String,Int,Vector{Int},Int}[]
        for _ in 1:nt
            name = rds(c); nd = Int(rd(c, UInt32))
            dims = [Int(rd(c, UInt64)) for _ in 1:nd]
            typ = Int(rd(c, UInt32)); off = Int(rd(c, UInt64))
            push!(infos, (name, typ, dims, off))
        end
        data = cld(c.p, align) * align
        for (name, typ, dims, off) in infos
            haskey(GGT, typ) || error("$name: unsupported type $typ")
            b = GGT[typ]
            nb = prod(dims) ÷ b[2] * b[3]
            (header_only || data + off + nb <= length(m)) || error("$name: past end of $path")
            tens[name] = GTensor(name, typ, dims, si, data + off, nb)
            push!(order, name)
        end
    end
    GModel(paths, maps, kv, tens, order)
end
# --merge OUT: the model's files as one GGUF (v3): the first file's metadata (its entries' bytes as they are, the
# split.* keys left out), every file's tensors in order, each at the next 32-byte aligned offset
function merge_gguf(paths::Vector{String}, out::String)
    g = gguf_open(paths)
    m = g.maps[1]; c = Cur(m, 8)
    nt0 = Int(rd(c, UInt64)); nkv = Int(rd(c, UInt64))
    kvs = Vector{UInt8}[]
    for _ in 1:nkv
        a = c.p; k = rds(c); t = Int(rd(c, UInt32)); rdval(c, t)
        startswith(k, "split.") || push!(kvs, m[a+1:c.p])
    end
    align = Int(get(g.kv, "general.alignment", 32))
    t0 = time()
    open(out, "w") do io
        write(io, b"GGUF", UInt32(3), UInt64(length(g.order)), UInt64(length(kvs)))
        foreach(b -> write(io, b), kvs)
        off = 0; offs = Int[]
        for n in g.order
            t = g.tensors[n]
            write(io, UInt64(ncodeunits(n)), n, UInt32(length(t.dims)), UInt64.(t.dims)..., UInt32(t.typ), UInt64(off))
            push!(offs, off); off = cld(off + t.nbytes, align) * align
        end
        write(io, zeros(UInt8, cld(position(io), align) * align - position(io)))
        base = position(io)
        for (n, o) in zip(g.order, offs)
            t = g.tensors[n]
            write(io, zeros(UInt8, base + o - position(io)))
            write(io, tbytes(g, t))
        end
    end
    note(@sprintf("merge: %d tensors, %d metadata entries, %.1f GB to %s in %.0f s", length(g.order), length(kvs), filesize(out) / 1e9, out, time() - t0))
    0
end
tbytes(g::GModel, t::GTensor) = view(g.maps[t.shard], t.off+1:t.off+t.nbytes)
tptr(g::GModel, t::GTensor) = pointer(g.maps[t.shard]) + t.off

const SHARDS = ["/home/nate/flash-ple/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf",
                "/home/nate/flash-ple/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf"]
# the model's files (--model): a merged GGUF alone, or every shard of a split given its first (NAME-0000k-of-0000n.gguf);
# tensors are found in whichever file holds them either way
const MODEL = Ref(SHARDS)
function model_paths(p::AbstractString)
    m = match(r"^(.*)-(\d{5})-of-(\d{5})\.gguf$", p)
    m === nothing && return [String(p)]
    n = parse(Int, m[3])
    [@sprintf("%s-%05d-of-%05d.gguf", m[1], i, n) for i in 1:n]
end
const DRAFT = "/home/nate/models-cache/flash-sass/dflash2-q4km/Qwen3.8-Flash-Next-DFlash2-Q4_K_M.gguf"

function show_meta(g::GModel)
    for k in sort(collect(keys(g.kv)))
        v = g.kv[k]
        if v isa Vector
            s = length(v) <= 24 ? string(v) : string(v[1:8], " ... (", length(v), ")")
            println(k, " = ", first(s, 400))
        else
            println(k, " = ", first(string(v), 400))
        end
    end
    kinds = Dict{String,Int}()
    for n in g.order
        t = g.tensors[n]
        k = replace(n, r"blk\.\d+\." => "blk.N.") * " " * tname(t.typ) * " " * join(t.dims, "x")
        kinds[k] = get(kinds, k, 0) + 1
    end
    for k in sort(collect(keys(kinds)))
        println(kinds[k], "\t", k)
    end
end

# ---------------------------------------------------------------------------------------------
# 3. Weight formats: host dequantisation (ggml layouts, ref/quant.c), and the device layout.
#    Device layout of a quantised tensor ("split"): all payloads (codes, 16-byte multiples) for
#    blocks in row-major block order, then all headers (scales) in the same order. Dense types
#    (BF16, F16, F32) stay row-major as stored, with 32-element "blocks" of 64 or 128 bytes.
# ---------------------------------------------------------------------------------------------
const KV_IQ4 = Float32[-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]
@inline lu8(p::Ptr{UInt8}, i) = unsafe_load(p + i)
@inline lu16(p::Ptr{UInt8}, i) = unsafe_load(Ptr{UInt16}(p + i))
@inline lu32(p::Ptr{UInt8}, i) = unsafe_load(Ptr{UInt32}(p + i))
@inline h2f(h::UInt16) = Float32(reinterpret(Float16, h))
@inline b2f(h::UInt16) = reinterpret(Float32, UInt32(h) << 16)

function scale_min_k4(j, q::Ptr{UInt8})
    if j < 4
        return Int(lu8(q, j) & 63), Int(lu8(q, j + 4) & 63)
    end
    return Int((lu8(q, j + 4) & 0x0f) | ((lu8(q, j - 4) >> 6) << 4)), Int((lu8(q, j + 4) >> 4) | ((lu8(q, j) >> 6) << 4))
end

function q3k_scales(x::Ptr{UInt8})
    a0, a1, tmp = lu32(x, 96), lu32(x, 100), lu32(x, 104)
    km1, km2 = 0x03030303, 0x0f0f0f0f
    a2 = ((a0 >> 4) & km2) | (((tmp >> 4) & km1) << 4)
    a3 = ((a1 >> 4) & km2) | (((tmp >> 6) & km1) << 4)
    a0n = (a0 & km2) | ((tmp & km1) << 4)
    a1n = (a1 & km2) | (((tmp >> 2) & km1) << 4)
    w = (a0n, a1n, a2, a3)
    ntuple(i -> Int(reinterpret(Int8, UInt8((w[(i-1)>>2+1] >> (8 * ((i - 1) & 3))) & 0xff))) - 32, 16)
end

# decode one block of type t at x (ggml layout) into y[o+1 : o+E]
function dequant_block!(y::AbstractVector{Float32}, o::Int, t::Int, x::Ptr{UInt8})
    if t == T_F32
        y[o+1] = unsafe_load(Ptr{Float32}(x))
    elseif t == T_F16
        y[o+1] = h2f(lu16(x, 0))
    elseif t == T_BF16
        y[o+1] = b2f(lu16(x, 0))
    elseif t == T_Q2_0
        d = h2f(lu16(x, 0))
        for j in 0:63
            y[o+j+1] = d * Float32(((lu8(x, 2 + (j >> 2)) >> (2 * (j & 3))) & 3) - 1)
        end
    elseif t == T_Q4_0 || t == T_IQ4_NL
        d = h2f(lu16(x, 0))
        for j in 0:15
            b = lu8(x, 2 + j)
            lo, hi = Int(b & 0x0f), Int(b >> 4)
            y[o+j+1] = t == T_Q4_0 ? d * (lo - 8) : d * KV_IQ4[lo+1]
            y[o+j+17] = t == T_Q4_0 ? d * (hi - 8) : d * KV_IQ4[hi+1]
        end
    elseif t == T_Q5_0
        d = h2f(lu16(x, 0)); qh = lu32(x, 2)
        for j in 0:15
            b = lu8(x, 6 + j)
            y[o+j+1] = d * Float32(Int((b & 0x0f) | (((qh >> j) & 1) << 4)) - 16)
            y[o+j+17] = d * Float32(Int((b >> 4) | (((qh >> (j + 16)) & 1) << 4)) - 16)
        end
    elseif t == T_Q8_0
        d = h2f(lu16(x, 0))
        for j in 0:31
            y[o+j+1] = d * Float32(reinterpret(Int8, lu8(x, 2 + j)))
        end
    elseif t == T_Q4_K || t == T_Q5_K
        d, dm = h2f(lu16(x, 0)), h2f(lu16(x, 2))
        qs = t == T_Q4_K ? 16 : 48
        for i in 0:3
            s1, m1 = scale_min_k4(2i, x + 4)
            s2, m2 = scale_min_k4(2i + 1, x + 4)
            for l in 0:31
                b = lu8(x, qs + 32i + l)
                lo, hi = Int(b & 0x0f), Int(b >> 4)
                if t == T_Q5_K
                    h = lu8(x, 16 + l)
                    lo += ((h >> (2i)) & 1) * 16
                    hi += ((h >> (2i + 1)) & 1) * 16
                end
                y[o+64i+l+1] = d * s1 * lo - dm * m1
                y[o+64i+32+l+1] = d * s2 * hi - dm * m2
            end
        end
    elseif t == T_Q6_K
        d = h2f(lu16(x, 208))
        for n in 0:1, l in 0:31
            ql0, ql1, qh = lu8(x, 64n + l), lu8(x, 64n + l + 32), lu8(x, 128 + 32n + l)
            sc(k) = Float32(reinterpret(Int8, lu8(x, 192 + 8n + (l >> 4) + 2k)))
            q1 = Int((ql0 & 0x0f) | ((qh & 3) << 4)) - 32
            q2 = Int((ql1 & 0x0f) | (((qh >> 2) & 3) << 4)) - 32
            q3 = Int((ql0 >> 4) | (((qh >> 4) & 3) << 4)) - 32
            q4 = Int((ql1 >> 4) | (((qh >> 6) & 3) << 4)) - 32
            y[o+128n+l+1] = d * sc(0) * q1
            y[o+128n+l+33] = d * sc(1) * q2
            y[o+128n+l+65] = d * sc(2) * q3
            y[o+128n+l+97] = d * sc(3) * q4
        end
    elseif t == T_Q3_K
        d = h2f(lu16(x, 108)); sc = q3k_scales(x)
        for h in 0:1, j in 0:3, l in 0:31
            q2 = (lu8(x, 32 + 32h + l) >> (2j)) & 3
            hb = (lu8(x, l) >> (4h + j)) & 1
            g = 8h + 2j + (l >> 4)
            y[o+128h+32j+l+1] = d * sc[g+1] * (Int(q2) - (hb == 0 ? 4 : 0))
        end
    elseif t == T_IQ4_XS
        d = h2f(lu16(x, 0)); sh = lu16(x, 2)
        for ib in 0:7
            ls = Int(((lu8(x, 4 + (ib >> 1)) >> (4 * (ib & 1))) & 0x0f) | (((sh >> (2ib)) & 3) << 4))
            dl = d * (ls - 32)
            for j in 0:15
                b = lu8(x, 8 + 16ib + j)
                y[o+32ib+j+1] = dl * KV_IQ4[(b&0x0f)+1]
                y[o+32ib+j+17] = dl * KV_IQ4[(b>>4)+1]
            end
        end
    else
        error("dequant: type $t")
    end
    y
end

function dequant_row!(y::AbstractVector{Float32}, t::Int, x::Ptr{UInt8}, K::Int)
    E, B = GGT[t][2], GGT[t][3]
    for b in 0:K÷E-1
        dequant_block!(y, b * E, t, x + b * B)
    end
    y
end
dequant_row(t, x, K) = dequant_row!(Vector{Float32}(undef, K), t, x, K)

# device layout per type: (payload source offset, payload bytes P, header bytes H, header pieces (src, len, dst))
const SPLIT = Dict{Int,Tuple{Int,Int,Int,Vector{NTuple{3,Int}}}}(
    T_Q2_0 => (2, 16, 2, [(0, 2, 0)]), T_Q4_0 => (2, 16, 2, [(0, 2, 0)]), T_IQ4_NL => (2, 16, 2, [(0, 2, 0)]),
    T_Q5_0 => (6, 16, 8, [(0, 2, 0), (2, 4, 4)]), T_Q8_0 => (2, 32, 2, [(0, 2, 0)]),
    T_Q4_K => (16, 128, 16, [(0, 16, 0)]), T_Q5_K => (16, 160, 16, [(0, 16, 0)]),
    T_Q6_K => (0, 208, 2, [(208, 2, 0)]), T_Q3_K => (0, 96, 16, [(96, 14, 0)]),
    T_IQ4_XS => (8, 128, 8, [(0, 8, 0)]))
# elements per device block, payload and header bytes (plain branches: these also run on the device)
devE(t) = t == T_Q2_0 ? 64 : t in (T_Q3_K, T_Q4_K, T_Q5_K, T_Q6_K, T_IQ4_XS) ? 256 : 32
devP(t) = t in (T_BF16, T_F16) ? 64 : t == T_F32 ? 128 : t in (T_Q2_0, T_Q4_0, T_IQ4_NL, T_Q5_0) ? 16 : t == T_Q8_0 ? 32 :
          t in (T_Q4_K, T_IQ4_XS) ? 128 : t == T_Q5_K ? 160 : t == T_Q6_K ? 208 : t == T_Q3_K ? 96 : 0
devH(t) = t in (T_Q2_0, T_Q4_0, T_IQ4_NL, T_Q8_0, T_Q6_K) ? 2 : t in (T_Q5_0, T_IQ4_XS) ? 8 : t in (T_Q4_K, T_Q5_K, T_Q3_K) ? 16 : 0


# ---------------------------------------------------------------------------------------------
# 4. Kernels. Int32 indices, raw global pointers, 128-bit weight loads.
# ---------------------------------------------------------------------------------------------
# a loop LLVM must not unroll (code size: see bdot(Q6_K))
macro nounroll(ex)
    (ex isa Expr && ex.head in (:for, :while)) || error("@nounroll: not a loop")
    push!(ex.args[2].args, Expr(:loopinfo, (Symbol("llvm.loop.unroll.disable"),)))
    esc(ex)
end
# the K-quant block loops: rolled, or (FLASHNEXT_UNROLL2=1) unrolled by two (two runs' loads in flight, twice the code:
# slower in situ, 27.5 -> 26.4 tok/s)
const UNROLL2 = get(ENV, "FLASHNEXT_UNROLL2", "0") == "1"
macro rolled(ex)
    (ex isa Expr && ex.head in (:for, :while)) || error("@rolled: not a loop")
    push!(ex.args[2].args, UNROLL2 ? Expr(:loopinfo, (Symbol("llvm.loop.unroll.count"), 2)) : Expr(:loopinfo, (Symbol("llvm.loop.unroll.disable"),)))
    esc(ex)
end
const GPtr = Core.LLVMPtr{UInt8,1}
const FPtr = Core.LLVMPtr{Float32,1}
const U64P = Core.LLVMPtr{UInt64,1}
const SPtr = Core.LLVMPtr{Float32,3}
const V4I = NTuple{4,VecElement{Int32}}
const V4F = NTuple{4,VecElement{Float32}}
const FULL = 0xffffffff
gptr(p::CuPtr) = reinterpret(GPtr, p)
fptr(p::CuPtr) = reinterpret(FPtr, p)
gptr(a::CuArray) = gptr(pointer(a))
fptr(a::CuArray{Float32}) = fptr(pointer(a))

@inline function ld16(p::GPtr, off::Int32)
    v = CUDA.unsafe_cached_load(reinterpret(Core.LLVMPtr{V4I,1}, p + off), 1, Val(16))
    (reinterpret(UInt32, v[1].value), reinterpret(UInt32, v[2].value), reinterpret(UInt32, v[3].value), reinterpret(UInt32, v[4].value))
end
@inline ldu16(p::GPtr, off::Int32) = CUDA.unsafe_cached_load(reinterpret(Core.LLVMPtr{UInt16,1}, p + off), 1, Val(2))
@inline ldu32(p::GPtr, off::Int32) = CUDA.unsafe_cached_load(reinterpret(Core.LLVMPtr{UInt32,1}, p + off), 1, Val(4))
@inline ldu64(p::GPtr, off::Int32) = CUDA.unsafe_cached_load(reinterpret(Core.LLVMPtr{UInt64,1}, p + off), 1, Val(8))
@inline sx4(xs::SPtr, i::Int32) = unsafe_load(reinterpret(Core.LLVMPtr{V4F,3}, xs + 4i), 1, Val(16))
@inline sld(xs::SPtr, i::Int32) = unsafe_load(xs + 4i, 1, Val(4))
@inline hf(h::UInt16) = Float32(reinterpret(Float16, h))
@inline hf(h::UInt32) = Float32(reinterpret(Float16, (h & 0xffff) % UInt16))
# exact small unsigned integer -> float (magic exponent)
@inline u2f(q::UInt32) = reinterpret(Float32, q | 0x4b000000) - 8388608.0f0
@inline lsr(x::UInt32, n) = x >> (n % UInt32)
# sum over the L lanes of a row (lanes rho L .. rho L + L - 1). L a power of two: butterfly, the sum in every lane;
# otherwise a tree of shfl_down, the sum in the row's lane j = 0
@inline function wsum(acc::Float32, ::Val{L}) where {L}
    L == 1 && return acc
    L & (L - 1) == 0 && return wsum(acc + shfl_xor_sync(FULL, acc, L ÷ 2), Val(L ÷ 2))
    wsumseg(acc, ((threadIdx().x - Int32(1)) % Int32(32)) % Int32(L), Val(L))
end
# (a loop over the constant offsets, unrolled by LLVM: the recursive form over Val(2O) was left as nested calls, 488
# CALs in k_token!'s SASS)
@inline function wsumseg(acc::Float32, j::Int32, ::Val{L}) where {L}
    o = Int32(1)
    while o < Int32(L)
        u = shfl_down_sync(FULL, acc, o)
        acc = j + o < Int32(L) ? acc + u : acc
        o <<= Int32(1)
    end
    acc
end

# xs layout, the same for every block size: x[k] at xu(k) = k + 4 (k ÷ 32) + 4 (k ÷ 256). Lanes reading blocks
# b = j + L i of 32, 64 or 256 elements then hit distinct banks (block strides 36, 72, 292 floats, mod 32 = 4, 8, 4).
# Within a block starting at a multiple of its size, element c sits at xu(start) + xo(c). xs16[g] = sum of x[16g : 16g + 15].
@inline xu(k::Int32) = k + Int32(4) * (k >> 5) + Int32(4) * (k >> 8)
xu(k::Integer) = k + 4 * (k >> 5) + 4 * (k >> 8)
@inline xo(c::Int) = c + 4 * (c ÷ 32)
@inline sxo(xs::SPtr, xb::Int32, c::Int) = sx4(xs, xb + Int32(xo(c)))

# the nb-bit code at bits p of w as the float 1 + q 2^-nb, exact: one shift and one LOP3 (u2f takes shift, and, or, add).
# Sums of these against x carry sum(x), which the callers remove with the 16-sums.
@inline function mant(w::UInt32, p::Int, nb::Int)
    sh = 23 - nb - p
    ws = sh >= 0 ? w << sh : w >> (-sh)
    reinterpret(Float32, (ws & (UInt32((1 << nb) - 1) << (23 - nb))) | 0x3f800000)
end
# sum (1 + q_e / 4) x_e over 16 2-bit codes of word w at x elements c0 .. c0+15 of the block at xb
@inline function dot2b16(w::UInt32, xs::SPtr, xb::Int32, c0::Int)
    s = 0.0f0
    Base.Cartesian.@nexprs 4 c -> begin
        xv = sxo(xs, xb, c0 + 4(c - 1))
        Base.Cartesian.@nexprs 4 e -> (s = fma(mant(w, 2 * (4(c - 1) + e - 1), 2), xv[e].value, s))
    end
    s
end
# 4-bit codes: elements 4c..4c+3 of word w (nibbles at bits sh, sh+8, sh+16, sh+24)
@inline function dot4b4(w::UInt32, sh, xv::V4F, s::Float32)
    Base.Cartesian.@nexprs 4 e -> (s = fma(u2f((w >> (8(e - 1) + sh)) & 0xf), xv[e].value, s))
    s
end
# the same as sum (1 + q_e / 16) x_e
@inline function dot4b4m(w::UInt32, sh, xv::V4F, s::Float32)
    Base.Cartesian.@nexprs 4 e -> (s = fma(mant(w, 8(e - 1) + sh, 4), xv[e].value, s))
    s
end
# the 16 IQ4_NL values + 128 as bytes, for PRMT lookups (entries 0-7 in NL0/NL1, 8-15 in NL2/NL3)
const NL0, NL1, NL2, NL3 = 0x3f2d1801, 0x766a5d4f, 0xa6998d81, 0xf1d9c5b5
# byte e of r as the low byte of 2^23 (PRMT), less 2^23 + 128: the IQ4_NL value, exact (all 8 bits of the byte are
# mantissa here; in byte 2 of 0.5, as dotp4, bit 7 would land in the exponent)
@inline nlf(r::Union{Int32,UInt32}, e::Int) =reinterpret(Float32, CUDA.byte_perm(r, 0x4b000000, UInt32(0x7540 | e))) - 8388736.0f0
# sum value x over the 8 codes of word q (low nibbles: x elements xa, high: xc): each half-word's 4 nibbles index both
# table halves (2 PRMT), bit 3 picks between them (1 PRMT); no shared-memory table reads
@inline function dotnl8(q::UInt32, xa::V4F, xc::V4F)
    v = q & 0x77777777; hb = q & 0x88888888
    lo0 = CUDA.byte_perm(NL0, NL1, v); hi0 = CUDA.byte_perm(NL2, NL3, v)
    lo1 = CUDA.byte_perm(NL0, NL1, v >> 16); hi1 = CUDA.byte_perm(NL2, NL3, v >> 16)
    r0 = CUDA.byte_perm(lo0, hi0, 0x3210 | ((hb >> 1) & 0x4444)); r1 = CUDA.byte_perm(lo1, hi1, 0x3210 | (hb >> 17))
    # r0 bytes: elements (0, 16, 1, 17) of the word's slots, r1: (2, 18, 3, 19)
    s = nlf(r0, 0) * xa[1].value; t = nlf(r0, 1) * xc[1].value
    s = fma(nlf(r0, 2), xa[2].value, s); t = fma(nlf(r0, 3), xc[2].value, t)
    s = fma(nlf(r1, 0), xa[3].value, s); t = fma(nlf(r1, 1), xc[3].value, t)
    s = fma(nlf(r1, 2), xa[4].value, s); t = fma(nlf(r1, 3), xc[4].value, t)
    s + t
end
const IQ4PRMT = get(ENV, "FLASHNEXT_IQ4PRMT", "1") == "1"
@inline function dot4b4lut(w::UInt32, sh, xv::V4F, s::Float32, lut::SPtr)
    Base.Cartesian.@nexprs 4 e -> (s = fma(sld(lut, Int32((w >> (8(e - 1) + sh)) & 0xf)), xv[e].value, s))
    s
end

# one block's dot product with x. pay/hdr point at the block; xb = x index of its first element (xs layout); g16 = its first 16-group
# The formats of 1-4 chunks per block are split into the block's loads (bload) and its arithmetic (bcomp), so a row
# can have block i + 1's loads in flight while block i is computed (rowdotp); bdot is the two together.
@inline bload(::Val{T_Q2_0}, pay::GPtr, hdr::GPtr) = (pld(pay, (0)), ldu16(hdr, Int32(0)))
@inline function bcomp(::Val{T_Q2_0}, r, xs::SPtr, x16::SPtr, xb::Int32, g16::Int32, lut::SPtr)
    v, h = r; d = hf(h)
    if Q2PRMT
        # codes k, 4 + k, 8 + k, 12 + k of a word (one per byte) rotated to bits 5-6 and permuted to 0.5 + q / 8:
        # sum (q - 1) x = 8 sum f x - 5 X
        X = (sld(x16, g16) + sld(x16, g16 + Int32(1))) + (sld(x16, g16 + Int32(2)) + sld(x16, g16 + Int32(3)))
        Base.Cartesian.@nexprs 4 w -> begin
            xa_w = sxo(xs, xb, 16(w - 1)); xb_w = sxo(xs, xb, 16(w - 1) + 4); xc_w = sxo(xs, xb, 16(w - 1) + 8); xd_w = sxo(xs, xb, 16(w - 1) + 12)
            s_w = 0.0f0
            Base.Cartesian.@nexprs 4 k -> begin
                q4 = bitrotate(v[w], 5 - 2(k - 1)) & 0x60606060
                s_w = fma(prmtf(q4, 0), xa_w[k].value, s_w); s_w = fma(prmtf(q4, 1), xb_w[k].value, s_w)
                s_w = fma(prmtf(q4, 2), xc_w[k].value, s_w); s_w = fma(prmtf(q4, 3), xd_w[k].value, s_w)
            end
        end
        return d * fma(8.0f0, (s_1 + s_2) + (s_3 + s_4), -5.0f0 * X)
    end
    s = dot2b16(v[1], xs, xb, 0) + dot2b16(v[2], xs, xb, 16) + dot2b16(v[3], xs, xb, 32) + dot2b16(v[4], xs, xb, 48)
    X = (sld(x16, g16) + sld(x16, g16 + Int32(1))) + (sld(x16, g16 + Int32(2)) + sld(x16, g16 + Int32(3)))
    d * fma(4.0f0, s, -5.0f0 * X)          # sum (q - 1) x = 4 (s - X) - X
end
@inline bload(::Val{T_Q4_0}, pay::GPtr, hdr::GPtr) = (pld(pay, (0)), ldu16(hdr, Int32(0)))
@inline function bcomp(::Val{T_Q4_0}, r, xs::SPtr, x16::SPtr, xb::Int32, g16::Int32, lut::SPtr)
    v, h = r; d = hf(h)
    Base.Cartesian.@nexprs 4 w -> (s_w = dot4b4m(v[w], 4, sxo(xs, xb, (16 + 4(w - 1))), dot4b4m(v[w], 0, sxo(xs, xb, (4(w - 1))), 0.0f0)))
    s = (s_1 + s_2) + (s_3 + s_4)                  # one chain per word
    d * fma(16.0f0, s, -24.0f0 * (sld(x16, g16) + sld(x16, g16 + Int32(1))))    # sum (q - 8) x = 16 (s - X) - 8 X
end
@inline bload(::Val{T_IQ4_NL}, pay::GPtr, hdr::GPtr) = (pld(pay, (0)), ldu16(hdr, Int32(0)))
@inline function bcomp(::Val{T_IQ4_NL}, r, xs::SPtr, x16::SPtr, xb::Int32, g16::Int32, lut::SPtr)
    v, h = r; d = hf(h)
    Base.Cartesian.@nexprs 4 w -> (s_w = dot4b4lut(v[w], 4, sxo(xs, xb, (16 + 4(w - 1))), dot4b4lut(v[w], 0, sxo(xs, xb, (4(w - 1))), 0.0f0, lut), lut))
    d * ((s_1 + s_2) + (s_3 + s_4))                # one chain per word
end
@inline bload(::Val{T_Q5_0}, pay::GPtr, hdr::GPtr) = (pld(pay, (0)), ldu64(hdr, Int32(0)))
@inline function bcomp(::Val{T_Q5_0}, r, xs::SPtr, x16::SPtr, xb::Int32, g16::Int32, lut::SPtr)
    v, h = r
    d = hf(UInt16(h & 0xffff)); qh = UInt32(h >> 32)
    s = 0.0f0
    Base.Cartesian.@nexprs 4 w -> begin
        xl = sxo(xs, xb, (4(w - 1))); xh = sxo(xs, xb, (16 + 4(w - 1)))
        Base.Cartesian.@nexprs 4 e -> begin
            j = 4(w - 1) + e - 1
            s = fma(u2f(((v[w] >> (8(e - 1))) & 0xf) | (((qh >> j) & 0x1) << 4)), xl[e].value, s)
            s = fma(u2f(((v[w] >> (8(e - 1) + 4)) & 0xf) | (((qh >> (j + 16)) & 0x1) << 4)), xh[e].value, s)
        end
    end
    d * (s - 16.0f0 * (sld(x16, g16) + sld(x16, g16 + Int32(1))))
end
@inline bload(::Val{T_Q8_0}, pay::GPtr, hdr::GPtr) = (pld(pay, (0)), pld(pay, (16)), ldu16(hdr, Int32(0)))
@inline function bcomp(::Val{T_Q8_0}, r, xs::SPtr, x16::SPtr, xb::Int32, g16::Int32, lut::SPtr)
    v1, v2, h = r; d = hf(h)
    Base.Cartesian.@nexprs 4 w -> (s_w = dot8b4(v2[w] ⊻ 0x80808080, sxo(xs, xb, (16 + 4(w - 1))), dot8b4(v1[w] ⊻ 0x80808080, sxo(xs, xb, (4(w - 1))), 0.0f0)))
    s = (s_1 + s_2) + (s_3 + s_4)                  # one chain per word
    d * (s - 128.0f0 * (sld(x16, g16) + sld(x16, g16 + Int32(1))))
end
@inline bload(::Val{T_BF16}, pay::GPtr, hdr::GPtr) = (pld(pay, (0)), pld(pay, (16)), pld(pay, (32)), pld(pay, (48)))
@inline function bcomp(::Val{T_BF16}, r, xs::SPtr, x16::SPtr, xb::Int32, g16::Int32, lut::SPtr)
    Base.Cartesian.@nexprs 4 c -> begin            # one chain per chunk
        v = r[c]
        xa = sxo(xs, xb, (8(c - 1))); xc = sxo(xs, xb, (8(c - 1) + 4))
        s_c = reinterpret(Float32, v[1] << 16) * xa[1].value; s_c = fma(reinterpret(Float32, v[1] & 0xffff0000), xa[2].value, s_c)
        s_c = fma(reinterpret(Float32, v[2] << 16), xa[3].value, s_c); s_c = fma(reinterpret(Float32, v[2] & 0xffff0000), xa[4].value, s_c)
        s_c = fma(reinterpret(Float32, v[3] << 16), xc[1].value, s_c); s_c = fma(reinterpret(Float32, v[3] & 0xffff0000), xc[2].value, s_c)
        s_c = fma(reinterpret(Float32, v[4] << 16), xc[3].value, s_c); s_c = fma(reinterpret(Float32, v[4] & 0xffff0000), xc[4].value, s_c)
    end
    (s_1 + s_2) + (s_3 + s_4)
end
const SPLITT = (T_Q2_0, T_Q4_0, T_IQ4_NL, T_Q5_0, T_Q8_0, T_BF16)
for T in SPLITT
    @eval @inline bdot(::Val{$T}, pay::GPtr, hdr::GPtr, xs::SPtr, x16::SPtr, xb::Int32, g16::Int32, lut::SPtr) =
        bcomp(Val($T), bload(Val($T), pay, hdr), xs, x16, xb, g16, lut)
end
@inline function dot8b4(w::UInt32, xv::V4F, s::Float32)
    Base.Cartesian.@nexprs 4 e -> (s = fma(u2f((w >> (8(e - 1))) & 0xff), xv[e].value, s))
    s
end
# K-quant scales: 6-bit scale and min j of the 12 bytes in words a (bytes 0-3), b (4-7), c (8-11)
@inline byte(w::UInt32, i) = (w >> (8i)) & 0xff
@inline function kscale(j, a::UInt32, b::UInt32, c::UInt32)
    if j < 4
        return byte(a, j) & 0x3f, byte(b, j) & 0x3f
    end
    return (byte(c, j - 4) & 0xf) | ((byte(a, j - 4) >> 6) << 4), (byte(c, j - 4) >> 4) | ((byte(b, j - 4) >> 6) << 4)
end
@inline function bdot(::Val{T_Q4_K}, pay::GPtr, hdr::GPtr, xs::SPtr, x16::SPtr, xb::Int32, g16::Int32, lut::SPtr)
    h = ld16(hdr, Int32(0))
    d = hf(h[1]); dm = hf(h[1] >> 16)
    acc = 0.0f0; macc = 0.0f0
    i = Int32(0)
    @rolled while i < Int32(4)                 # 64-element runs (rolled, see bdot(Q6_K))
        q1 = ld16(pay, Int32(2) * i * Int32(CHS)); q2 = ld16(pay, (Int32(2) * i + Int32(1)) * Int32(CHS))
        xq = xb + Int32(72) * i                  # xu of element 64 i
        X1 = sld(x16, g16 + Int32(4) * i) + sld(x16, g16 + Int32(4) * i + Int32(1))
        X2 = sld(x16, g16 + Int32(4) * i + Int32(2)) + sld(x16, g16 + Int32(4) * i + Int32(3))
        if Q4PRMT                              # nibbles to bits 3-6 of the bytes, each byte to 0.5 + q / 32
            Base.Cartesian.@nexprs 4 w -> begin
                a_w = dotp4(bitrotate(q1[w], 3) & 0x78787878, sx4(xs, xq + Int32(xo(4(w - 1)))), 0.0f0)
                a_w = dotp4(bitrotate(q2[w], 3) & 0x78787878, sx4(xs, xq + Int32(xo(16 + 4(w - 1)))), a_w)
                b_w = dotp4(bitrotate(q1[w], -1) & 0x78787878, sx4(xs, xq + Int32(xo(32 + 4(w - 1)))), 0.0f0)
                b_w = dotp4(bitrotate(q2[w], -1) & 0x78787878, sx4(xs, xq + Int32(xo(48 + 4(w - 1)))), b_w)
            end
            s1 = fma(32.0f0, (a_1 + a_2) + (a_3 + a_4), -16.0f0 * X1); s2 = fma(32.0f0, (b_1 + b_2) + (b_3 + b_4), -16.0f0 * X2)
        else
            Base.Cartesian.@nexprs 4 w -> begin        # chains per word (8 in flight)
                a_w = dot4b4m(q1[w], 0, sx4(xs, xq + Int32(xo(4(w - 1)))), 0.0f0)
                a_w = dot4b4m(q2[w], 0, sx4(xs, xq + Int32(xo(16 + 4(w - 1)))), a_w)
                b_w = dot4b4m(q1[w], 4, sx4(xs, xq + Int32(xo(32 + 4(w - 1)))), 0.0f0)
                b_w = dot4b4m(q2[w], 4, sx4(xs, xq + Int32(xo(48 + 4(w - 1)))), b_w)
            end
            s1 = (a_1 + a_2) + (a_3 + a_4); s2 = (b_1 + b_2) + (b_3 + b_4)
            s1 = 16.0f0 * (s1 - X1); s2 = 16.0f0 * (s2 - X2)       # sum q x
        end
        sc1, m1 = kscale(Int32(2) * i, h[2], h[3], h[4]); sc2, m2 = kscale(Int32(2) * i + Int32(1), h[2], h[3], h[4])
        acc = fma(u2f(sc1), s1, acc); acc = fma(u2f(sc2), s2, acc)
        macc = fma(u2f(m1), X1, macc); macc = fma(u2f(m2), X2, macc)
        i += Int32(1)
    end
    d * acc - dm * macc
end
@inline function bdot(::Val{T_Q5_K}, pay::GPtr, hdr::GPtr, xs::SPtr, x16::SPtr, xb::Int32, g16::Int32, lut::SPtr)
    h = ld16(hdr, Int32(0))
    d = hf(h[1]); dm = hf(h[1] >> 16)
    qa = pld(pay, (0)); qb = pld(pay, (16))      # qh[0:16], qh[16:32]
    acc = 0.0f0; macc = 0.0f0
    i = Int32(0)
    @rolled while i < Int32(4)                 # 64-element runs (rolled, see bdot(Q6_K))
        q1 = ld16(pay, (Int32(2) + Int32(2) * i) * Int32(CHS)); q2 = ld16(pay, (Int32(3) + Int32(2) * i) * Int32(CHS))
        xq = xb + Int32(72) * i
        X1 = sld(x16, g16 + Int32(4) * i) + sld(x16, g16 + Int32(4) * i + Int32(1))
        X2 = sld(x16, g16 + Int32(4) * i + Int32(2)) + sld(x16, g16 + Int32(4) * i + Int32(3))
        if Q56PRMT
            # 5-bit codes at bits 2-6 of the bytes (nibble rotated to 2-5, high bit to 6), each byte to 0.5 + q / 64
            ra = (Int32(6) - Int32(2) * i) & Int32(31); rb = (Int32(5) - Int32(2) * i) & Int32(31)
            Base.Cartesian.@nexprs 4 w -> begin
                a_w = dotp4((bitrotate(q1[w], 2) & 0x3c3c3c3c) | (bitrotate(qa[w], ra) & 0x40404040), sx4(xs, xq + Int32(xo(4(w - 1)))), 0.0f0)
                a_w = dotp4((bitrotate(q2[w], 2) & 0x3c3c3c3c) | (bitrotate(qb[w], ra) & 0x40404040), sx4(xs, xq + Int32(xo(16 + 4(w - 1)))), a_w)
                b_w = dotp4((bitrotate(q1[w], -2) & 0x3c3c3c3c) | (bitrotate(qa[w], rb) & 0x40404040), sx4(xs, xq + Int32(xo(32 + 4(w - 1)))), 0.0f0)
                b_w = dotp4((bitrotate(q2[w], -2) & 0x3c3c3c3c) | (bitrotate(qb[w], rb) & 0x40404040), sx4(xs, xq + Int32(xo(48 + 4(w - 1)))), b_w)
            end
            s1 = fma(64.0f0, (a_1 + a_2) + (a_3 + a_4), -32.0f0 * X1); s2 = fma(64.0f0, (b_1 + b_2) + (b_3 + b_4), -32.0f0 * X2)
        else
            Base.Cartesian.@nexprs 4 w -> begin        # chains per word (8 in flight)
                a_w = dot5b4(q1[w], qa[w], 0, Int32(2) * i, sx4(xs, xq + Int32(xo(4(w - 1)))), 0.0f0)
                a_w = dot5b4(q2[w], qb[w], 0, Int32(2) * i, sx4(xs, xq + Int32(xo(16 + 4(w - 1)))), a_w)
                b_w = dot5b4(q1[w], qa[w], 4, Int32(2) * i + Int32(1), sx4(xs, xq + Int32(xo(32 + 4(w - 1)))), 0.0f0)
                b_w = dot5b4(q2[w], qb[w], 4, Int32(2) * i + Int32(1), sx4(xs, xq + Int32(xo(48 + 4(w - 1)))), b_w)
            end
            s1 = (a_1 + a_2) + (a_3 + a_4); s2 = (b_1 + b_2) + (b_3 + b_4)
        end
        sc1, m1 = kscale(Int32(2) * i, h[2], h[3], h[4]); sc2, m2 = kscale(Int32(2) * i + Int32(1), h[2], h[3], h[4])
        acc = fma(u2f(sc1), s1, acc); acc = fma(u2f(sc2), s2, acc)
        macc = fma(u2f(m1), X1, macc); macc = fma(u2f(m2), X2, macc)
        i += Int32(1)
    end
    d * acc - dm * macc
end
# 5-bit: nibble at bits sh + 8e of w, high bit at bit hb + 8e of qh word
@inline function dot5b4(w::UInt32, qh::UInt32, sh, hb, xv::V4F, s::Float32)
    Base.Cartesian.@nexprs 4 e -> (s = fma(u2f(((w >> (8(e - 1) + sh)) & 0xf) | (((qh >> (8(e - 1) + hb)) & 0x1) << 4)), xv[e].value, s))
    s
end
# Rolled over its four (n, hh) quarters: a quarter of the code. Unrolled block dots are tens of KB of SASS, and a
# phase that runs them once per warp pays mostly for instruction fetch.
@inline function bdot(::Val{T_Q6_K}, pay::GPtr, hdr::GPtr, xs::SPtr, x16::SPtr, xb::Int32, g16::Int32, lut::SPtr)
    d = hf(ldu16(hdr, Int32(0)))
    sc = pld(pay, (192))
    acc = 0.0f0
    q = Int32(0)
    # quarter q's three chunks; the next quarter's are loaded before this one is computed (in flight together)
    @inline q6c(q) = (ld16(pay, (Int32(4) * (q >> 1) + (q & Int32(1))) * Int32(CHS)), ld16(pay, (Int32(4) * (q >> 1) + Int32(2) + (q & Int32(1))) * Int32(CHS)),
        ld16(pay, (Int32(8) + Int32(2) * (q >> 1) + (q & Int32(1))) * Int32(CHS)))
    cur = q6c(q)
    @rolled while q < Int32(4)
        n = q >> 1; hh = q & Int32(1)            # l in 16 hh .. 16 hh + 15
        nxt = q < Int32(3) ? q6c(q + Int32(1)) : cur
        l0, l1, qh = cur
        s1 = 0.0f0; s2 = 0.0f0; s3 = 0.0f0; s4 = 0.0f0
        xq = xb + Int32(144) * n + Int32(16) * hh
        g = Int32(8) * n + hh                    # scale index for q1 (and its 16-group); q2..q4 at +2, +4, +6
        if Q56PRMT
            # 6-bit codes at bits 1-6 of the bytes (nibble to 1-4, high pair to 5-6), each byte to 0.5 + q / 128:
            # sum (q - 32) x = 128 sum f x - 96 X
            Base.Cartesian.@nexprs 4 w -> begin
                o = 4(w - 1)
                s1 = dotp4((bitrotate(l0[w], 1) & 0x1e1e1e1e) | (bitrotate(qh[w], 5) & 0x60606060), sx4(xs, xq + Int32(xo(o))), s1)
                s2 = dotp4((bitrotate(l1[w], 1) & 0x1e1e1e1e) | (bitrotate(qh[w], 3) & 0x60606060), sx4(xs, xq + Int32(xo(o + 32))), s2)
                s3 = dotp4((bitrotate(l0[w], -3) & 0x1e1e1e1e) | (bitrotate(qh[w], 1) & 0x60606060), sx4(xs, xq + Int32(xo(o + 64))), s3)
                s4 = dotp4((bitrotate(l1[w], -3) & 0x1e1e1e1e) | (bitrotate(qh[w], -1) & 0x60606060), sx4(xs, xq + Int32(xo(o + 96))), s4)
            end
            acc = fma(sbyter(sc, g), fma(128.0f0, s1, -96.0f0 * sld(x16, g16 + g)), acc)
            acc = fma(sbyter(sc, g + Int32(2)), fma(128.0f0, s2, -96.0f0 * sld(x16, g16 + g + Int32(2))), acc)
            acc = fma(sbyter(sc, g + Int32(4)), fma(128.0f0, s3, -96.0f0 * sld(x16, g16 + g + Int32(4))), acc)
            acc = fma(sbyter(sc, g + Int32(6)), fma(128.0f0, s4, -96.0f0 * sld(x16, g16 + g + Int32(6))), acc)
        else
            Base.Cartesian.@nexprs 4 w -> begin
                o = 4(w - 1)
                s1 = dot6b4(l0[w], 0, qh[w], 0, sx4(xs, xq + Int32(xo(o))), s1)
                s2 = dot6b4(l1[w], 0, qh[w], 2, sx4(xs, xq + Int32(xo(o + 32))), s2)
                s3 = dot6b4(l0[w], 4, qh[w], 4, sx4(xs, xq + Int32(xo(o + 64))), s3)
                s4 = dot6b4(l1[w], 4, qh[w], 6, sx4(xs, xq + Int32(xo(o + 96))), s4)
            end
            acc = fma(sbyter(sc, g), s1 - 32.0f0 * sld(x16, g16 + g), acc)
            acc = fma(sbyter(sc, g + Int32(2)), s2 - 32.0f0 * sld(x16, g16 + g + Int32(2)), acc)
            acc = fma(sbyter(sc, g + Int32(4)), s3 - 32.0f0 * sld(x16, g16 + g + Int32(4)), acc)
            acc = fma(sbyter(sc, g + Int32(6)), s4 - 32.0f0 * sld(x16, g16 + g + Int32(6)), acc)
        end
        cur = nxt
        q += Int32(1)
    end
    d * acc
end
# signed byte i (runtime) of 16 bytes
@inline function sbyter(v::NTuple{4,UInt32}, i::Int32)
    k = i >> 2
    w = k == Int32(0) ? v[1] : k == Int32(1) ? v[2] : k == Int32(2) ? v[3] : v[4]
    Float32(reinterpret(Int8, UInt8((w >> (8 * (i & Int32(3)))) & 0xff)))
end
@inline sbyte(v::NTuple{4,UInt32}, i) = Float32(reinterpret(Int8, UInt8((v[i>>2+1] >> (8(i & 3))) & 0xff)))
@inline function dot6b4(w::UInt32, sh, qh::UInt32, hs, xv::V4F, s::Float32)
    Base.Cartesian.@nexprs 4 e -> (s = fma(u2f(((w >> (8(e - 1) + sh)) & 0xf) | (((qh >> (8(e - 1) + hs)) & 0x3) << 4)), xv[e].value, s))
    s
end
@inline function bdot(::Val{T_Q3_K}, pay::GPtr, hdr::GPtr, xs::SPtr, x16::SPtr, xb::Int32, g16::Int32, lut::SPtr)
    hd = ld16(hdr, Int32(0))
    d = hf(hd[4])
    km1, km2 = 0x03030303, 0x0f0f0f0f
    a0, a1, tmp = hd[1], hd[2], hd[3]
    sc = ((a0 & km2) | ((tmp & km1) << 4), (a1 & km2) | (((tmp >> 2) & km1) << 4),
          ((a0 >> 4) & km2) | (((tmp >> 4) & km1) << 4), ((a1 >> 4) & km2) | (((tmp >> 6) & km1) << 4))
    hm1 = pld(pay, (0)); hm2 = pld(pay, (16))
    acc = 0.0f0
    h = Int32(0)
    cur = (ld16(pay, Int32(2) * Int32(CHS)), ld16(pay, Int32(3) * Int32(CHS)))
    @rolled while h < Int32(2)                 # 128-element halves (rolled, see bdot(Q6_K)); the next half's loads first
        nxt = h == Int32(0) ? (ld16(pay, Int32(4) * Int32(CHS)), ld16(pay, Int32(5) * Int32(CHS))) : cur
        qa, qb = cur
        xq = xb + Int32(144) * h                 # xu of element 128 h
        Base.Cartesian.@nexprs 4 j -> begin
            sa = 0.0f0; sb = 0.0f0
            g = Int32(8) * h + Int32(2(j - 1))   # scale index = 16-group of the sa elements
            if Q3PRMT
                # the 3-bit codes q = low2 + 4 hbit of a word's 4 elements at bits 4-6 of its bytes (2 rotates, 2 LOP3),
                # then each byte to the float 0.5 + q / 16 (one PRMT): sum (q - 4) x = 16 sum f x - 12 X
                rh = (Int32(6) - (Int32(4) * h + Int32(j - 1))) & Int32(31)
                Base.Cartesian.@nexprs 4 w -> begin
                    qa4 = (bitrotate(qa[w], 4 - 2(j - 1)) & 0x30303030) | (bitrotate(hm1[w], rh) & 0x40404040)
                    qb4 = (bitrotate(qb[w], 4 - 2(j - 1)) & 0x30303030) | (bitrotate(hm2[w], rh) & 0x40404040)
                    sa = dotp4(qa4, sx4(xs, xq + Int32(xo(32(j - 1) + 4(w - 1)))), sa)
                    sb = dotp4(qb4, sx4(xs, xq + Int32(xo(32(j - 1) + 16 + 4(w - 1)))), sb)
                end
                acc = fma(sbyter(sc, g) - 32.0f0, fma(16.0f0, sa, -12.0f0 * sld(x16, g16 + g)), acc)
                acc = fma(sbyter(sc, g + Int32(1)) - 32.0f0, fma(16.0f0, sb, -12.0f0 * sld(x16, g16 + g + Int32(1))), acc)
            else
                Base.Cartesian.@nexprs 4 w -> begin
                    sa = dot3b4(qa[w], hm1[w], 2(j - 1), Int32(4) * h + Int32(j - 1), sx4(xs, xq + Int32(xo(32(j - 1) + 4(w - 1)))), sa)
                    sb = dot3b4(qb[w], hm2[w], 2(j - 1), Int32(4) * h + Int32(j - 1), sx4(xs, xq + Int32(xo(32(j - 1) + 16 + 4(w - 1)))), sb)
                end
                acc = fma(sbyter(sc, g) - 32.0f0, sa - 4.0f0 * sld(x16, g16 + g), acc)
                acc = fma(sbyter(sc, g + Int32(1)) - 32.0f0, sb - 4.0f0 * sld(x16, g16 + g + Int32(1)), acc)
            end
        end
        cur = nxt
        h += Int32(1)
    end
    d * acc
end
const Q3PRMT = get(ENV, "FLASHNEXT_Q3PRMT", "1") == "1"
const Q56PRMT = get(ENV, "FLASHNEXT_Q56PRMT", "1") == "1"
const Q4PRMT = get(ENV, "FLASHNEXT_Q4PRMT", "1") == "1"
const Q2PRMT = get(ENV, "FLASHNEXT_Q2PRMT", "1") == "1"
# byte e of q4 into byte 2 of 0x3f00_0000 (PRMT): the float 0.5 (1 + c 2^-7), c = the byte (a code q at bits b..6 of
# it: 0.5 + q 2^(b-8)); s + sum_e f_e x_e
@inline function dotp4(q4::UInt32, xv::V4F, s::Float32)
    Base.Cartesian.@nexprs 4 e -> (s = fma(reinterpret(Float32, CUDA.byte_perm(q4, 0x3f000000, UInt32(0x7054 | ((e - 1) << 8)))), xv[e].value, s))
    s
end
# 3-bit: 2-bit code at bits sh + 8e of w, high bit (code + 4) at bit hb + 8e of hm
@inline function dot3b4(w::UInt32, hm::UInt32, sh, hb, xv::V4F, s::Float32)
    Base.Cartesian.@nexprs 4 e -> (s = fma(u2f(((w >> (8(e - 1) + sh)) & 0x3) | (((hm >> (8(e - 1) + hb)) & 0x1) << 2)), xv[e].value, s))
    s
end
@inline function bdot(::Val{T_IQ4_XS}, pay::GPtr, hdr::GPtr, xs::SPtr, x16::SPtr, xb::Int32, g16::Int32, lut::SPtr)
    h = ldu64(hdr, Int32(0))
    d = hf(UInt16(h & 0xffff)); sh = UInt32((h >> 16) & 0xffff); sl = UInt32(h >> 32)
    acc = 0.0f0
    p = Int32(0)
    # the next pair's chunks are loaded before this pair is computed (in flight together)
    cur = (ld16(pay, Int32(0)), ld16(pay, Int32(CHS)))
    @rolled while p < Int32(4)                 # pairs of 32-element runs (rolled, see bdot(Q6_K))
        xq = xb + Int32(72) * p                  # xu of element 64 p
        nxt = p < Int32(3) ? (ld16(pay, (Int32(2) * p + Int32(2)) * Int32(CHS)), ld16(pay, (Int32(2) * p + Int32(3)) * Int32(CHS))) : cur
        Base.Cartesian.@nexprs 2 k -> begin
            ib = Int32(2) * p + Int32(k - 1)
            q = cur[k]
            if IQ4PRMT
                Base.Cartesian.@nexprs 4 w -> (s_w = dotnl8(q[w], sx4(xs, xq + Int32(xo(32(k - 1) + 4(w - 1)))), sx4(xs, xq + Int32(xo(32(k - 1) + 16 + 4(w - 1))))))
                s = (s_1 + s_2) + (s_3 + s_4)
            else
                Base.Cartesian.@nexprs 4 w -> begin            # one chain per word (4 in flight, not 32 FMAs in a row)
                    s_w = dot4b4lut(q[w], 0, sx4(xs, xq + Int32(xo(32(k - 1) + 4(w - 1)))), 0.0f0, lut)
                    s_w = dot4b4lut(q[w], 4, sx4(xs, xq + Int32(xo(32(k - 1) + 16 + 4(w - 1)))), s_w, lut)
                end
                s = (s_1 + s_2) + (s_3 + s_4)
            end
            ls = ((sl >> (Int32(4) * ib)) & 0xf) | (((sh >> (Int32(2) * ib)) & 0x3) << 4)
            acc = fma(u2f(ls) - 32.0f0, s, acc)
        end
        cur = nxt
        p += Int32(1)
    end
    d * acc
end

const IPtr = Core.LLVMPtr{Int32,1}
# Device layout ("warp-interleaved"): rows form groups of R = 32 ÷ L (L lanes per row); lane = ρ L + j of a warp holds
# row ρ of the group and blocks b = j + L i. Chunk c (16 bytes) of block (group g, iteration i, lane) sits at
#   ((((g NI + i) NC + c) 32 + lane) 16)        (NI = blocks per row ÷ L, NC = payload bytes ÷ 16)
# so every warp-wide 128-bit load reads 512 contiguous bytes; its header at ((g NI + i) 32 + lane) H.
const CHS = 512
@inline pld(pay::GPtr, o) = ld16(pay, Int32(o >> 4) * Int32(CHS))
# lanes per row: the largest power of two <= 32 dividing the row's block count. wide: the largest divisor that keeps
# 30 or more of the 32 lanes busy (5, 6, 10 lanes: 6, 5, 3 rows per warp), for fewer blocks per lane and smaller row
# groups (the tail of a phase is one group); only for plain rows, as the expert, MoE down and hc mix layouts need R | 32
function lanes_for(t, K; wide = false)
    nb = K ÷ devE(t)
    if wide
        for L in min(nb, 32):-1:1
            nb % L == 0 && (32 ÷ L) * L >= 30 && return L
        end
    end
    L = 32
    while nb % L != 0
        L ÷= 2
    end
    L
end
gemv_shmem(t, K) = 4 * (xu(K) + (t == T_BF16 ? 0 : K ÷ 16))   # BF16 does not read the 16-sums
padrows(t, K, rows, L = lanes_for(t, K)) = (R = 32 ÷ L; cld(rows, R) * R)
# 32 lane slots per (group, iteration), used or not (L R = 30 for 5, 6 or 10 lanes per row)
slotrows(t, K, rows, L = lanes_for(t, K)) = cld(rows, 32 ÷ L) * 32      # row groups x 32 lanes
devpay(t, K, rows, L = lanes_for(t, K)) = slotrows(t, K, rows, L) * ((K ÷ devE(t)) ÷ L) * devP(t)
devbytes_rows(t, K, rows, L = lanes_for(t, K)) = slotrows(t, K, rows, L) * ((K ÷ devE(t)) ÷ L) * (devP(t) + devH(t))

# a weight matrix on the device: rows x K in the interleaved layout with L lanes per row (rows padded to the group size)
struct DevMat
    t::Int
    rows::Int32
    K::Int32
    pay::GPtr
    hdr::GPtr
    L::Int32
end
DevMat(t, rows, K, base::CuPtr, L = lanes_for(t, K)) = DevMat(t, Int32(rows), Int32(K), gptr(base), gptr(base) + devpay(t, K, rows, L), Int32(L))

# y[r] = sum_k W[r, k] x[k] for r < rows. 256 threads; each warp takes one row group at a time (grid-stride over groups).
# MODE 0: y[r] = acc
# MODE 1: gathered rows (experts): row r reads matrix row idx[r ÷ RPE] RPE + r % RPE (R divides RPE)
# MODE 2: y[r] += sc[1] acc
# MODE 3: MoE down: x holds NS slots of K; y[r] = sum_e sc[e] (W row idx[e] RPE + r) . x_e
function k_gemv!(::Val{T}, ::Val{L}, ::Val{MODE}, y::FPtr, pay::GPtr, hdr::GPtr, x::FPtr, rows::Int32, K::Int32,
                 idx::IPtr, sc::FPtr, RPE::Int32, NS::Int32) where {T,L,MODE}
    E = Int32(devE(T)); H = Int32(devH(T)); NC = Int32(devP(T) ÷ 16); R = Int32(32 ÷ L)
    nb = K ÷ E; NI = nb ÷ Int32(L)
    Kx = MODE == 3 ? K * NS : K
    tid = threadIdx().x - Int32(1)
    xsa = CuDynamicSharedArray(Float32, (Int(xu(Kx) + (T == T_BF16 ? Int32(0) : Kx ÷ Int32(16))),))   # = gemv_shmem
    xs = pointer(xsa)
    x16 = xs + 4 * xu(Kx)
    luta = CuStaticSharedArray(Float32, 16)
    @inbounds begin
        tid < Int32(16) && (luta[tid+1] = kv4val(tid))
        k = tid
        while k < Kx
            unsafe_store!(xs + 4 * xu(k), unsafe_load(x, k + 1, Val(4)), 1, Val(4))
            k += Int32(256)
        end
        sync_threads()
        g = tid
        while T != T_BF16 && g < Kx ÷ Int32(16)
            s = 0.0f0
            for i in Int32(0):Int32(15)
                s += sld(xs, xu(Int32(16) * g) + i)
            end
            unsafe_store!(x16 + 4g, s, 1, Val(4))
            g += Int32(256)
        end
        sync_threads()
        lane = tid % Int32(32); warp = tid ÷ Int32(32)
        rho = lane ÷ Int32(L); j = lane % Int32(L)
        G = (rows + R - Int32(1)) ÷ R
        wg = (blockIdx().x - Int32(1)) * Int32(8) + warp
        while wg < G
            r = wg * R + rho
            acc = 0.0f0
            if MODE == 4                       # output segments of RPE elements (<= 4): y[(s + NS) rows + r] = W[r, s RPE : (s+1) RPE] . x[same]
                a1 = 0.0f0; a2 = 0.0f0; a3 = 0.0f0; a4 = 0.0f0; bseg = RPE ÷ E
                for i in Int32(0):NI-Int32(1)
                    b = j + Int32(L) * i; gi = wg * NI + i
                    v = bdot(Val(T), pay + (gi * NC * Int32(32) + lane) * Int32(16), hdr + (gi * Int32(32) + lane) * H, xs, x16,
                        xu(b * E), b * (E ÷ Int32(16)), pointer(luta))
                    sg = b ÷ bseg
                    a1 += sg == Int32(0) ? v : 0.0f0; a2 += sg == Int32(1) ? v : 0.0f0
                    a3 += sg == Int32(2) ? v : 0.0f0; a4 += sg == Int32(3) ? v : 0.0f0
                end
                a1 = wsum(a1, Val(L)); a2 = wsum(a2, Val(L)); a3 = wsum(a3, Val(L)); a4 = wsum(a4, Val(L))
                if j == Int32(0) && rho < R && r < rows
                    nseg = nb ÷ bseg
                    unsafe_store!(y, a1, NS * rows + r + 1, Val(4))
                    nseg > Int32(1) && unsafe_store!(y, a2, (NS + Int32(1)) * rows + r + 1, Val(4))
                    nseg > Int32(2) && unsafe_store!(y, a3, (NS + Int32(2)) * rows + r + 1, Val(4))
                    nseg > Int32(3) && unsafe_store!(y, a4, (NS + Int32(3)) * rows + r + 1, Val(4))
                end
                wg += gridDim().x * Int32(8)
                continue
            end
            if MODE == 3
                for e in Int32(0):NS-Int32(1)
                    pg = (unsafe_load(idx, e + 1, Val(4)) * RPE + wg * R) ÷ R
                    a2 = 0.0f0
                    for i in Int32(0):NI-Int32(1)
                        b = j + Int32(L) * i; gi = pg * NI + i
                        a2 += bdot(Val(T), pay + (gi * NC * Int32(32) + lane) * Int32(16), hdr + (gi * Int32(32) + lane) * H, xs, x16,
                            xu((e * nb + b) * E), (e * nb + b) * (E ÷ Int32(16)), pointer(luta))
                    end
                    acc = fma(unsafe_load(sc, e + 1, Val(4)), a2, acc)
                end
            else
                pg = MODE == 1 ? (unsafe_load(idx, (wg * R) ÷ RPE + 1, Val(4)) * RPE + (wg * R) % RPE) ÷ R : wg
                for i in Int32(0):NI-Int32(1)
                    b = j + Int32(L) * i; gi = pg * NI + i
                    acc += bdot(Val(T), pay + (gi * NC * Int32(32) + lane) * Int32(16), hdr + (gi * Int32(32) + lane) * H, xs, x16,
                        xu(b * E), b * (E ÷ Int32(16)), pointer(luta))
                end
            end
            acc = wsum(acc, Val(L))
            if j == Int32(0) && rho < R && r < rows
                if MODE == 2
                    unsafe_store!(y, unsafe_load(y, r + 1, Val(4)) + unsafe_load(sc, 1, Val(4)) * acc, r + 1, Val(4))
                else
                    unsafe_store!(y, acc, r + 1, Val(4))
                end
            end
            wg += gridDim().x * Int32(8)
        end
    end
    return
end
@inline kv4val(i::Int32) = i < Int32(8) ? (i < Int32(4) ? (i < Int32(2) ? (i == Int32(0) ? -127.0f0 : -104.0f0) : (i == Int32(2) ? -83.0f0 : -65.0f0)) :
    (i < Int32(6) ? (i == Int32(4) ? -49.0f0 : -35.0f0) : (i == Int32(6) ? -22.0f0 : -10.0f0))) :
    (i < Int32(12) ? (i < Int32(10) ? (i == Int32(8) ? 1.0f0 : 13.0f0) : (i == Int32(10) ? 25.0f0 : 38.0f0)) :
    (i < Int32(14) ? (i == Int32(12) ? 53.0f0 : 69.0f0) : (i == Int32(14) ? 89.0f0 : 113.0f0)))
const KV4 = (-127.0f0, -104.0f0, -83.0f0, -65.0f0, -49.0f0, -35.0f0, -22.0f0, -10.0f0, 1.0f0, 13.0f0, 25.0f0, 38.0f0, 53.0f0, 69.0f0, 89.0f0, 113.0f0)

# the GEMV's loads alone (decode replaced by an XOR of the words): the access pattern's ceiling, for the bench
function k_gemv_stream!(::Val{T}, ::Val{L}, y::FPtr, pay::GPtr, hdr::GPtr, rows::Int32, K::Int32) where {T,L}
    E = Int32(devE(T)); H = Int32(devH(T)); NC = Int32(devP(T) ÷ 16); R = Int32(32 ÷ L)
    nb = K ÷ E; NI = nb ÷ Int32(L)
    tid = threadIdx().x - Int32(1); lane = tid % Int32(32); warp = tid ÷ Int32(32)
    G = (rows + R - Int32(1)) ÷ R
    wg = (blockIdx().x - Int32(1)) * Int32(8) + warp
    @inbounds while wg < G
        acc = UInt32(0)
        for i in Int32(0):NI-Int32(1)
            gi = wg * NI + i; p = pay + (gi * NC * Int32(32) + lane) * Int32(16)
            for c in Int32(0):NC-Int32(1)
                v = ld16(p, c * Int32(CHS)); acc ⊻= v[1] ⊻ v[2] ⊻ v[3] ⊻ v[4]
            end
            H > 0 && (acc ⊻= UInt32(ldu16(hdr + (gi * Int32(32) + lane) * H, Int32(0))))
        end
        acc = wsum(Float32(acc & 0xff), Val(L))
        (lane % Int32(L) == Int32(0) && lane ÷ Int32(L) < R && wg * R + lane ÷ Int32(L) < rows) && unsafe_store!(y, acc, wg * R + lane ÷ Int32(L) + 1, Val(4))
        wg += gridDim().x * Int32(8)
    end
    return
end

const NOMAT = DevMat(T_F32, Int32(0), Int32(0), GPtr(0), GPtr(0), Int32(1))
const NSM = Ref(16)
const KCACHE = Dict{Any,Any}()
const NOI = IPtr(0)
const NOF = FPtr(0)
gemv_tt(t, L, mode) = Tuple{Val{t},Val{L},Val{mode},FPtr,GPtr,GPtr,FPtr,Int32,Int32,IPtr,FPtr,Int32,Int32}
function gemv_kernel(t, L, mode)
    get!(KCACHE, (:gemv, t, L, mode, CUDA.device())) do
        getkernel(k_gemv!, gemv_tt(t, L, mode))
    end
end
# y = W x (MODE 0), gathered expert rows (1), y += sc[1] W x (2), MoE down over ns slots (3)
function gemv!(y::FPtr, W::DevMat, x::FPtr; rows = W.rows, mode = 0, idx = NOI, sc = NOF, rpe = 0, ns = 1, s = CUDA.stream())
    L = Int(W.L)
    k = gemv_kernel(W.t, L, mode)
    groups = cld(Int(rows), 32 ÷ L)
    k(Val(W.t), Val(L), Val(mode), y, W.pay, W.hdr, x, Int32(rows), W.K, idx, sc, Int32(rpe), Int32(ns);
      threads = 256, blocks = min(cld(groups, 8), 8 * NSM[]), shmem = gemv_shmem(W.t, mode == 3 ? W.K * ns : W.K), stream = s)
end

# ---- block-level kernels (one token position; the reference is flash-sass/ref/*.c) ----------
@inline ldf(p::FPtr, i) = unsafe_load(p, i + 1, Val(4))
@inline stf(p::FPtr, i, v::Float32) = unsafe_store!(p, v, i + 1, Val(4))
@inline sigm(x::Float32) = 1.0f0 / (1.0f0 + exp(-x))
@inline silu(x::Float32) = x * sigm(x)
@inline softplus(x::Float32) = x > 20.0f0 ? x : log1p(exp(x))
@inline gtid() = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)

# sum over the CTA (every thread gets the total); uses 32 floats of shared memory
@inline function bsum(v::Float32)
    sh = CuStaticSharedArray(Float32, 32)
    lane = (threadIdx().x - Int32(1)) % Int32(32); wid = (threadIdx().x - Int32(1)) ÷ Int32(32)
    v = wsum(v, Val(32))
    lane == Int32(0) && (sh[wid+1] = v)
    sync_threads()
    nw = blockDim().x ÷ Int32(32)
    t = lane < nw ? sh[lane+1] : 0.0f0
    t = wsum(t, Val(32))
    sync_threads()
    t
end
@inline wmax(v::Float32, ::Val{L}) where {L} = L == 1 ? v : wmax(max(v, shfl_xor_sync(FULL, v, L ÷ 2)), Val(L ÷ 2))

# y[i n + c] = x[i n + c] r_i w[i ws + c], r_i = 1/sqrt(mean x_i^2 + eps); one CTA per vector i
function k_rmsnorm!(y::FPtr, x::FPtr, w::FPtr, n::Int32, ws::Int32, eps::Float32)
    i = blockIdx().x - Int32(1); tid = threadIdx().x - Int32(1)
    @inbounds begin
        s = 0.0f0
        c = tid
        while c < n
            v = ldf(x, i * n + c); s = fma(v, v, s); c += blockDim().x
        end
        s = bsum(s)
        r = 1.0f0 / sqrt(s / Float32(n) + eps)
        c = tid
        while c < n
            stf(y, i * n + c, ldf(x, i * n + c) * r * ldf(w, i * ws + c)); c += blockDim().x
        end
    end
    return
end

# t[i] = silu(d[i] / 4), i < n
function k_hc_silu!(t::FPtr, d::FPtr, n::Int32)
    i = gtid()
    @inbounds i < n && stf(t, i, silu(ldf(d, i) * 0.25f0))
    return
end
# z[c] = (1/4) sum_l sigmoid(up[2560 l + c]) xh[2560 l + c]
function k_hc_mix!(z::FPtr, up::FPtr, xh::FPtr)
    c = gtid()
    @inbounds if c < Int32(2560)
        a = 0.0f0
        for l in Int32(0):Int32(3)
            a = fma(sigm(ldf(up, Int32(2560) * l + c)), ldf(xh, Int32(2560) * l + c), a)
        end
        stf(z, c, a * 0.25f0)
    end
    return
end
# S[2560 l + c] += 2 sigmoid(inj[l] / 4) y[c]
function k_hc_write!(S::FPtr, y::FPtr, inj::FPtr)
    i = gtid()
    @inbounds if i < Int32(10240)
        l = i ÷ Int32(2560); c = i % Int32(2560)
        stf(S, i, fma(2.0f0 * sigm(0.25f0 * ldf(inj, l)), ldf(y, c), ldf(S, i)))
    end
    return
end

# depthwise causal conv (k = 4) + SiLU; st[4][n] holds x(t') in slot t' mod 4; w[4c + k]
function k_conv1d!(u::FPtr, x::FPtr, st::FPtr, w::FPtr, t::Int32, n::Int32)
    c = gtid()
    @inbounds if c < n
        xc = ldf(x, c)
        y = ldf(w, Int32(4) * c) * ldf(st, ((t + Int32(1)) & Int32(3)) * n + c)
        y = fma(ldf(w, Int32(4) * c + Int32(1)), ldf(st, ((t + Int32(2)) & Int32(3)) * n + c), y)
        y = fma(ldf(w, Int32(4) * c + Int32(2)), ldf(st, ((t + Int32(3)) & Int32(3)) * n + c), y)
        y = fma(ldf(w, Int32(4) * c + Int32(3)), xc, y)
        stf(u, c, silu(y))
        stf(st, (t & Int32(3)) * n + c, xc)
    end
    return
end

# 16 consecutive floats (64-byte aligned) as four 128-bit accesses
@inline function ld16f(p::FPtr)
    a = unsafe_load(reinterpret(Core.LLVMPtr{V4F,1}, p), 1, Val(16)); b = unsafe_load(reinterpret(Core.LLVMPtr{V4F,1}, p), 2, Val(16))
    c = unsafe_load(reinterpret(Core.LLVMPtr{V4F,1}, p), 3, Val(16)); d = unsafe_load(reinterpret(Core.LLVMPtr{V4F,1}, p), 4, Val(16))
    (a[1].value, a[2].value, a[3].value, a[4].value, b[1].value, b[2].value, b[3].value, b[4].value,
     c[1].value, c[2].value, c[3].value, c[4].value, d[1].value, d[2].value, d[3].value, d[4].value)
end
@inline function st16f(p::FPtr, v::NTuple{16,Float32})
    q = reinterpret(Core.LLVMPtr{V4F,1}, p)
    for j in 0:3
        unsafe_store!(q, (VecElement(v[4j+1]), VecElement(v[4j+2]), VecElement(v[4j+3]), VecElement(v[4j+4])), j + 1, Val(16))
    end
end
@inline dn_update(sv::NTuple{16,Float32}, ks, i0::Int32, be::Float32, alpha::Float32) = ntuple(k -> fma(ks[i0+k], be, alpha * sv[k]), Val(16))
# gated delta rule for value head h (one CTA of 256 threads), then the gated RMS readout norm.
# u = conv output [q 2048 | k 2048 | v 6144]; value head h uses key head h % 16. State S_h[c][i] (value c, key i).
# thread t owns values c = 4 (t ÷ 8) .. +3 and keys i = 16 (t % 8) .. +15.
function k_deltanet!(on::FPtr, St::FPtr, u::FPtr, ab::FPtr, abb::FPtr, ssa::FPtr, gate::FPtr, nw::FPtr, eps::Float32, nkh::Int32)
    h = blockIdx().x - Int32(1); tid = threadIdx().x - Int32(1)
    qs = CuStaticSharedArray(Float32, 128); ks = CuStaticSharedArray(Float32, 128)
    vs = CuStaticSharedArray(Float32, 128); os = CuStaticSharedArray(Float32, 128)
    @inbounds begin
        kh = h % nkh                       # u = [q (nkh heads) | k (nkh) | v (3 nkh)]; value head h reads key head h % nkh
        qv = tid < Int32(128) ? ldf(u, Int32(128) * kh + tid) : 0.0f0
        kv = tid < Int32(128) ? ldf(u, Int32(128) * nkh + Int32(128) * kh + tid) : 0.0f0
        sq = bsum(qv * qv); sk = bsum(kv * kv)
        if tid < Int32(128)
            qs[tid+1] = qv / sqrt(sq + eps)
            ks[tid+1] = kv / sqrt(sk + eps)
            vs[tid+1] = ldf(u, Int32(256) * nkh + Int32(128) * h + tid)
        end
        sync_threads()
        a = ldf(ab, h) + ldf(abb, h); b = ldf(ab, Int32(3) * nkh + h) + ldf(abb, Int32(3) * nkh + h)
        alpha = exp(ldf(ssa, h) * softplus(a)); beta = sigm(b)
        c0 = Int32(4) * (tid ÷ Int32(8)); i0 = Int32(16) * (tid % Int32(8))
        base = St + 4 * (Int32(16384) * h)
        for cc in Int32(0):Int32(3)
            c = c0 + cc
            row = base + 4 * (Int32(128) * c + i0)
            sv = ld16f(row)
            r = 0.0f0
            for k in 1:16
                r = fma(ks[i0+k], sv[k], r)
            end
            r = wsum(r, Val(8)) * alpha
            be = beta * (vs[c+1] - r)
            o = 0.0f0
            nv = dn_update(sv, ks, i0, be, alpha)
            st16f(row, nv)
            for k in 1:16
                o = fma(qs[i0+k], nv[k], o)
            end
            o = wsum(o, Val(8)) * 0.088388347648318440550f0
            (tid % Int32(8)) == Int32(0) && (os[c+1] = o)
        end
        sync_threads()
        ov = tid < Int32(128) ? os[tid+1] : 0.0f0
        ss = bsum(ov * ov)
        if tid < Int32(128)
            r = 1.0f0 / sqrt(ss / 128.0f0 + eps)
            stf(on, Int32(128) * h + tid, ov * r * ldf(nw, tid) * sigm(ldf(gate, Int32(128) * h + tid)))
        end
    end
    return
end

# per-head RMS norm (256 wide, weight w) and NeoX RoPE on dims 0..63 (pairs p, p + 32); cs = [cos 32 | sin 32]
@inline function norm_rope256(x::Float32, d::Int32, w::FPtr, cs::FPtr, eps::Float32, sh)
    ss = bsum(x * x)
    y = x / sqrt(ss / 256.0f0 + eps) * ldf(w, d)
    sh[d+1] = y
    sync_threads()
    if d < Int32(64)
        p = d % Int32(32)
        c = ldf(cs, p); s = ldf(cs, Int32(32) + p)
        y = d < Int32(32) ? fma(-sh[d+33], s, sh[d+1] * c) : fma(sh[d-31], s, sh[d+1] * c)
    end
    sync_threads()
    y
end
# int8 + fp16 scale per 32 values (the warp holds 32 consecutive values, lane = value)
@inline function kvq32(x::Float32, q::GPtr, qi::Int32, s::GPtr, si::Int32)
    m = max(wmax(abs(x), Val(32)), 7.888609f-31)
    v = round(x * (127.0f0 / m))
    unsafe_store!(q + qi, reinterpret(UInt8, Int8(clamp(v, -127.0f0, 127.0f0))))
    (threadIdx().x - Int32(1)) % Int32(32) == Int32(0) && unsafe_store!(reinterpret(Core.LLVMPtr{Float16,1}, s + 2si), Float16(m / 127.0f0), 1, Val(2))
    return
end
@inline kvd(q::GPtr, qi::Int32, s::GPtr, si::Int32) = Float32(reinterpret(Int8, unsafe_load(q + qi))) * Float32(unsafe_load(reinterpret(Core.LLVMPtr{Float16,1}, s + 2si), 1, Val(2)))

# attention prologue at position t: CTAs 0..23 normalise + rotate the q heads (qg = 24 x [q 256 | gate 256]) into qo,
# CTAs 24, 25 the k heads, quantised into the cache with v. 256 threads (one per dim).
# KV cache: kq/vq int8 [pos][2][256], ks/vs fp16 [pos][2][8]
function k_attn_prep!(qo::FPtr, kq::GPtr, ks::GPtr, vq::GPtr, vs::GPtr, qg::FPtr, k::FPtr, v::FPtr, qn::FPtr, kn::FPtr, cs::FPtr, t::Int32, eps::Float32, nq::Int32, nkv::Int32)
    hb = blockIdx().x - Int32(1); d = threadIdx().x - Int32(1)
    sh = CuStaticSharedArray(Float32, 256)
    @inbounds if hb < nq
        y = norm_rope256(ldf(qg, Int32(512) * hb + d), d, qn, cs, eps, sh)
        stf(qo, Int32(256) * hb + d, y)
    else
        kv = hb - nq
        y = norm_rope256(ldf(k, Int32(256) * kv + d), d, kn, cs, eps, sh)
        row = t * nkv + kv
        kvq32(y, kq, row * Int32(256) + d, ks, row * Int32(8) + d ÷ Int32(32))
        kvq32(ldf(v, Int32(256) * kv + d), vq, row * Int32(256) + d, vs, row * Int32(8) + d ÷ Int32(32))
    end
    return
end

# indexer at position t: CTAs 0..3 the query heads (RMS 128 * q_norm, RoPE(t)) into iq; CTA 4 keeps the raw key in
# the ring and, when t completes a block of 4, pools it (mean, RMS * k_norm, RoPE at the block's first position) as fp16.
# ix = [q_proj 512 | k_proj 128]; 128 threads
function k_idx_prep!(iq::FPtr, ring::FPtr, pool::GPtr, ix::FPtr, qn::FPtr, kn::FPtr, cs::FPtr, cs3::FPtr, t::Int32, eps::Float32)
    hb = blockIdx().x - Int32(1); d = threadIdx().x - Int32(1)
    sh = CuStaticSharedArray(Float32, 128)
    @inbounds begin
        if hb < Int32(4)
            x = ldf(ix, Int32(128) * hb + d); w = qn; c = cs
        else
            stf(ring, (t & Int32(3)) * Int32(128) + d, ldf(ix, Int32(512) + d))
            x = (((ldf(ring, d) + ldf(ring, Int32(128) + d)) + ldf(ring, Int32(256) + d)) + ldf(ring, Int32(384) + d)) * 0.25f0
            w = kn; c = cs3
        end
        ss = bsum(x * x)
        y = x / sqrt(ss / 128.0f0 + eps) * ldf(w, d)
        sh[d+1] = y
        sync_threads()
        if d < Int32(64)
            p = d % Int32(32)
            cc = ldf(c, p); s = ldf(c, Int32(32) + p)
            y = d < Int32(32) ? fma(-sh[d+33], s, sh[d+1] * cc) : fma(sh[d-31], s, sh[d+1] * cc)
        end
        if hb < Int32(4)
            stf(iq, Int32(128) * hb + d, y)
        elseif (t & Int32(3)) == Int32(3)
            unsafe_store!(reinterpret(Core.LLVMPtr{Float16,1}, pool), Float16(y), (t ÷ Int32(4)) * Int32(128) + d + Int32(1), Val(2))
        end
    end
    return
end
# score[b] = sum_h relu(iq_h . pool_b), b < nv (one thread per block)
function k_idx_score!(score::FPtr, iq::FPtr, pool::GPtr, nv::Int32)
    b = gtid()
    qsh = CuStaticSharedArray(Float32, 512)
    @inbounds begin
        for i in threadIdx().x:blockDim().x:Int32(512)
            qsh[i] = ldf(iq, i - Int32(1))
        end
        sync_threads()
        if b < nv
            pp = reinterpret(Core.LLVMPtr{Float16,1}, pool) + 256 * b
            s = 0.0f0
            for h in Int32(0):Int32(3)
                a = 0.0f0
                for d in Int32(0):Int32(127)
                    a = fma(qsh[Int32(128)*h+d+1], Float32(unsafe_load(pp, d + Int32(1), Val(2))), a)
                end
                s += max(a, 0.0f0)
            end
            stf(score, b, s)
        end
    end
    return
end
# the k = min(nv, 512) best blocks (score descending, lower index first on ties) as ascending indices in sel
# (one CTA of 1024 threads; radix select of the threshold over order-preserving keys, then a stable compaction)
@inline okey(f::Float32) = (u = reinterpret(UInt32, f); (u & 0x80000000) != 0 ? ~u : u | 0x80000000)
function k_idx_select!(sel::IPtr, score::FPtr, nv::Int32, kk::Int32)
    tid = threadIdx().x - Int32(1); nt = blockDim().x
    cnt = CuStaticSharedArray(Int32, 32)
    @inbounds begin
        # largest key T with count(key >= T) >= kk
        T = UInt32(0)
        for bit in 31:-1:0
            cand = T | (UInt32(1) << bit)
            c = Int32(0)
            i = tid
            while i < nv
                c += okey(ldf(score, i)) >= cand ? Int32(1) : Int32(0); i += nt
            end
            tot = isum(c, cnt)
            tot >= kk && (T = cand)
        end
        # count of keys strictly above T; the rest come from keys == T in index order
        c = Int32(0); i = tid
        while i < nv
            c += okey(ldf(score, i)) > T ? Int32(1) : Int32(0); i += nt
        end
        gt = isum(c, cnt)
        need_eq = kk - gt
        # stable compaction in chunks of nt
        outp = Int32(0); eqseen = Int32(0)
        base = Int32(0)
        while base < nv
            i = base + tid
            kq = i < nv ? okey(ldf(score, i)) : UInt32(0)
            iseq = i < nv && kq == T
            eqrank = eqseen + iscan(iseq ? Int32(1) : Int32(0), cnt) - (iseq ? Int32(1) : Int32(0))
            take = i < nv && (kq > T || (iseq && eqrank < need_eq))
            pos = outp + iscan(take ? Int32(1) : Int32(0), cnt) - (take ? Int32(1) : Int32(0))
            take && unsafe_store!(sel, i, pos + Int32(1), Val(4))
            outp += isum(take ? Int32(1) : Int32(0), cnt)
            eqseen += isum(iseq ? Int32(1) : Int32(0), cnt)
            base += nt
        end
    end
    return
end
# CTA-wide integer sum and inclusive scan (thread order)
@inline function isum(v::Int32, sh)
    lane = (threadIdx().x - Int32(1)) % Int32(32); wid = (threadIdx().x - Int32(1)) ÷ Int32(32)
    for o in (16, 8, 4, 2, 1)
        v += shfl_xor_sync(FULL, v, o)
    end
    lane == Int32(0) && (sh[wid+1] = v)
    sync_threads()
    t = lane < blockDim().x ÷ Int32(32) ? sh[lane+1] : Int32(0)
    for o in (16, 8, 4, 2, 1)
        t += shfl_xor_sync(FULL, t, o)
    end
    sync_threads()
    t
end
@inline function iscan(v::Int32, sh)
    lane = (threadIdx().x - Int32(1)) % Int32(32); wid = (threadIdx().x - Int32(1)) ÷ Int32(32)
    for o in (1, 2, 4, 8, 16)
        n = shfl_up_sync(FULL, v, o)
        lane >= o && (v += n)
    end
    lane == Int32(31) && (sh[wid+1] = v)
    sync_threads()
    if wid == Int32(0)
        w = lane < blockDim().x ÷ Int32(32) ? sh[lane+1] : Int32(0)
        for o in (1, 2, 4, 8, 16)
            n = shfl_up_sync(FULL, w, o)
            lane >= o && (w += n)
        end
        sh[lane+1] = w
    end
    sync_threads()
    r = v + (wid > Int32(0) ? sh[wid] : Int32(0))
    sync_threads()
    r
end

# decode attention partials: grid (nch, 24 heads); 256 threads = 8 warps; warp w takes keys j = c0 + w, + 8, ...;
# lane l holds dims 8l .. 8l + 7. Key j is token j when the selection is dense (nsel < 0), else block sel[j ÷ 4]
# for j < 4 nsel and the tail block after. Writes m, sum and o[256] per (head, chunk).
function k_attn_part!(pm::FPtr, pd::FPtr, po::FPtr, qo::FPtr, kq::GPtr, ks::GPtr, vq::GPtr, vs::GPtr, sel::IPtr,
                      nsel::Int32, nkeys::Int32, tailb::Int32, ck::Int32, nkv::Int32)
    ch = blockIdx().x - Int32(1); h = blockIdx().y - Int32(1); nch = gridDim().x
    tid = threadIdx().x - Int32(1); lane = tid % Int32(32); w = tid ÷ Int32(32)
    kv = h ÷ Int32(12)
    msh = CuStaticSharedArray(Float32, 8); dsh = CuStaticSharedArray(Float32, 8)
    osh = CuStaticSharedArray(Float32, (256, 8))
    @inbounds begin
        q = ntuple(i -> ldf(qo, Int32(256) * h + Int32(8) * lane + Int32(i - 1)), Val(8))
        m = -1.0f30; dsum = 0.0f0
        acc = ntuple(i -> 0.0f0, Val(8))
        j = ch * ck + w
        jend = min(nkeys, (ch + Int32(1)) * ck)
        while j < jend
            tok = nsel < Int32(0) ? j : (j < Int32(4) * nsel ? Int32(4) * unsafe_load(sel, j ÷ Int32(4) + Int32(1), Val(4)) + j % Int32(4) : Int32(4) * tailb + (j - Int32(4) * nsel))
            row = tok * nkv + kv
            s = 0.0f0
            for i in 1:8
                d = Int32(8) * lane + Int32(i - 1)
                s = fma(q[i], kvd(kq, row * Int32(256) + d, ks, row * Int32(8) + d ÷ Int32(32)), s)
            end
            s = wsum(s, Val(32)) * 0.0625f0
            mn = max(m, s)
            sc = exp(m - mn); p = exp(s - mn)
            dsum = fma(dsum, sc, p)
            acc = vacc(acc, p, sc, vq, vs, row, lane)
            m = mn
            j += Int32(8)
        end
        if lane == Int32(0)
            msh[w+1] = m; dsh[w+1] = dsum
        end
        for i in 1:8
            osh[Int32(8)*lane+Int32(i), w+1] = acc[i]
        end
        sync_threads()
        M = msh[1]
        for ww in 2:8
            M = max(M, msh[ww])
        end
        o = 0.0f0; D = 0.0f0
        for ww in 1:8
            e = exp(msh[ww] - M)
            o = fma(osh[tid+1, ww], e, o); D = fma(dsh[ww], e, D)
        end
        part = h * nch + ch
        stf(po, Int32(256) * part + tid, o)
        if tid == Int32(0)
            stf(pm, part, M); stf(pd, part, D)
        end
    end
    return
end
@inline vacc(acc::NTuple{8,Float32}, p::Float32, sc::Float32, vq::GPtr, vs::GPtr, row::Int32, lane::Int32) =
    Base.Cartesian.@ntuple 8 i -> fma(p, kvd(vq, row * Int32(256) + Int32(8) * lane + Int32(i - 1), vs, row * Int32(8) + lane ÷ Int32(4)), acc[i] * sc)
# merge the chunks of head h (CTA h, 256 threads) and apply the output gate: out = o / sum * sigmoid(gate)
function k_attn_merge!(out::FPtr, pm::FPtr, pd::FPtr, po::FPtr, qg::FPtr, nch::Int32)
    h = blockIdx().x - Int32(1); d = threadIdx().x - Int32(1)
    @inbounds begin
        M = -1.0f30
        for c in Int32(0):nch-Int32(1)
            M = max(M, ldf(pm, h * nch + c))
        end
        o = 0.0f0; D = 0.0f0
        for c in Int32(0):nch-Int32(1)
            e = exp(ldf(pm, h * nch + c) - M)
            o = fma(ldf(po, Int32(256) * (h * nch + c) + d), e, o); D = fma(ldf(pd, h * nch + c), e, D)
        end
        stf(out, Int32(256) * h + d, o / D * sigm(ldf(qg, Int32(512) * h + Int32(256) + d)))
    end
    return
end

# router: top 10 of r[0:512] (ties: lower index), softmax over them; w[10] = sigmoid(r[512]) (shared-expert gate).
# one CTA of 512 threads
function k_router!(idx::IPtr, w::FPtr, r::FPtr)
    tid = threadIdx().x - Int32(1)
    bv = CuStaticSharedArray(Float32, 16); bi = CuStaticSharedArray(Int32, 16)
    top = CuStaticSharedArray(Float32, 10)
    @inbounds begin
        v = ldf(r, tid)
        lane = tid % Int32(32); wid = tid ÷ Int32(32)
        for j in Int32(0):Int32(9)
            mv = v; mi = tid
            for o in (16, 8, 4, 2, 1)
                ov = shfl_xor_sync(FULL, mv, o); oi = shfl_xor_sync(FULL, mi, o)
                if ov > mv || (ov == mv && oi < mi)
                    mv = ov; mi = oi
                end
            end
            if lane == Int32(0)
                bv[wid+1] = mv; bi[wid+1] = mi
            end
            sync_threads()
            if tid == Int32(0)
                BV = bv[1]; BI = bi[1]
                for k in 2:16
                    if bv[k] > BV || (bv[k] == BV && bi[k] < BI)
                        BV = bv[k]; BI = bi[k]
                    end
                end
                top[j+1] = BV
                unsafe_store!(idx, BI, j + Int32(1), Val(4))
                bi[1] = BI
            end
            sync_threads()
            tid == bi[1] && (v = -Inf32)
            sync_threads()
        end
        if tid == Int32(0)
            den = 0.0f0
            for j in 1:10
                den += exp(top[j] - top[1])
            end
            for j in 1:10
                stf(w, Int32(j - 1), exp(top[j] - top[1]) / den)
            end
            stf(w, Int32(10), sigm(ldf(r, Int32(512))))
        end
    end
    return
end
# h[640 e + i] = silu(gu[1280 e + i]) gu[1280 e + 640 + i], e < 11
# h[F e + i] = silu(gu[2F e + i]) gu[2F e + F + i], e < ns
# slots e < il hold interleaved rows (gate i at 2F e + 2i, up i at 2F e + 2i + 1), the others gate then up
function k_swiglu!(h::FPtr, gu::FPtr, F::Int32, ns::Int32, il::Int32)
    i = gtid()
    @inbounds if i < F * ns
        e = i ÷ F; j = i % F
        if e < il
            stf(h, i, silu(ldf(gu, Int32(2) * F * e + Int32(2) * j)) * ldf(gu, Int32(2) * F * e + Int32(2) * j + Int32(1)))
        else
            stf(h, i, silu(ldf(gu, Int32(2) * F * e + j)) * ldf(gu, Int32(2) * F * e + F + j))
        end
    end
    return
end

# PLE prefix of block 1 (CTA l = hyper-connection lane, 256 threads): K, V = key/value of the n-gram rows;
# kn = RMS(K_l) nk_l, qn = RMS(S_l) nq_l, s = kn.qn / sqrt(2560), gate = sigmoid(sign(s) sqrt(max(|s|, 1e-6))),
# G = gate V, x = RMS(G) nc_l, y = dilated causal conv (taps t-9, t-6, t-3, t), S_l += G + silu(y); ring[t mod 10] = x
function k_ple!(S::FPtr, K::FPtr, V::FPtr, nk::FPtr, nq::FPtr, nc::FPtr, cw::FPtr, ring::FPtr, t::Int32, eps::Float32)
    l = blockIdx().x - Int32(1); tid = threadIdx().x - Int32(1)
    @inbounds begin
        o = Int32(2560) * l
        sK = 0.0f0; sS = 0.0f0
        for c in tid:Int32(256):Int32(2559)
            a = ldf(K, o + c); b = ldf(S, o + c)
            sK = fma(a, a, sK); sS = fma(b, b, sS)
        end
        rK = 1.0f0 / sqrt(bsum(sK) / 2560.0f0 + eps); rS = 1.0f0 / sqrt(bsum(sS) / 2560.0f0 + eps)
        dt = 0.0f0
        for c in tid:Int32(256):Int32(2559)
            dt = fma(ldf(K, o + c) * rK * ldf(nk, o + c), ldf(S, o + c) * rS * ldf(nq, o + c), dt)
        end
        s = bsum(dt) * (1.0f0 / sqrt(2560.0f0))
        g = sigm(sign(s) * sqrt(max(abs(s), 1.0f-6)))
        sG = 0.0f0
        for c in tid:Int32(256):Int32(2559)
            G = g * ldf(V, c); sG = fma(G, G, sG)
        end
        rG = 1.0f0 / sqrt(bsum(sG) / 2560.0f0 + eps)
        for c in tid:Int32(256):Int32(2559)
            ch = o + c
            G = g * ldf(V, c)
            x = G * rG * ldf(nc, ch)
            x9 = t >= Int32(9) ? ldf(ring, ((t - Int32(9)) % Int32(10)) * Int32(10240) + ch) : 0.0f0
            x6 = t >= Int32(6) ? ldf(ring, ((t - Int32(6)) % Int32(10)) * Int32(10240) + ch) : 0.0f0
            x3 = t >= Int32(3) ? ldf(ring, ((t - Int32(3)) % Int32(10)) * Int32(10240) + ch) : 0.0f0
            y = ldf(cw, Int32(4) * ch) * x9
            y = fma(ldf(cw, Int32(4) * ch + Int32(1)), x6, y)
            y = fma(ldf(cw, Int32(4) * ch + Int32(2)), x3, y)
            y = fma(ldf(cw, Int32(4) * ch + Int32(3)), x, y)
            stf(S, ch, ldf(S, ch) + (G + silu(y)))
            stf(ring, (t % Int32(10)) * Int32(10240) + ch, x)
        end
    end
    return
end

# ---------------------------------------------------------------------------------------------
# Tokenizer: byte-level BPE (gpt2 model, qwen35 pre-split), as llama.cpp / flash-sass/kern/tok.c
# ---------------------------------------------------------------------------------------------
const F_UNDEF, F_NUMBER, F_LETTER, F_SEP, F_MARK, F_PUNCT, F_SYM, F_CTRL, F_WS = 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x100
const WS_CPTS = Set{UInt32}([0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x20, 0x85, 0xa0, 0x1680, 0x2000, 0x2001, 0x2002, 0x2003, 0x2004, 0x2005,
    0x2006, 0x2007, 0x2008, 0x2009, 0x200a, 0x2028, 0x2029, 0x202f, 0x205f, 0x3000])
function cflags(c::UInt32)
    c >= 0x110000 && return 0
    cat = Base.Unicode.category_code(Char(c))
    f = cat == 0 ? F_UNDEF : cat <= 5 ? F_LETTER : cat <= 8 ? F_MARK : cat <= 11 ? F_NUMBER : cat <= 18 ? F_PUNCT :
        cat <= 22 ? F_SYM : cat <= 25 ? F_SEP : F_CTRL
    c in WS_CPTS ? f | F_WS : f
end

struct Tok
    text::Vector{String}
    typ::Vector{Int32}
    ids::Dict{String,Int32}
    ranks::Dict{String,Int32}
    special::Vector{Int32}          # control / user-defined / unknown ids, longest text first
    byte2u::Vector{String}          # byte -> its byte-level character (UTF-8)
    u2byte::Dict{Char,UInt8}
    eos::Int32
end
function Tok(g::GModel)
    kv = g.kv
    kv["tokenizer.ggml.model"] == "gpt2" || error("tokenizer model is not gpt2")
    kv["tokenizer.ggml.pre"] == "qwen35" || error("tokenizer pre is $(kv["tokenizer.ggml.pre"]), only qwen35 is implemented")
    text = kv["tokenizer.ggml.tokens"]::Vector{String}
    typ = Int32.(kv["tokenizer.ggml.token_type"])
    ids = Dict{String,Int32}()
    for (i, s) in enumerate(text)
        haskey(ids, s) || (ids[s] = i - 1)
    end
    ranks = Dict{String,Int32}()
    for (i, s) in enumerate(kv["tokenizer.ggml.merges"]::Vector{String})
        haskey(ranks, s) || (ranks[s] = i - 1)
    end
    special = Int32[i - 1 for i in eachindex(typ) if typ[i] in (2, 3, 4)]
    sort!(special; by = i -> (-ncodeunits(text[i+1]), i))
    byte2u = Vector{String}(undef, 256); u2byte = Dict{Char,UInt8}()
    extra = 0
    for b in 0:255
        keep = (0x21 <= b <= 0x7e) || (0xa1 <= b <= 0xac) || (0xae <= b <= 0xff)
        c = keep ? Char(b) : Char(256 + extra)
        keep || (extra += 1)
        byte2u[b+1] = string(c); u2byte[c] = UInt8(b)
    end
    Tok(text, typ, ids, ranks, special, byte2u, u2byte, Int32(kv["tokenizer.ggml.eos_token_id"]))
end

# llama.cpp cpts_from_utf8: an invalid sequence yields U+FFFD and consumes one byte
function cpts(s::Vector{UInt8})
    out = UInt32[]; i = 1; n = length(s)
    while i <= n
        b = s[i]
        if b < 0x80
            push!(out, b); i += 1; continue
        end
        ok = false
        if (b & 0x40) != 0
            if (b & 0x20) == 0
                if i + 1 <= n && (s[i+1] & 0xc0) == 0x80
                    r = (UInt32(b & 0x1f) << 6) | (s[i+1] & 0x3f)
                    r >= 0x80 && (push!(out, r); i += 2; ok = true)
                end
            elseif (b & 0x10) == 0
                if i + 2 <= n && (s[i+1] & 0xc0) == 0x80 && (s[i+2] & 0xc0) == 0x80
                    r = (UInt32(b & 0x0f) << 12) | (UInt32(s[i+1] & 0x3f) << 6) | (s[i+2] & 0x3f)
                    r >= 0x800 && (push!(out, r); i += 3; ok = true)
                end
            elseif (b & 0x08) == 0
                if i + 3 <= n && (s[i+1] & 0xc0) == 0x80 && (s[i+2] & 0xc0) == 0x80 && (s[i+3] & 0xc0) == 0x80
                    r = (UInt32(b & 0x07) << 18) | (UInt32(s[i+1] & 0x3f) << 12) | (UInt32(s[i+2] & 0x3f) << 6) | (s[i+3] & 0x3f)
                    0x10000 <= r <= 0x10ffff && (push!(out, r); i += 4; ok = true)
                end
            end
        end
        ok || (push!(out, 0xfffd); i += 1)
    end
    out
end

# unicode_regex_split_custom_qwen35 over cp: word lengths in codepoints
function split_qwen35(cp::Vector{UInt32})
    n = length(cp); OUT = 0xffffffff
    C(p) = p < n ? cp[p+1] : OUT
    FL(p) = p < n ? cflags(cp[p+1]) : 0
    lower(c) = (0x41 <= c <= 0x5a) ? c + 0x20 : c
    lens = Int[]; prev = 0
    add(e) = (e > prev && push!(lens, e - prev); prev = e)
    pos = 0
    while pos < n
        c = C(pos); f = FL(pos)
        if c == UInt32('\'') && pos + 1 < n
            c1 = lower(C(pos + 1))
            if c1 in (UInt32('s'), UInt32('t'), UInt32('m'), UInt32('d'))
                add(pos + 2); pos += 2; continue
            end
            if pos + 2 < n
                c2 = lower(C(pos + 2))
                if (c1 == UInt32('r') && c2 == UInt32('e')) || (c1 == UInt32('v') && c2 == UInt32('e')) || (c1 == UInt32('l') && c2 == UInt32('l'))
                    add(pos + 3); pos += 3; continue
                end
            end
        end
        if !(c == 0x0d || c == 0x0a || (f & F_NUMBER) != 0)
            if (f & (F_LETTER | F_MARK)) != 0 || (FL(pos + 1) & (F_LETTER | F_MARK)) != 0
                pos += 1
                while (FL(pos) & (F_LETTER | F_MARK)) != 0
                    pos += 1
                end
                add(pos); continue
            end
        end
        if (f & F_NUMBER) != 0
            pos += 1; add(pos); continue
        end
        f2 = c == 0x20 ? FL(pos + 1) : f
        if (f2 & (F_WS | F_LETTER | F_MARK | F_NUMBER)) == 0 && f != 0
            pos += (c == 0x20)
            while (f2 & (F_WS | F_LETTER | F_MARK | F_NUMBER)) == 0 && f2 != 0
                pos += 1; f2 = FL(pos)
            end
            c2 = C(pos)
            while c2 == 0x0d || c2 == 0x0a
                pos += 1; c2 = C(pos)
            end
            add(pos); continue
        end
        nws = 0; last_rn = 0
        while (FL(pos + nws) & F_WS) != 0
            c2 = C(pos + nws)
            (c2 == 0x0d || c2 == 0x0a) && (last_rn = pos + nws + 1)
            nws += 1
        end
        if last_rn > 0
            pos = last_rn; add(pos); continue
        end
        if nws > 1 && C(pos + nws) != OUT
            pos += nws - 1; add(pos); continue
        end
        if nws > 0
            pos += nws; add(pos); continue
        end
        pos += 1; add(pos)
    end
    lens
end

# BPE of one byte-level word (llm_tokenizer_bpe_session::tokenize)
function bpe_word!(out::Vector{Int32}, tk::Tok, w::String)
    syms = String[]                         # symbol texts (UTF-8 characters of the byte-level word)
    for c in w
        push!(syms, string(c))
    end
    n = length(syms)
    prv = collect(-1:n-2); nxt = [i == n - 1 ? -1 : i + 1 for i in 0:n-1]
    heap = Tuple{Int32,Int,Int,Int}[]       # (rank, left, right, combined length)
    function addb(l, r)
        (l < 0 || r < 0) && return
        rk = get(tk.ranks, syms[l+1] * " " * syms[r+1], Int32(-1))
        rk < 0 && return
        hpush!(heap, (rk, l, r, ncodeunits(syms[l+1]) + ncodeunits(syms[r+1])))
    end
    for i in 1:n-1
        addb(i - 1, i)
    end
    while !isempty(heap)
        rk, l, r, len = hpop!(heap)
        (isempty(syms[l+1]) || isempty(syms[r+1]) || ncodeunits(syms[l+1]) + ncodeunits(syms[r+1]) != len || nxt[l+1] != r) && continue
        syms[l+1] = syms[l+1] * syms[r+1]; syms[r+1] = ""
        nxt[l+1] = nxt[r+1]
        nxt[r+1] >= 0 && (prv[nxt[r+1]+1] = l)
        addb(prv[l+1], l); addb(l, nxt[l+1])
    end
    i = n == 0 ? -1 : 0
    while i != -1
        s = syms[i+1]
        if !isempty(s)
            id = get(tk.ids, s, Int32(-1))
            if id >= 0
                push!(out, id)
            else
                for b in codeunits(s)
                    b1 = get(tk.ids, String([b]), Int32(-1))
                    b1 >= 0 && push!(out, b1)
                end
            end
        end
        i = nxt[i+1]
    end
    out
end
hless(a, b) = a[1] < b[1] || (a[1] == b[1] && a[2] < b[2])
function hpush!(h, x)
    push!(h, x); i = length(h)
    while i > 1
        p = i ÷ 2
        hless(h[i], h[p]) || break
        h[i], h[p] = h[p], h[i]; i = p
    end
end
function hpop!(h)
    top = h[1]; last = pop!(h)
    if !isempty(h)
        h[1] = last; i = 1; n = length(h)
        while true
            l = 2i; r = l + 1; m = i
            l <= n && hless(h[l], h[m]) && (m = l)
            r <= n && hless(h[r], h[m]) && (m = r)
            m == i && break
            h[i], h[m] = h[m], h[i]; i = m
        end
    end
    top
end

function encode_raw!(out::Vector{Int32}, tk::Tok, s::Vector{UInt8})
    isempty(s) && return out
    cp = cpts(s)
    at = 0
    for wl in split_qwen35(cp)
        io = IOBuffer()
        for i in at+1:at+wl
            for b in codeunits(string(Char(cp[i])))
                write(io, tk.byte2u[b+1])
            end
        end
        at += wl
        bpe_word!(out, tk, String(take!(io)))
    end
    out
end
# text -> ids; special tokens written as text become their ids (longest first, left to right)
function encode(tk::Tok, text::AbstractString; parse_special = true)
    frags = Any[Vector{UInt8}(codeunits(text))]
    for id in tk.special
        (!parse_special && tk.typ[id+1] in (2, 3)) && continue
        st = Vector{UInt8}(codeunits(tk.text[id+1]))
        isempty(st) && continue
        nf = Any[]
        for f in frags
            if f isa Int32
                push!(nf, f); continue
            end
            rest = f
            while true
                r = findfirst(st, rest)
                if r === nothing
                    isempty(rest) || push!(nf, rest)
                    break
                end
                first(r) > 1 && push!(nf, rest[1:first(r)-1])
                push!(nf, id)
                rest = rest[last(r)+1:end]
            end
        end
        frags = nf
    end
    out = Int32[]
    for f in frags
        f isa Int32 ? push!(out, f) : encode_raw!(out, tk, f)
    end
    out
end
function piece(tk::Tok, id)
    t = tk.text[id+1]
    tk.typ[id+1] != 1 && return Vector{UInt8}(codeunits(t))
    out = UInt8[]
    for c in t
        b = get(tk.u2byte, c, nothing)
        b === nothing ? append!(out, codeunits(string(c))) : push!(out, b)
    end
    out
end
hidden(tk::Tok, id) = tk.typ[id+1] in (2, 3)
chat_prompt(user) = "<|im_start|>user\n" * user * "<|im_end|>\n<|im_start|>assistant\n<think>\n"

# ---------------------------------------------------------------------------------------------
# PLE n-gram rows (ref/ple.c, llama.cpp build_inp_ple), embeddings on the host
# ---------------------------------------------------------------------------------------------
struct PLE
    mult::Vector{UInt64}
    off::Vector{UInt64}
    voc::Vector{UInt64}
    eos::UInt64
    tab::Ptr{UInt8}          # IQ4_NL rows of 160 values (90 bytes)
end
function PLE(g::GModel)
    tab = tptr(g, g.tensors["per_layer_token_embd.weight"])
    # random rows of the host tables: no readahead around a page that has to come from disk (MADV_RANDOM; an embedding
    # row from disk was seen taking 20 ms)
    for t in host_tables(g)
        madv(tptr(g, t), t.nbytes, 1)
    end
    te = g.tensors["token_embd.weight"]; willneed(tptr(g, te), te.nbytes)    # 0.27 GB read ahead in the background
    PLE(UInt64.(g.kv["qwen4exp.ple.layer_multipliers"]), UInt64.(g.kv["qwen4exp.ple.head_offsets"]),
        UInt64.(g.kv["qwen4exp.ple.head_vocab_sizes"]), UInt64(g.kv["qwen4exp.ple.eos_token_id"]), tab)
end
# the 16 rows of position pos (0-based) of ids
function ple_rows(p::PLE, ids::Vector{Int32}, pos::Int)
    ctx = UInt64[UInt64(ids[pos+1])]
    cut = false
    for s in 1:2
        cut = cut || s > pos || UInt64(ids[pos-s+1]) == p.eos
        push!(ctx, cut ? p.eos : UInt64(ids[pos-s+1]))
    end
    rows = UInt64[]
    for n in 2:3
        mixed = UInt64(0)
        for j in 1:n
            mixed ⊻= ctx[j] * p.mult[j]
        end
        for k in 1:8
            h = (n - 2) * 8 + k
            push!(rows, p.off[h] + mixed % p.voc[h])
        end
    end
    rows
end
function ple_input(p::PLE, ids::Vector{Int32}, pos::Int)
    g = Vector{Float32}(undef, 2560)
    rows = ple_rows(p, ids, pos)
    # rows whose pages left RAM: all 16 reads queued at once (MADV_WILLNEED) instead of 16 faults one after another
    for r in rows
        willneed(p.tab + Int(r) * 90, 90)
    end
    for (h, r) in enumerate(rows)
        dequant_row!(view(g, 160(h-1)+1:160h), T_IQ4_NL, p.tab + Int(r) * 90, 160)
    end
    g
end
function embed(g::GModel, id)
    t = g.tensors["token_embd.weight"]
    dequant_row(t.typ, tptr(g, t) + Int(id) * rowbytes(t), 2560)
end

# ---------------------------------------------------------------------------------------------
# 5. Placement and weights: one arena per die, filled from the mmap'd shards (or attached from a holder)
# ---------------------------------------------------------------------------------------------
mutable struct Arena
    base::CuPtr{UInt8}
    size::Int
    off::Int
    dry::Bool
    fill::Bool             # false with dry false: attach to an arena a holder filled (pointers only)
end
Arena() = Arena(CuPtr{UInt8}(0), 0, 0, true, false)
Arena(base, size; fill = true) = Arena(base, size, 0, false, fill)
function alloc!(A::Arena, n::Integer)
    o = cld(A.off, 256) * 256
    A.off = o + n
    A.dry ? CuPtr{UInt8}(0) : (A.off <= A.size || error("arena overflow"); A.base + o)
end
function rawalloc(n)
    r = Ref{CUC.CUdeviceptr}()
    CUC.cuMemAlloc_v2(r, n)
    reinterpret(CuPtr{UInt8}, r[])
end
upload!(dst::CuPtr, src::Ptr, n) = (n > 0 && CUC.cuMemcpyHtoD_v2(reinterpret(CUC.CUdeviceptr, dst), src, n); nothing)


# n rows of a tensor (row length K, ggml layout at src, srb bytes per row) into the interleaved layout at
# logical rows r0 .. r0 + n - 1: payloads under pd, headers under hd
function repack_il!(pd::Ptr{UInt8}, hd::Ptr{UInt8}, t::Int, K::Int, r0::Int, src::Ptr{UInt8}, n::Int, srb::Int, L::Int = lanes_for(t, K))
    R = 32 ÷ L; nb = K ÷ devE(t); NI = nb ÷ L; P = devP(t); H = devH(t); NC = P ÷ 16
    dense = !haskey(SPLIT, t)
    ps, _, _, hp = dense ? (0, P, 0, NTuple{3,Int}[]) : SPLIT[t]
    B = dense ? P : GGT[t][3]
    @inbounds for rr in 0:n-1
        g, rho = divrem(r0 + rr, R)
        srow = src + rr * srb
        for b in 0:nb-1
            i, j = divrem(b, L); lane = rho * L + j
            s = srow + b * B + ps
            d = pd + ((g * NI + i) * NC * 32 + lane) * 16
            for c in 0:NC-1
                unsafe_store!(Ptr{UInt128}(d + c * CHS), unsafe_load(Ptr{UInt128}(s + 16c)))
            end
            if H > 0
                h = hd + ((g * NI + i) * 32 + lane) * H
                H > 2 && ccall(:memset, Ptr{Cvoid}, (Ptr{UInt8}, Cint, Csize_t), h, 0, H)
                for (so, ln, dd) in hp
                    ln == 2 ? unsafe_store!(Ptr{UInt16}(h + dd), unsafe_load(Ptr{UInt16}(srow + b * B + so))) :
                    ln == 16 ? unsafe_store!(Ptr{UInt128}(h + dd), unsafe_load(Ptr{UInt128}(srow + b * B + so))) :
                    unsafe_copyto!(h + dd, srow + b * B + so, ln)
                end
            end
        end
    end
end

# start reading n bytes of a mapping at p (MADV_WILLNEED)
willneed(p::Ptr, n::Integer) = madv(p, n, 3)
# a matrix made of row ranges of tensors (same type): srcs = [(tensor, first row, rows[, first column, columns]), ...];
# the columns (block aligned) default to the whole row
srcspec(s) = length(s) == 3 ? (s[1], s[2], s[3], 0, s[1].dims[1]) : s
function put_mat!(A::Arena, g::GModel, srcs; wide = false, lanes = 0)
    t0, _, _, _, K = srcspec(srcs[1]); typ = t0.typ
    rows = sum(s[3] for s in srcs)
    L = lanes > 0 ? lanes : lanes_for(typ, K; wide)
    (K ÷ devE(typ)) % L == 0 || error("put_mat!: $L lanes do not divide the $(K ÷ devE(typ)) blocks of $(t0.name)")
    bytes = devbytes_rows(typ, K, rows, L)
    dst = alloc!(A, bytes)
    A.fill || return DevMat(typ, rows, K, dst, L)
    buf = zeros(UInt8, bytes)
    for s in srcs                  # the rows' pages read ahead together, not one fault at a time
        t, r0, n = srcspec(s)
        willneed(tptr(g, t) + r0 * rowbytes(t), n * rowbytes(t))
    end
    r = 0
    GC.@preserve buf for s in srcs
        t, r0, n, c0, kc = srcspec(s)
        (t.typ == typ && kc == K) || error("put_mat!: $(t.name) does not match $(t0.name)")
        eb, bb = GGT[typ][2], GGT[typ][3]
        c0 % (typ in (T_BF16, T_F16, T_F32) ? 32 : eb) == 0 || error("put_mat!: column $c0 of $(t.name) is not block aligned")
        repack_il!(pointer(buf), pointer(buf) + devpay(typ, K, rows, L), typ, K, r, tptr(g, t) + r0 * rowbytes(t) + (c0 ÷ eb) * bb, n, rowbytes(t), L)
        r += n
    end
    GC.@preserve buf upload!(dst, pointer(buf), bytes)
    DevMat(typ, rows, K, dst, L)
end
put_mat!(A::Arena, g::GModel, name::String) = (t = g.tensors[name]; put_mat!(A, g, [(t, 0, nrows(t))]))
function put_f32!(A::Arena, v::Vector{Float32})
    dst = alloc!(A, 4 * length(v))
    A.fill && GC.@preserve v upload!(dst, pointer(v), 4 * length(v))
    fptr(dst)
end
function tensor_f32(g::GModel, name)
    t = g.tensors[name]
    n = prod(t.dims)
    t.typ == T_F32 ? copy(unsafe_wrap(Array, Ptr{Float32}(tptr(g, t)), n)) : dequant_row(t.typ, tptr(g, t), n)
end
put_f32!(A::Arena, g::GModel, name::String) = put_f32!(A, A.fill ? tensor_f32(g, name) : Float32[])
# dry-run sizes must not depend on the data: f32 vectors of known length
put_f32n!(A::Arena, g::GModel, name::String) = A.fill ? put_f32!(A, tensor_f32(g, name)) : fptr(alloc!(A, 4 * prod(g.tensors[name].dims)))

struct DBlk
    l::Int
    attn::Bool
    hcA_norm::FPtr; hcA_dj::DevMat; hcA_up::DevMat
    hcF_norm::FPtr; hcF_dj::DevMat; hcF_up::DevMat
    qkv::DevMat; gate::DevMat; ab::DevMat; ab_bias::FPtr; conv_w::FPtr; ssm_a::FPtr; ssm_norm::FPtr; ssm_out::DevMat
    wq::DevMat; wk::DevMat; wv::DevMat; wo::DevMat; q_norm::FPtr; k_norm::FPtr; idx_x::DevMat; idx_qn::FPtr; idx_kn::FPtr
    router::DevMat; gu::DevMat; down::DevMat; sh_gate::DevMat; sh_up::DevMat; sh_down::DevMat
    ple_key::DevMat; ple_value::DevMat; ple_nk::FPtr; ple_nq::FPtr; ple_nc::FPtr; ple_conv::FPtr
end

function put_block!(A::Arena, g::GModel, l::Int)
    p = "blk.$l."
    T(n) = g.tensors[p*n]
    rows(n) = nrows(T(n))
    attn = !haskey(g.tensors, p * "ssm_a")
    hcA_norm = put_f32n!(A, g, p * "hc_attn_norm.weight")
    hcA_dj = put_mat!(A, g, [(T("hc_attn_down.weight"), 0, 320), (T("hc_attn_inject.weight"), 0, 4)])
    hcA_up = put_mat!(A, g, p * "hc_attn_up.weight")
    hcF_norm = put_f32n!(A, g, p * "hc_ffn_norm.weight")
    hcF_dj = put_mat!(A, g, [(T("hc_ffn_down.weight"), 0, 320), (T("hc_ffn_inject.weight"), 0, 4)])
    hcF_up = put_mat!(A, g, p * "hc_ffn_up.weight")
    qkv = gate = ab = ssm_out = wq = wk = wv = wo = idx_x = NOMAT
    ab_bias = conv_w = ssm_a = ssm_norm = q_norm = k_norm = idx_qn = idx_kn = NOF
    if !attn
        qkv = put_mat!(A, g, p * "attn_qkv.weight"); gate = put_mat!(A, g, p * "attn_gate.weight")
        ab = put_mat!(A, g, [(T("ssm_alpha.weight"), 0, 48), (T("ssm_beta.weight"), 0, 48)])
        ab_bias = A.fill ? put_f32!(A, vcat(tensor_f32(g, p * "ssm_dt.bias"), zeros(Float32, 48))) : fptr(alloc!(A, 4 * 96))
        conv_w = put_f32n!(A, g, p * "ssm_conv1d.weight")
        ssm_a = put_f32n!(A, g, p * "ssm_a"); ssm_norm = put_f32n!(A, g, p * "ssm_norm.weight")
        ssm_out = put_mat!(A, g, p * "ssm_out.weight")
    else
        wq = put_mat!(A, g, p * "attn_q.weight"); wk = put_mat!(A, g, p * "attn_k.weight"); wv = put_mat!(A, g, p * "attn_v.weight")
        wo = put_mat!(A, g, p * "attn_output.weight")
        q_norm = put_f32n!(A, g, p * "attn_q_norm.weight"); k_norm = put_f32n!(A, g, p * "attn_k_norm.weight")
        idx_x = put_mat!(A, g, [(T("indexer.q_proj.weight"), 0, 512), (T("indexer.k_proj.weight"), 0, 128)])
        idx_qn = put_f32n!(A, g, p * "indexer.q_norm.weight"); idx_kn = put_f32n!(A, g, p * "indexer.k_norm.weight")
    end
    router = put_mat!(A, g, [(T("ffn_gate_inp.weight"), 0, 512), (T("ffn_gate_inp_shexp.weight"), 0, 1)])
    ge, ue = T("ffn_gate_exps.weight"), T("ffn_up_exps.weight")
    gu = put_mat!(A, g, vcat([[(ge, 640e, 640), (ue, 640e, 640)] for e in 0:511]...))
    down = put_mat!(A, g, p * "ffn_down_exps.weight")
    sh_gate = put_mat!(A, g, p * "ffn_gate_shexp.weight"); sh_up = put_mat!(A, g, p * "ffn_up_shexp.weight")
    sh_down = put_mat!(A, g, p * "ffn_down_shexp.weight")
    ple_key = ple_value = NOMAT; ple_nk = ple_nq = ple_nc = ple_conv = NOF
    if haskey(g.tensors, p * "ple_key.weight")
        ple_key = put_mat!(A, g, p * "ple_key.weight"); ple_value = put_mat!(A, g, p * "ple_value.weight")
        ple_nk = put_f32n!(A, g, p * "ple_norm_key.weight"); ple_nq = put_f32n!(A, g, p * "ple_norm_query.weight")
        ple_nc = put_f32n!(A, g, p * "ple_norm_conv.weight"); ple_conv = put_f32n!(A, g, p * "ple_conv1d.weight")
    end
    DBlk(l, attn, hcA_norm, hcA_dj, hcA_up, hcF_norm, hcF_dj, hcF_up, qkv, gate, ab, ab_bias, conv_w, ssm_a, ssm_norm, ssm_out,
        wq, wk, wv, wo, q_norm, k_norm, idx_x, idx_qn, idx_kn, router, gu, down, sh_gate, sh_up, sh_down,
        ple_key, ple_value, ple_nk, ple_nq, ple_nc, ple_conv)
end

# the output head: the collapse (hc norm, down, up; NOF/NOMAT where absent) and output rows [r0, r0 + out.rows)
struct DHead
    norm::FPtr; down::DevMat; up::DevMat; out::DevMat; r0::Int
    down2::DevMat                     # B1, n == 1: the down's lanes 2-3 (down: lanes 0-1)
end
DHead(norm, down, up, out, r0) = DHead(norm, down, up, out, r0, NOMAT)
put_head!(A::Arena, g::GModel) = DHead(put_f32n!(A, g, "output_hc_norm.weight"), put_mat!(A, g, "output_hc_down.weight"),
    put_mat!(A, g, "output_hc_up.weight"), put_mat!(A, g, "output.weight"), 0)
# B1: output rows [r0, r0 + nr) and, on die d of the n dies of the last group (n > 0), its share of the final
# collapse, split as a block's hc: the norm weights and down columns of my 4 nc channels, the up rows of my channels
function put_head_slice!(A::Arena, g::GModel, n::Int, d::Int, r0, nr)
    out = put_mat!(A, g, [(g.tensors["output.weight"], r0, nr)])
    n == 0 && return DHead(NOF, NOMAT, NOMAT, out, r0)
    nc = 2560 ÷ n; segs = [(l2 * 2560 + nc * d, nc) for l2 in 0:3]
    w = put_f32_sel!(A, g, "output_hc_norm.weight", lanesel(nc, d))
    rs = [(g.tensors["output_hc_down.weight"], r) for r in 0:319]
    dn = put_mat_cols!(A, g, rs, n == 1 ? segs[1:2] : segs); dn2 = n == 1 ? put_mat_cols!(A, g, rs, segs[3:4]) : NOMAT
    up = put_mat!(A, g, [(g.tensors["output_hc_up.weight"], l2 * 2560 + nc * d + c, 1) for c in 0:nc-1 for l2 in 0:3])
    DHead(w, dn, up, out, r0, dn2)
end

# Stage A placement: die i holds blocks layers[i] (consecutive); the last die also holds the head (full model only)
struct PlacementA
    dies::Vector{Int}                 # CUDA ordinals (CUDA_DEVICE_ORDER=PCI_BUS_ID: nvidia-smi indices)
    layers::Vector{UnitRange{Int}}
    head::Bool
end
function placement_a(dies, counts, nlayers)
    lay = UnitRange{Int}[]; l = 0
    for c in counts
        push!(lay, l:min(l + c, nlayers)-1); l += c
    end
    l >= nlayers || error("placement covers $l of $nlayers blocks")
    PlacementA(dies, lay, nlayers == 48)
end

# fill (or size) die i's arena
function build_die!(A::Arena, g::GModel, pl::PlacementA, i::Int)
    blks = [put_block!(A, g, l) for l in pl.layers[i]]
    head = i == length(pl.dies) && pl.head ? put_head!(A, g) : nothing
    blks, head
end

# ---------------------------------------------------------------------------------------------
# Runtime (Stage A: one die at a time, plain launches)
# ---------------------------------------------------------------------------------------------
const EPS = 1.0f-6
const MAXCTX = Ref(4096)
const CK = 256                                  # attention keys per chunk
const NCHMAX = 16

# (the kernel's concrete type asserted: getkernel's cache returns Any, and the call through it was dynamic dispatch,
# compiled on first use: the juliac executable spent its 60 s compiling the draft's launches)
function launch(f::F, args::Vararg{Any,N}; threads, blocks, shmem = 0) where {F,N}
    TT = Tuple{map(typeof, args)...}
    k = getkernel(f, TT)::CUC.HostKernel{F,TT}
    k(args...; threads, blocks, shmem)
end

struct LState
    conv::FPtr; dn::FPtr
    kq::GPtr; ks::GPtr; vq::GPtr; vs::GPtr; pool::GPtr; ring::FPtr; score::FPtr; sel::IPtr
    kf::FPtr                          # B1 attention: 1 / rms(k) of each cached position (the cache holds rope(k w) unnormalised)
end
LState(conv, dn, kq, ks, vq, vs, pool, ring, score, sel) = LState(conv, dn, kq, ks, vq, vs, pool, ring, score, sel, NOF)
mutable struct DieRT
    ord::Int
    blks::Vector{DBlk}
    head::Union{DHead,Nothing}
    st::Vector{LState}
    keep::Vector{Any}
    b::Dict{Symbol,FPtr}
    idx::IPtr
    rope::FPtr
    pler::FPtr
end
function dbuf!(rt::DieRT, T, n)
    a = CUDA.zeros(T, n)
    push!(rt.keep, a)
    a
end
const BUFS = (:S => 10240, :xh => 10240, :down => 324, :t => 320, :up => 10240, :z => 2560, :qkv => 10240, :gate => 6144,
    :ab => 96, :u => 10240, :on => 6144, :y => 2560, :qg => 12288, :k => 512, :v => 512, :ix => 640, :qo => 6144, :ao => 6144,
    :iq => 512, :pm => 24 * NCHMAX, :pd => 24 * NCHMAX, :po => 24 * NCHMAX * 256, :r => 513, :w => 11, :gu => 14080, :h => 7040,
    :out => 2560, :pleK => 10240, :pleV => 2560, :pleg => 2560)
function rope_table(n)
    tab = Vector{Float32}(undef, 64n)
    for t in 0:n-1, p in 0:31
        th = 1.0e7^(-2.0 * p / 64.0)
        tab[64t+p+1] = Float32(cos(t * th)); tab[64t+32+p+1] = Float32(sin(t * th))
    end
    tab
end
function DieRT(ord, blks, head)
    device!(ord)
    rt = DieRT(ord, blks, head, LState[], Any[], Dict{Symbol,FPtr}(), NOI, NOF, NOF)
    for (k, n) in BUFS
        rt.b[k] = fptr(dbuf!(rt, Float32, n))
    end
    head === nothing || (rt.b[:logits] = fptr(dbuf!(rt, Float32, 248320)))
    rt.idx = reinterpret(IPtr, pointer(dbuf!(rt, Int32, 16)))
    ra = CuArray(rope_table(MAXCTX[])); push!(rt.keep, ra); rt.rope = fptr(ra)
    for B in blks
        if B.attn
            mc = MAXCTX[]
            kq = dbuf!(rt, UInt8, mc * 512); vq = dbuf!(rt, UInt8, mc * 512)
            ks = dbuf!(rt, UInt8, mc * 32); vs = dbuf!(rt, UInt8, mc * 32)
            pool = dbuf!(rt, UInt8, (mc ÷ 4) * 256)
            push!(rt.st, LState(NOF, NOF, gptr(kq), gptr(ks), gptr(vq), gptr(vs), gptr(pool), fptr(dbuf!(rt, Float32, 512)),
                fptr(dbuf!(rt, Float32, mc ÷ 4)), reinterpret(IPtr, pointer(dbuf!(rt, Int32, 512)))))
        else
            push!(rt.st, LState(fptr(dbuf!(rt, Float32, 4 * 10240)), fptr(dbuf!(rt, Float32, 48 * 16384)), GPtr(0), GPtr(0), GPtr(0), GPtr(0), GPtr(0), NOF, NOF, NOI))
        end
        B.l == 1 && (rt.pler = fptr(dbuf!(rt, Float32, 10 * 10240)))
    end
    rt
end
function reset!(rt::DieRT)
    device!(rt.ord)
    for st in rt.st
        st.conv == NOF || CUC.cuMemsetD8_v2(reinterpret(CUC.CUdeviceptr, st.conv), 0, 4 * 4 * 10240)
        st.dn == NOF || CUC.cuMemsetD8_v2(reinterpret(CUC.CUdeviceptr, st.dn), 0, 4 * 48 * 16384)
    end
    rt.pler == NOF || CUC.cuMemsetD8_v2(reinterpret(CUC.CUdeviceptr, rt.pler), 0, 4 * 10 * 10240)
end
B_(rt, k) = rt.b[k]
off(p::FPtr, n) = p + 4n

function hc_read!(rt::DieRT, norm::FPtr, dj::DevMat, up::DevMat)
    b = rt.b
    launch(k_rmsnorm!, b[:xh], b[:S], norm, Int32(2560), Int32(2560), EPS; threads = 256, blocks = 4)
    gemv!(b[:down], dj, b[:xh])
    launch(k_hc_silu!, b[:t], b[:down], Int32(320); threads = 320, blocks = 1)
    gemv!(b[:up], up, b[:t])
    launch(k_hc_mix!, b[:z], b[:up], b[:xh]; threads = 256, blocks = 10)
end

function block_fwd!(rt::DieRT, i::Int, t::Int)
    B = rt.blks[i]; st = rt.st[i]; b = rt.b
    t32 = Int32(t)
    if B.l == 1
        gemv!(b[:pleK], B.ple_key, b[:pleg]); gemv!(b[:pleV], B.ple_value, b[:pleg])
        launch(k_ple!, b[:S], b[:pleK], b[:pleV], B.ple_nk, B.ple_nq, B.ple_nc, B.ple_conv, rt.pler, t32, EPS; threads = 256, blocks = 4)
    end
    hc_read!(rt, B.hcA_norm, B.hcA_dj, B.hcA_up)
    if !B.attn
        gemv!(b[:qkv], B.qkv, b[:z]); gemv!(b[:gate], B.gate, b[:z]); gemv!(b[:ab], B.ab, b[:z])
        launch(k_conv1d!, b[:u], b[:qkv], st.conv, B.conv_w, t32, Int32(10240); threads = 256, blocks = 40)
        dlaunch(k_deltanet!, b[:on], st.dn, b[:u], b[:ab], B.ab_bias, B.ssm_a, b[:gate], B.ssm_norm, EPS, Int32(16); threads = 256, blocks = 48)
        gemv!(b[:y], B.ssm_out, b[:on])
    else
        gemv!(b[:qg], B.wq, b[:z]); gemv!(b[:k], B.wk, b[:z]); gemv!(b[:v], B.wv, b[:z]); gemv!(b[:ix], B.idx_x, b[:z])
        cs = off(rt.rope, 64t); cs3 = off(rt.rope, 64 * max(t - 3, 0))
        launch(k_attn_prep!, b[:qo], st.kq, st.ks, st.vq, st.vs, b[:qg], b[:k], b[:v], B.q_norm, B.k_norm, cs, t32, EPS, Int32(24), Int32(2); threads = 256, blocks = 26)
        launch(k_idx_prep!, b[:iq], st.ring, st.pool, b[:ix], B.idx_qn, B.idx_kn, cs, cs3, t32, EPS; threads = 128, blocks = 5)
        nv = (t + 1) ÷ 4; tail = (t + 1) % 4
        if nv > 512
            launch(k_idx_score!, st.score, b[:iq], st.pool, Int32(nv); threads = 256, blocks = cld(nv, 256))
            launch(k_idx_select!, st.sel, st.score, Int32(nv), Int32(512); threads = 1024, blocks = 1)
            nsel = 512; nkeys = 4 * 512 + tail; tailb = nv
        else
            nsel = -1; nkeys = t + 1; tailb = 0
        end
        nch = cld(nkeys, CK)
        launch(k_attn_part!, b[:pm], b[:pd], b[:po], b[:qo], st.kq, st.ks, st.vq, st.vs, st.sel, Int32(nsel), Int32(nkeys), Int32(tailb), Int32(CK), Int32(2);
            threads = 256, blocks = (nch, 24))
        launch(k_attn_merge!, b[:ao], b[:pm], b[:pd], b[:po], b[:qg], Int32(nch); threads = 256, blocks = 24)
        gemv!(b[:y], B.wo, b[:ao])
    end
    launch(k_hc_write!, b[:S], b[:y], off(b[:down], 320); threads = 256, blocks = 40)
    hc_read!(rt, B.hcF_norm, B.hcF_dj, B.hcF_up)
    gemv!(b[:r], B.router, b[:z])
    launch(k_router!, rt.idx, b[:w], b[:r]; threads = 512, blocks = 1)
    gemv!(b[:gu], B.gu, b[:z]; rows = 12800, mode = 1, idx = rt.idx, rpe = 1280)
    gemv!(off(b[:gu], 12800), B.sh_gate, b[:z]); gemv!(off(b[:gu], 13440), B.sh_up, b[:z])
    launch(k_swiglu!, b[:h], b[:gu], Int32(640), Int32(11), Int32(0); threads = 256, blocks = cld(7040, 256))
    gemv!(b[:out], B.down, b[:h]; rows = 2560, mode = 3, idx = rt.idx, sc = b[:w], rpe = 2560, ns = 10)
    gemv!(b[:out], B.sh_down, off(b[:h], 6400); mode = 2, sc = off(b[:w], 10))
    launch(k_hc_write!, b[:S], b[:out], off(b[:down], 320); threads = 256, blocks = 40)
end
function head_fwd!(rt::DieRT)
    H = rt.head; b = rt.b
    launch(k_rmsnorm!, b[:xh], b[:S], H.norm, Int32(2560), Int32(2560), EPS; threads = 256, blocks = 4)
    gemv!(b[:down], H.down, b[:xh])
    launch(k_hc_silu!, b[:t], b[:down], Int32(320); threads = 320, blocks = 1)
    gemv!(b[:up], H.up, b[:t])
    launch(k_hc_mix!, b[:z], b[:up], b[:xh]; threads = 256, blocks = 10)
    gemv!(b[:logits], H.out, b[:z])
end
dl(p::FPtr, n) = (v = Vector{Float32}(undef, n); GC.@preserve v CUC.cuMemcpyDtoH_v2(pointer(v), reinterpret(CUC.CUdeviceptr, p), 4n); v)
ul!(p::FPtr, v::Vector{Float32}) = GC.@preserve v upload!(reinterpret(CuPtr{UInt8}, p), pointer(v), 4 * length(v))

mutable struct Engine
    g::GModel
    tk::Tok
    ple::PLE
    dies::Vector{DieRT}
    ids::Vector{Int32}          # tokens fed so far (position = index - 1)
    syncs::Int
end

# feed token id at the next position; returns the logits (host) when the model includes the head
function step!(E::Engine, id::Integer; logits = true)
    t = length(E.ids)
    push!(E.ids, Int32(id))
    S = repeat(embed(E.g, id), 4)
    for (di, rt) in enumerate(E.dies)
        device!(rt.ord)
        ul!(rt.b[:S], S)
        any(B -> B.l == 1, rt.blks) && ul!(rt.b[:pleg], ple_input(E.ple, E.ids, t))
        for i in eachindex(rt.blks)
            block_fwd!(rt, i, t)
        end
        S = dl(rt.b[:S], 10240)
    end
    rt = E.dies[end]
    if logits && rt.head !== nothing
        device!(rt.ord)
        head_fwd!(rt)
        return dl(rt.b[:logits], 248320)
    end
    S
end

# ---------------------------------------------------------------------------------------------
# Stage B: the two dies of a board split every block of their range; boards form a pipeline.
#   die position d of a group of n (1 or 2) owns channels C = [nc d, nc d + nc), nc = 2560 / n, of all four
#   hyper-connection lanes (S is kept lane-major: S[l nc + c']), key heads [nkh d, +nkh) (nkh = 16 / n) with their
#   value heads 16 g + nkh d + i (local h' = g nkh + i), attention heads [nq d, +nq) (nq = 24 / n) with KV heads
#   [nkv d, +nkv), and FFN rows [F d, +F) (F = 640 / n) of every expert and the shared expert.
#   Per block the pair exchanges six small vectors: the hc down partials (twice), z halves (twice, with the router
#   partial), and the partial sums of the attention/DeltaNet output and of the MoE output.
# ---------------------------------------------------------------------------------------------
struct PBlk
    l::Int; attn::Bool; n::Int; d::Int
    hcA_w::FPtr; hcA_dj::DevMat; hcA_dj2::DevMat; hcA_up::DevMat     # n == 1: dj holds lanes 0-1 and dj2 lanes 2-3
    hcF_w::FPtr; hcF_dj::DevMat; hcF_dj2::DevMat; hcF_up::DevMat
    qkv::DevMat; gate::DevMat; ab::DevMat; ab_bias::FPtr; conv_w::FPtr; ssm_a::FPtr; ssm_norm::FPtr; ssm_out::DevMat
    wq::DevMat; wk::DevMat; wv::DevMat; wo::DevMat; q_norm::FPtr; k_norm::FPtr; idx_x::DevMat; idx_qn::FPtr; idx_kn::FPtr
    router::DevMat; gu::DevMat; down::DevMat; sh_gate::DevMat; sh_up::DevMat; sh_down::DevMat
    ple_key::DevMat; ple_value::DevMat; ple_nk::FPtr; ple_nq::FPtr; ple_nc::FPtr; ple_conv::FPtr
    router_sh::DevMat                 # the shared expert's gate row, my nc columns (the router itself: my 512 / n rows)
    # copy trials (the holder keeps both: the module picks): the whole router and shared gate row (ROUTERCOPY); my
    # nq query heads whole and my KV head's k / v rows whole with their norm weights (ATTNCOPY)
    router_f::DevMat; wq_h::DevMat; wk_f::DevMat; wv_f::DevMat; qn_f::FPtr; kn_f::FPtr
end

# rows of tensors with columns gathered from several block-aligned ranges, as a ggml-layout host buffer; then repacked
function put_mat_cols!(A::Arena, g::GModel, rowsrc::Vector{Tuple{GTensor,Int}}, segs::Vector{Tuple{Int,Int}}; wide = false)
    t0 = rowsrc[1][1]; typ = t0.typ
    K = sum(s[2] for s in segs)
    rows = length(rowsrc)
    L = lanes_for(typ, K; wide)
    bytes = devbytes_rows(typ, K, rows, L)
    dst = alloc!(A, bytes)
    A.fill || return DevMat(typ, rows, K, dst, L)
    eb, bb = GGT[typ][2], GGT[typ][3]
    rb = K ÷ eb * bb
    tmp = Vector{UInt8}(undef, rows * rb)
    GC.@preserve tmp for (i, (t, r)) in enumerate(rowsrc)
        t.typ == typ || error("put_mat_cols!: $(t.name) type")
        o = 0
        for (c0, len) in segs
            n = len ÷ eb * bb
            unsafe_copyto!(pointer(tmp) + (i - 1) * rb + o, tptr(g, t) + r * rowbytes(t) + c0 ÷ eb * bb, n)
            o += n
        end
    end
    buf = zeros(UInt8, bytes)
    GC.@preserve buf tmp repack_il!(pointer(buf), pointer(buf) + devpay(typ, K, rows, L), typ, K, 0, pointer(tmp), rows, rb, L)
    GC.@preserve buf upload!(dst, pointer(buf), bytes)
    DevMat(typ, rows, K, dst, L)
end
function put_f32_sel!(A::Arena, g::GModel, name, idx::Vector{Int})      # elements idx (0-based) of a vector tensor
    A.fill || return fptr(alloc!(A, 4 * length(idx)))
    v = tensor_f32(g, name)
    put_f32!(A, Float32[v[i+1] for i in idx])
end
lanesel(nc, d) = [l * 2560 + nc * d + c for l in 0:3 for c in 0:nc-1]          # lane-major channels of die d
# attention over n dies (n = 1, 2, 4, 8), every die an equal share of every tensor and no copies: the 2 KV heads split
# by dims, GS = n / 2 dies a head (dims [DH s, DH (s + 1)) on die d = GS g + s of head g), and the 12 query heads of
# head g on each of its dies, the same dims (q, gate, the norm weights); the KV cache likewise. Scores are sums of the
# GS dies' partials. (n = 1: both heads, all dims.) The indexer's key: IW = 128 / n dims a die, rope pairs (p, p + 32)
# kept on one die (the first n / 2 dies hold dims [w d, w d + w) and [32 + w d, ...), w = 64 / n; the others 64 ..).
attn_geom(n) = n == 1 ? (GS = 1, DH = 256, QH = 24, nkvl = 2, IW = 128) : (GS = n ÷ 2, DH = 512 ÷ n, QH = 12, nkvl = 1, IW = 128 ÷ n)
attn_gs(n, d) = n == 1 ? (0, 0) : divrem(d, n ÷ 2)                # (kv head g, dims part s) of die d
idims(n, d) = n == 1 ? collect(0:127) : d < n ÷ 2 ? vcat((64 ÷ n) * d .+ (0:64÷n-1), 32 .+ (64 ÷ n) * d .+ (0:64÷n-1)) :
    collect(64 .+ (128 ÷ n) * (d - n ÷ 2) .+ (0:128÷n-1))

# the output projections (K = 3072 a die: 12 blocks a row) in wide rows, 6 lanes (2 blocks each, 5 rows a group): row
# groups of 8 KB, not 14 KB (4 lanes, 3 blocks, 8 rows), for a shorter phase tail
const OWIDE = get(ENV, "FLASHNEXT_OWIDE", "1") == "1"
# Every model line of a block split into n equal shares, none copied (but ssm_norm: the 128 weights every value head's
# readout is scaled by, on each die), each by rows or by block-aligned columns:
#   hc down/inject, ssm_out, attn_output: columns (my channels / value heads / query heads); the rest: rows.
#   The router: my 512 / n rows over all of x; its shared-expert row: my nc columns. Expert and shared down: my nc
#   output rows of all 640 inner columns (640 / n columns would cut Q2_0's 64-element blocks for n = 8).
function put_pblock!(A::Arena, g::GModel, l::Int, n::Int, d::Int)
    p = "blk.$l."
    T(nm) = g.tensors[p*nm]
    attn = !haskey(g.tensors, p * "ssm_a")
    nc = 2560 ÷ n; nkh = 16 ÷ n; nq = 24 ÷ n; F = 640 ÷ n
    lsel = lanesel(nc, d)
    chmaj = [l2 * 2560 + nc * d + c for c in 0:nc-1 for l2 in 0:3]              # channel-major rows (4 c' + l)
    hcsegs = [(l2 * 2560 + nc * d, nc) for l2 in 0:3]
    function hc(which)
        w = put_f32_sel!(A, g, p * "hc_$(which)_norm.weight", lsel)
        dn, inj = T("hc_$(which)_down.weight"), T("hc_$(which)_inject.weight")
        rs = vcat([(dn, r) for r in 0:319], [(inj, r) for r in 0:3])
        dj = put_mat_cols!(A, g, rs, n == 1 ? hcsegs[1:2] : hcsegs)
        dj2 = n == 1 ? put_mat_cols!(A, g, rs, hcsegs[3:4]) : NOMAT
        up = put_mat!(A, g, [(T("hc_$(which)_up.weight"), r, 1) for r in chmaj])
        w, dj, dj2, up
    end
    hcA_w, hcA_dj, hcA_dj2, hcA_up = hc("attn")
    hcF_w, hcF_dj, hcF_dj2, hcF_up = hc("ffn")
    qkv = gate = ab = ssm_out = wq = wk = wv = wo = idx_x = NOMAT
    ab_bias = conv_w = ssm_a = ssm_norm = q_norm = k_norm = idx_qn = idx_kn = NOF
    vh = [16gg + nkh * d + i for gg in 0:2 for i in 0:nkh-1]                     # my value heads (global), local order
    if !attn
        # u channels: q of my key heads, k of my key heads, v of my value heads
        uch = vcat([128 * (nkh * d) + c for c in 0:128nkh-1], [2048 + 128 * (nkh * d) + c for c in 0:128nkh-1],
                   [4096 + 128h + c for h in vh for c in 0:127])
        qkv = put_mat!(A, g, vcat([(T("attn_qkv.weight"), 128 * nkh * d, 128nkh), (T("attn_qkv.weight"), 2048 + 128 * nkh * d, 128nkh)],
                                  [(T("attn_qkv.weight"), 4096 + 128 * (16gg + nkh * d), 128nkh) for gg in 0:2]))
        gate = put_mat!(A, g, [(T("attn_gate.weight"), 128 * (16gg + nkh * d), 128nkh) for gg in 0:2])
        ab = put_mat!(A, g, vcat([(T("ssm_alpha.weight"), 16gg + nkh * d, nkh) for gg in 0:2], [(T("ssm_beta.weight"), 16gg + nkh * d, nkh) for gg in 0:2]))
        ab_bias = A.fill ? put_f32!(A, vcat(tensor_f32(g, p * "ssm_dt.bias")[vh.+1], zeros(Float32, 3nkh))) : fptr(alloc!(A, 4 * 6nkh))
        conv_w = put_f32_sel!(A, g, p * "ssm_conv1d.weight", [4c + k for c in uch for k in 0:3])
        ssm_a = put_f32_sel!(A, g, p * "ssm_a", vh); ssm_norm = put_f32n!(A, g, p * "ssm_norm.weight")
        ssm_out = put_mat_cols!(A, g, [(T("ssm_out.weight"), r) for r in 0:2559], [(128 * (16gg + nkh * d), 128nkh) for gg in 0:2]; wide = OWIDE)
    else
        ag = attn_geom(n); gk, s = attn_gs(n, d); DH, QH = ag.DH, ag.QH
        qh = n == 1 ? (0:23) : 12gk .+ (0:11)                                     # my query heads (global)
        kvh = n == 1 ? (0:1) : (gk:gk)
        # per query head: its q dims, then its gate dims (QG[2 DH h' ..])
        wq = put_mat!(A, g, vcat([[(T("attn_q.weight"), 512h + DH * s, DH), (T("attn_q.weight"), 512h + 256 + DH * s, DH)] for h in qh]...))
        wk = put_mat!(A, g, [(T("attn_k.weight"), 256kv + DH * s, DH) for kv in kvh]; wide = true)
        wv = put_mat!(A, g, [(T("attn_v.weight"), 256kv + DH * s, DH) for kv in kvh]; wide = true)
        wo = put_mat_cols!(A, g, [(T("attn_output.weight"), r) for r in 0:2559], [(256 * nq * d, 256nq)]; wide = OWIDE)
        q_norm = put_f32_sel!(A, g, p * "attn_q_norm.weight", collect(DH * s .+ (0:DH-1)))
        k_norm = put_f32_sel!(A, g, p * "attn_k_norm.weight", collect(DH * s .+ (0:DH-1)))
        # indexer: my 512 / n query rows (unused below 2048 keys) and my key dims
        idd = idims(n, d)
        idx_x = put_mat!(A, g, vcat([(T("indexer.q_proj.weight"), (512 ÷ n) * d, 512 ÷ n)], [(T("indexer.k_proj.weight"), r, 1) for r in idd]))
        idx_qn = put_f32_sel!(A, g, p * "indexer.q_norm.weight", collect((128 ÷ n) * d .+ (0:128÷n-1)))
        idx_kn = put_f32_sel!(A, g, p * "indexer.k_norm.weight", idd)
    end
    rt_ = T("ffn_gate_inp.weight"); rs_ = T("ffn_gate_inp_shexp.weight")
    # FLASHNEXT_ROUTER_L lanes a row (default the wide layout's)
    RR = 512 ÷ n
    router = router_sh = router_f = NOMAT
    # (FLASHNEXT_ROUTERBOTH=1: the copy and the split both held, the module's ROUTERCOPY picks: an A/B on one holder)
    if ROUTERCOPY || ROUTERBOTH        # (kept: 40.43 tok/s against 39.57 split, alternating A/B)
        router_f = put_mat!(A, g, [(rt_, 0, 512), (rs_, 0, 1)]; wide = true, lanes = ROUTER_L)
    end
    if ROUTERPAIR                      # (FLASHNEXT_ROUTERPAIR: my board's half of the router, the even die the shared row)
        router = put_mat!(A, g, [(rt_, 256 * (d % 2), 256)]; wide = true, lanes = ROUTER_L)
        router_sh = d % 2 == 0 ? put_mat!(A, g, [(rs_, 0, 1)]; wide = true, lanes = ROUTER_L) : NOMAT
    elseif !ROUTERCOPY || ROUTERBOTH
        router = put_mat!(A, g, [(rt_, RR * d, RR)]; wide = true, lanes = ROUTER_L)
        router_sh = put_mat!(A, g, [(rs_, 0, 1, nc * d, nc)]; wide = true)
    end
    wq_h = wk_f = wv_f = NOMAT; qn_f = kn_f = NOF
    if attn && ATTNCOPY
        gk = attn_gs(n, d)[1]; kvh = n == 1 ? (0:1) : (gk:gk)
        wq_h = put_mat!(A, g, [(T("attn_q.weight"), 512 * nq * d, 512nq)])
        wk_f = put_mat!(A, g, [(T("attn_k.weight"), 256kv, 256) for kv in kvh]; wide = true)
        wv_f = put_mat!(A, g, [(T("attn_v.weight"), 256kv, 256) for kv in kvh]; wide = true)
        qn_f = put_f32n!(A, g, p * "attn_q_norm.weight"); kn_f = put_f32n!(A, g, p * "attn_k_norm.weight")
    end
    ge, ue = T("ffn_gate_exps.weight"), T("ffn_up_exps.weight")
    # gate and up rows interleaved per expert (row 2i gate i, 2i + 1 up i): the GEMV writes silu(gate) up itself
    gu = put_mat!(A, g, [(isodd(k) ? ue : ge, 640e + F * d + k ÷ 2, 1) for e in 0:511 for k in 0:2F-1])
    sh_gate = put_mat!(A, g, [(T("ffn_gate_shexp.weight"), F * d, F)]; wide = true); sh_up = put_mat!(A, g, [(T("ffn_up_shexp.weight"), F * d, F)]; wide = true)
    if dfast(n)                        # the first half's dies: my nc rows, then my slow partner's (die d + n / 2), per expert
        down = sh_down = NOMAT
        if d < n ÷ 2
            down = put_mat!(A, g, [(T("ffn_down_exps.weight"), 2560e + nc * r, nc) for e in 0:511 for r in (d, d + n ÷ 2)])
            sh_down = put_mat!(A, g, [(T("ffn_down_shexp.weight"), nc * r, nc) for r in (d, d + n ÷ 2)])
        elseif DFASTBOTH               # (and the others their own, so a module without DFAST runs on the same holder)
            down = put_mat!(A, g, [(T("ffn_down_exps.weight"), 2560e + nc * d, nc) for e in 0:511])
            sh_down = put_mat!(A, g, [(T("ffn_down_shexp.weight"), nc * d, nc)])
        end
    else
        down = put_mat!(A, g, [(T("ffn_down_exps.weight"), 2560e + nc * d, nc) for e in 0:511])
        sh_down = put_mat!(A, g, [(T("ffn_down_shexp.weight"), nc * d, nc)])
    end
    ple_key = ple_value = NOMAT; ple_nk = ple_nq = ple_nc = ple_conv = NOF
    if haskey(g.tensors, p * "ple_key.weight")
        ple_key = put_mat!(A, g, [(T("ple_key.weight"), r, 1) for r in lsel])
        ple_value = put_mat!(A, g, [(T("ple_value.weight"), nc * d, nc)])
        ple_nk = put_f32_sel!(A, g, p * "ple_norm_key.weight", lsel); ple_nq = put_f32_sel!(A, g, p * "ple_norm_query.weight", lsel)
        ple_nc = put_f32_sel!(A, g, p * "ple_norm_conv.weight", lsel)
        ple_conv = put_f32_sel!(A, g, p * "ple_conv1d.weight", [4c + k for c in lsel for k in 0:3])
    end
    PBlk(l, attn, n, d, hcA_w, hcA_dj, hcA_dj2, hcA_up, hcF_w, hcF_dj, hcF_dj2, hcF_up, qkv, gate, ab, ab_bias, conv_w, ssm_a, ssm_norm, ssm_out,
        wq, wk, wv, wo, q_norm, k_norm, idx_x, idx_qn, idx_kn, router, gu, down, sh_gate, sh_up, sh_down,
        ple_key, ple_value, ple_nk, ple_nq, ple_nc, ple_conv, router_sh, router_f, wq_h, wk_f, wv_f, qn_f, kn_f)
end

struct PlacementB
    groups::Vector{Vector{Int}}
    layers::Vector{UnitRange{Int}}
end
function placement_b(groups, counts)
    lay = UnitRange{Int}[]; l = 0
    for c in counts
        push!(lay, l:l+c-1); l += c
    end
    l <= 48 || error("stage B placement covers $l blocks")
    PlacementB(groups, lay)
end
placement_key(pl::PlacementB) = string("B;", join([join(gr, "+") for gr in pl.groups], ","), ";", join(pl.layers, ","), ";v", LAYOUT_VERSION,
    ROUTER_L > 0 ? ";router$ROUTER_L" : "", ROUTERBOTH ? ";rboth" : ROUTERCOPY ? ";rcopy" : "", ROUTERPAIR ? ";rpair" : "", ATTNCOPY ? ";acopy" : "",
    DFAST ? (DFASTBOTH ? ";dfastb" : ";dfast") : "", length(pl.groups) > 1 ? ";headlast" : "")
const ROUTERBOTH = get(ENV, "FLASHNEXT_ROUTERBOTH", "0") == "1"
# FLASHNEXT_ROUTERPAIR=1: the router split over each board's 2 dies (rows 256 d mod 2 .. of 512, the shared gate row on
# the even die), each half's logits to both dies over the on-board link (10.4 GB/s, about 1 us): with ROUTERCOPY every
# die read the whole 2.6 MB of BF16 rows a block (the router's 41 us of work), split 8-way the logits crossed the
# slow boards' links
const ROUTERPAIR = get(ENV, "FLASHNEXT_ROUTERPAIR", "0") == "1"
# FLASHNEXT_DFAST (AOT shares from the links: boards 2 and 3, dies 4-7, take 0.60-0.63 GB/s in against 3.4-10.4 for
# boards 0 and 1): the MoE down's rows (routed and shared) on the first half's dies only, each die d < n / 2 its nc
# channels and those of die d + n / 2, whose output it writes into that die's OUTP. Dies n / 2 .. then take no h (the
# 11 x 640 values a block over their links: about 21 KB, 35 us at 0.6 GB/s) and receive their nc outputs instead.
# Off by default: every model line is split equally over the dies (ffn_down_exps and ffn_down_shexp 1/8 each).
const DFAST = get(ENV, "FLASHNEXT_DFAST", "0") == "1"   # (41.36-41.44 tok/s against 40.67-40.77 without, ABBA x2)
@inline dfast(n::Integer) = DFAST && n >= 4 && n % 4 == 0
# FLASHNEXT_DFASTBOTH=1 (with DFAST, holder side): the second half's dies keep their own rows too, and a module built
# without DFAST reads a fast die's first nc rows of each expert's 2 nc (rows per expert from the matrix): an A/B on one
# holder
const DFASTBOTH = get(ENV, "FLASHNEXT_DFASTBOTH", "0") == "1"
# die k of group i: its blocks and, when the placement ends with block 47 and i is the last group, its share of the
# output head (the k-th of the group's n dies holds output rows [(k - 1) per, +per), the last die the rest) and of the
# collapse. Only the last group runs the collapse (block 48 of its k_token!) and has its Z in its LL region, so its
# dies hold the whole vocabulary between them (earlier groups' slices would wait for a Z that never reaches them)
function build_die!(A::Arena, g::GModel, pl::PlacementB, i::Int, k::Int)
    n = length(pl.groups[i])
    blks = [put_pblock!(A, g, l, n, k - 1) for l in pl.layers[i]]
    head = nothing
    if last(pl.layers[end]) == 47 && i == length(pl.groups)
        per = (248320 ÷ n) ÷ 16 * 16
        head = put_head_slice!(A, g, n, k - 1, (k - 1) * per, k == n ? 248320 - (n - 1) * per : per)
    end
    blks, head
end

# ---- Stage B kernels (lane-major die slices) ----
# xw = S w (4 nc), ss[l] = sum over my channels of S_l^2 (written to ss[0..3]); one CTA per lane
function k_hc_prep!(xw::FPtr, ss::FPtr, S::FPtr, w::FPtr, nc::Int32)
    l = blockIdx().x - Int32(1); tid = threadIdx().x - Int32(1)
    @inbounds begin
        s = 0.0f0; c = tid
        while c < nc
            v = ldf(S, l * nc + c); s = fma(v, v, s); stf(xw, l * nc + c, v * ldf(w, l * nc + c)); c += blockDim().x
        end
        s = bsum(s)
        tid == Int32(0) && stf(ss, l, s)
    end
    return
end
# down[r] = sum_l r_l sum_p part[p][l][r]; r_l = 1/sqrt(sum_p ss[p][l] / 2560 + eps); t = silu(down / 4) (r < 320), inj = down[320:324]
# part: np x [4][324] then ss: np x [4] (part at pt + p 1300 floats: 1296 partials then 4 ss)
function k_hc_comb!(t::FPtr, rl::FPtr, pt::FPtr, np::Int32, eps::Float32)
    r = threadIdx().x - Int32(1)
    @inbounds if r < Int32(324)
        rr = ntuple(Val(4)) do l
            s = 0.0f0
            for p in Int32(0):np-Int32(1)
                s += ldf(pt, p * Int32(1300) + Int32(1296) + Int32(l - 1))
            end
            1.0f0 / sqrt(s / 2560.0f0 + eps)
        end
        dsum = 0.0f0
        for l in 1:4, p in Int32(0):np-Int32(1)
            dsum = fma(rr[l], ldf(pt, p * Int32(1300) + Int32(324) * Int32(l - 1) + r), dsum)
        end
        stf(t, r, r < Int32(320) ? silu(dsum * 0.25f0) : dsum)
        r < Int32(4) && stf(rl, r, rr[r+1])
    end
    return
end
# z[zoff + c'] = (1/4) sum_l sigmoid(up[4 c' + l]) S[l nc + c'] r_l w[l nc + c']
function k_hc_mix2!(z::FPtr, up::FPtr, S::FPtr, w::FPtr, rl::FPtr, nc::Int32, zoff::Int32)
    c = gtid()
    @inbounds if c < nc
        a = 0.0f0
        for l in Int32(0):Int32(3)
            a = fma(sigm(ldf(up, Int32(4) * c + l)), ldf(S, l * nc + c) * ldf(rl, l) * ldf(w, l * nc + c), a)
        end
        stf(z, zoff + c, a * 0.25f0)
    end
    return
end
# S[l nc + c'] += 2 sigmoid(inj[l] / 4) (sum_p part[p][coff + c'])
function k_hc_write2!(S::FPtr, part::FPtr, np::Int32, inj::FPtr, nc::Int32, coff::Int32)
    i = gtid()
    @inbounds if i < Int32(4) * nc
        l = i ÷ nc; c = i % nc
        y = 0.0f0
        for p in Int32(0):np-Int32(1)
            y += ldf(part, p * Int32(2560) + coff + c)
        end
        stf(S, i, fma(2.0f0 * sigm(0.25f0 * ldf(inj, l)), y, ldf(S, i)))
    end
    return
end
# the router over r = sum_p rp[p] (513 floats each)
function k_router2!(idx::IPtr, w::FPtr, rs::FPtr, rp::FPtr, np::Int32)
    tid = threadIdx().x - Int32(1)
    @inbounds if tid < Int32(513)
        s = 0.0f0
        for p in Int32(0):np-Int32(1)
            s += ldf(rp, p * Int32(513) + tid)
        end
        stf(rs, tid, s)
    end
    return
end
# PLE partials of my channels per lane: [ssK_l, ssS_l, sum K nk S nq (4 each), sum V^2]; CTA l (the last CTA sums V^2)
function k_ple_part!(pt::FPtr, K::FPtr, V::FPtr, S::FPtr, nk::FPtr, nq::FPtr, nc::Int32)
    l = blockIdx().x - Int32(1); tid = threadIdx().x - Int32(1)
    @inbounds begin
        a = 0.0f0; b = 0.0f0; dd = 0.0f0; vv = 0.0f0; c = tid
        while c < nc
            k = ldf(K, l * nc + c); s = ldf(S, l * nc + c)
            a = fma(k, k, a); b = fma(s, s, b); dd = fma(k * ldf(nk, l * nc + c), s * ldf(nq, l * nc + c), dd)
            l == Int32(0) && (v = ldf(V, c); vv = fma(v, v, vv))
            c += blockDim().x
        end
        a = bsum(a); b = bsum(b); dd = bsum(dd); vv = bsum(vv)
        if tid == Int32(0)
            stf(pt, l, a); stf(pt, Int32(4) + l, b); stf(pt, Int32(8) + l, dd)
            l == Int32(0) && stf(pt, Int32(12), vv)
        end
    end
    return
end
@inline function psum16(pt::FPtr, np::Int32, o::Int32)
    s = 0.0f0
    for p in Int32(0):np-Int32(1)
        s += ldf(pt, p * Int32(16) + o)
    end
    s
end
# PLE finish on my channels (thread per lane-major element): gate_l from the summed partials, G = gate V, x = G rG nc,
# dilated conv with the ring (10 x 4 nc), S += G + silu(y)
function k_ple_fin!(S::FPtr, V::FPtr, pt::FPtr, np::Int32, ncw::FPtr, cw::FPtr, ring::FPtr, t::Int32, nc::Int32, eps::Float32)
    i = gtid()
    @inbounds if i < Int32(4) * nc
        l = i ÷ nc; c = i % nc
        rK = 1.0f0 / sqrt(psum16(pt, np, l) / 2560.0f0 + eps); rS = 1.0f0 / sqrt(psum16(pt, np, Int32(4) + l) / 2560.0f0 + eps)
        s = rK * rS * psum16(pt, np, Int32(8) + l) * (1.0f0 / sqrt(2560.0f0))
        gt = sigm(sign(s) * sqrt(max(abs(s), 1.0f-6)))
        rG = 1.0f0 / sqrt(gt * gt * psum16(pt, np, Int32(12)) / 2560.0f0 + eps)
        G = gt * ldf(V, c)
        x = G * rG * ldf(ncw, i)
        n4 = Int32(4) * nc
        x9 = t >= Int32(9) ? ldf(ring, ((t - Int32(9)) & Int32(15)) * n4 + i) : 0.0f0
        x6 = t >= Int32(6) ? ldf(ring, ((t - Int32(6)) & Int32(15)) * n4 + i) : 0.0f0
        x3 = t >= Int32(3) ? ldf(ring, ((t - Int32(3)) & Int32(15)) * n4 + i) : 0.0f0
        y = ldf(cw, Int32(4) * i) * x9
        y = fma(ldf(cw, Int32(4) * i + Int32(1)), x6, y)
        y = fma(ldf(cw, Int32(4) * i + Int32(2)), x3, y)
        y = fma(ldf(cw, Int32(4) * i + Int32(3)), x, y)
        stf(S, i, ldf(S, i) + (G + silu(y)))
        stf(ring, (t & Int32(15)) * n4 + i, x)
    end
    return
end

# ---- Stage B0 runtime: launches per op, exchanges between the pair through the host ----
mutable struct PDie
    ord::Int; n::Int; d::Int
    blks::Vector{PBlk}
    head::Union{DHead,Nothing}
    st::Vector{LState}
    keep::Vector{Any}
    b::Dict{Symbol,FPtr}
    idx::IPtr
    rope::FPtr
    pler::FPtr
end
function PDie(ord, n, d, blks, head)
    device!(ord)
    nc = 2560 ÷ n; nkh = 16 ÷ n; nq = 24 ÷ n; nkv = 2 ÷ n; F = 640 ÷ n
    rt = PDie(ord, n, d, blks, head, LState[], Any[], Dict{Symbol,FPtr}(), NOI, NOF, NOF)
    mk(T, k) = (a = CUDA.zeros(T, k); push!(rt.keep, a); a)
    for (k, m) in (:S => 4nc, :xw => 4nc, :pt => 2 * 1300, :t => 324, :rl => 4, :up => 4nc, :z => 2560, :qkv => 640nkh, :gate => 384nkh,
                   :ab => 6nkh, :u => 640nkh, :on => 384nkh, :yp => 2 * 2560, :qg => 512nq, :k => 256nkv, :v => 256nkv, :ix => 640,
                   :qo => 256nq, :ao => 256nq, :iq => 512, :pm => nq * NCHMAX, :pd => nq * NCHMAX, :po => nq * NCHMAX * 256,
                   :rp => 2 * 513, :r => 513, :w => 11, :gu => 11 * 2F, :h => 11 * F, :pK => 4nc, :pV => nc, :pg => 2560, :ppt => 2 * 16,
                   :xh => 10240, :down => 324, :ht => 320, :hup => 10240, :hz => 2560)
        rt.b[k] = fptr(mk(Float32, m))
    end
    head === nothing || (rt.b[:logits] = fptr(mk(Float32, Int(head.out.rows))); rt.b[:Sfull] = fptr(mk(Float32, 10240));
        rt.b[:hzll] = reinterpret(FPtr, pointer(mk(UInt64, 2560))); rt.b[:am] = fptr(mk(Float32, 2P8MAX)))     # (a verify round's positions: value, index each)
    rt.idx = reinterpret(IPtr, pointer(mk(Int32, 16)))
    ra = CuArray(rope_table(MAXCTX[])); push!(rt.keep, ra); rt.rope = fptr(ra)
    mc = MAXCTX[]
    ag = attn_geom(n); kd = ag.nkvl * (KVCOPY ? 256 : ag.DH)     # B1: my dims of my KV heads a position (KVCOPY: all)
    for B in blks
        if B.attn
            kq = mk(UInt8, mc * kd); vq = mk(UInt8, mc * kd); ks = mk(UInt8, mc * kd ÷ 16); vs = mk(UInt8, mc * kd ÷ 16)
            pool = mk(UInt8, (mc ÷ 4) * 2 * ag.IW)
            # (the indexer's key ring: SRN slots of IW, by position; Stage A uses its first 4)
            push!(rt.st, LState(NOF, NOF, gptr(kq), gptr(ks), gptr(vq), gptr(vs), gptr(pool), fptr(mk(Float32, max(512, SRN * ag.IW))),
                fptr(mk(Float32, mc ÷ 4)), reinterpret(IPtr, pointer(mk(Int32, 512))), fptr(mk(Float32, mc * ag.nkvl))))
        else
            # (B1: the conv inputs in a ring of SRN positions, the recurrent state in SRN copies, both indexed by position)
            push!(rt.st, LState(fptr(mk(Float32, SRN * 640nkh)), fptr(mk(Float32, SRN * 3nkh * 16384)), GPtr(0), GPtr(0), GPtr(0), GPtr(0), GPtr(0), NOF, NOF, NOI))
        end
        B.l == 1 && (rt.pler = fptr(mk(Float32, 16 * 4nc)))          # the PLE conv ring: 16 positions
    end
    rt
end
sync!(rt::PDie) = (device!(rt.ord); synchronize())
# copy n floats from die a's buffer (offset oa) into die b's buffer (offset ob); both dies synchronised first
function xcopy!(a::PDie, ka::Symbol, oa, b::PDie, kb::Symbol, ob, n)
    CUC.cuMemcpy(reinterpret(CUC.CUdeviceptr, b.b[kb] + 4ob), reinterpret(CUC.CUdeviceptr, a.b[ka] + 4oa), 4n)
end
# exchange slot 0 (my part) into the peer's slot 1, both ways
dsync!(G::Vector{PDie}) = for rt in G
    device!(rt.ord); CUDA.device_synchronize()
end
function xchg!(G::Vector{PDie}, k::Symbol, n::Int)
    length(G) == 1 && return
    sync!(G[1]); sync!(G[2])
    xcopy!(G[1], k, 0, G[2], k, n, n); xcopy!(G[2], k, 0, G[1], k, n, n)
    dsync!(G)
end
# z halves: my half is at z[nc d], the peer's goes to the same offset on the other die
function xchg_z!(G::Vector{PDie})
    length(G) == 1 && return
    sync!(G[1]); sync!(G[2])
    xcopy!(G[1], :z, 0, G[2], :z, 0, 1280); xcopy!(G[2], :z, 1280, G[1], :z, 1280, 1280)
    dsync!(G)
end

function hc_read_b!(G::Vector{PDie}, i::Int, which::Symbol; router = false)
    np = Int32(length(G))
    for rt in G
        device!(rt.ord); B = rt.blks[i]; b = rt.b; nc = Int32(2560 ÷ rt.n)
        w, dj, dj2 = which == :attn ? (B.hcA_w, B.hcA_dj, B.hcA_dj2) : (B.hcF_w, B.hcF_dj, B.hcF_dj2)
        launch(k_hc_prep!, b[:xw], off(b[:pt], 1296), b[:S], w, nc; threads = 256, blocks = 4)
        gemv!(b[:pt], dj, b[:xw]; mode = 4, rpe = nc, ns = 0)
        dj2.rows > 0 && gemv!(b[:pt], dj2, off(b[:xw], 5120); mode = 4, rpe = nc, ns = 2)
    end
    xchg!(G, :pt, 1300)
    for rt in G
        device!(rt.ord); B = rt.blks[i]; b = rt.b; nc = Int32(2560 ÷ rt.n)
        w, up = which == :attn ? (B.hcA_w, B.hcA_up) : (B.hcF_w, B.hcF_up)
        launch(k_hc_comb!, b[:t], b[:rl], b[:pt], np, EPS; threads = 324, blocks = 1)
        gemv!(b[:up], up, b[:t])
        launch(k_hc_mix2!, b[:z], b[:up], b[:S], w, b[:rl], nc, Int32(nc * rt.d); threads = 256, blocks = cld(Int(nc), 256))
        router && gemv!(b[:rp], B.router, off(b[:z], nc * rt.d))
    end
    xchg_z!(G)
    router && xchg!(G, :rp, 513)
end
function out_write_b!(G::Vector{PDie}, i::Int)
    xchg!(G, :yp, 2560)
    np = Int32(length(G))
    for rt in G
        device!(rt.ord); b = rt.b; nc = Int32(2560 ÷ rt.n)
        launch(k_hc_write2!, b[:S], b[:yp], np, off(b[:t], 320), nc, Int32(nc * rt.d); threads = 256, blocks = cld(4 * Int(nc), 256))
    end
end

function block_fwd_b!(G::Vector{PDie}, i::Int, t::Int)
    error("stage B0 runs the pair layout (K-split router, expert down by columns): not the tensor-split layout v$LAYOUT_VERSION")
    t32 = Int32(t); np = Int32(length(G))
    B1 = G[1].blks[i]
    if B1.l == 1
        for rt in G
            device!(rt.ord); B = rt.blks[i]; b = rt.b; nc = Int32(2560 ÷ rt.n)
            gemv!(b[:pK], B.ple_key, b[:pg]); gemv!(b[:pV], B.ple_value, b[:pg])
            launch(k_ple_part!, b[:ppt], b[:pK], b[:pV], b[:S], B.ple_nk, B.ple_nq, nc; threads = 256, blocks = 4)
        end
        xchg!(G, :ppt, 16)
        for rt in G
            device!(rt.ord); B = rt.blks[i]; b = rt.b; nc = Int32(2560 ÷ rt.n)
            launch(k_ple_fin!, b[:S], b[:pV], b[:ppt], np, B.ple_nc, B.ple_conv, rt.pler, t32, nc, EPS; threads = 256, blocks = cld(4 * Int(nc), 256))
        end
    end
    hc_read_b!(G, i, :attn)
    for rt in G
        device!(rt.ord); B = rt.blks[i]; st = rt.st[i]; b = rt.b
        nkh = Int32(16 ÷ rt.n); nq = Int32(24 ÷ rt.n); nkv = Int32(2 ÷ rt.n)
        if !B.attn
            gemv!(b[:qkv], B.qkv, b[:z]); gemv!(b[:gate], B.gate, b[:z]); gemv!(b[:ab], B.ab, b[:z])
            launch(k_conv1d!, b[:u], b[:qkv], st.conv, B.conv_w, t32, Int32(640) * nkh; threads = 256, blocks = cld(640 * Int(nkh), 256))
            dlaunch(k_deltanet!, b[:on], st.dn, b[:u], b[:ab], B.ab_bias, B.ssm_a, b[:gate], B.ssm_norm, EPS, nkh; threads = 256, blocks = 3 * Int(nkh))
            gemv!(b[:yp], B.ssm_out, b[:on])
        else
            gemv!(b[:qg], B.wq, b[:z]); gemv!(b[:k], B.wk, b[:z]); gemv!(b[:v], B.wv, b[:z]); gemv!(b[:ix], B.idx_x, b[:z])
            cs = off(rt.rope, 64t); cs3 = off(rt.rope, 64 * max(t - 3, 0))
            launch(k_attn_prep!, b[:qo], st.kq, st.ks, st.vq, st.vs, b[:qg], b[:k], b[:v], B.q_norm, B.k_norm, cs, t32, EPS, nq, nkv;
                threads = 256, blocks = Int(nq + nkv))
            launch(k_idx_prep!, b[:iq], st.ring, st.pool, b[:ix], B.idx_qn, B.idx_kn, cs, cs3, t32, EPS; threads = 128, blocks = 5)
            nv = (t + 1) ÷ 4; tail = (t + 1) % 4
            if nv > 512
                launch(k_idx_score!, st.score, b[:iq], st.pool, Int32(nv); threads = 256, blocks = cld(nv, 256))
                launch(k_idx_select!, st.sel, st.score, Int32(nv), Int32(512); threads = 1024, blocks = 1)
                nsel = 512; nkeys = 4 * 512 + tail; tailb = nv
            else
                nsel = -1; nkeys = t + 1; tailb = 0
            end
            nch = cld(nkeys, CK)
            launch(k_attn_part!, b[:pm], b[:pd], b[:po], b[:qo], st.kq, st.ks, st.vq, st.vs, st.sel, Int32(nsel), Int32(nkeys), Int32(tailb), Int32(CK), nkv;
                threads = 256, blocks = (nch, Int(nq)))
            launch(k_attn_merge!, b[:ao], b[:pm], b[:pd], b[:po], b[:qg], Int32(nch); threads = 256, blocks = Int(nq))
            gemv!(b[:yp], B.wo, b[:ao])
        end
    end
    out_write_b!(G, i)
    hc_read_b!(G, i, :ffn; router = true)
    for rt in G
        device!(rt.ord); B = rt.blks[i]; b = rt.b; F = 640 ÷ rt.n
        launch(k_router2!, rt.idx, b[:w], b[:r], b[:rp], np; threads = 544, blocks = 1)
        launch(k_router!, rt.idx, b[:w], b[:r]; threads = 512, blocks = 1)
        gemv!(b[:gu], B.gu, b[:z]; rows = 20F, mode = 1, idx = rt.idx, rpe = 2F)
        gemv!(off(b[:gu], 20F), B.sh_gate, b[:z]); gemv!(off(b[:gu], 21F), B.sh_up, b[:z])
        launch(k_swiglu!, b[:h], b[:gu], Int32(F), Int32(11), Int32(10); threads = 256, blocks = cld(11F, 256))
        gemv!(b[:yp], B.down, b[:h]; rows = 2560, mode = 3, idx = rt.idx, sc = b[:w], rpe = 2560, ns = 10)
        gemv!(b[:yp], B.sh_down, off(b[:h], 10F); mode = 2, sc = off(b[:w], 10))
    end
    out_write_b!(G, i)
end

mutable struct EngineB
    g::GModel
    tk::Tok
    ple::PLE
    groups::Vector{Vector{PDie}}
    ids::Vector{Int32}
end
function load_engine_b(g::GModel, pl::PlacementB)
    m = read_manifest()
    holder = (m !== nothing && get(m, "key", "") == placement_key(pl) && alive(parse(Int, m["pid"]))) ? parse(Int, m["pid"]) : 0
    check_dies(vcat(pl.groups...), holder_pid(m))
    groups = Vector{PDie}[]
    for (i, gr) in enumerate(pl.groups)
        G = PDie[]
        for (k, ord) in enumerate(gr)
            device!(ord)
            NSM[] = CUDA.attribute(device(), CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
            A = Arena(); build_die!(A, g, pl, i, k); sz = A.off
            t0 = time()
            if holder != 0
                hs, hb = m["dies"][ord]
                hs == sz || error("holder arena on die $ord is $hs bytes, expected $sz")
                blks, head = build_die!(Arena(ipc_open(hb), sz; fill = false), g, pl, i, k)
            else
                blks, head = build_die!(Arena(rawalloc(sz), sz), g, pl, i, k)
            end
            note(@sprintf("die %d: group %d/%d blocks %s%s, %.2f GB %s in %.1f s", ord, k, length(gr), string(pl.layers[i]),
                head === nothing ? "" : " + head", sz / 1e9, holder != 0 ? "attached from holder $holder" : "loaded", time() - t0))
            push!(G, PDie(ord, length(gr), k - 1, blks, head))
        end
        push!(groups, G)
    end
    # every die may write into every other one (the pair exchanges, and each board's handoff to the next)
    ords = [rt.ord for G in groups for rt in G]
    for a in ords, b in ords
        a == b && continue
        device!(a)
        try
            CUDA.enable_peer_access(context(CuDevice(b)))
        catch e
            occursin("ALREADY", sprint(showerror, e)) || note("peer access $a -> $b: ", first(sprint(showerror, e), 200))
        end
    end
    EngineB(g, Tok(g), PLE(g), groups, Int32[])
end

# S of a group: full lane-major [4][2560] on the host from each die's slice, and back
function getS(G::Vector{PDie})
    S = Vector{Float32}(undef, 10240)
    for rt in G
        nc = 2560 ÷ rt.n; part = dl(rt.b[:S], 4nc)
        for l in 0:3
            S[l*2560+nc*rt.d+1:l*2560+nc*rt.d+nc] = part[l*nc+1:l*nc+nc]
        end
    end
    S
end
function putS!(G::Vector{PDie}, S::Vector{Float32})
    for rt in G
        device!(rt.ord); nc = 2560 ÷ rt.n
        ul!(rt.b[:S], vcat([S[l*2560+nc*rt.d+1:l*2560+nc*rt.d+nc] for l in 0:3]...))
    end
end
function step!(E::EngineB, id::Integer; logits = true)
    t = length(E.ids)
    push!(E.ids, Int32(id))
    S = repeat(embed(E.g, id), 4)
    for G in E.groups
        putS!(G, S)
        if any(B -> B.l == 1, G[1].blks)
            gi = ple_input(E.ple, E.ids, t)
            for rt in G
                device!(rt.ord); ul!(rt.b[:pg], gi)
            end
        end
        for i in eachindex(G[1].blks)
            block_fwd_b!(G, i, t)
        end
        S = getS(G)
    end
    logits && E.groups[end][end].head !== nothing && return head_logits(vcat(E.groups...), S)
    S
end
# logits over the dies holding head slices: the collapse where it lives, its hz to every die, each die its rows
function head_logits(dies::Vector{PDie}, S::Vector{Float32})
    hd = filter(rt -> rt.head !== nothing, dies)
    c = hd[findfirst(rt -> rt.head.down.rows > 0, hd)]
    device!(c.ord); H = c.head; b = c.b
    ul!(b[:Sfull], S)
    launch(k_rmsnorm!, b[:xh], b[:Sfull], H.norm, Int32(2560), Int32(2560), EPS; threads = 256, blocks = 4)
    gemv!(b[:down], H.down, b[:xh])
    launch(k_hc_silu!, b[:ht], b[:down], Int32(320); threads = 320, blocks = 1)
    gemv!(b[:hup], H.up, b[:ht])
    launch(k_hc_mix!, b[:hz], b[:hup], b[:xh]; threads = 256, blocks = 10)
    hz = dl(b[:hz], 2560)
    for rt in hd
        device!(rt.ord)
        rt === c || ul!(rt.b[:hz], hz)
        gemv!(rt.b[:logits], rt.head.out, rt.b[:hz])
    end
    out = Vector{Float32}(undef, 248320)
    for rt in hd
        device!(rt.ord); n = Int(rt.head.out.rows)
        out[rt.head.r0+1:rt.head.r0+n] = dl(rt.b[:logits], n)
    end
    out
end

function run_check_b(E::EngineB, ids::Vector{Int32})
    g = E.g
    R = Dict(B.l => RefState() for G in E.groups for B in G[1].blks)
    worst = 0.0
    for (t0, id) in enumerate(ids)
        t = t0 - 1
        push!(E.ids, id)
        S = repeat(embed(g, id), 4)
        for G in E.groups
            putS!(G, S)
            if any(B -> B.l == 1, G[1].blks)
                gi = ple_input(E.ple, E.ids, t)
                for rt in G
                    device!(rt.ord); ul!(rt.b[:pg], gi)
                end
            end
            for i in eachindex(G[1].blks)
                l = G[1].blks[i].l
                Sin = Float64.(getS(G))
                block_fwd_b!(G, i, t)
                Sg = getS(G)
                Sr = ref_block!(g, l, copy(Sin), R[l], t, E.ids, E.ple)
                e = relerr(Sg .- Sin, Sr .- Sin)
                worst = max(worst, e)
                @printf("check t=%d blk=%2d %s dies %s  |dS| rel err %.3e\n", t, l, G[1].blks[i].attn ? "attn" : "gdn ", join([rt.ord for rt in G], "+"), e)
                metric("check.block_relerr", "ratio", e; position = t, block = l, stage = "B")
            end
            S = getS(G)
        end
    end
    ok = worst < 2e-3
    @printf("check: worst block-update relative error %.3e: %s\n", worst, ok ? "PASS" : "FAIL")
    metric("check.pass", "bool", ok; worst = worst, stage = "B")
    ok
end


# ---------------------------------------------------------------------------------------------
# Stage B1: one persistent kernel per die per token. Every intermediate vector is an LL word ({seq, value} in
# 8 bytes) written into the memory of each die that reads it (the pair peer, the next board); readers poll the
# words they use, so ops need no grid barriers and no fences. seq = seq0 + 32 (l + 1) + slot for block l's ops.
# ---------------------------------------------------------------------------------------------
@inline function llget(p::U64P, i::Int32, sq::UInt32)
    a = p + 8i
    v = UA.load(a, UA.monotonic, UA.system)
    # (LLMARK: the host stopped this position and overwrote the die's LL words, see abort_positions!)
    dl = UInt32(POLLMIN)
    while (v >> 32) % UInt32 != sq && (v >> 32) % UInt32 != LLMARK
        POLLEXP ? (spin(dl); dl = min(dl + dl, UInt32(POLLDELAY))) : (POLLDELAY > 0 && spin(POLLDELAY))
        v = UA.load(a, UA.monotonic, UA.system)
    end
    reinterpret(Float32, v % UInt32)
end
# between two polls of a word not yet written: n SM cycles (FLASHNEXT_POLLDELAY; 0: poll back to back), so thousands
# of waiting threads do not load the lines their writers are storing to (8 dies: 400 cycles 41.56-41.62 tok/s against
# 40.62-40.73 without, 1600: 40.78-40.82). FLASHNEXT_POLLEXP=1: the delay doubles from POLLMIN (32) up to POLLDELAY.
const POLLDELAY = parse(Int, get(ENV, "FLASHNEXT_POLLDELAY", "400"))
const POLLEXP = get(ENV, "FLASHNEXT_POLLEXP", "0") == "1"
const POLLMIN = 32
@inline function spin(n::Integer)
    t0 = CUDA.clock(UInt32)
    while CUDA.clock(UInt32) - t0 < n % UInt32
    end
end
# Early stop of a speculative round: the host writes an abort window [lo, hi) of seq numbers into each die's ABW
# (device memory, by a copy on a stream of its own while the kernels run). k_token! keeps [ABMAGIC, its ABW pointer]
# in the first 16 bytes of its dynamic shared memory, and a CTA leaves at a block's start when its seq0 is in the
# window. A CTA inside a block may wait for words a CTA that left will not write: the same copy engine then fills the
# die's LL words with seq LLMARK, which ends every wait (each die's waits are on its own memory: producers write into
# the reader's), and the words are zeroed behind the stopped kernels. (A test of the window in the wait itself cost
# plain decode 29.6 -> 28.9 tok/s in code size; the per-block test costs nothing measurable.)
const ABMAGIC = 0xab07ab07ab07ab07
const LLMARK = 0xffffffff
CUDA.CUDACore.@device_function @inline dsmem_size() = Base.llvmcall("""%s = call i32 asm sideeffect "mov.u32 \$0, %dynamic_smem_size;", "=r"()
    ret i32 %s""", UInt32, Tuple{})
@noinline function aborted(sq::UInt32)
    dsmem_size() >= UInt32(16) || return false
    h = @inbounds CuDynamicSharedArray(UInt64, 2)     # (its own size check is a throw path at every inlined wait)
    @inbounds h[1] == ABMAGIC || return false
    p = reinterpret(Core.LLVMPtr{UInt32,1}, @inbounds h[2])
    p == Core.LLVMPtr{UInt32,1}(0) && return false
    lo = UA.load(p, UA.monotonic, UA.device); hi = UA.load(p + 4, UA.monotonic, UA.device)
    lo <= sq && sq < hi
end
@inline llput(p::U64P, i::Int32, sq::UInt32, v::Float32) =UA.store!(p + 8i, (UInt64(sq) << 32) | reinterpret(UInt32, v), UA.monotonic, UA.system)
@inline sqof(seq0::UInt32, l, slot) = seq0 + UInt32(32 * (l + 1) + slot)

struct MatI
    t::Int32; L::Int32; rows::Int32; K::Int32; pay::GPtr; hdr::GPtr
end
MatI(m::DevMat) = m.rows == 0 ? MatI(Int32(0), Int32(1), Int32(0), Int32(0), GPtr(0), GPtr(0)) :
    MatI(Int32(m.t), m.L, m.rows, m.K, m.pay, m.hdr)

# x source of a GEMV phase. kind 1: LL p1[0:n1) then p2[0:n2); 2: LL S times plain w (and the four lane sums of S^2);
# 3: hc t[k] = silu(sum_l r_l sum_p P_p[l][k] / 4) from the down partials (n1 slots of 1300 at p1, seq s1);
# 4: swiglu h[k] = silu(g) u from GU (LL, F per slot: g at 2F e + j, u at 2F e + F + j) starting at slot n2;
# 5: LL p1[0:n1) (seq s1), then silu(p2[j]) p2[F + j] for j = k - n1 (seq s2)
# 6: S after an hc write, S_old p1[i] (seq s1) + g_l (sum of n1 partials p2[p nc + c], seq s2), times w (as 2; i = k + n2,
# g_l in rl); the CTA owning element i (a 1/grid share) stores it to dst (seq sd)
struct XF
    kind::Int32
    p1::U64P; n1::Int32; s1::UInt32
    p2::U64P; n2::Int32; s2::UInt32
    w::FPtr; F::Int32; nc::Int32
    dst::U64P; sd::UInt32
end
XF(kind, p1, n1, s1; p2 = U64P(0), n2 = 0, s2 = UInt32(0), w = NOF, F = 0, nc = 0, dst = U64P(0), sd = UInt32(0)) =
    XF(Int32(kind), p1, Int32(n1), s1, p2, Int32(n2), s2, w, Int32(F), Int32(nc), dst, sd)
# output of a GEMV phase. kind 1: LL d1[o1 + r] (and d2[o1 + r]); 2: row partial to the channel owner: rows
# [lo, lo + nc) -> d1[o1 + r - lo], others -> d2[o1 + r - plo] (n == 2); with addp: + sc * LL ad[r]; 3: four segments
# s -> d1/d2 [o1 + 324 s + r]; 4: hc mix: z[zoff + c'] (rows 4 c' + l) to d1 and d2
struct EM
    kind::Int32
    d1::U64P; d2::U64P; sq::UInt32; o1::Int32
    lo::Int32; plo::Int32; nc::Int32
    S::U64P; sS::UInt32; w::FPtr; zoff::Int32
    ad::U64P; asq::UInt32
    # B1 over a group of bc > 0 dies (the peer table pt: the dies' LL region bases; d1 at byte doff of its region):
    # kinds 1, 4, 5 write every die's d1, kind 2 the channel owner's (r ÷ nc), kind 3 the row owner's (r ÷ plo)
    bc::Int32; pt::Core.LLVMPtr{UInt64,1}; doff::UInt32
end
EM(kind, d1, sq; d2 = U64P(0), o1 = 0, lo = 0, plo = 0, nc = 0, S = U64P(0), sS = UInt32(0), w = NOF, zoff = 0, ad = U64P(0), asq = UInt32(0),
   bc = 0, pt = Core.LLVMPtr{UInt64,1}(0), doff = UInt32(0)) =
    EM(Int32(kind), d1, d2, sq, Int32(o1), Int32(lo), Int32(plo), Int32(nc), S, sS, w, Int32(zoff), ad, asq, Int32(bc), pt, doff % UInt32)
# d1 on die e of the group
@inline emdst(em::EM, e::Int32) = reinterpret(U64P, CUDA.unsafe_cached_load(em.pt, e + Int32(1), Val(8)) + UInt64(em.doff))
# word i of d1: on every die of the group (bc > 0), else d1 (and d2)
@inline function emput!(em::EM, i::Int32, v::Float32)
    if em.bc > Int32(0)
        for e in Int32(0):em.bc-Int32(1)
            llput(emdst(em, e), i, em.sq, v)
        end
    else
        llput(em.d1, i, em.sq, v)
        em.d2 != U64P(0) && llput(em.d2, i, em.sq, v)
    end
end
# hc down partial of segment sg, row r (EM kind 3): to its owner, [segment][plo rows] at o1 (bc > 0), else [segment][324]
@inline function put_seg!(em::EM, sg::Int32, r::Int32, v::Float32)
    if em.bc > Int32(0)
        ow = r ÷ em.plo
        llput(emdst(em, ow), em.o1 + em.plo * (sg + em.lo) + (r - ow * em.plo), em.sq, v)
    else
        llput(em.d1, em.o1 + Int32(324) * (sg + em.lo) + r, em.sq, v)
        em.d2 != U64P(0) && llput(em.d2, em.o1 + Int32(324) * (sg + em.lo) + r, em.sq, v)
    end
end

@inline function xval(xf::XF, k::Int32, rl::NTuple{4,Float32})
    if xf.kind == Int32(1)
        return k < xf.n1 ? llget(xf.p1, k, xf.s1) : llget(xf.p2, k - xf.n1, xf.s2)
    elseif xf.kind == Int32(2)
        return llget(xf.p1, k + xf.n2, xf.s1) * ldf(xf.w, k + xf.n2)       # n2: offset into S and w
    elseif xf.kind == Int32(3)
        s = 0.0f0
        for l in Int32(0):Int32(3)
            ps = 0.0f0
            for p in Int32(0):xf.n1-Int32(1)
                ps += llget(xf.p1, p * Int32(1300) + Int32(324) * l + k, xf.s1)
            end
            s = fma(rl[l+1], ps, s)
        end
        return silu(s * 0.25f0)
    else
        e = k ÷ xf.F + xf.n2; j = k % xf.F
        return silu(llget(xf.p1, Int32(2) * xf.F * e + j, xf.s1)) * llget(xf.p1, Int32(2) * xf.F * e + xf.F + j, xf.s1)
    end
end
# LL reads in batches: issue every load, then check each word's seq and poll only the ones not yet written
@inline llraw(p::U64P, i::Int32) = LLRAWPLAIN ? unsafe_load(p + 8i, 1, Val(8)) : UA.load(p + 8i, UA.monotonic, UA.system)
const LLRAWPLAIN = get(ENV, "FLASHNEXT_LLRAW_PLAIN", "0") == "1"     # the first look at an LL word as a plain load
@inline llfix(p::U64P, i::Int32, sq::UInt32, w::UInt64) = (w >> 32) % UInt32 == sq ? reinterpret(Float32, w % UInt32) : llget(p, i, sq)
# the source word of x[k] for kinds 1, 2 (pointer, index, seq)
@inline xsrc(xf::XF, k::Int32) = xf.kind == Int32(1) ? (k < xf.n1 ? (xf.p1, k, xf.s1) : (xf.p2, k - xf.n1, xf.s2)) : (xf.p1, k + xf.n2, xf.s1)
# x[kk] into shared memory, and the 16-sum of its 16-group from its half-warp: the lanes of a half-warp hold 16
# consecutive elements, and the callers' loops are uniform per warp (Kx a multiple of 32), so every lane shuffles
@inline function xput!(xs::SPtr, x16::SPtr, kk::Int32, v::Float32, ok::Bool)
    ok && unsafe_store!(xs + 4 * xu(kk), v, 1, Val(4))
    s = ok ? v : 0.0f0
    s += shfl_xor_sync(FULL, s, 8); s += shfl_xor_sync(FULL, s, 4); s += shfl_xor_sync(FULL, s, 2); s += shfl_xor_sync(FULL, s, 1)
    (ok && (kk & Int32(15)) == Int32(0)) && unsafe_store!(x16 + 4 * (kk >> 4), s, 1, Val(4))
    nothing
end
# x[0:Kx) and its 16-sums into shared memory, XLU elements per thread per round with their loads in flight together
# (FLASHNEXT_XLU, default 4: 8 measured no faster in the phase bench, alone on the die)
const XLU = parse(Int, get(ENV, "FLASHNEXT_XLU", "4"))
macro xlu(ex)
    esc(Expr(:macrocall, GlobalRef(Base.Cartesian, Symbol("@nexprs")), __source__, XLU, ex))
end
@inline function xload!(xs::SPtr, x16::SPtr, xf::XF, Kx::Int32, rl::NTuple{4,Float32})
    tid = threadIdx().x - Int32(1); nt = blockDim().x
    k = tid
    ls1 = 0.0f0; ls2 = 0.0f0; ls3 = 0.0f0; ls4 = 0.0f0        # kind 2: this thread's sums of S^2 per lane (nc channels each)
    if xf.kind == Int32(1) || xf.kind == Int32(2)
        while k < Kx
            @xlu q -> begin
                kk_q = k + Int32(q - 1) * nt
                ok_q = kk_q < Kx
                p_q, i_q, s_q = ok_q ? xsrc(xf, kk_q) : (xf.p1, Int32(0), UInt32(0))
                w_q = ok_q ? llraw(p_q, i_q) : UInt64(0)
            end
            @xlu q -> begin
                v_q = ok_q ? llfix(p_q, i_q, s_q, w_q) : 0.0f0
                if ok_q && xf.kind == Int32(2)
                    ln_q = Int32(i_q >= xf.nc) + Int32(i_q >= Int32(2) * xf.nc) + Int32(i_q >= Int32(3) * xf.nc); v2_q = v_q * v_q
                    ls1 += ln_q == Int32(0) ? v2_q : 0.0f0; ls2 += ln_q == Int32(1) ? v2_q : 0.0f0
                    ls3 += ln_q == Int32(2) ? v2_q : 0.0f0; ls4 += ln_q == Int32(3) ? v2_q : 0.0f0
                    v_q *= ldf(xf.w, i_q)
                end
                xput!(xs, x16, kk_q, v_q, ok_q)
            end
            k += Int32(XLU) * nt
        end
    elseif xf.kind == Int32(6)
        own = cld(Int32(4) * xf.nc, gridDim().x); b = blockIdx().x - Int32(1)
        while k < Kx
            @xlu q -> begin
                kk_q = k + Int32(q - 1) * nt
                ok_q = kk_q < Kx
                i_q = kk_q + xf.n2; c_q = i_q % xf.nc
                ws_q = ok_q ? llraw(xf.p1, i_q) : UInt64(0)
                w0_q = ok_q ? llraw(xf.p2, c_q) : UInt64(0)
                w1_q = ok_q && xf.n1 > Int32(1) ? llraw(xf.p2, xf.nc + c_q) : UInt64(0)
            end
            @xlu q -> begin
                v_q = 0.0f0
                if ok_q
                    ln_q = i_q ÷ xf.nc
                    y_q = llfix(xf.p2, c_q, xf.s2, w0_q)
                    xf.n1 > Int32(1) && (y_q += llfix(xf.p2, xf.nc + c_q, xf.s2, w1_q))
                    v_q = fma(rl4(rl, ln_q), y_q, llfix(xf.p1, i_q, xf.s1, ws_q))
                    i_q ÷ own == b && llput(xf.dst, i_q, xf.sd, v_q)
                    v2_q = v_q * v_q
                    ls1 += ln_q == Int32(0) ? v2_q : 0.0f0; ls2 += ln_q == Int32(1) ? v2_q : 0.0f0
                    ls3 += ln_q == Int32(2) ? v2_q : 0.0f0; ls4 += ln_q == Int32(3) ? v2_q : 0.0f0
                    v_q *= ldf(xf.w, i_q)
                end
                xput!(xs, x16, kk_q, v_q, ok_q)
            end
            k += Int32(XLU) * nt
        end
    elseif xf.kind == Int32(9)
        # hc t[k] = silu(sum_p P_p[k] / 4) from the n1 dies' lane-summed hc down partials ([n1][324] at p1, seq s1;
        # W2 when F == 1)
        while k < Kx
            xput!(xs, x16, k, silu(0.25f0 * dsum_el(xf.p1, xf.n1, k, xf.s1, xf.F == Int32(1))), true)
            k += nt
        end
    elseif xf.kind == Int32(8)
        # a W2 vector (2 values a word, see W2): x[k] in word k ÷ 2 of p1, seq s1
        while k < Kx
            @xlu q -> begin
                kk_q = k + Int32(q - 1) * nt
                ok_q = kk_q < Kx
                w_q = ok_q ? llraw(xf.p1, kk_q >> 1) : UInt64(0)
            end
            @xlu q -> xput!(xs, x16, kk_q, ok_q ? w2fix(xf.p1, kk_q, xf.s1, w_q) : 0.0f0, ok_q)
            k += Int32(XLU) * nt
        end
    elseif xf.kind == Int32(5)
        while k < xf.n1
            Base.Cartesian.@nexprs 4 q -> begin
                kk_q = k + Int32(q - 1) * nt
                ok_q = kk_q < xf.n1
                w_q = ok_q ? llraw(xf.p1, kk_q) : UInt64(0)
            end
            Base.Cartesian.@nexprs 4 q -> xput!(xs, x16, kk_q, ok_q ? llfix(xf.p1, kk_q, xf.s1, w_q) : 0.0f0, ok_q)
            k += Int32(4) * nt
        end
        k = xf.n1 + tid
        while k < Kx
            j = k - xf.n1
            wg = llraw(xf.p2, j); wu = llraw(xf.p2, xf.F + j)
            xput!(xs, x16, k, silu(llfix(xf.p2, j, xf.s2, wg)) * llfix(xf.p2, xf.F + j, xf.s2, wu), true)
            k += nt
        end
    elseif xf.kind == Int32(4)
        while k < Kx
            Base.Cartesian.@nexprs 2 q -> begin
                kk_q = k + Int32(q - 1) * nt
                ok_q = kk_q < Kx
                e_q = kk_q ÷ xf.F + xf.n2; j_q = kk_q % xf.F
                ig_q = Int32(2) * xf.F * e_q + j_q; iu_q = ig_q + xf.F
                wg_q = ok_q ? llraw(xf.p1, ig_q) : UInt64(0); wu_q = ok_q ? llraw(xf.p1, iu_q) : UInt64(0)
            end
            Base.Cartesian.@nexprs 2 q -> begin
                v_q = ok_q ? silu(llfix(xf.p1, ig_q, xf.s1, wg_q)) * llfix(xf.p1, iu_q, xf.s1, wu_q) : 0.0f0
                xput!(xs, x16, kk_q, v_q, ok_q)
            end
            k += Int32(2) * nt
        end
    else
        while k < Kx
            xput!(xs, x16, k, xval3(xf, k, rl), true); k += nt
        end
    end
    (ls1, ls2, ls3, ls4)
end
# hc t[k] with its 4 lanes x n1 slots of partials loaded together
@inline function xval3(xf::XF, k::Int32, rl::NTuple{4,Float32})
    w0 = ntuple(l -> llraw(xf.p1, Int32(324) * Int32(l - 1) + k), Val(4))
    w1 = ntuple(l -> xf.n1 > Int32(1) ? llraw(xf.p1, Int32(1300) + Int32(324) * Int32(l - 1) + k) : UInt64(0), Val(4))
    s = 0.0f0
    Base.Cartesian.@nexprs 4 l -> begin
        i0_l = Int32(324) * Int32(l - 1) + k
        ps_l = llfix(xf.p1, i0_l, xf.s1, w0[l])
        xf.n1 > Int32(1) && (ps_l += llfix(xf.p1, Int32(1300) + i0_l, xf.s1, w1[l]))
        s = fma(rl[l], ps_l, s)
    end
    silu(s * 0.25f0)
end
# r_l = 1/sqrt(sum_p ss_p[l] / 2560 + eps) from the partials (xf kind 3)
# r_l = 1/sqrt(sum S_l^2 / 2560 + eps) from the lane sums LS (this parity's [die][warp][lane], 32 words per die):
# lanes 0..31 of the warp load die 0's 32 words (and die 1's), sum over warps by shuffles
@inline function hc_rl(p::U64P, np::Int32, sq::UInt32, eps::Float32)
    lane = (threadIdx().x - Int32(1)) % Int32(32)
    w0 = llraw(p, lane); w1 = np > Int32(1) ? llraw(p, Int32(32) + lane) : UInt64(0)
    v = llfix(p, lane, sq, w0)
    np > Int32(1) && (v += llfix(p, Int32(32) + lane, sq, w1))
    # lane 4 w + l holds warp w's lane-l sum: add over w (lanes with the same l = lane % 4)
    v += shfl_xor_sync(FULL, v, 4); v += shfl_xor_sync(FULL, v, 8); v += shfl_xor_sync(FULL, v, 16)
    Base.Cartesian.@ntuple 4 l -> (1.0f0 / sqrt(shfl_sync(FULL, v, Int32(l)) / 2560.0f0 + eps))   # CUDA.jl: 1-based source lane
end

# ---- B1 GEMV phases. A phase is the matrices of one op that share its x: every CTA loads x into shared memory once,
# then the row groups of all the matrices, in order, are cut into one contiguous range of equal weight bytes per warp.
const SIP = Core.LLVMPtr{Int32,3}
const NOSI = SIP(0)
const NOSF = SPtr(0)
const NOSTP = (Core.LLVMPtr{UInt64,1}(0), Int32(-1))
# the MoE down over (row group, expert) tasks: a task stores P[s stride + r] = w_s (W_s x_s)[r] for its rows and counts
# itself on its 32-row block; the warp that brings a block's count to target sums the block's ns slots in order
# (deterministic) and writes the rows
struct Red
    P::FPtr; cnt::IPtr; target::Int32; ns::Int32; stride::Int32
    ord::Core.LLVMPtr{Int32,3}         # (B8) each position's 10 experts' union slots in rank order, ord[10 p + k]; or null
end
Red(P, cnt, target, ns, stride) = Red(P, cnt, target, ns, stride, Core.LLVMPtr{Int32,3}(0))
const NORED = Red(FPtr(0), IPtr(0), Int32(0), Int32(0), Int32(0))
# one matrix of a phase. MODE 0: rows; 1: gathered expert rows (rpe rows per expert, from idx); 3: MoE down tasks
# (group, expert e < ns) with x slot e at xo + e K, weight wsc[eo + e] and output slot eo + e (rpe = 0: no gather);
# 4: hc down, four segments of em.nc elements. xo = this matrix's first x element.
struct Job{MODE}
    m::MatI; rows::Int32; em::EM; xo::Int32; eo::Int32; ns::Int32; rpe::Int32
end
Job{M}(m::MatI, rows, em::EM; xo = 0, eo = 0, ns = 1, rpe = 0) where {M} = Job{M}(m, rows % Int32, em, xo % Int32, eo % Int32, ns % Int32, rpe % Int32)
# (groups, weight words per group)
@inline function jobshape(j::Job{M}) where {M}
    m = j.m
    m.rows == Int32(0) && return Int32(0), Int32(1)
    R = Int32(32) ÷ m.L
    G = (j.rows + R - Int32(1)) ÷ R
    M == 3 && (G *= j.ns)
    c = R * (m.K ÷ (devE(m.t) % Int32)) * ((devP(m.t) + devH(m.t)) % Int32) ÷ Int32(4)
    if M == 4 && HCSPLIT && m.L == Int32(32)          # hc down: one task per (row, segment)
        ns4 = m.K ÷ j.em.nc
        return G * ns4, c ÷ ns4
    end
    G, c
end
const HCSPLIT = get(ENV, "FLASHNEXT_HCSPLIT", "1") == "1"
# one phase: x[0:Kx) into shared memory and its 16-sums, then this warp's share of the jobs' row groups
@inline jobs_groups(::Tuple{}) = Int32(0)
@inline jobs_groups(js::Tuple) = jobshape(first(js))[1] + jobs_groups(Base.tail(js))
const PFENTRY = get(ENV, "FLASHNEXT_PFENTRY", "0") == "1"     # slower in situ (25.9 -> 25.0 tok/s)
# L2 prefetch of row group g's first block (plain rows; one prefetch per 128-byte line of each chunk)
@inline pf_one!(::Tuple{}, g::Int32, lane::Int32) = nothing
@inline function pf_one!(js::Tuple, g::Int32, lane::Int32)
    j = first(js); G = jobshape(j)[1]
    g < G ? pfgrp!(j, g, lane) : pf_one!(Base.tail(js), g - G, lane)
end
@inline function pfgrp!(j::Job{M}, g::Int32, lane::Int32) where {M}
    (M == 0 || M == 4) || return           # plain rows and hc down share the row-group layout
    m = j.m
    E = devE(m.t) % Int32; NC = (devP(m.t) ÷ 16) % Int32
    gi = g * ((m.K ÷ E) ÷ m.L)
    if lane & Int32(7) == Int32(0)
        c = Int32(0)
        while c < NC
            pfl2(m.pay + ((gi * NC + c) * Int32(32) + lane) * Int32(16))
            c += Int32(1)
        end
    end
end
CUDA.CUDACore.@device_function @inline pfl2(p::GPtr) = Base.llvmcall("""call void asm sideeffect "prefetch.global.L2 [\$0];", "l"(i64 %0)
    ret void""", Cvoid, Tuple{UInt64}, reinterpret(UInt64, p))
# FLASHNEXT_PFALL=1: while x is awaited, the phase's plain-row matrices (payload, then headers; jobs in order, the first
# PFMAX bytes: the groups the counter hands out first) to L2, each CTA a 1/grid share of the 128-byte lines. A die's
# 8-way share of a phase is 0.2-2.5 MB, which streamed at 31-54 GB/s after x (the same code reads 84-113 GB/s on whole
# matrices): the phases are short, and the x waits (17-42 us) leave the DRAM idle
const PFALL = get(ENV, "FLASHNEXT_PFALL", "0") == "1"
const PFMAX = parse(Int, get(ENV, "FLASHNEXT_PFMAX", "1536")) * 1024
@inline pfall!(::Tuple{}, left::Int32) = nothing
@inline function pfall!(js::Tuple, left::Int32)
    j = first(js)
    left = pfmat!(j, left)
    left > Int32(0) && pfall!(Base.tail(js), left)
end
@inline function pfmat!(j::Job{M}, left::Int32) where {M}
    (M == 0 || M == 4) || return left
    m = j.m; m.rows == Int32(0) && return left
    E = devE(m.t) % Int32; NC = (devP(m.t) ÷ 16) % Int32; H = devH(m.t) % Int32
    G = (m.rows + Int32(32) ÷ m.L - Int32(1)) ÷ (Int32(32) ÷ m.L); NI = (m.K ÷ E) ÷ m.L
    for (p, nb) in ((m.pay, G * NI * NC * Int32(512)), (m.hdr, G * NI * Int32(32) * H))
        nb = min(nb, left); left -= nb
        nl = (nb + Int32(127)) >> 7; sh = (nl + gridDim().x - Int32(1)) ÷ gridDim().x
        l = (blockIdx().x - Int32(1)) * sh + threadIdx().x - Int32(1); le = min(nl, (blockIdx().x) * sh)
        while l < le
            pfl2(p + Int32(128) * l)
            l += blockDim().x
        end
    end
    left
end
# the warp's groups from index v on (fetched from ctr): job by job in order (indices only grow); the first index
# past the last job is returned
@inline run_dyn!(::Tuple{}, s::Int32, v::Int32, ctr, idx, wsc, red, xs, x16, lut, rl) = v
@inline function run_dyn!(js::Tuple, s::Int32, v::Int32, ctr, idx, wsc, red, xs, x16, lut, rl)
    j = first(js); G = jobshape(j)[1]
    v < s + G && (v = jobdyn!(j, v, s, G, ctr, idx, wsc, red, xs, x16, lut, rl))
    run_dyn!(Base.tail(js), s + G, v, ctr, idx, wsc, red, xs, x16, lut, rl)
end
@inline function grpdyn!(::Val{T}, ::Val{L}, ::Val{MODE}, m::MatI, v::Int32, s::Int32, G::Int32, ctr::IPtr, rows::Int32, em::EM, xo::Int32, eo::Int32,
                         ns::Int32, rpe::Int32, idx::SIP, wsc::SPtr, red::Red, xs::SPtr, x16::SPtr, lut::SPtr, rl::NTuple{4,Float32}) where {T,L,MODE}
    lane = (threadIdx().x - Int32(1)) % Int32(32)
    while v < s + G
        nx = shfl_sync(FULL, lane == Int32(0) ? ctrnext(ctr) : Int32(0), Int32(1))
        grp!(Val(T), Val(L), Val(MODE), m, v - s, rows, em, xo, eo, ns, rpe, idx, wsc, red, xs, x16, lut, rl)
        v = nx
    end
    v
end
# the next row-group index of a phase: a relaxed atomic add (the counter only hands out work; CUDA.atomic_add! is
# sequentially consistent, a membar.gl on each side, and each fence waited for all the warp's loads in flight)
@inline ctrnext(ctr::IPtr) = UA.add!(ctr, Int32(1), UA.monotonic, UA.device)
# a device trap (a matrix layout or a path the module was not built for: loud, not a silent skip)
@inline devtrap() = ccall("llvm.trap", llvmcall, Cvoid, ())
# ctr: the phase's group counter (zero between uses). Warps take the jobs' row groups in order from it, each fetching its
# next index before computing the current one, so CTAs and formats that run slower take fewer groups; the fetch loop
# runs inside each job's format dispatch (one dispatch per job a warp reaches, not per group: per group, 26.3 -> 27.5
# tok/s). The warp whose fetch is the last one (index groups + warps - 1) zeroes the counter again.
@inline function cphase!(xf::XF, Kx::Int32, xs::SPtr, rl::NTuple{4,Float32}, idx::SIP, wsc::SPtr, red::Red, js::Tuple, ctr::IPtr,
                         stp::Tuple{Core.LLVMPtr{UInt64,1},Int32} = NOSTP)
    tid = threadIdx().x - Int32(1); nt = blockDim().x; nw = nt ÷ Int32(32); warp = tid ÷ Int32(32); lane = tid % Int32(32)
    x16 = xs + 4 * xu(Kx)
    luta = CuStaticSharedArray(Float32, 16)
    @inbounds begin
        ctr == IPtr(0) && devtrap()
        tid < Int32(16) && (luta[tid+1] = kv4val(tid))
        W = gridDim().x * nw; w = (blockIdx().x - Int32(1)) * nw + warp
        # the first block of row group w (about the group this warp takes first) to L2 while x is awaited
        PFENTRY && w < jobs_groups(js) && pf_one!(js, w, lane)
        PFALL && pfall!(js, Int32(PFMAX))
        ls = xload!(xs, x16, xf, Kx, rl)                  # per-lane sums of S^2 for x kind 2 (else zeros), returned
        sync_threads()
        stp[2] >= Int32(0) && stamp!(stp[1], stp[2])      # x ready (--prof)
        G = jobs_groups(js)
        v = shfl_sync(FULL, lane == Int32(0) ? ctrnext(ctr) : Int32(0), Int32(1))   # CUDA.jl: 1-based source lane
        v = run_dyn!(js, Int32(0), v, ctr, idx, wsc, red, xs, x16, pointer(luta), rl)
        if lane == Int32(0) && v == G + W - Int32(1)
            UA.store!(ctr, Int32(0), UA.monotonic, UA.device); threadfence()
        end
        sync_threads()
        return ls
    end
end

# sum over the blocks of one row of the lane's share: blocks b = j + L i, x from element x0 (block-size aligned)
# block i of the lane's share of row group pg (blocks b = j + L i, x from element x0)
@inline function blkdot(::Val{T}, ::Val{L}, m::MatI, pg::Int32, NI::Int32, i::Int32, lane::Int32, j::Int32, xs::SPtr, x16::SPtr, x0::Int32, lut::SPtr) where {T,L}
    E = Int32(devE(T)); H = Int32(devH(T)); NC = Int32(devP(T) ÷ 16)
    b = j + Int32(L) * i; gi = pg * NI + i; k0 = x0 + b * E
    bdot(Val(T), m.pay + (gi * NC * Int32(32) + lane) * Int32(16), m.hdr + (gi * Int32(32) + lane) * H, xs, x16, xu(k0), k0 >> 4, lut)
end
@inline issplit(T) = T == T_Q2_0 || T == T_Q4_0 || T == T_IQ4_NL || T == T_Q5_0 || T == T_Q8_0 || T == T_BF16
# FLASHNEXT_PFNEXT=1: the formats of more than 4 chunks a block (no bload / bcomp split: their loads wait in each
# block's dot) have the warp's next block (its NC chunks of 512 bytes, then its 32 H header bytes) prefetched to L2,
# one 128-byte line a lane, before the current block is computed
const PFNEXT = get(ENV, "FLASHNEXT_PFNEXT", "0") == "1"
@inline function pfblk!(::Val{T}, m::MatI, gi::Int32, lane::Int32) where {T}
    NC = Int32(devP(T) ÷ 16); H = Int32(devH(T))
    np = Int32(4) * NC; nh = (Int32(32) * H + Int32(127)) >> 7
    l = lane
    while l < np + nh
        pfl2(l < np ? m.pay + gi * NC * Int32(512) + Int32(128) * l : m.hdr + gi * Int32(32) * H + Int32(128) * (l - np))
        l += Int32(32)
    end
end
@inline function rowdot(::Val{T}, ::Val{L}, m::MatI, pg::Int32, lane::Int32, j::Int32, xs::SPtr, x16::SPtr, x0::Int32, lut::SPtr) where {T,L}
    NI = (m.K ÷ Int32(devE(T))) ÷ Int32(L)
    if issplit(T)                  # block i + 1's loads issued before block i is computed
        E = Int32(devE(T)); H = Int32(devH(T)); NC = Int32(devP(T) ÷ 16)
        gi = pg * NI
        r = bload(Val(T), m.pay + (gi * NC * Int32(32) + lane) * Int32(16), m.hdr + (gi * Int32(32) + lane) * H)
        acc = 0.0f0; i = Int32(0)
        @nounroll while i < NI
            nx = i + Int32(1) < NI ? bload(Val(T), m.pay + ((gi + Int32(1)) * NC * Int32(32) + lane) * Int32(16), m.hdr + ((gi + Int32(1)) * Int32(32) + lane) * H) : r
            k0 = x0 + (j + Int32(L) * i) * E
            acc += bcomp(Val(T), r, xs, x16, xu(k0), k0 >> 4, lut)
            r = nx; gi += Int32(1); i += Int32(1)
        end
        return acc
    end
    acc = 0.0f0
    @nounroll for i in Int32(0):NI-Int32(1)
        PFNEXT && i + Int32(1) < NI && pfblk!(Val(T), m, pg * NI + i + Int32(1), lane)
        acc += blkdot(Val(T), Val(L), m, pg, NI, i, lane, j, xs, x16, x0, lut)
    end
    acc
end
# formats of at most 4 payload chunks per block: a row's NI = 3 or 5 blocks are one straight-line piece of code, so
# their loads are in flight together (a warp has 2-4 times the bytes in flight of the loop's one block). Written with
# @ntuple, not closures: those stay calls on the device.
@inline unrollable(T) = UNROLL && (T == T_Q2_0 || T == T_Q4_0 || T == T_IQ4_NL || T == T_Q5_0 || T == T_Q8_0 || T == T_BF16)
const UNROLL = false      # the unrolled rows cost more in instruction fetch than they win in loads in flight
@inline function rowdotn(::Val{T}, ::Val{L}, ::Val{N}, m::MatI, pg::Int32, lane::Int32, j::Int32, xs::SPtr, x16::SPtr, x0::Int32, lut::SPtr) where {T,L,N}
    if N == 5
        v = Base.Cartesian.@ntuple 5 u -> blkdot(Val(T), Val(L), m, pg, Int32(5), Int32(u - 1), lane, j, xs, x16, x0, lut)
        return (((v[1] + v[2]) + v[3]) + v[4]) + v[5]
    end
    v = Base.Cartesian.@ntuple 3 u -> blkdot(Val(T), Val(L), m, pg, Int32(3), Int32(u - 1), lane, j, xs, x16, x0, lut)
    (v[1] + v[2]) + v[3]
end
@inline function rowdotu(::Val{T}, ::Val{L}, m::MatI, pg::Int32, lane::Int32, j::Int32, xs::SPtr, x16::SPtr, x0::Int32, lut::SPtr) where {T,L}
    if unrollable(T)
        NI = (m.K ÷ Int32(devE(T))) ÷ Int32(L)
        NI == Int32(5) && return rowdotn(Val(T), Val(L), Val(5), m, pg, lane, j, xs, x16, x0, lut)
        NI == Int32(3) && return rowdotn(Val(T), Val(L), Val(3), m, pg, lane, j, xs, x16, x0, lut)
    end
    rowdot(Val(T), Val(L), m, pg, lane, j, xs, x16, x0, lut)
end
@inline rl4(rl::NTuple{4,Float32}, l::Int32) = l == Int32(0) ? rl[1] : l == Int32(1) ? rl[2] : l == Int32(2) ? rl[3] : rl[4]
# the row partial to its channel owner (EM kind 2): rows [lo, lo + nc) -> d1[o1 + r - lo], others -> d2[o1 + r - plo]
@inline function put_owner!(em::EM, r::Int32, v::Float32)
    if em.kind == Int32(1)
        emput!(em, em.o1 + r, v)
    elseif em.bc > Int32(0)            # kind 2 over the group: the owner of channel r, word o1 + r mod nc
        ow = r ÷ em.nc
        llput(emdst(em, ow), em.o1 + r - ow * em.nc, em.sq, v)
    elseif r >= em.lo && r < em.lo + em.nc
        llput(em.d1, em.o1 + r - em.lo, em.sq, v)
    else
        llput(em.d2, em.o1 + r - em.plo, em.sq, v)
    end
end
@inline ldvolf(p::FPtr, i::Int32) = reinterpret(Float32, UA.load(reinterpret(Core.LLVMPtr{UInt32,1}, p + 4i), UA.monotonic, UA.device))

# one row group (MODE 0, 1, 4) or one (row group, expert) task (MODE 3) of a phase
@inline function grp!(::Val{T}, ::Val{L}, ::Val{MODE}, m::MatI, g::Int32, rows::Int32, em::EM, xo::Int32, eo::Int32, ns::Int32, rpe::Int32,
                      idx::SIP, wsc::SPtr, red::Red, xs::SPtr, x16::SPtr, lut::SPtr, rl::NTuple{4,Float32}) where {T,L,MODE}
    E = Int32(devE(T)); H = Int32(devH(T)); NC = Int32(devP(T) ÷ 16); R = Int32(32 ÷ L)
    nb = m.K ÷ E; NI = nb ÷ Int32(L)
    lane = (threadIdx().x - Int32(1)) % Int32(32)
    rho = lane ÷ Int32(L); j = lane % Int32(L)
    @inbounds if MODE == 4 && HCSPLIT && L == 32
        # task g = (row r, segment sg) of the row's nseg segments of nc elements (R = 1: L = 32, a row's block b at
        # iteration b ÷ 32, lane b mod 32): lane u takes the segment's blocks u, u + 32, ... (40 blocks for nc = 1280:
        # not a multiple of 32, so the iterations of the row layout do not follow the segments), two loads in flight;
        # 4 tasks per row, not one row a warp
        nseg = m.K ÷ em.nc; r = g ÷ nseg; sg = g - r * nseg
        bseg = em.nc ÷ E; b0 = sg * bseg
        a = 0.0f0; u = lane
        while u < bseg
            bA = b0 + u; bB = bA + Int32(32)
            vA = blkdot(Val(T), Val(L), m, r, NI, bA >> 5, bA & Int32(31), bA & Int32(31), xs, x16, xo, lut)
            vB = u + Int32(32) < bseg ? blkdot(Val(T), Val(L), m, r, NI, bB >> 5, bB & Int32(31), bB & Int32(31), xs, x16, xo, lut) : 0.0f0
            a += vA + vB
            u += Int32(64)
        end
        a = wsum(a, Val(L))
        lane == Int32(0) && r < rows && put_seg!(em, sg, r, a)
    elseif MODE == 4
        r = g * R + rho
        a1 = 0.0f0; a2 = 0.0f0; a3 = 0.0f0; a4 = 0.0f0; bseg = em.nc ÷ E   # segments of nc elements; em.lo = first segment index
        if NI == Int32(5)               # the five blocks' loads together (see rowdotn)
            vs = Base.Cartesian.@ntuple 5 u -> blkdot(Val(T), Val(L), m, g, Int32(5), Int32(u - 1), lane, j, xs, x16, xo, lut)
            Base.Cartesian.@nexprs 5 u -> begin
                sg_u = (j + Int32(L) * Int32(u - 1)) ÷ bseg
                a1 += sg_u == Int32(0) ? vs[u] : 0.0f0; a2 += sg_u == Int32(1) ? vs[u] : 0.0f0
                a3 += sg_u == Int32(2) ? vs[u] : 0.0f0; a4 += sg_u == Int32(3) ? vs[u] : 0.0f0
            end
        else
            for i in Int32(0):NI-Int32(1)
                v = blkdot(Val(T), Val(L), m, g, NI, i, lane, j, xs, x16, xo, lut)
                sg = (j + Int32(L) * i) ÷ bseg
                a1 += sg == Int32(0) ? v : 0.0f0; a2 += sg == Int32(1) ? v : 0.0f0
                a3 += sg == Int32(2) ? v : 0.0f0; a4 += sg == Int32(3) ? v : 0.0f0
            end
        end
        a1 = wsum(a1, Val(L)); a2 = wsum(a2, Val(L)); a3 = wsum(a3, Val(L)); a4 = wsum(a4, Val(L))
        if j == Int32(0) && rho < R && r < rows
            nseg = nb ÷ bseg
            for (sg, v) in ((Int32(0), a1), (Int32(1), a2), (Int32(2), a3), (Int32(3), a4))
                sg < nseg && put_seg!(em, sg, r, v)
            end
        end
    elseif MODE == 3
        gr = g ÷ ns; e = g % ns
        pg = rpe == Int32(0) ? gr : (unsafe_load(idx, e + 1, Val(4)) * rpe + gr * R) ÷ R
        acc = wsum(rowdotu(Val(T), Val(L), m, pg, lane, j, xs, x16, xo + e * m.K, lut), Val(L))
        r = gr * R + rho
        (j == Int32(0) && rho < R && r < rows) && stf(red.P, (eo + e) * red.stride + r, unsafe_load(wsc, eo + e + 1, Val(4)) * acc)
        threadfence()
        sync_warp()
        blk = (gr * R) ÷ Int32(32)
        old = lane == Int32(0) ? CUDA.atomic_add!(red.cnt + 4blk, Int32(1)) : Int32(0)
        old = shfl_sync(FULL, old, Int32(1))   # CUDA.jl: 1-based source lane
        if old == red.target - Int32(1)
            threadfence()
            rr = blk * Int32(32) + lane
            y = 0.0f0
            for s in Int32(0):red.ns-Int32(1)
                y += ldvolf(red.P, s * red.stride + rr)
            end
            lane == Int32(0) && UA.store!(red.cnt + 4blk, Int32(0), UA.monotonic, UA.device)
            threadfence()
            rr < rows && put_owner!(em, rr, y)
        end
    else
        pg = MODE == 1 ? (unsafe_load(idx, (g * R) ÷ rpe + 1, Val(4)) * rpe + (g * R) % rpe) ÷ R : g
        acc = wsum(rowdotu(Val(T), Val(L), m, pg, lane, j, xs, x16, xo, lut), Val(L))
        r = g * R + rho
        if em.kind == Int32(4)
            # row r = 4 c + l: its lane (j = 0) takes sigmoid(acc) S_l[c] r_l w_l[c], all S words in flight at once; the four
            # rows of channel c are the lanes rho L .. (rho + 3) L, summed into the first by shuffles
            l = r & Int32(3); c = r >> 2
            live = j == Int32(0) && rho < R && r < rows
            tt = live ? sigm(acc) * llget(em.S, l * em.nc + c, em.sS) * rl4(rl, l) * ldf(em.w, l * em.nc + c) : 0.0f0
            tt += shfl_down_sync(FULL, tt, Int32(L)); tt += shfl_down_sync(FULL, tt, Int32(2L))
            live && l == Int32(0) && emput!(em, em.zoff + c, tt * 0.25f0)
        elseif em.kind == Int32(5)
            # interleaved gate/up rows (2i, 2i + 1): d1[o1 + i] = silu(gate) up; with nc > 0 pair i is inner channel
            # i mod nc of expert slot i ÷ nc: d1[(i ÷ nc) zoff + o1 + i mod nc] (on every die of the group: bc > 0)
            u = shfl_sync(FULL, acc, (rho + Int32(1)) * Int32(L) + Int32(1))   # CUDA.jl: 1-based source lane
            if j == Int32(0) && rho < R && rho % Int32(2) == Int32(0) && r < rows
                q = r ÷ Int32(2)
                if em.nc > Int32(0)
                    es = q ÷ em.nc
                    emput!(em, es * em.zoff + em.o1 + (q - es * em.nc), silu(acc) * u)
                else
                    llput(em.d1, em.o1 + q, em.sq, silu(acc) * u)
                end
            end
        elseif j == Int32(0) && rho < R && r < rows
            put_owner!(em, r, acc)
        end
    end
    return
end

# the (format, lanes) layouts the placement builds, by phase mode: the --combos listing of the placed blocks (mode 0
# also Q5_K and Q6_K at 2 lanes, for --bench phase; mode 1 is (Q2_0, 8), mode 4 (BF16, 32)). The engine checks its
# matrices against these lists when it loads; a layout outside them traps on the device.
const ROUTER_L = parse(Int, get(ENV, "FLASHNEXT_ROUTER_L", "0"))     # 0: lanes_for's choice
const COMBOS = ((T_BF16, 20), (T_Q2_0, 8), (T_Q2_0, 10), (T_IQ4_XS, 2), (T_IQ4_XS, 4), (T_IQ4_XS, 8), (T_IQ4_XS, 10), (T_Q3_K, 2), (T_Q3_K, 4),
    (T_Q3_K, 10), (T_BF16, 2), (T_BF16, 10), (T_BF16, 16), (T_Q4_K, 2), (T_Q4_K, 4), (T_Q4_K, 8), (T_Q4_K, 10), (T_Q5_K, 2),
    (T_Q5_K, 4), (T_Q5_K, 8), (T_Q5_K, 10), (T_Q6_K, 2), (T_Q6_K, 4), (T_Q6_K, 10), (T_IQ4_XS, 6), (T_Q4_K, 6), (T_Q5_K, 6), (T_Q6_K, 6), (T_Q3_K, 6),   # the 6-lane ones: OWIDE
    (T_IQ4_XS, 5), (T_Q3_K, 5),                                            # (5 lanes: the projections' trial)
    (T_Q3_K, 3), (T_Q4_K, 3), (T_Q5_K, 3), (T_Q6_K, 3), (T_IQ4_XS, 3))     # (the output projections of 8 dies: K = 768, wide)
const COMBOS3 = ((T_Q2_0, 1), (T_Q2_0, 2), (T_Q4_0, 2), (T_Q4_0, 4), (T_Q5_0, 2), (T_Q5_0, 4), (T_Q8_0, 2), (T_Q8_0, 4), (T_IQ4_NL, 2), (T_IQ4_NL, 4))
const COMBOS_BY_MODE = Dict(0 => COMBOS, 1 => ((T_Q2_0, 8),), 3 => COMBOS3, 4 => ((T_BF16, 32), (T_BF16, 8)))
# flat dispatchers, one per mode, with the fetch loop inside (grpdyn!): return the next index
for (name, mode) in ((:gdd0, 0), (:gdd1, 1), (:gdd3, 3), (:gdd4, 4))
    body = Expr(:block)
    for (t, L) in COMBOS_BY_MODE[mode]
        push!(body.args, :(m.t == Int32($t) && m.L == Int32($L) && return grpdyn!(Val($t), Val($L), Val($mode), m, v, s, G, ctr, rows, em, xo, eo, ns, rpe, idx, wsc, red, xs, x16, lut, rl)))
    end
    @eval @noinline function $name(m::MatI, v::Int32, s::Int32, G::Int32, ctr::IPtr, rows::Int32, em::EM, xo::Int32, eo::Int32, ns::Int32, rpe::Int32,
                                   idx::SIP, wsc::SPtr, red::Red, xs::SPtr, x16::SPtr, lut::SPtr, rl::NTuple{4,Float32})
        $body
        devtrap()
        return s + G
    end
end
# every placed block matrix's layout is in the lists (else the kernels would trap)
function check_combos(E)
    for (mode, c) in placed_combos(E), (t, L) in c
        (t, L) in COMBOS_BY_MODE[mode] || error("mode $mode layout ($(tname(t)), $L) is not in the kernels' dispatch lists (see --combos)")
    end
end
@inline function jobdyn!(j::Job{M}, v::Int32, s::Int32, G::Int32, ctr::IPtr, idx, wsc, red, xs, x16, lut, rl) where {M}
    a = (j.m, v, s, G, ctr, j.rows, j.em, j.xo, j.eo, j.ns, j.rpe, idx, wsc, red, xs, x16, lut, rl)
    M == 0 ? gdd0(a...) : M == 1 ? gdd1(a...) : M == 3 ? gdd3(a...) : gdd4(a...)
end
# a phase of plain matrices (MODE 0) on x
@inline phase0!(xf::XF, Kx, xs, rl, js::Tuple, ctr::IPtr, stp = NOSTP) = cphase!(xf, Kx % Int32, xs, rl, NOSI, NOSF, NORED, js, ctr, stp)
const NORL = (1.0f0, 1.0f0, 1.0f0, 1.0f0)

# the per-die descriptor and the per-block descriptor (device memory); fields are read one at a time.
# B1 (k_token!): a group of n dies, every LL buffer at the same offset of each die's LL region (REG; PEER: the group's
# region bases, by die index), so a die writes word i of buffer X on die e at PEER[e] + (X - REG) + 8i.
struct DD
    n::Int32; d::Int32; nc::Int32; nkh::Int32; nq::Int32; F::Int32; RO::Int32; ctx::Int32
    GS::Int32; g::Int32; s::Int32; DH::Int32; QH::Int32; nkvl::Int32; IW::Int32; pad_::Int32   # attention (attn_geom)
    S::U64P; DP::U64P; LS::U64P; T::U64P; Z::U64P; RP::U64P
    QKV::U64P; GATE::U64P; AB::U64P; ON::U64P; YP::U64P
    QG::U64P; KK::U64P; VV::U64P; IX::U64P; QO::U64P; NRM::U64P; IXN::U64P; SC::U64P; AO::U64P; KVR::U64P
    GU::U64P; IDXW::U64P; H::U64P; OUTP::U64P
    PK::U64P; PV::U64P; PPT::U64P
    EMB::U64P; PLEG::U64P
    rope::FPtr
    RED::FPtr; CNT::IPtr              # MoE down reduction: 11 x nc partials, block counters
    tim::Core.LLVMPtr{UInt64,1}       # profile stamps (null without --prof)
    CTR::IPtr                         # phase group counters, 32 per block
    PLD::U64P                         # the PLE input copied into device memory (null: read from the host-mapped PLEG)
    ABW::Core.LLVMPtr{UInt32,1}       # abort window [lo, hi) of seq numbers (early stop of a speculative round), or null
    REG::U64P; PEER::Core.LLVMPtr{UInt64,1}
    DPS::U64P; YPS::U64P              # my hc down / out projection partials of every row (forwarded to their owners)
    QF::U64P; QN::U64P; KF::U64P; VF::U64P; KN::U64P                # KVCOPY's gathered q / gate, k, v and their sums of squares
    ZW::U64P; HW::U64P; YW::U64P      # z, h, the out projection partials as W2 (2 values a word)
    DSUM::U64P; DAG::U64P             # HCAG: the lane-summed hc down partials [q][n][324] (LL; W2)
    YCB::U64P; DCB::U64P              # RHIER: the board partner's out projection / hc down partials to combine
    PSTR::Int32; UIDX_::Int32         # B8: words between a round's position copies of the region (B1: 0)
    UIDX::U64P                        # B8: the experts' union of a round: [par][U, ids 40, weights 4 x 41, slots 4 x 41]
end
# B4's (k_token4!: pairs, partner pointers)
struct DD4
    n::Int32; d::Int32; nc::Int32; nkh::Int32; nq::Int32; nkv::Int32; F::Int32; seq_in::Int32
    S::U64P; DP::U64P; DPp::U64P; Z::U64P; Zp::U64P; INJ::U64P; RP::U64P; RPp::U64P
    QKV::U64P; GATE::U64P; AB::U64P; ON::U64P; YP::U64P; YPp::U64P
    QG::U64P; KK::U64P; VV::U64P; IX::U64P; QO::U64P; AO::U64P; KVR::U64P
    GU::U64P; IDXW::U64P; SHO::U64P; OUTP::U64P; OUTPp::U64P
    PK::U64P; PV::U64P; PPT::U64P; PPTp::U64P
    LS::U64P; LSp::U64P               # hc lane sums of S^2: [parity][die][8 warps of CTA 0][4 lanes]
    EMB::U64P; PLEG::U64P
    rope::FPtr
    RED::FPtr; CNT::IPtr              # MoE down reduction: 11 x 2560 partials, 80 block counters
    tim::Core.LLVMPtr{UInt64,1}       # profile stamps (null without --prof)
    CTR::IPtr                         # phase group counters, 32 per block
    S2::U64P                          # S after a block's attention half (FUSEHCW)
    PLD::U64P                         # the PLE input copied into device memory (null: read from the host-mapped PLEG)
    ABW::Core.LLVMPtr{UInt32,1}       # abort window [lo, hi) of seq numbers (early stop of a speculative round), or null
end
struct BD
    l::Int32; attn::Int32; ple::Int32; hand_ls::Int32; hand_off::Int32; k0::Int32     # k0: the block's first stamp slot
    hcA_w::FPtr; hcA_dj::MatI; hcA_dj2::MatI; hcA_up::MatI; hcF_w::FPtr; hcF_dj::MatI; hcF_dj2::MatI; hcF_up::MatI
    qkv::MatI; gate::MatI; ab::MatI; ab_bias::FPtr; conv_w::FPtr; ssm_a::FPtr; ssm_norm::FPtr; ssm_out::MatI
    wq::MatI; wk::MatI; wv::MatI; wo::MatI; q_norm::FPtr; k_norm::FPtr; idx_x::MatI; idx_qn::FPtr; idx_kn::FPtr
    router::MatI; gu::MatI; down::MatI; sh_gate::MatI; sh_up::MatI; sh_down::MatI
    ple_key::MatI; ple_value::MatI; ple_nk::FPtr; ple_nq::FPtr; ple_nc::FPtr; ple_conv::FPtr
    conv::FPtr; dn::FPtr; kq::GPtr; ks::GPtr; vq::GPtr; vs::GPtr; pool::GPtr; ring::FPtr; pler::FPtr
    hand::U64P
    tap::FPtr                         # DFlash2 feature tap of this block (its 4 lanes of S, into die 6), or null
    router_sh::MatI; kf::FPtr
    router_f::MatI; wq_h::MatI; wk_f::MatI; wv_f::MatI; qn_f::FPtr; kn_f::FPtr     # the copy trials (PBlk)
end
@inline @generated function fld(p::Core.LLVMPtr{T,1}, ::Val{name}) where {T,name}
    i = Base.fieldindex(T, name)
    off = fieldoffset(T, i); FT = fieldtype(T, i)
    :(unsafe_load(reinterpret(Core.LLVMPtr{$FT,1}, p + $(Int(off))), 1, Val($(min(8, sizeof(FT))))))
end
macro f(p, name)
    :(fld($(esc(p)), Val($(QuoteNode(name)))))
end
# the group counter of phase s of the current block (32 per block)
@inline ctrof(dd, bd, s) = @f(dd, CTR) + 4 * (Int32(32) * ((@f(bd, k0) - Int32(1)) ÷ Int32(NSTB)) + Int32(s))
# a stamp slot of the current block for --prof (DD.tim is null without it)
@inline stslot(dd, bd, o) = (@f(dd, tim), @f(bd, k0) + Int32(o))

# ---- ops (every CTA runs every op in order and takes its share) ----
@inline gstride() = gridDim().x * blockDim().x

# every die of the group / the GS dies of my KV head's group: word i of my buffer p on each
@inline peerb(dd, e::Int32) = CUDA.unsafe_cached_load(@f(dd, PEER), e + Int32(1), Val(8))
@inline onpeer(dd, p::U64P, e::Int32) = reinterpret(U64P, peerb(dd, e) + (reinterpret(UInt64, p) - reinterpret(UInt64, @f(dd, REG))))
@inline function bput!(dd, p::U64P, i::Int32, sq::UInt32, v::Float32)
    n = @f(dd, n); o = reinterpret(UInt64, p) - reinterpret(UInt64, @f(dd, REG))
    for e in Int32(0):n-Int32(1)
        llput(reinterpret(U64P, peerb(dd, e) + o), i, sq, v)
    end
end
@inline function gput!(dd, p::U64P, i::Int32, sq::UInt32, v::Float32)
    GS = @f(dd, GS); e0 = @f(dd, g) * GS; o = reinterpret(UInt64, p) - reinterpret(UInt64, @f(dd, REG))
    for e in e0:e0+GS-Int32(1)
        llput(reinterpret(U64P, peerb(dd, e) + o), i, sq, v)
    end
end
# Forwarding: a phase writes its outputs into my own region (one or two words a row group: as remote stores, each an
# 8-byte PCIe write), then warps forward them in chunks of 32 consecutive words, so a warp's store is one 256-byte
# write a die. Chunk c goes to the grid's warp (c + ws) mod W (ws spreads several forwards of an op over the warps);
# every thread of every CTA calls these (the warps without a chunk return at once).
@inline function fwd_warp()
    nwp = blockDim().x ÷ Int32(32)
    (blockIdx().x - Int32(1)) * nwp + (threadIdx().x - Int32(1)) ÷ Int32(32), gridDim().x * nwp
end
# every die's words of buffer p (seq sq) to every die: die s's are [base + stride s, + nw) (+ nwl for the last die).
# My chunks go to every other die, or (FLASHNEXT_HIER, n even: the group's die pairs are boards, 2b and 2b + 1) to my
# board partner and to the die of my parity on each other board, which relays them to its partner: each word crosses
# each board's uplink once, not once per die. The relays (chunks of the other boards' dies of my parity, to my
# partner) are items after my own chunks, waiting for their words to arrive.
const HIER = get(ENV, "FLASHNEXT_HIER", "1") == "1"
@inline function fwd_bcast!(dd, p::U64P, base::Int32, stride::Int32, nw::Int32, nwl::Int32, sq::UInt32, ws::Int32)
    n = @f(dd, n); n == Int32(1) && return
    d = @f(dd, d); o = reinterpret(UInt64, p) - reinterpret(UInt64, @f(dd, REG))
    lane = (threadIdx().x - Int32(1)) % Int32(32)
    gw, W = fwd_warp()
    c = gw - ws % W; c < Int32(0) && (c += W)
    hier = HIER && (n & Int32(1)) == Int32(0)
    nch = (nw + Int32(31)) ÷ Int32(32)                # chunks a die (nwl <= nw needs no more)
    nb = n ÷ Int32(2); myb = d ÷ Int32(2); pa = d & Int32(1)
    tot = hier ? nch * nb : nch                       # my chunks, then (nb - 1) dies' relays
    while c < tot
        if c < nch
            k = Int32(32) * c + lane
            if k < (d == n - Int32(1) ? nwl : nw)
                i = base + stride * d + k
                v = llget(p, i, sq)
                if hier
                    llput(reinterpret(U64P, peerb(dd, d ⊻ Int32(1)) + o), i, sq, v)
                    for j in Int32(1):nb-Int32(1)
                        b = myb + j; b >= nb && (b -= nb)
                        llput(reinterpret(U64P, peerb(dd, Int32(2) * b + pa) + o), i, sq, v)
                    end
                else
                    for j in Int32(1):n-Int32(1)
                        e = d + j; e >= n && (e -= n)
                        llput(reinterpret(U64P, peerb(dd, e) + o), i, sq, v)
                    end
                end
            end
        else
            r = c - nch; bi = r ÷ nch; k = Int32(32) * (r - bi * nch) + lane
            b = bi < myb ? bi : bi + Int32(1); s = Int32(2) * b + pa
            if k < (s == n - Int32(1) ? nwl : nw)
                i = base + stride * s + k
                llput(reinterpret(U64P, peerb(dd, d ⊻ Int32(1)) + o), i, sq, llget(p, i, sq))
            end
        end
        c += W
    end
end
# my partials src[sg R1 + r] (nseg segments of R1 rows, rows r < nr valid; seq sq) to their owners: row r of segment
# sg to die e = r ÷ per, word (d nseg + sg) per + r - e per of its dst (dst: that buffer's address in my region)
@inline function fwd_owners!(dd, src::U64P, dst::U64P, R1::Int32, nseg::Int32, nr::Int32, per::Int32, sq::UInt32, ws::Int32)
    d = @f(dd, d); o = reinterpret(UInt64, dst) - reinterpret(UInt64, @f(dd, REG))
    lane = (threadIdx().x - Int32(1)) % Int32(32)
    gw, W = fwd_warp()
    c = gw - ws % W; c < Int32(0) && (c += W)
    tot = R1 * nseg
    while Int32(32) * c < tot
        k = Int32(32) * c + lane
        sg = k ÷ R1; r = k - sg * R1
        if k < tot && r < nr
            e = r ÷ per
            llput(e == d ? dst : reinterpret(U64P, peerb(dd, e) + o), (d * nseg + sg) * per + r - e * per, sq, llget(src, k, sq))
        end
        c += W
    end
end
# ---- W2 (FLASHNEXT_W2): two W2B-bit floats (fp32 rounded; 24: to 15 mantissa bits) and a flag in one 8-byte word,
# for the bulk exchanges (z, the experts' h, the out projection partials): half the PCIe bytes of LL words and, unlike
# a flag a chunk, no fence (each word is still written whole). The seq's low 16 bits repeat every 32 tokens; every word
# of these buffers is rewritten every block of its parity, so a stale word never carries the awaited seq (and 0, a
# buffer's initial word, is never one). Element k is in word k ÷ 2 (k even: bits 24-47; odd: bits 0-23).
# FLASHNEXT_W2: the buffers that use it, z / h / y (the out projection partials, into the residual stream)
const W2SET = get(ENV, "FLASHNEXT_W2", "zhdr")         # (d: the hc down partials of HCAG)
const W2Z = 'z' in W2SET; const W2H = 'h' in W2SET; const W2Y = 'y' in W2SET; const W2D = 'd' in W2SET
# r: the out projection partials as W2 through RHIER's board sums (B1 and B8; over y without its board sums): with
# W2B 27, 40.91 tok/s against 40.63 (24-bit floats, LL partials), alternating A/B; oracle 33 of 128 as before and
# speculative output identical to greedy (24-bit partials had moved the oracle's first difference to output 22)
const W2R = 'r' in W2SET
# t: the hc down's lane-summed partials to their owners (RHIER's board sums) and t itself to every die as W2 (t of half
# q in HCAG's DAG words, unused without it); the dies' row shares RO even (W2 pairs), 42 for 8 dies
const W2T = 't' in W2SET
hcro(n) = 2cld(324, 2n)
@inline tw(dd, q) = @f(dd, DAG) + 8 * (q * Int32(162))
@inline w2t(n::Int32) = W2T && rhier(n)
# FLASHNEXT_W2B (24 or 27): the floats' width. 27: 18 mantissa bits (8 times finer than 24's 15) and a 10-bit flag,
# (sq >> 6) + 1: a buffer is written once a block of its parity at a fixed slot, so the flag steps by 1 a write (9 at a
# token's start) and repeats every 32 tokens as the 16-bit seq does; its low 5 bits are 1..25, never 0 (a buffer's
# initial word) or 31 (LLMARK's all-ones)
const W2B = parse(Int, get(ENV, "FLASHNEXT_W2B", "27"))
const W2FM = UInt64((1 << W2B) - 1); const W2SM = UInt64((1 << (64 - 2W2B)) - 1)
@inline f24(v::Float32) = ((reinterpret(UInt32, v) + UInt32(1 << (31 - W2B))) >> (32 - W2B)) % UInt64
@inline uf24(u::UInt64) = reinterpret(Float32, (u % UInt32) << (32 - W2B))
@inline s16(sq::UInt32) = W2B == 24 ? (sq % UInt64) & UInt64(0xffff) : (((sq >> 6) + UInt32(1)) % UInt64) & W2SM
@inline w2put(p::U64P, i::Int32, sq::UInt32, a::Float32, b::Float32) =
    UA.store!(p + 8i, (s16(sq) << 2W2B) | (f24(a) << W2B) | f24(b), UA.monotonic, UA.system)
@inline w2ok(v::UInt64, s::UInt64) = (v >> 2W2B) == s || (v >> 2W2B) == W2SM     # (all ones: LLMARK's)
@inline function w2get(p::U64P, i::Int32, sq::UInt32)
    a = p + 8i; s = s16(sq)
    v = UA.load(a, UA.monotonic, UA.system)
    dl = UInt32(POLLMIN)
    while !w2ok(v, s)
        POLLEXP ? (spin(dl); dl = min(dl + dl, UInt32(POLLDELAY))) : (POLLDELAY > 0 && spin(POLLDELAY))
        v = UA.load(a, UA.monotonic, UA.system)
    end
    v
end
@inline w2half(v::UInt64, k::Int32) = (k & Int32(1)) == Int32(0) ? uf24((v >> W2B) & W2FM) : uf24(v & W2FM)
@inline w2el(p::U64P, k::Int32, sq::UInt32) = w2half(w2get(p, k >> 1, sq), k)
@inline w2fix(p::U64P, k::Int32, sq::UInt32, v::UInt64) = w2half(w2ok(v, s16(sq)) ? v : w2get(p, k >> 1, sq), k)
# fwd_bcast! into a W2 buffer D (the same layout, 2 values a word): die s's values src[base + stride s, + nw) (LL, base,
# stride and nw even) to every die, a warp's chunk 32 words (64 values); relays as fwd_bcast!
@inline function fwd_bcast_w2!(dd, src::U64P, D::U64P, base::Int32, stride::Int32, nw::Int32, sq::UInt32, ws::Int32, rf::Bool = false,
                               nwl::Int32 = nw)           # (nwl: the last die's values, even)
    n = @f(dd, n); d = @f(dd, d); o = reinterpret(UInt64, D) - reinterpret(UInt64, @f(dd, REG))
    lane = (threadIdx().x - Int32(1)) % Int32(32)
    gw, W = fwd_warp()
    c = gw - ws % W; c < Int32(0) && (c += W)
    hier = HIER && (n & Int32(1)) == Int32(0)
    nch = (nw ÷ Int32(2) + Int32(31)) ÷ Int32(32); nwd = (d == n - Int32(1) ? nwl : nw) ÷ Int32(2)
    nb = n ÷ Int32(2); myb = d ÷ Int32(2); pa = d & Int32(1)
    # rf (DFAST, hier): to the first half's boards only; the others' dies send but neither receive nor relay
    nfb = rf ? nb ÷ Int32(2) : nb; fast = myb < nfb
    # SRELAY (4 boards): boards 2 and 3 send their chunks over their uplinks once, to the same-parity die of board 0 or
    # 1 (their proxy), which passes them to the other boards in its relay of them
    sr = SRELAY && hier && nb == Int32(4)
    tot = hier ? (fast ? nch * nb : nch) : nch
    while c < tot
        if c < nch
            k = Int32(32) * c + lane
            if k < nwd
                i = (base + stride * d) ÷ Int32(2) + k
                a = llget(src, Int32(2) * i, sq); b = llget(src, Int32(2) * i + Int32(1), sq)
                fast && w2put(D, i, sq, a, b)
                if hier
                    n > Int32(1) && fast && w2put(reinterpret(U64P, peerb(dd, d ⊻ Int32(1)) + o), i, sq, a, b)
                    if sr && myb >= Int32(2)   # (SRELAY: a slow board's die to its proxy only)
                        w2put(reinterpret(U64P, peerb(dd, Int32(2) * (myb - Int32(2)) + pa) + o), i, sq, a, b)
                    else
                        for j in Int32(1):nb-Int32(1)
                            bb = myb + j; bb >= nb && (bb -= nb)
                            bb < nfb && w2put(reinterpret(U64P, peerb(dd, Int32(2) * bb + pa) + o), i, sq, a, b)
                        end
                    end
                else
                    for j in Int32(1):n-Int32(1)
                        e = d + j; e >= n && (e -= n)
                        w2put(reinterpret(U64P, peerb(dd, e) + o), i, sq, a, b)
                    end
                end
            end
        else
            r = c - nch; bi = r ÷ nch; k = Int32(32) * (r - bi * nch) + lane
            bb = bi < myb ? bi : bi + Int32(1); s = Int32(2) * bb + pa
            if k < (s == n - Int32(1) ? nwl : nw) ÷ Int32(2)
                i = (base + stride * s) ÷ Int32(2) + k
                v = w2get(D, i, sq)
                UA.store!(reinterpret(U64P, peerb(dd, d ⊻ Int32(1)) + o) + 8i, v, UA.monotonic, UA.system)
                if sr && bb == myb + Int32(2)  # (SRELAY: I am s's proxy: to the other fast board and, unless rf, slow board)
                    UA.store!(reinterpret(U64P, peerb(dd, Int32(2) * (Int32(1) - myb) + pa) + o) + 8i, v, UA.monotonic, UA.system)
                    rf || UA.store!(reinterpret(U64P, peerb(dd, Int32(2) * (Int32(5) - bb) + pa) + o) + 8i, v, UA.monotonic, UA.system)
                end
            end
        end
        c += W
    end
end
# FLASHNEXT_SRELAY: see fwd_bcast_w2! (z and h)
const SRELAY = get(ENV, "FLASHNEXT_SRELAY", "0") == "1"
# my partials src[r] (LL, r < nrows, even) to their owners as W2: rows r, r + 1 (r even) of owner e = r ÷ per at word
# (d per + r - e per) / 2 of its D (per a multiple of 64: one owner a chunk)
@inline function fwd_owners_w2!(dd, src::U64P, D::U64P, nrows::Int32, per::Int32, sq::UInt32, ws::Int32)
    d = @f(dd, d); o = reinterpret(UInt64, D) - reinterpret(UInt64, @f(dd, REG))
    lane = (threadIdx().x - Int32(1)) % Int32(32)
    gw, W = fwd_warp()
    c = gw - ws % W; c < Int32(0) && (c += W)
    while Int32(64) * c < nrows
        r = Int32(64) * c + Int32(2) * lane
        if r < nrows
            e = r ÷ per
            w2put(e == d ? D : reinterpret(U64P, peerb(dd, e) + o), (d * per + r - e * per) ÷ Int32(2), sq, llget(src, r, sq), llget(src, r + Int32(1), sq))
        end
        c += W
    end
end
# an EM over the group: its destination d1 (a buffer of my region) on every die
@inline EMP(kind, dd, d1::U64P, sq; kw...) = EM(kind, d1, sq; bc = @f(dd, n), pt = @f(dd, PEER),
    doff = (reinterpret(UInt64, d1) - reinterpret(UInt64, @f(dd, REG))) % UInt32, kw...)
# the hc buffers of a block half (hf: 0 attention, 1 ffn) and parity: q = 2 hf + par
@inline hcq(l, W::Symbol) = Int32(W == :ffn ? 2 : 0) + (l & Int32(1))     # (W a static parameter of the callers)

# hc down: the 324 rows (320 down + 4 inject) over my 4 nc channels of x = S w, the row partials to their owners (die
# r ÷ RO holds rows [RO e, RO (e + 1))): DP[q][source die][4 lanes][RO]; the x loads also sum S^2 per lane over my
# channels (every CTA reads all of S), CTA 0's 8 warps' sums to every die: LS[q][source die][8 warps][4 lanes]
function op_hcd!(dd, bd, l, slot, which::Val{W}, sS, seq0, xs, tim, ks) where {W}
    nc = @f(dd, nc); d = @f(dd, d); n = @f(dd, n); RO = @f(dd, RO); q = hcq(l, W)
    m = W == :attn ? @f(bd, hcA_dj) : @f(bd, hcF_dj)
    w = W == :attn ? @f(bd, hcA_w) : @f(bd, hcF_w)
    sq = sqof(seq0, l, slot)
    DP = @f(dd, DP) + 8 * (q * n * RO)               # [source die][RO]: the lane-summed partials of my rows
    DPS = @f(dd, DPS) + 8 * (q * Int32(1296))        # my partials of every row, [4 segments][324], forwarded to the owners
    xf = XF(2, @f(dd, S), 4nc, sS; w = w, nc = nc)
    # n > 1: one matrix of 4 lane segments (K = 4 nc); n == 1: two matrices of 2 lane segments each (K = 5120), so x fits
    m2 = W == :attn ? @f(bd, hcA_dj2) : @f(bd, hcF_dj2)
    K1 = m.K
    em = EM(3, DPS, sq; o1 = 0, nc = nc, lo = 0)
    m.rows == Int32(0) && return
    # the lane sums of S^2 over my channels first (CTA 0, from S itself: per warp, 4 words a warp), to every die, so
    # the lane norms are there when the partials are
    LS = @f(dd, LS); o = (q * n + d) * Int32(32)
    if blockIdx().x == Int32(1)
        tid = threadIdx().x - Int32(1); lane = tid % Int32(32); warp = tid ÷ Int32(32)
        S = @f(dd, S)
        a1 = 0.0f0; a2 = 0.0f0; a3 = 0.0f0; a4 = 0.0f0
        i = tid
        while i < Int32(4) * nc
            v = llget(S, i, sS); v2 = v * v; ln = i ÷ nc
            a1 += ln == Int32(0) ? v2 : 0.0f0; a2 += ln == Int32(1) ? v2 : 0.0f0
            a3 += ln == Int32(2) ? v2 : 0.0f0; a4 += ln == Int32(3) ? v2 : 0.0f0
            i += blockDim().x
        end
        a1 = wsum(a1, Val(32)); a2 = wsum(a2, Val(32)); a3 = wsum(a3, Val(32)); a4 = wsum(a4, Val(32))
        # the 8 warps' sums added in warp order (shared memory: xs is free before the phase), 4 words a die
        sh = CuDeviceArray{Float32,1,AS.Shared}(xs, (64,))
        if lane == Int32(0)
            sh[Int32(4)*warp+Int32(1)] = a1; sh[Int32(4)*warp+Int32(2)] = a2; sh[Int32(4)*warp+Int32(3)] = a3; sh[Int32(4)*warp+Int32(4)] = a4
        end
        sync_threads()
        if tid < Int32(4)
            s = 0.0f0
            for w8 in Int32(0):blockDim().x÷Int32(32)-Int32(1)
                s += sh[Int32(4)*w8+tid+Int32(1)]
            end
            llput(LS, o + tid, sq, s)
        end
    end
    fwd_bcast!(dd, LS, q * n * Int32(32), Int32(32), Int32(LSW), Int32(LSW), sq, Int32(0))
    phase0!(xf, K1, xs, NORL, (Job{4}(m, m.rows, em),), ctrof(dd, bd, W == :ffn ? 5 : 0), stslot(dd, bd, W == :ffn ? 24 : 15))
    if m2.rows > Int32(0)
        xf2 = XF(2, @f(dd, S), 4nc, sS; w = w, n2 = K1, nc = nc)
        em2 = EM(3, DPS, sq; o1 = 0, nc = nc, lo = K1 ÷ nc)
        phase0!(xf2, m2.K, xs, NORL, (Job{4}(m2, m2.rows, em2),), ctrof(dd, bd, W == :ffn ? 6 : 1))
    end
    if HCAG                            # my lane-summed partials of all rows to every die (DSUM / DAG [source die][324])
        DSUM = @f(dd, DSUM) + 8 * (q * n * Int32(324))
        fwd_dp!(dd, DPS, DSUM, LS + 8 * (q * n * Int32(32)), m.rows, RO, sq, Int32(8), true)
        W2D ? fwd_bcast_w2!(dd, DSUM, @f(dd, DAG) + 8 * (q * n * Int32(162)), Int32(0), Int32(324), m.rows, sq, Int32(24)) :
              fwd_bcast!(dd, DSUM, Int32(0), Int32(324), m.rows, m.rows, sq, Int32(24))
    else
        if rhier(n)                    # lane-summed into my slot of DSUM, then summed a board on the way to the owners
            DSUM = @f(dd, DSUM) + 8 * (q * n * Int32(324) + Int32(324) * d)
            fwd_dp!(dd, DPS, DSUM, LS + 8 * (q * n * Int32(32)), m.rows, RO, sq, Int32(8), true, false)
            W2T ? fwd_red_hier_w2!(dd, DSUM, DP, @f(dd, DCB) + 8 * (q * n * RO), m.rows, RO, sq, Int32(24)) :
                  fwd_red_hier!(dd, DSUM, DP, @f(dd, DCB) + 8 * (q * n * RO), m.rows, RO, sq, Int32(24))
        else
            fwd_dp!(dd, DPS, DP, LS + 8 * (q * n * Int32(32)), m.rows, RO, sq, Int32(8))
        end
    end
    stamp!(tim, ks)
end
# my hc down partials DPS[lane][324] (seq sq) summed over the 4 lanes with the lane norms r_l (from every die's LS),
# to their rows' owners: word d RO + r - e RO of DP on die e = r ÷ RO
# (ag: all rows into my own [d][324] slot of DP instead, for an all-gather; with slot false, at DP[r])
@inline function fwd_dp!(dd, DPS::U64P, DP::U64P, LSq::U64P, rows::Int32, RO::Int32, sq::UInt32, ws::Int32, ag::Bool = false, slot::Bool = true)
    n = @f(dd, n); d = @f(dd, d); o = reinterpret(UInt64, DP) - reinterpret(UInt64, @f(dd, REG))
    lane = (threadIdx().x - Int32(1)) % Int32(32)
    gw, W = fwd_warp()
    c = gw - ws % W; c < Int32(0) && (c += W)
    nchk = (rows + Int32(31)) ÷ Int32(32)
    c < nchk || return
    rl = hc_rln(LSq, n, sq, EPS)
    while c < nchk
        r = Int32(32) * c + lane
        if r < rows
            w0 = Base.Cartesian.@ntuple 4 ln -> llraw(DPS, Int32(324 * (ln - 1)) + r)
            v = 0.0f0
            Base.Cartesian.@nexprs 4 ln -> (v = fma(rl[ln], llfix(DPS, Int32(324 * (ln - 1)) + r, sq, w0[ln]), v))
            if ag
                llput(DP, (slot ? Int32(324) * d : Int32(0)) + r, sq, v)
            else
                e = r ÷ RO
                llput(e == d ? DP : reinterpret(U64P, peerb(dd, e) + o), d * RO + r - e * RO, sq, v)
            end
        end
        c += W
    end
end
const FUSEHCW = false
# FLASHNEXT_HCAG (default off): the hc down's lane-summed partials all-gathered (324 a die), each die summing them for
# t and the inject gates, instead of the owners' reduce and their t to every die: one exchange round less a half
const HCAG = get(ENV, "FLASHNEXT_HCAG", "0") == "1"     # (tried: 35.29 tok/s vs 37.22 without, W2 partials)
# the all-gathered partials of half q: [n][324], W2 (DAG) or LL (DSUM)
@inline hcagp(dd, q) = (n = @f(dd, n); W2D ? @f(dd, DAG) + 8 * (q * n * Int32(162)) : @f(dd, DSUM) + 8 * (q * n * Int32(324)))
# element k of the hc down output summed over the n dies' partials ([n][324] at p, W2 or LL)
@inline function dsum_el(p::U64P, n::Int32, k::Int32, sq::UInt32, w2::Bool)
    s = 0.0f0
    for e in Int32(0):n-Int32(1)
        s += w2 ? w2el(p, Int32(324) * e + k, sq) : llget(p, Int32(324) * e + k, sq)
    end
    s
end
# r_l = 1/sqrt(sum S_l^2 / 2560 + eps) from the lane sums LS of n dies (LSW = 4 words each, at stride 32: CTA 0's 8
# warps summed on its die; were 32, a word a warp and lane: 8 times the bytes over the slow boards' links): lane 4 e + l
# loads die e's word l, then the sums over the dies by shuffles
const LSW = 4
@inline function hc_rln(p::U64P, n::Int32, sq::UInt32, eps::Float32)
    lane = (threadIdx().x - Int32(1)) % Int32(32)
    e = lane >> 2; l = lane & Int32(3)
    v = e < n ? llget(p, Int32(32) * e + l, sq) : 0.0f0
    # lane 4 e + l holds die e's lane-l sum: add over e (lanes with the same l = lane % 4)
    v += shfl_xor_sync(FULL, v, 4); v += shfl_xor_sync(FULL, v, 8); v += shfl_xor_sync(FULL, v, 16)
    Base.Cartesian.@ntuple 4 l -> (1.0f0 / sqrt(shfl_sync(FULL, v, Int32(l)) / 2560.0f0 + eps))   # CUDA.jl: 1-based source lane
end
# my rows of the hc down output (rows [RO d, RO d + RO) of 324): T[q][r] = silu(sum_p DP_p[r] / 4) from the n dies'
# lane-summed partials (r < 320; the 4 inject rows 320.. without silu, the hc write's gates), forwarded by op_hcu!.
# Warp 0 of CTA b does row RO d + b, lane p die p's partial
function op_hct!(dd, l, slot, which::Val{W}, seq0, rows::Int32) where {W}
    n = @f(dd, n); d = @f(dd, d); RO = @f(dd, RO); q = hcq(l, W)
    sqp = sqof(seq0, l, W == :ffn ? 11 : 4); sq = sqof(seq0, l, slot)
    tid = threadIdx().x - Int32(1); lane = tid % Int32(32)
    r0 = RO * d; nr = min(rows, r0 + RO) - r0         # (rows: 324, or the final collapse's 320)
    b = blockIdx().x - Int32(1)
    (tid >= Int32(32) || b >= nr) && return
    DP = @f(dd, DP) + 8 * (q * n * RO); T = @f(dd, T) + 8 * (q * Int32(324))
    while b < nr
        # (the n dies' partials, or the board sums; W2 with W2T)
        v = lane < ynp(n) ? (w2t(n) ? w2el(DP, lane * RO + b, sqp) : llget(DP, lane * RO + b, sqp)) : 0.0f0
        v = wsum(v, Val(32))
        r = r0 + b
        lane == Int32(0) && llput(T, r, sq, r < Int32(320) ? silu(v * 0.25f0) : v)     # (forwarded by op_hcu!)
        b += gridDim().x
    end
end
# hc up + mix: x = T[q][0:320); my 4 nc rows (channel-major 4 c' + l), z[nc d + c'] = (1/4) sum_l sigmoid(up) S r_l w to
# every die (Z[q][2560])
function op_hcu!(dd, bd, l, slot, which::Val{W}, sS, seq0, xs, tim, ks) where {W}
    nc = @f(dd, nc); d = @f(dd, d); n = @f(dd, n); q = hcq(l, W)
    m = W == :attn ? @f(bd, hcA_up) : @f(bd, hcF_up)
    w = W == :attn ? @f(bd, hcA_w) : @f(bd, hcF_w)
    st = W == :ffn ? Int32(19) : Int32(17)
    rows = (W == :attn ? @f(bd, hcA_dj) : @f(bd, hcF_dj)).rows
    sq = sqof(seq0, l, slot); sqp = sqof(seq0, l, W == :ffn ? 11 : 4)
    if HCAG                            # t from every die's partials (x kind 9)
        xf = XF(9, hcagp(dd, q), n, sqp; F = W2D ? 1 : 0)
    else
        op_hct!(dd, l, st, which, seq0, rows)
        RO = @f(dd, RO); T = @f(dd, T) + 8 * (q * Int32(324))
        if w2t(n)
            fwd_bcast_w2!(dd, T, tw(dd, q), Int32(0), RO, RO, sqof(seq0, l, st), Int32(64), false, rows - RO * (n - Int32(1)))
            xf = XF(8, tw(dd, q), 320, sqof(seq0, l, st))
        else
            fwd_bcast!(dd, T, Int32(0), RO, RO, rows - RO * (n - Int32(1)), sqof(seq0, l, st), Int32(64))
            xf = XF(1, T, 320, sqof(seq0, l, st))
        end
    end
    rl = hc_rln(@f(dd, LS) + 8 * (q * n * Int32(32)), n, sqp, EPS)
    stamp!(tim, ks)
    Z = @f(dd, Z) + 8 * (q * Int32(2560))
    em = EM(4, Z, sq; nc = nc, S = @f(dd, S), sS = sS, w = w, zoff = nc * d)
    phase0!(xf, Int32(320), xs, rl, (Job{0}(m, m.rows, em),), ctrof(dd, bd, W == :ffn ? 7 : 2), W == :ffn ? stslot(dd, bd, 25) : NOSTP)
    W2Z ? fwd_bcast_w2!(dd, Z, @f(dd, ZW) + 8 * (q * Int32(1280)), Int32(0), nc, nc, sq, Int32(0)) : fwd_bcast!(dd, Z, Int32(0), nc, nc, nc, sq, Int32(0))
end
# z of half q (all 2560: W2 or LL) as a GEMV's x
@inline zx(dd, q, sq) = W2Z ? XF(8, @f(dd, ZW) + 8 * (q * Int32(1280)), 2560, sq) : XF(1, @f(dd, Z) + 8 * (q * Int32(2560)), 2560, sq)
# S[i] += 2 sigmoid(inj_l / 4) (sum_p P_p[c']), P = src: np slots of nc a parity (YP: the out projection's partials
# of the n dies, W2 when w2; OUTP: the experts' output, my rows); inj = T[q][320 + l] (seq sinj); optional handoff of
# the new S
function op_hcw!(dd, bd, l, slot, src::U64P, np::Int32, ssrc::UInt32, q::Int32, sinj::UInt32, sS::UInt32, seq0, last::Bool, t::Int32 = Int32(0),
                 w2::Bool = false)
    nc = @f(dd, nc); par = l & Int32(1); n = @f(dd, n)
    sq = sqof(seq0, l, slot)
    # (HCAG: the gates from every die's partials of rows 320.., seq sinj = the hc down's)
    S = @f(dd, S); INJ = HCAG ? hcagp(dd, q) : @f(dd, T) + 8 * (q * Int32(324) + Int32(320))
    hand = last ? @f(bd, hand) : U64P(0); hls = @f(bd, hand_ls); hoff = @f(bd, hand_off)
    # the DFlash2 tap: lane ln of channel c at slot (t mod 8) of die 6's tap buffer, [lane][2560] per tap
    tap = last ? @f(bd, tap) : NOF
    tap != NOF && (tap += 4 * ((t & Int32(7)) * Int32(TAPSLOT) + nc * @f(dd, d)))
    i = gtid()
    @inbounds while i < Int32(4) * nc
        ln = i ÷ nc; c = i % nc
        y = 0.0f0
        for p in Int32(0):np-Int32(1)
            y += w2 ? w2el(src, (par * np + p) * nc + c, ssrc) : llget(src, (par * np + p) * nc + c, ssrc)
        end
        inj = HCAG ? dsum_el(INJ, n, Int32(320) + ln, sinj, W2D) : w2t(n) ? w2el(tw(dd, q), Int32(320) + ln, sinj) : llget(INJ, ln, sinj)
        v = fma(2.0f0 * sigm(0.25f0 * inj), y, llget(S, i, sS))
        llput(S, i, sq, v)
        hand != U64P(0) && llput(hand, ln * hls + hoff + c, sq, v)
        tap != NOF && stf(tap, ln * Int32(2560) + c, v)
        i += gstride()
    end
end
const TAPSLOT = 5 * 4 * 2560          # floats per position slot of the tap buffer: 5 taps x 4 lanes x 2560
function op_proj_dn!(dd, bd, l, seq0, xs)
    par = l & Int32(1)
    sz = sqof(seq0, l, 5); sq = sqof(seq0, l, 6)
    xf = zx(dd, par, sz)
    nkh = @f(dd, nkh)
    qkv = @f(bd, qkv); gate = @f(bd, gate); ab = @f(bd, ab)
    # (a/b first, then q/k/v, then the gate, in the order the DeltaNet op needs them: measured neutral, 29.78 vs
    # 29.77 tok/s plain)
    phase0!(xf, 2560, xs, NORL, (Job{0}(qkv, qkv.rows, EM(1, @f(dd, QKV) + 8 * (par * Int32(640) * nkh), sq)),
        Job{0}(gate, gate.rows, EM(1, @f(dd, GATE) + 8 * (par * Int32(384) * nkh), sq)), Job{0}(ab, ab.rows, EM(1, @f(dd, AB) + 8 * (par * Int32(6) * nkh), sq))), ctrof(dd, bd, 3), stslot(dd, bd, 26))
end
# conv1d + gated delta rule + gated RMS readout for local value head h' = CTA index (< 3 nkh)
function op_dn!(dd, bd, l, seq0, t::Int32, xs::SPtr)
    nkh = @f(dd, nkh); par = l & Int32(1)
    h = blockIdx().x - Int32(1)
    tim = @f(dd, tim); k0 = @f(bd, k0)
    while h < Int32(3) * nkh
        sq_in = sqof(seq0, l, 6); sq = sqof(seq0, l, 7)
        QKV = @f(dd, QKV) + 8 * (par * Int32(640) * nkh); GATE = @f(dd, GATE) + 8 * (par * Int32(384) * nkh); AB = @f(dd, AB) + 8 * (par * Int32(6) * nkh)
        cw = @f(bd, conv_w); st = @f(bd, conv); ST = @f(bd, dn)
        n = Int32(640) * nkh
        tid = threadIdx().x - Int32(1)
        us = CuStaticSharedArray(Float32, 384)
        kh = h % nkh
        # the head's state copy (64 KB) to L2 while its inputs are awaited (it does not depend on them)
        if DNPF
            cps = Int32(3) * nkh * Int32(16384)
            src = reinterpret(GPtr, ST + 4 * ((t & SRM) * cps + Int32(16384) * h))
            ln = tid
            while ln < Int32(512)              # 512 lines of 128 bytes
                pfl2(src + Int32(128) * ln); ln += blockDim().x
            end
        end
        # u channels: q [128 kh, +128), k [128 nkh + 128 kh, +128), v [256 nkh + 128 h, +128)
        c = tid
        @inbounds while c < Int32(384)
            ch = c < Int32(128) ? Int32(128) * kh + c : c < Int32(256) ? Int32(128) * nkh + Int32(128) * kh + (c - Int32(128)) : Int32(256) * nkh + Int32(128) * h + (c - Int32(256))
            xc = llget(QKV, ch, sq_in)
            # the conv inputs in a ring of SRN positions: a position's slot outlives rejected speculative positions
            y = ldf(cw, Int32(4) * ch) * ldf(st, ((t - Int32(3)) & SRM) * n + ch)
            y = fma(ldf(cw, Int32(4) * ch + Int32(1)), ldf(st, ((t - Int32(2)) & SRM) * n + ch), y)
            y = fma(ldf(cw, Int32(4) * ch + Int32(2)), ldf(st, ((t - Int32(1)) & SRM) * n + ch), y)
            y = fma(ldf(cw, Int32(4) * ch + Int32(3)), xc, y)
            us[c+1] = silu(y)
            # the state slot of t: q/k channels belong to the first value head of the key head, v to its head
            (c >= Int32(256) || h < nkh) && stf(st, (t & SRM) * n + ch, xc)
            c += blockDim().x
        end
        sync_threads()
        h < gridDim().x && stamp!(tim, k0 + Int32(30))         # --prof: this CTA's first head has its inputs
        # the recurrent state of position t is read from copy t mod SRN and written to copy (t + 1) mod SRN
        cps = Int32(3) * nkh * Int32(16384)
        dn_core2!(ST + 4 * ((t & SRM) * cps), ST + 4 * (((t + Int32(1)) & SRM) * cps), us, xs, AB, GATE, @f(bd, ab_bias), @f(bd, ssm_a),
            @f(bd, ssm_norm), h, nkh, sq_in, @f(dd, ON) + 8 * (par * Int32(384) * nkh), sq)
        sync_threads()
        h < gridDim().x && stamp!(tim, k0 + Int32(31))
        h += gridDim().x
    end
end
const DNPF = get(ENV, "FLASHNEXT_DNPF", "1") == "1"
# the recurrent state copies, the conv inputs' ring and the indexer's key ring: SRN slots by position (mod SRN). A round
# of up to P8MAX = 8 positions writes slots its rejected positions share with the state the next round reads unless
# SRN > P8MAX + 3 (the conv reads 3 positions back)
const SRN = 16; const SRM = Int32(SRN - 1)
# FLASHNEXT_DNSPLIT: each value head's 128 state rows over 4 CTAs (32 rows each: a thread one row's 16 columns, as
# dn_core2!'s), 24 CTAs busy for 6 heads, not 6. The rows are independent; the readout's sum of squares is each part's
# warp sum, added as bsum adds dn_core2!'s 4 warps ((w0 + w2) + (w1 + w3)) through LL words (DNS: in HCAG's DAG
# words after W2T's t), so the output equals dn_core2!'s bit for bit (B8 keeps dn_core2!)
const DNSPLIT = get(ENV, "FLASHNEXT_DNSPLIT", "1") == "1"   # (40.91-40.99 tok/s against 40.69-40.76 without)
@inline dnss(dd, par) = @f(dd, DAG) + 8 * (Int32(648) + par * Int32(32))
function op_dn_split!(dd, bd, l, seq0, t::Int32, xs::SPtr)
    nkh = @f(dd, nkh); par = l & Int32(1)
    b = blockIdx().x - Int32(1)
    tim = @f(dd, tim); k0 = @f(bd, k0)
    tid = threadIdx().x - Int32(1); lane = tid % Int32(32); warp = tid ÷ Int32(32)
    sq_in = sqof(seq0, l, 6); sq = sqof(seq0, l, 7)
    QKV = @f(dd, QKV) + 8 * (par * Int32(640) * nkh); GATE = @f(dd, GATE) + 8 * (par * Int32(384) * nkh); AB = @f(dd, AB) + 8 * (par * Int32(6) * nkh)
    ON = @f(dd, ON) + 8 * (par * Int32(384) * nkh); DNS = dnss(dd, par)
    cw = @f(bd, conv_w); st = @f(bd, conv); ST = @f(bd, dn)
    n = Int32(640) * nkh; cps = Int32(3) * nkh * Int32(16384)
    us = CuStaticSharedArray(Float32, 384)
    qs = CuDeviceArray{Float32,1,AS.Shared}(xs, (128,)); ks = CuDeviceArray{Float32,1,AS.Shared}(xs + 512, (128,))
    vs = CuDeviceArray{Float32,1,AS.Shared}(xs + 1024, (128,)); os = CuDeviceArray{Float32,1,AS.Shared}(xs + 1536, (128,))
    while b < Int32(12) * nkh
        h = b >> 2; p = b & Int32(3); kh = h % nkh; r0 = Int32(32) * p
        STs = ST + 4 * ((t & SRM) * cps); STd = ST + 4 * (((t + Int32(1)) & SRM) * cps)
        if DNPF                        # my 32 rows of the state (16 KB) to L2 while the inputs are awaited
            src = reinterpret(GPtr, STs + 4 * (Int32(16384) * h + Int32(128) * r0))
            ln = tid
            while ln < Int32(128)
                pfl2(src + Int32(128) * ln); ln += blockDim().x
            end
        end
        # u: q and k whole (every part), v of my rows; the conv ring: q/k by part 0 of the key head's first value head
        c = tid
        @inbounds while c < Int32(288)
            cu = c < Int32(256) ? c : Int32(256) + r0 + (c - Int32(256))
            ch = cu < Int32(128) ? Int32(128) * kh + cu : cu < Int32(256) ? Int32(128) * nkh + Int32(128) * kh + (cu - Int32(128)) : Int32(256) * nkh + Int32(128) * h + (cu - Int32(256))
            xc = llget(QKV, ch, sq_in)
            y = ldf(cw, Int32(4) * ch) * ldf(st, ((t - Int32(3)) & SRM) * n + ch)
            y = fma(ldf(cw, Int32(4) * ch + Int32(1)), ldf(st, ((t - Int32(2)) & SRM) * n + ch), y)
            y = fma(ldf(cw, Int32(4) * ch + Int32(2)), ldf(st, ((t - Int32(1)) & SRM) * n + ch), y)
            y = fma(ldf(cw, Int32(4) * ch + Int32(3)), xc, y)
            us[cu+1] = silu(y)
            (cu >= Int32(256) || (h < nkh && p == Int32(0))) && stf(st, (t & SRM) * n + ch, xc)
            c += blockDim().x
        end
        sync_threads()
        b < gridDim().x && stamp!(tim, k0 + Int32(30))
        @inbounds begin
            qv = tid < Int32(128) ? us[tid+1] : 0.0f0
            kv = tid < Int32(128) ? us[tid+129] : 0.0f0
            sq2 = bsum(qv * qv); sk = bsum(kv * kv)
            if tid < Int32(128)
                qs[tid+1] = qv / sqrt(sq2 + EPS)
                ks[tid+1] = kv / sqrt(sk + EPS)
            end
            tid < Int32(32) && (vs[r0+tid+1] = us[r0+tid+257])
            sync_threads()
            a = llget(AB, h, sq_in) + ldf(@f(bd, ab_bias), h); bb = llget(AB, Int32(3) * nkh + h, sq_in) + ldf(@f(bd, ab_bias), Int32(3) * nkh + h)
            alpha = exp(ldf(@f(bd, ssm_a), h) * softplus(a)); beta = sigm(bb)
            if tid < Int32(256)
                cr = r0 + tid ÷ Int32(8); i0 = Int32(16) * (tid % Int32(8))
                off = 4 * (Int32(16384) * h + Int32(128) * cr + i0)
                dn_row2!(STd + off, ld16f(STs + off), ks, qs, vs, os, i0, cr, alpha, beta, tid)
            end
            sync_threads()
            # my rows' sum of squares (warp 0: dn_core2!'s warp p), then the 4 parts' as bsum adds them
            if warp == Int32(0)
                ov = os[r0+lane+1]
                w = wsum(ov * ov, Val(32))
                lane == Int32(0) && llput(DNS, Int32(4) * h + p, sq, w)
                w0 = llget(DNS, Int32(4) * h, sq); w1 = llget(DNS, Int32(4) * h + Int32(1), sq)
                w2 = llget(DNS, Int32(4) * h + Int32(2), sq); w3 = llget(DNS, Int32(4) * h + Int32(3), sq)
                ss = (w0 + w2) + (w1 + w3)
                rr = 1.0f0 / sqrt(ss / 128.0f0 + EPS)
                c2 = r0 + lane
                llput(ON, Int32(128) * h + c2, sq, ov * rr * ldf(@f(bd, ssm_norm), c2) * sigm(llget(GATE, Int32(128) * h + c2, sq_in)))
            end
        end
        sync_threads()
        b < gridDim().x && stamp!(tim, k0 + Int32(31))
        b += gridDim().x
    end
end
@inline function dn_core!(ST::FPtr, us, AB::U64P, GATE::U64P, abb::FPtr, ssa::FPtr, nw::FPtr, h::Int32, nkh::Int32, sq_in::UInt32, ON::U64P, sq::UInt32)
    tid = threadIdx().x - Int32(1)
    qs = CuStaticSharedArray(Float32, 128); ks = CuStaticSharedArray(Float32, 128)
    vs = CuStaticSharedArray(Float32, 128); os = CuStaticSharedArray(Float32, 128)
    @inbounds begin
        qv = tid < Int32(128) ? us[tid+1] : 0.0f0
        kv = tid < Int32(128) ? us[tid+129] : 0.0f0
        sq2 = bsum(qv * qv); sk = bsum(kv * kv)
        if tid < Int32(128)
            qs[tid+1] = qv / sqrt(sq2 + EPS)
            ks[tid+1] = kv / sqrt(sk + EPS)
            vs[tid+1] = us[tid+257]
        end
        sync_threads()
        a = llget(AB, h, sq_in) + ldf(abb, h); b = llget(AB, Int32(3) * nkh + h, sq_in) + ldf(abb, Int32(3) * nkh + h)
        alpha = exp(ldf(ssa, h) * softplus(a)); beta = sigm(b)
        if tid < Int32(256)
            c0 = Int32(4) * (tid ÷ Int32(8)); i0 = Int32(16) * (tid % Int32(8))
            base = ST + 4 * (Int32(16384) * h)
            for cc in Int32(0):Int32(3)
                c = c0 + cc
                row = base + 4 * (Int32(128) * c + i0)
                sv = ld16f(row)
                r = 0.0f0
                for k in 1:16
                    r = fma(ks[i0+k], sv[k], r)
                end
                r = wsum(r, Val(8)) * alpha
                be = beta * (vs[c+1] - r)
                o = 0.0f0
                nv = dn_update(sv, ks, i0, be, alpha)
                st16f(row, nv)
                for k in 1:16
                    o = fma(qs[i0+k], nv[k], o)
                end
                o = wsum(o, Val(8)) * 0.088388347648318440550f0
                (tid % Int32(8)) == Int32(0) && (os[c+1] = o)
            end
        end
        sync_threads()
        ov = tid < Int32(128) ? os[tid+1] : 0.0f0
        ss = bsum(ov * ov)
        if tid < Int32(128)
            r = 1.0f0 / sqrt(ss / 128.0f0 + EPS)
            llput(ON, Int32(128) * h + tid, sq, ov * r * ldf(nw, tid) * sigm(llget(GATE, Int32(128) * h + tid, sq_in)))
        end
    end
end
function op_proj_attn!(dd, bd, l, seq0, xs)
    par = l & Int32(1)
    sz = sqof(seq0, l, 5); sq = sqof(seq0, l, 6)
    xf = zx(dd, par, sz)
    # (ATTNCOPY: my query heads whole, my KV head's k / v whole: QG [nq][q 256 | gate 256], KK / VV [nkvl][256])
    wq = ATTNCOPY ? @f(bd, wq_h) : @f(bd, wq); wk = ATTNCOPY ? @f(bd, wk_f) : @f(bd, wk); wv = ATTNCOPY ? @f(bd, wv_f) : @f(bd, wv)
    ix = @f(bd, idx_x)
    phase0!(xf, 2560, xs, NORL, (Job{0}(wq, wq.rows, EM(1, @f(dd, QG), sq)), Job{0}(wk, wk.rows, EM(1, @f(dd, KK), sq)),
        Job{0}(wv, wv.rows, EM(1, @f(dd, VV), sq)), Job{0}(ix, ix.rows, EM(1, @f(dd, IX), sq))), ctrof(dd, bd, 3), stslot(dd, bd, 26))
end
# x w (thread i < DH, x = 0 beyond), rotated (NeoX, pairs (i, i + 32), i < 32) when rot: the die holding dims 0..63
@inline function wrope(x::Float32, i::Int32, DH::Int32, w::FPtr, cs::FPtr, sh, rot::Bool)
    y = i < DH ? x * ldf(w, i) : 0.0f0
    i < Int32(256) && (sh[i+1] = y)
    sync_threads()
    if rot && i < Int32(64)
        p = i % Int32(32)
        c = ldf(cs, p); s = ldf(cs, Int32(32) + p)
        y = i < Int32(32) ? fma(-sh[i+33], s, sh[i+1] * c) : fma(sh[i-31], s, sh[i+1] * c)
    end
    sync_threads()
    y
end
# attention prologue on my dims [DH s, +DH) of my KV head(s) (attn_geom). CTAs 0 .. QH-1: query head h' rotated (q w,
# not yet normalised: its 1 / rms is a factor of the score) into QO; CTAs QH ..: k (rotated k w) and v into the cache,
# then KVR[kv]; each head's sum of squares over my dims to the GS dies of my KV head (NRM[s][0:QH) q, [QH + kv] k). Next
# CTA: the indexer key's raw ring (my IW dims) and its pooled key's sum of squares over my dims to every die (IXN[d]).
function op_aprep!(dd, bd, l, seq0, t::Int32)
    QH = @f(dd, QH); nkvl = @f(dd, nkvl); DH = @f(dd, DH); s = @f(dd, s); IW = @f(dd, IW); nh = @f(dd, n)
    sq_in = sqof(seq0, l, 6); sq = sqof(seq0, l, 7)
    cs = @f(dd, rope) + 4 * (Int32(64) * t)
    hb = blockIdx().x - Int32(1); i = threadIdx().x - Int32(1)
    sh = CuStaticSharedArray(Float32, 256)
    NRM = @f(dd, NRM)
    while hb < QH + nkvl + Int32(1)
        if hb < QH
            x = i < DH ? llget(@f(dd, QG), Int32(2) * DH * hb + i, sq_in) : 0.0f0
            ss = bsum(x * x)
            y = wrope(x, i, DH, @f(bd, q_norm), cs, sh, s == Int32(0))
            i < DH && llput(@f(dd, QO), DH * hb + i, sq, y)
            i == Int32(0) && gput!(dd, NRM, Int32(32) * s + hb, sq, ss)
        elseif hb < QH + nkvl
            kv = hb - QH
            x = i < DH ? llget(@f(dd, KK), DH * kv + i, sq_in) : 0.0f0
            ss = bsum(x * x)
            y = wrope(x, i, DH, @f(bd, k_norm), cs, sh, s == Int32(0))
            row = t * nkvl + kv
            if i < DH
                kvq32(y, @f(bd, kq), row * DH + i, @f(bd, ks), row * (DH ÷ Int32(32)) + i ÷ Int32(32))
                kvq32(llget(@f(dd, VV), DH * kv + i, sq_in), @f(bd, vq), row * DH + i, @f(bd, vs), row * (DH ÷ Int32(32)) + i ÷ Int32(32))
            end
            threadfence()
            sync_threads()
            if i == Int32(0)
                llput(@f(dd, KVR), kv, sq, 1.0f0)
                gput!(dd, NRM, Int32(32) * s + QH + kv, sq, ss)
            end
        else
            # the indexer's raw key ring and pooled key (blocks of 4); its query side is only needed past 2048 keys
            ring = @f(bd, ring)
            xr = i < IW ? llget(@f(dd, IX), Int32(512) ÷ nh + i, sq_in) : 0.0f0
            i < IW && stf(ring, (t & SRM) * IW + i, xr)
            x = i < IW ? ipool(ring, IW, i, t, xr) : 0.0f0
            ss = bsum(x * x)
            i == Int32(0) && bput!(dd, @f(dd, IXN), @f(dd, d), sq, ss)
        end
        sync_threads()
        hb += gridDim().x
    end
end
# the pooled indexer key of dim i (the ring's 4 slots in order, slot t & 3 being x, the current position's)
@inline function ipool(ring::FPtr, IW::Int32, i::Int32, t::Int32, x::Float32)
    c = t & Int32(3)
    # (the last 4 positions, summed in the order of their residues mod 4; position q in ring slot q mod SRN)
    r = Base.Cartesian.@ntuple 4 k -> (Int32(k - 1) == c ? x : ldf(ring, ((t - ((t - Int32(k - 1)) & Int32(3))) & SRM) * IW + i))
    (((r[1] + r[2]) + r[3]) + r[4]) * 0.25f0
end
# acc[h] += q_h[i] k (q_h[i] at qb + h DH floats)
@inline qkacc(acc::NTuple{12,Float32}, qb::SPtr, DH::Int32, kx::Float32) =
    Base.Cartesian.@ntuple 12 hh -> fma(unsafe_load(qb + 4 * (Int32(hh - 1) * DH), 1, Val(4)), kx, acc[hh])
# the partial scores over my dims: thread per (KV head, key j <= t), sum_i q_h'[i] k_j[i] for the 12 query heads of the
# KV head, to the GS dies of the KV head: SC[s][h'][j] (ctx keys a head)
function op_score!(dd, bd, l, seq0, t::Int32, xs::SPtr)
    QH = @f(dd, QH); DH = @f(dd, DH); nkvl = @f(dd, nkvl); s = @f(dd, s); ctx = @f(dd, ctx)
    sq_in = sqof(seq0, l, 7); sq = sqof(seq0, l, 18)
    QO = @f(dd, QO); SC = @f(dd, SC); kq = @f(bd, kq); ks = @f(bd, ks)
    tid = threadIdx().x - Int32(1)
    k = tid
    @inbounds while k < QH * DH
        unsafe_store!(xs + 4k, llget(QO, k, sq_in), 1, Val(4)); k += blockDim().x
    end
    sync_threads()
    nk = t + Int32(1)
    it = gtid()
    @inbounds while it < nk * nkvl
        kv = it ÷ nk; j = it - kv * nk
        j == t && llget(@f(dd, KVR), kv, sq_in)
        row = j * nkvl + kv
        acc = ntuple(_ -> 0.0f0, Val(12))
        for i in Int32(0):DH-Int32(1)
            kx = kvd(kq, row * DH + i, ks, row * (DH ÷ Int32(32)) + i ÷ Int32(32))
            acc = qkacc(acc, xs + 4 * ((Int32(12) * kv) * DH + i), DH, kx)
        end
        for hh in 1:12
            gput!(dd, SC, (s * QH + Int32(12) * kv + Int32(hh - 1)) * ctx + j, sq, acc[hh])
        end
        it += gstride()
    end
end
# norm + rope for 256-dim heads with any block size (threads >= 256 contribute zeros)
@inline function norm_rope256x(x::Float32, d::Int32, w::FPtr, cs::FPtr, sh)
    ss = bsum(x * x)
    y = d < Int32(256) ? x / sqrt(ss / 256.0f0 + EPS) * ldf(w, d) : 0.0f0
    d < Int32(256) && (sh[d+1] = y)
    sync_threads()
    if d < Int32(64)
        p = d % Int32(32)
        c = ldf(cs, p); s = ldf(cs, Int32(32) + p)
        y = d < Int32(32) ? fma(-sh[d+33], s, sh[d+1] * c) : fma(sh[d-31], s, sh[d+1] * c)
    end
    sync_threads()
    y
end
@inline function ldq8(QO::U64P, h::Int32, lane::Int32, sq::UInt32)
    b = Int32(256) * h + Int32(8) * lane
    ws = ntuple(i -> llraw(QO, b + Int32(i - 1)), Val(8))
    ntuple(i -> llfix(QO, b + Int32(i - 1), sq, ws[i]), Val(8))
end
# the router logits r[16 lane .. +16) summed over the np partial slots, loads in flight together
@inline function rsum16(RP::U64P, np::Int32, lane::Int32, sq::UInt32)
    b = Int32(16) * lane
    w0 = Base.Cartesian.@ntuple 16 k -> llraw(RP, b + Int32(k - 1))
    w1 = Base.Cartesian.@ntuple 16 k -> (np > Int32(1) ? llraw(RP, Int32(513) + b + Int32(k - 1)) : UInt64(0))
    Base.Cartesian.@ntuple 16 k -> (np > Int32(1) ? llfix(RP, b + Int32(k - 1), sq, w0[k]) + llfix(RP, Int32(513) + b + Int32(k - 1), sq, w1[k]) :
                                    llfix(RP, b + Int32(k - 1), sq, w0[k]))
end
# acc[i] = p v[b0 + i] + sc acc[i] for the lane's dims i < DPL (v int8, its fp16 scale at si)
@inline vaccd(acc::NTuple{8,Float32}, p::Float32, sc::Float32, vq::GPtr, vs::GPtr, b0::Int32, si::Int32, DPL::Int32) =
    Base.Cartesian.@ntuple 8 i -> (Int32(i - 1) < DPL ? fma(p, kvd(vq, b0 + Int32(i - 1), vs, si), acc[i] * sc) : 0.0f0)
# attention of my query head h' = CTA index (< QH) over keys 0..t (dense; the selection is needed only past 2048 keys)
# on my dims: score_j = (1/16) r_q f_j sum_s SC[s][h'][j] (r_q, f_t from the GS dies' sums of squares; f_j < t cached),
# o = sum_j softmax_j v_j[my dims], gated, to the owner of the global head's output projection columns (die h ÷ nq,
# AO[256 (h mod nq) + DH s + i]). CTA QH: the indexer's pooled key (every 4th position), from every die's IXN.
function op_attn!(dd, bd, l, seq0, t::Int32, xs::SPtr)
    QH = @f(dd, QH); DH = @f(dd, DH); nkvl = @f(dd, nkvl); GS = @f(dd, GS); s = @f(dd, s); ctx = @f(dd, ctx); nh = @f(dd, n)
    sq_n = sqof(seq0, l, 7); sq_sc = sqof(seq0, l, 18); sq = sqof(seq0, l, 8); sqg = sqof(seq0, l, 6)
    hp = blockIdx().x - Int32(1)
    tid = threadIdx().x - Int32(1); lane = tid % Int32(32); w = tid ÷ Int32(32); nw = blockDim().x ÷ Int32(32)
    msh = CuStaticSharedArray(Float32, 16); dsh = CuStaticSharedArray(Float32, 16)
    osh = CuDeviceArray{Float32,2,AS.Shared}(xs, (256, 16))      # the GEMV x buffer is free during attention
    vq = @f(bd, vq); vs = @f(bd, vs); kf = @f(bd, kf); NRM = @f(dd, NRM); SC = @f(dd, SC)
    DPL = DH ÷ Int32(32)                                          # dims a lane: lane DPL .. + DPL
    @inbounds while hp < QH + Int32(1)
        if hp < QH
            kv = hp ÷ Int32(12)
            sq2 = 0.0f0; sk2 = 0.0f0
            for e in Int32(0):GS-Int32(1)
                sq2 += llget(NRM, Int32(32) * e + hp, sq_n); sk2 += llget(NRM, Int32(32) * e + QH + kv, sq_n)
            end
            rq = 1.0f0 / sqrt(sq2 / 256.0f0 + EPS); ft = 1.0f0 / sqrt(sk2 / 256.0f0 + EPS)
            m = -1.0f30; dsum = 0.0f0
            acc = ntuple(i -> 0.0f0, Val(8))
            j = w
            while j <= t
                sc = 0.0f0
                for e in Int32(0):GS-Int32(1)
                    sc += llget(SC, (e * QH + hp) * ctx + j, sq_sc)
                end
                f = j == t ? ft : ldf(kf, j * nkvl + kv)
                sv = sc * rq * f * 0.0625f0
                mn = max(m, sv)
                scl = exp(m - mn); p = exp(sv - mn)
                dsum = fma(dsum, scl, p)
                row = j * nkvl + kv
                acc = vaccd(acc, p, scl, vq, vs, row * DH + lane * DPL, row * (DH ÷ Int32(32)) + (lane * DPL) ÷ Int32(32), DPL)
                m = mn
                j += nw
            end
            if lane == Int32(0)
                msh[w+1] = m; dsh[w+1] = dsum
            end
            for i in 1:8
                Int32(i - 1) < DPL && (osh[lane*DPL+Int32(i), w+1] = acc[i])
            end
            sync_threads()
            if tid < DH
                M = msh[1]
                for ww in Int32(2):nw
                    M = max(M, msh[ww])
                end
                o = 0.0f0; D = 0.0f0
                for ww in Int32(1):nw
                    e = exp(msh[ww] - M)
                    o = fma(osh[tid+1, ww], e, o); D = fma(dsh[ww], e, D)
                end
                hg = nh == Int32(1) ? hp : Int32(12) * @f(dd, g) + hp; nq = @f(dd, nq)
                ow = hg ÷ nq
                llput(onpeer(dd, @f(dd, AO), ow), Int32(256) * (hg - ow * nq) + DH * s + tid, sq,
                    o / D * sigm(llget(@f(dd, QG), Int32(2) * DH * hp + DH + tid, sqg)))
            end
            tid == Int32(0) && hp % Int32(12) == Int32(0) && stf(kf, t * nkvl + kv, ft)
        elseif (t & Int32(3)) == Int32(3)
            ipool_fin!(dd, bd, l, seq0, t)
        end
        sync_threads()
        hp += gridDim().x
    end
end
# the indexer's pooled key of positions t - 3 .. t (a CTA): normalised over all 128 dims (every die's IXN), rotated
# at t - 3 (the first n / 2 dies hold the rope pairs), fp16 into the pool
function ipool_fin!(dd, bd, l, seq0, t::Int32)
    nh = @f(dd, n); IW = @f(dd, IW); d = @f(dd, d); tid = threadIdx().x - Int32(1)
    sq_n = sqof(seq0, l, 7); sqg = sqof(seq0, l, 6)
    sh = CuStaticSharedArray(Float32, 256)
    @inbounds begin
        x = tid < IW ? ipool(@f(bd, ring), IW, tid, t, llget(@f(dd, IX), Int32(512) ÷ nh + tid, sqg)) : 0.0f0
        ss = 0.0f0
        for e in Int32(0):nh-Int32(1)
            ss += llget(@f(dd, IXN), e, sq_n)
        end
        y = tid < IW ? x / sqrt(ss / 128.0f0 + EPS) * ldf(@f(bd, idx_kn), tid) : 0.0f0
        tid < Int32(256) && (sh[tid+1] = y)
        sync_threads()
        wr = nh == Int32(1) ? Int32(32) : Int32(64) ÷ nh       # rope pairs (i, i + wr) of my dims
        if (nh == Int32(1) || d < nh ÷ Int32(2)) && tid < Int32(2) * wr
            cs3 = @f(dd, rope) + 4 * (Int32(64) * max(t - Int32(3), Int32(0)))
            pp = (nh == Int32(1) ? Int32(0) : wr * d) + tid % wr
            cc = ldf(cs3, pp); sn = ldf(cs3, Int32(32) + pp)
            y = tid < wr ? fma(-sh[tid+wr+1], sn, sh[tid+1] * cc) : fma(sh[tid-wr+1], sn, sh[tid+1] * cc)
        end
        tid < IW && unsafe_store!(reinterpret(Core.LLVMPtr{Float16,1}, @f(bd, pool)), Float16(y), (t ÷ Int32(4)) * IW + tid + Int32(1), Val(2))
    end
end

# ---- FLASHNEXT_KVCOPY: the KV cache of a KV head copied on each of its GS dies (whole 256-dim rows), so attention
# needs no partial scores: the weights stay split (q, gate, k, v rows by dims; the q / k norm weights applied on the
# die holding them), and their outputs are gathered instead. op_aprep_kc!: my dims of the 12 query heads' q w_q and
# gate to the die holding the head's output columns (h ÷ nq: QF[h mod nq][q 256 | gate 256], QN[h mod nq][s] the sums
# of squares of q), my dims of k w_k and v to the GS dies of the head (KF, VF [kv][256], KN[kv][s]); CTA QH + 1: the
# indexer key (as op_aprep!)
const KVCOPY = get(ENV, "FLASHNEXT_KVCOPY", "1") == "1"     # (kept: 33.85 vs 33.51 tok/s at 192 positions, 34.15 vs 26.60 at 640-768)
function op_aprep_kc!(dd, bd, l, seq0, t::Int32)
    QH = @f(dd, QH); nkvl = @f(dd, nkvl); DH = @f(dd, DH); s = @f(dd, s); IW = @f(dd, IW); nh = @f(dd, n); nq = @f(dd, nq); GS = @f(dd, GS)
    sq_in = sqof(seq0, l, 6); sq = sqof(seq0, l, 7)
    hb = blockIdx().x - Int32(1); i = threadIdx().x - Int32(1)
    @inbounds while hb < QH + Int32(2)
        if hb < QH
            x = i < DH ? llget(@f(dd, QG), Int32(2) * DH * hb + i, sq_in) : 0.0f0
            ss = bsum(x * x)
            hg = nh == Int32(1) ? hb : Int32(12) * @f(dd, g) + hb; ow = hg ÷ nq; hl = hg - ow * nq
            QF = onpeer(dd, @f(dd, QF), ow)
            if i < DH
                llput(QF, Int32(512) * hl + DH * s + i, sq, x * ldf(@f(bd, q_norm), i))
                llput(QF, Int32(512) * hl + Int32(256) + DH * s + i, sq, llget(@f(dd, QG), Int32(2) * DH * hb + DH + i, sq_in))
            end
            i == Int32(0) && llput(onpeer(dd, @f(dd, QN), ow), GS * hl + s, sq, ss)
        elseif hb == QH
            for kv in Int32(0):nkvl-Int32(1)
                x = i < DH ? llget(@f(dd, KK), DH * kv + i, sq_in) : 0.0f0
                ss = bsum(x * x)
                if i < DH
                    gput!(dd, @f(dd, KF), Int32(256) * kv + DH * s + i, sq, x * ldf(@f(bd, k_norm), i))
                    gput!(dd, @f(dd, VF), Int32(256) * kv + DH * s + i, sq, llget(@f(dd, VV), DH * kv + i, sq_in))
                end
                i == Int32(0) && gput!(dd, @f(dd, KN), GS * kv + s, sq, ss)
            end
        else
            ring = @f(bd, ring)
            xr = i < IW ? llget(@f(dd, IX), Int32(512) ÷ nh + i, sq_in) : 0.0f0
            i < IW && stf(ring, (t & SRM) * IW + i, xr)
            x = i < IW ? ipool(ring, IW, i, t, xr) : 0.0f0
            ss = bsum(x * x)
            i == Int32(0) && bput!(dd, @f(dd, IXN), @f(dd, d), sq, ss)
        end
        sync_threads()
        hb += gridDim().x
    end
end
# the rotation of a 256-dim head (thread d, NeoX pairs (d, d + 32), d < 32) of y
@inline function rope256(y::Float32, d::Int32, cs::FPtr, sh)
    d < Int32(256) && (sh[d+1] = y)
    sync_threads()
    if d < Int32(64)
        p = d % Int32(32)
        c = ldf(cs, p); s = ldf(cs, Int32(32) + p)
        y = d < Int32(32) ? fma(-sh[d+33], s, sh[d+1] * c) : fma(sh[d-31], s, sh[d+1] * c)
    end
    sync_threads()
    y
end
# KVCOPY: CTAs 0 .. nkvl-1: k (k w / rms(k), rotated) and v of KV head kv into my copy of the cache [pos][nkvl][256],
# then KVR[kv]; CTAs nkvl ..: my nq query heads' q (q w / rms(q), rotated) into QO
function op_aprep2_kc!(dd, bd, l, seq0, t::Int32)
    nq = @f(dd, nq); nkvl = @f(dd, nkvl); GS = @f(dd, GS)
    sq_in = sqof(seq0, l, 7); sq = sqof(seq0, l, 18)
    cs = @f(dd, rope) + 4 * (Int32(64) * t)
    hb = blockIdx().x - Int32(1); d = threadIdx().x - Int32(1)
    sh = CuStaticSharedArray(Float32, 256)
    @inbounds while hb < nkvl + nq
        if hb < nkvl
            kv = hb
            ss = 0.0f0
            for e in Int32(0):GS-Int32(1)
                ss += llget(@f(dd, KN), GS * kv + e, sq_in)
            end
            x = d < Int32(256) ? llget(@f(dd, KF), Int32(256) * kv + d, sq_in) / sqrt(ss / 256.0f0 + EPS) : 0.0f0
            y = rope256(x, d, cs, sh)
            row = t * nkvl + kv
            if d < Int32(256)
                kvq32(y, @f(bd, kq), row * Int32(256) + d, @f(bd, ks), row * Int32(8) + d ÷ Int32(32))
                kvq32(llget(@f(dd, VF), Int32(256) * kv + d, sq_in), @f(bd, vq), row * Int32(256) + d, @f(bd, vs), row * Int32(8) + d ÷ Int32(32))
            end
            threadfence()
            sync_threads()
            d == Int32(0) && llput(@f(dd, KVR), kv, sq, 1.0f0)
        else
            hl = hb - nkvl
            ss = 0.0f0
            for e in Int32(0):GS-Int32(1)
                ss += llget(@f(dd, QN), GS * hl + e, sq_in)
            end
            x = d < Int32(256) ? llget(@f(dd, QF), Int32(512) * hl + d, sq_in) / sqrt(ss / 256.0f0 + EPS) : 0.0f0
            y = rope256(x, d, cs, sh)
            d < Int32(256) && llput(@f(dd, QO), Int32(256) * hl + d, sq, y)
        end
        sync_threads()
        hb += gridDim().x
    end
end
# FLASHNEXT_ATTNCOPY (a copy trial, with KVCOPY's cache): my nq query heads whole (q split by heads, no copy) and my
# KV head's k / v rows whole on each of its GS dies (attn_k / attn_v and the q / k norm weights copied there): no
# gathers. CTAs 0 .. nq-1: q heads (norm, rope) into QO; CTAs nq ..: k (norm, rope) and v into the cache, KVR[kv];
# next CTA: the indexer key (as op_aprep!)
const ATTNCOPY = get(ENV, "FLASHNEXT_ATTNCOPY", "0") == "1"     # (tried: 40.50 tok/s with ROUTERCOPY against 40.43 without it: off)
function op_aprep_ac!(dd, bd, l, seq0, t::Int32)
    nq = @f(dd, nq); nkvl = @f(dd, nkvl); IW = @f(dd, IW); nh = @f(dd, n)
    sq_in = sqof(seq0, l, 6); sq = sqof(seq0, l, 18); sq7 = sqof(seq0, l, 7)
    cs = @f(dd, rope) + 4 * (Int32(64) * t)
    hb = blockIdx().x - Int32(1); d = threadIdx().x - Int32(1)
    sh = CuStaticSharedArray(Float32, 256)
    @inbounds while hb < nq + nkvl + Int32(1)
        if hb < nq
            x = d < Int32(256) ? llget(@f(dd, QG), Int32(512) * hb + d, sq_in) : 0.0f0
            y = norm_rope256x(x, d, @f(bd, qn_f), cs, sh)
            d < Int32(256) && llput(@f(dd, QO), Int32(256) * hb + d, sq, y)
        elseif hb < nq + nkvl
            kv = hb - nq
            x = d < Int32(256) ? llget(@f(dd, KK), Int32(256) * kv + d, sq_in) : 0.0f0
            y = norm_rope256x(x, d, @f(bd, kn_f), cs, sh)
            row = t * nkvl + kv
            if d < Int32(256)
                kvq32(y, @f(bd, kq), row * Int32(256) + d, @f(bd, ks), row * Int32(8) + d ÷ Int32(32))
                kvq32(llget(@f(dd, VV), Int32(256) * kv + d, sq_in), @f(bd, vq), row * Int32(256) + d, @f(bd, vs), row * Int32(8) + d ÷ Int32(32))
            end
            threadfence()
            sync_threads()
            d == Int32(0) && llput(@f(dd, KVR), kv, sq, 1.0f0)
        else
            ring = @f(bd, ring)
            xr = d < IW ? llget(@f(dd, IX), Int32(512) ÷ nh + d, sq_in) : 0.0f0
            d < IW && stf(ring, (t & SRM) * IW + d, xr)
            x = d < IW ? ipool(ring, IW, d, t, xr) : 0.0f0
            ss = bsum(x * x)
            d == Int32(0) && bput!(dd, @f(dd, IXN), @f(dd, d), sq7, ss)
        end
        sync_threads()
        hb += gridDim().x
    end
end
# KVCOPY: attention of my query head hl = CTA index (< nq) over keys 0..t (dense) on whole rows, gated, into AO
# (local); CTA nq: the indexer's pooled key
function op_attn_kc!(dd, bd, l, seq0, t::Int32, xs::SPtr)
    nq = @f(dd, nq); nkvl = @f(dd, nkvl)
    sq_in = sqof(seq0, l, 18); sq = sqof(seq0, l, 8); sqg = sqof(seq0, l, 7)
    h = blockIdx().x - Int32(1)
    tid = threadIdx().x - Int32(1); lane = tid % Int32(32); w = tid ÷ Int32(32); nw = blockDim().x ÷ Int32(32)
    msh = CuStaticSharedArray(Float32, 16); dsh = CuStaticSharedArray(Float32, 16)
    osh = CuDeviceArray{Float32,2,AS.Shared}(xs, (256, 16))
    kq = @f(bd, kq); ks = @f(bd, ks); vq = @f(bd, vq); vs = @f(bd, vs)
    hg0 = @f(dd, n) == Int32(1) ? Int32(0) : nq * @f(dd, d)        # my first global head
    @inbounds while h < nq + Int32(1)
        if h < nq
            kv = nkvl == Int32(1) ? Int32(0) : (hg0 + h) ÷ Int32(12)
            llget(@f(dd, KVR), kv, sq_in)
            q = ldq8(@f(dd, QO), h, lane, sq_in)
            m = -1.0f30; dsum = 0.0f0
            acc = ntuple(i -> 0.0f0, Val(8))
            j = w
            while j <= t
                row = j * nkvl + kv
                s = 0.0f0
                for i in 1:8
                    dd_ = Int32(8) * lane + Int32(i - 1)
                    s = fma(q[i], kvd(kq, row * Int32(256) + dd_, ks, row * Int32(8) + dd_ ÷ Int32(32)), s)
                end
                s = wsum(s, Val(32)) * 0.0625f0
                mn = max(m, s)
                sc = exp(m - mn); p = exp(s - mn)
                dsum = fma(dsum, sc, p)
                acc = vacc(acc, p, sc, vq, vs, row, lane)
                m = mn
                j += nw
            end
            if lane == Int32(0)
                msh[w+1] = m; dsh[w+1] = dsum
            end
            for i in 1:8
                osh[Int32(8)*lane+Int32(i), w+1] = acc[i]
            end
            sync_threads()
            if tid < Int32(256)
                M = msh[1]
                for ww in Int32(2):nw
                    M = max(M, msh[ww])
                end
                o = 0.0f0; D = 0.0f0
                for ww in Int32(1):nw
                    e = exp(msh[ww] - M)
                    o = fma(osh[tid+1, ww], e, o); D = fma(dsh[ww], e, D)
                end
                gt = ATTNCOPY ? llget(@f(dd, QG), Int32(512) * h + Int32(256) + tid, sqof(seq0, l, 6)) : llget(@f(dd, QF), Int32(512) * h + Int32(256) + tid, sqg)
                llput(@f(dd, AO), Int32(256) * h + tid, sq, o / D * sigm(gt))
            end
        elseif (t & Int32(3)) == Int32(3)
            ipool_fin!(dd, bd, l, seq0, t)
        end
        sync_threads()
        h += gridDim().x
    end
end
# the out projection over my columns (query heads / value heads), each row's partial to its channel's owner:
# YP[par][source die][nc]
function op_out!(dd, bd, l, seq0, xs)
    nc = @f(dd, nc); d = @f(dd, d); n = @f(dd, n); nkh = @f(dd, nkh); nq = @f(dd, nq); par = l & Int32(1)
    attn = @f(bd, attn) == Int32(1)
    sq = sqof(seq0, l, 9)
    xf = attn ? XF(1, @f(dd, AO), 256nq, sqof(seq0, l, 8)) : XF(1, @f(dd, ON) + 8 * (par * Int32(384) * nkh), 384nkh, sqof(seq0, l, 7))
    m = attn ? @f(bd, wo) : @f(bd, ssm_out)
    YPS = @f(dd, YPS) + 8 * (par * Int32(2560))      # my partials of all 2560 rows, forwarded to the channel owners
    phase0!(xf, m.K, xs, NORL, (Job{0}(m, m.rows, EM(1, YPS, sq)),), ctrof(dd, bd, 4), stslot(dd, bd, 27))
    if w2r(n)
        fwd_red_hier_w2!(dd, YPS, @f(dd, YW) + 8 * (par * ((ynp(n) * nc) ÷ Int32(2))), @f(dd, YCB) + 8 * (par * n * nc), Int32(2560), nc, sq, Int32(0))
    elseif W2Y
        fwd_owners_w2!(dd, YPS, @f(dd, YW) + 8 * (par * ((n * nc) ÷ Int32(2))), Int32(2560), nc, sq, Int32(0))
    elseif rhier(n)
        fwd_red_hier!(dd, YPS, @f(dd, YP) + 8 * (par * ynp(n) * nc), @f(dd, YCB) + 8 * (par * n * nc), Int32(2560), nc, sq, Int32(0))
    else
        fwd_owners!(dd, YPS, @f(dd, YP) + 8 * (par * n * nc), Int32(2560), Int32(1), Int32(2560), nc, sq, Int32(0))
    end
end
# FLASHNEXT_RHIER (default on, n even): partials to their owners summed a board first: the die of the owner's parity
# on each other board adds its partner's partial (sent on the board) to its own and sends the sum, so each board's
# uplink carries one partial a row, not two. The owner's slots: [own, partner, the other boards' sums by board].
const RHIER = get(ENV, "FLASHNEXT_RHIER", "1") == "1"
@inline rhier(n::Int32) = RHIER && (n & Int32(1)) == Int32(0) && n > Int32(2)
@inline ynp(n::Int32) = rhier(n) ? n ÷ Int32(2) + Int32(1) : n          # the owners' partial slots
@inline w2r(n::Int32) = W2R && rhier(n)
# the out projection partials' buffer, slot count and format a block's hc write reads (the owners' slots of parity par)
@inline ysrc(dd, n::Int32) = w2r(n) ? (@f(dd, YW), ynp(n), true) : W2Y ? (@f(dd, YW), n, true) : (@f(dd, YP), ynp(n), false)
# my partials src[r] (r < rows, LL) to the owners (rows [per e, per (e + 1)) of die e): dst slots [ynp][per] on each
# owner, comb [nb - 1][per] the combining die's partner partials
@inline function fwd_red_hier!(dd, src::U64P, dst::U64P, comb::U64P, rows::Int32, per::Int32, sq::UInt32, ws::Int32)
    n = @f(dd, n); d = @f(dd, d)
    od = reinterpret(UInt64, dst) - reinterpret(UInt64, @f(dd, REG)); oc = reinterpret(UInt64, comb) - reinterpret(UInt64, @f(dd, REG))
    lane = (threadIdx().x - Int32(1)) % Int32(32)
    gw, W = fwd_warp()
    c = gw - ws % W; c < Int32(0) && (c += W)
    nb = n ÷ Int32(2); myb = d ÷ Int32(2); pa = d & Int32(1); pt = d ⊻ Int32(1)
    nchp = (per + Int32(31)) ÷ Int32(32)
    while c < n * nchp
        e = c ÷ nchp; o = Int32(32) * (c - e * nchp) + lane; r = per * e + o
        if o < per && r < rows
            v = llget(src, r, sq)
            be = e ÷ Int32(2)
            if e == d
                llput(dst, o, sq, v)
            elseif e == pt
                llput(reinterpret(U64P, peerb(dd, pt) + od), per + o, sq, v)
            else
                j = be < myb ? be : be - Int32(1)     # e's board among my other boards
                if (e & Int32(1)) != pa               # my partner combines for e
                    llput(reinterpret(U64P, peerb(dd, pt) + oc), per * j + o, sq, v)
                else                                  # I combine: + my partner's, to e's slot of my board
                    s = v + llget(comb, per * j + o, sq)
                    jb = myb < be ? myb : myb - Int32(1)
                    llput(reinterpret(U64P, peerb(dd, e) + od), per * (Int32(2) + jb) + o, sq, s)
                end
            end
        end
        c += W
    end
end
# fwd_red_hier! with W2 owner slots (W2R): rows o, o + 1 a lane (per even), dst words [ynp][per / 2]; the board
# partner's partials to combine stay LL (comb, on the board)
@inline function fwd_red_hier_w2!(dd, src::U64P, dst::U64P, comb::U64P, rows::Int32, per::Int32, sq::UInt32, ws::Int32)
    n = @f(dd, n); d = @f(dd, d)
    od = reinterpret(UInt64, dst) - reinterpret(UInt64, @f(dd, REG)); oc = reinterpret(UInt64, comb) - reinterpret(UInt64, @f(dd, REG))
    lane = (threadIdx().x - Int32(1)) % Int32(32)
    gw, W = fwd_warp()
    c = gw - ws % W; c < Int32(0) && (c += W)
    nb = n ÷ Int32(2); myb = d ÷ Int32(2); pa = d & Int32(1); pt = d ⊻ Int32(1)
    nchp = (per + Int32(63)) ÷ Int32(64)
    while c < n * nchp
        e = c ÷ nchp; o = Int32(64) * (c - e * nchp) + Int32(2) * lane; r = per * e + o
        if o < per && r < rows
            a = llget(src, r, sq); b = llget(src, r + Int32(1), sq)
            be = e ÷ Int32(2)
            if e == d
                w2put(dst, o ÷ Int32(2), sq, a, b)
            elseif e == pt
                w2put(reinterpret(U64P, peerb(dd, pt) + od), (per + o) ÷ Int32(2), sq, a, b)
            else
                j = be < myb ? be : be - Int32(1)     # e's board among my other boards
                if (e & Int32(1)) != pa               # my partner combines for e
                    cp = reinterpret(U64P, peerb(dd, pt) + oc)
                    llput(cp, per * j + o, sq, a); llput(cp, per * j + o + Int32(1), sq, b)
                else                                  # I combine: + my partner's, to e's slot of my board
                    sa = a + llget(comb, per * j + o, sq); sb = b + llget(comb, per * j + o + Int32(1), sq)
                    jb = myb < be ? myb : myb - Int32(1)
                    w2put(reinterpret(U64P, peerb(dd, e) + od), (per * (Int32(2) + jb) + o) ÷ Int32(2), sq, sa, sb)
                end
            end
        end
        c += W
    end
end
# the router: my 512 / n rows over all of x (the logits to every die: RP[par][0:512)) and the shared expert's gate row
# over my nc channels (the partial to every die: RP[par][512 + d]); the shared expert's gate/up rows (my F of each) in
# the same phase (one x load, one tail), into GU[par][20F ..] with the experts' seq
const RPW = 680                       # RP words a parity: 512 logits, the shared gate's partials of up to 8 dies, then (RTOP)
                                      # each die's 10 candidates (id, logit) at 520 + 20 d
# FLASHNEXT_RTOP (with the router split, ROUTERCOPY=0): each die's top 10 of its 512 / n logits to every die (20 words,
# not its 512 / n logits), the top 10 of the 10 n candidates there: the same experts, order and weights as the top 10
# of all 512 (a logit in the top 10 by (value, lower id) is in its die's top 10 by the same order)
const RTOP = get(ENV, "FLASHNEXT_RTOP", "1") == "1"
# the top 10 by (value descending, id ascending) of the warp's N values a lane (ids < 0: none), every lane the same
@inline function sel10(vals::NTuple{N,Float32}, ids::NTuple{N,Int32}) where {N}
    taken = UInt32(0)
    Base.Cartesian.@nexprs 10 j -> begin
        bv_j = -Inf32; bi_j = Int32(-1); bk = Int32(-1)
        for k in 1:N
            if (taken >> (k - 1)) & 0x1 == 0x0 && ids[k] >= Int32(0) && (bi_j < Int32(0) || vals[k] > bv_j || (vals[k] == bv_j && ids[k] < bi_j))
                bv_j = vals[k]; bi_j = ids[k]; bk = Int32(k - 1)
            end
        end
        mine = bi_j
        for o in (16, 8, 4, 2, 1)
            ov = shfl_xor_sync(FULL, bv_j, o); oi = shfl_xor_sync(FULL, bi_j, o)
            if oi >= Int32(0) && (bi_j < Int32(0) || ov > bv_j || (ov == bv_j && oi < bi_j))
                bv_j = ov; bi_j = oi
            end
        end
        (bk >= Int32(0) && mine == bi_j) && (taken |= UInt32(1) << bk)
    end
    (Base.Cartesian.@ntuple 10 j -> bv_j), (Base.Cartesian.@ntuple 10 j -> bi_j)
end
# RTOP, warp 0 of CTA 0: my die's top 10 of its RR logits RP[r0 ..] (lane: r0 + lane + 32 k) into RC[20 d ..]
@inline function rtop_local!(RC::U64P, RP::U64P, r0::Int32, RR::Int32, d::Int32, sq::UInt32)
    lane = (threadIdx().x - Int32(1)) % Int32(32)
    vals = Base.Cartesian.@ntuple 16 k -> (lane + Int32(32(k - 1)) < RR ? llget(RP, r0 + lane + Int32(32(k - 1)), sq) : -Inf32)
    ids = Base.Cartesian.@ntuple 16 k -> (lane + Int32(32(k - 1)) < RR ? r0 + lane + Int32(32(k - 1)) : Int32(-1))
    v, ix = sel10(vals, ids)
    Base.Cartesian.@nexprs 10 j -> if lane == Int32(j - 1)
        llput(RC, Int32(20) * d + Int32(2(j - 1)), sq, reinterpret(Float32, ix[j])); llput(RC, Int32(20) * d + Int32(2(j - 1) + 1), sq, v[j])
    end
end
# RTOP: the top 10 of the n dies' candidates RC and the shared gate's partials RP[512 ..] (warp 0 of the CTA), as
# router_top10!: idx into ix[1:10], w into wv[1:11]
@inline function router_top10c!(ix, wv, RC::U64P, RP::U64P, n::Int32, sq::UInt32)
    tid = threadIdx().x - Int32(1); lane = tid % Int32(32)
    @inbounds if tid < Int32(32)
        nc = Int32(10) * n
        vals = Base.Cartesian.@ntuple 3 k -> (lane + Int32(32(k - 1)) < nc ? llget(RC, Int32(2) * (lane + Int32(32(k - 1))) + Int32(1), sq) : -Inf32)
        ids = Base.Cartesian.@ntuple 3 k -> (lane + Int32(32(k - 1)) < nc ? reinterpret(Int32, llget(RC, Int32(2) * (lane + Int32(32(k - 1))), sq)) : Int32(-1))
        sh = 0.0f0
        for p in Int32(0):n-Int32(1)
            sh += llget(RP, Int32(512) + p, sq)
        end
        v, ixv = sel10(vals, ids)
        if lane == Int32(0)
            den = 0.0f0
            Base.Cartesian.@nexprs 10 j -> (w_j = exp(v[j] - v[1]); unsafe_store!(ix, ixv[j], j, Val(4)); unsafe_store!(wv, w_j, j, Val(4)); den += w_j)
            Base.Cartesian.@nexprs 10 j -> unsafe_store!(wv, unsafe_load(wv, j, Val(4)) / den, j, Val(4))
            unsafe_store!(wv, sigm(sh), 11, Val(4))
        end
    end
    sync_threads()
end
function op_rtp!(dd, bd, l, seq0, xs)
    nc = @f(dd, nc); d = @f(dd, d); n = @f(dd, n); par = l & Int32(1)
    sq = sqof(seq0, l, 13)
    RP = @f(dd, RP) + 8 * (par * Int32(RPW)); RR = Int32(512) ÷ n
    F = @f(dd, F); sqe = sqof(seq0, l, 14)
    GU = @f(dd, GU) + 8 * (par * Int32(22) * F)
    shg = @f(bd, sh_gate); shu = @f(bd, sh_up)
    if ROUTERPAIR                      # my board's half (and on the even die the shared row), to me and my board partner
        rf = @f(bd, router); rsh = @f(bd, router_sh); RPq = onpeer(dd, RP, d ⊻ Int32(1))
        phase0!(zx(dd, Int32(2) + par, sqof(seq0, l, 12)), 2560, xs, NORL,
            (Job{0}(rf, Int32(256), EM(1, RP, sq; d2 = RPq, o1 = Int32(256) * (d & Int32(1)))),
             Job{0}(rsh, min(rsh.rows, Int32(1)), EM(1, RP, sq; d2 = RPq, o1 = 512)),
             Job{0}(shg, F, EM(1, GU, sqe; o1 = 20F)), Job{0}(shu, F, EM(1, GU, sqe; o1 = 21F))),
            ctrof(dd, bd, 8), stslot(dd, bd, 17))
        return
    end
    if ROUTERCOPY                      # the whole router and shared gate row on every die: the 513 logits here, no exchange
        rf = @f(bd, router_f)
        phase0!(zx(dd, Int32(2) + par, sqof(seq0, l, 12)), 2560, xs, NORL,
            (Job{0}(rf, rf.rows, EM(1, RP, sq)), Job{0}(shg, F, EM(1, GU, sqe; o1 = 20F)), Job{0}(shu, F, EM(1, GU, sqe; o1 = 21F))),
            ctrof(dd, bd, 8), stslot(dd, bd, 17))
        return
    end
    phase0!(zx(dd, Int32(2) + par, sqof(seq0, l, 12)), 2560, xs, NORL,
        (Job{0}(@f(bd, router), RR, EM(1, RP, sq; o1 = RR * d)), Job{0}(@f(bd, router_sh), 1, EM(1, RP, sq; o1 = 512 + d); xo = nc * d),
         Job{0}(shg, F, EM(1, GU, sqe; o1 = 20F)), Job{0}(shu, F, EM(1, GU, sqe; o1 = 21F))), ctrof(dd, bd, 8), stslot(dd, bd, 17))
    if RTOP                            # my 10 candidates (warp 0 of CTA 0, then the same warp forwards them: chunk 0, ws 0)
        RC = RP + 8 * Int32(520)
        blockIdx().x == Int32(1) && threadIdx().x <= Int32(32) && rtop_local!(RC, RP, RR * d, RR, d, sq)
        fwd_bcast!(dd, RC, Int32(0), Int32(20), Int32(20), Int32(20), sq, Int32(0))
    else
        fwd_bcast!(dd, RP, Int32(0), RR, RR, RR, sq, Int32(0))
    end
    fwd_bcast!(dd, RP, Int32(512), Int32(1), Int32(1), Int32(1), sq, Int32(16))
end
const ROUTERCOPY = get(ENV, "FLASHNEXT_ROUTERCOPY", "1") == "1"      # (the second copied item: ffn_gate_inp and the shared gate row whole)
const T10STAMP = get(ENV, "FLASHNEXT_T10STAMP", "0") == "1"          # (--prof: "router top 10" stamped once the logits are in)
# the router top 10 from the logits RP[0:512) and the shared gate's nsh partials RP[512 ..] (warp 0 of the CTA; lane
# holds 16 values): idx into ix[1:10], w into wv[1:11]
@inline function router_top10!(ix, wv, RP::U64P, nsh::Int32, sq::UInt32, tim = Core.LLVMPtr{UInt64,1}(0), ks = Int32(-1))
    tid = threadIdx().x - Int32(1); lane = tid % Int32(32)
    @inbounds if tid < Int32(32)
        vals = rsum16(RP, Int32(1), lane, sq)
        sh = 0.0f0
        for p in Int32(0):nsh-Int32(1)
            sh += llget(RP, Int32(512) + p, sq)
        end
        ks >= Int32(0) && (sync_warp(); stamp!(tim, ks))   # (FLASHNEXT_T10STAMP: the logits in, before the selection)
        taken = UInt32(0)
        top = ntuple(i -> 0.0f0, Val(10))
        m0 = 0.0f0
        for j in 1:10
            # my best untaken value (lowest index on ties)
            bv = -Inf32; bi = Int32(-1)
            for k in 1:16
                if (taken >> (k - 1)) & 0x1 == 0x0 && vals[k] > bv
                    bv = vals[k]; bi = Int32(16) * lane + Int32(k - 1)
                end
            end
            for o in (16, 8, 4, 2, 1)
                ov = shfl_xor_sync(FULL, bv, o); oi = shfl_xor_sync(FULL, bi, o)
                if ov > bv || (ov == bv && oi >= Int32(0) && (bi < Int32(0) || oi < bi))
                    bv = ov; bi = oi
                end
            end
            bi ÷ Int32(16) == lane && (taken |= UInt32(1) << (bi % Int32(16)))
            j == 1 && (m0 = bv)
            if lane == Int32(0)
                unsafe_store!(ix, bi, j, Val(4)); unsafe_store!(wv, exp(bv - m0), j, Val(4))
            end
        end
        if lane == Int32(0)
            den = 0.0f0
            for j in 1:10
                den += unsafe_load(wv, j, Val(4))
            end
            for j in 1:10
                unsafe_store!(wv, unsafe_load(wv, j, Val(4)) / den, j, Val(4))
            end
            unsafe_store!(wv, sigm(sh), 11, Val(4))
        end
    end
    sync_threads()
end
# the top 10 (every die the same, from the same logits), then the routed experts' gate/up: my F inner rows of each, h =
# silu(gate) up to every die (H[par][expert slot][640], my channels [F d, F d + F)); the shared expert's h (its
# gate/up rows ran in op_rtp!) from CTA 0, slot 10
function op_exp!(dd, bd, l, seq0, xs, tim, ks)
    F = @f(dd, F); n = @f(dd, n); d = @f(dd, d); par = l & Int32(1)
    sq = sqof(seq0, l, 14)
    ixs = CuStaticSharedArray(Int32, 16); wvs = CuStaticSharedArray(Float32, 16)
    RPp = @f(dd, RP) + 8 * (par * Int32(RPW))
    if !ROUTERCOPY && RTOP
        router_top10c!(pointer(ixs), pointer(wvs), RPp + 8 * Int32(520), RPp, n, sqof(seq0, l, 13))
    else
        router_top10!(pointer(ixs), pointer(wvs), RPp, ROUTERCOPY ? Int32(1) : n, sqof(seq0, l, 13), tim, T10STAMP ? ks : Int32(-1))
    end
    T10STAMP || stamp!(tim, ks)
    H = @f(dd, H) + 8 * (par * Int32(7040))
    if blockIdx().x == Int32(1)
        i = threadIdx().x - Int32(1)
        i < Int32(21) && llput(@f(dd, IDXW), Int32(24) * par + i, sq, i < Int32(10) ? reinterpret(Float32, ixs[i+1]) : wvs[i-Int32(9)])
        GU = @f(dd, GU) + 8 * (par * Int32(22) * F)
        while i < F
            llput(H, Int32(6400) + F * d + i, sq, silu(llget(GU, Int32(20) * F + i, sq)) * llget(GU, Int32(21) * F + i, sq))
            i += blockDim().x
        end
    end
    xf = zx(dd, Int32(2) + par, sqof(seq0, l, 12))
    gu = @f(bd, gu)
    cphase!(xf, Int32(2560), xs, NORL, pointer(ixs), pointer(wvs), NORED,
        (Job{1}(gu, 20F, EM(5, H, sq; o1 = F * d, nc = F, zoff = 640); rpe = 2F),), ctrof(dd, bd, 10), stslot(dd, bd, 28))
    nch = (F + Int32(31)) ÷ Int32(32) * (n ÷ Int32(2) + Int32(1))
    for es in Int32(0):Int32(10)       # my F channels of each expert slot to every die
        W2H ? fwd_bcast_w2!(dd, H, @f(dd, HW) + 8 * (par * Int32(3520)), Int32(640) * es, F, F, sq, es * nch, dfast(n)) :
             fwd_bcast!(dd, H, Int32(640) * es, F, F, F, sq, es * nch)
    end
    stamp!(tim, ks + Int32(1))
end
# routed (10 experts) and shared down over x = h of all 640 inner channels (H), my nc output rows: one phase of (row
# group, expert) tasks; each 32-row block is summed by the task that completes it (P slots 0-9 routed, 10 shared), into
# OUTP[par][nc] (my channels: no exchange)
function op_down!(dd, bd, l, seq0, xs, tim, ks)
    nc = @f(dd, nc); par = l & Int32(1)
    sqi = sqof(seq0, l, 14); sq = sqof(seq0, l, 15)
    ixs = CuStaticSharedArray(Int32, 16); wvs = CuStaticSharedArray(Float32, 16)
    tid = threadIdx().x - Int32(1)
    if tid < Int32(21)
        v = llget(@f(dd, IDXW), Int32(24) * par + tid, sqi)
        tid < Int32(10) ? (ixs[tid+1] = reinterpret(Int32, v)) : (wvs[tid-Int32(9)] = v)
    end
    sync_threads()
    stamp!(tim, ks)
    xf = W2H ? XF(8, @f(dd, HW) + 8 * (par * Int32(3520)), 7040, sqi) : XF(1, @f(dd, H) + 8 * (par * Int32(7040)), 7040, sqi)
    OUTP = @f(dd, OUTP) + 8 * (par * nc)
    dn = @f(bd, down); shd = @f(bd, sh_down)
    n = @f(dd, n); d = @f(dd, d)
    # DFAST: my rows then my slow partner's (rows nc ..: into its OUTP); its dies hold none
    df = dfast(n)
    df && d >= n ÷ Int32(2) && return
    em = df ? EM(2, OUTP, sq; d2 = reinterpret(U64P, peerb(dd, d + n ÷ Int32(2)) + (reinterpret(UInt64, OUTP) - reinterpret(UInt64, @f(dd, REG)))),
                 o1 = 0, lo = 0, plo = nc, nc = nc) : EM(1, OUTP, sq)
    nr = df ? Int32(2) * nc : nc
    # contributions per 32-row block: 32 / R = L tasks per expert
    red = Red(@f(dd, RED), @f(dd, CNT), Int32(10) * dn.L + shd.L, Int32(11), nr)
    # (rows a stored expert: nc, or 2 nc on a fast die of DFASTBOTH's holder, of which a module without DFAST takes nc)
    cphase!(xf, Int32(7040), xs, NORL, pointer(ixs), pointer(wvs), red, (Job{3}(dn, nr, em; ns = 10, rpe = dn.rows ÷ Int32(512)),
        Job{3}(shd, nr, em; xo = 6400, eo = 10, ns = 1, rpe = 0)), ctrof(dd, bd, 11), stslot(dd, bd, 29))
    stamp!(tim, ks + Int32(1))
end
function op_sinit!(dd, seq0)
    nc = @f(dd, nc); d = @f(dd, d)
    i = gtid()
    while i < Int32(4) * nc
        llput(@f(dd, S), i, sqof(seq0, 0, 0), llget(@f(dd, EMB), nc * d + i % nc, seq0 + UInt32(1)))
        i += gstride()
    end
end
function op_ple!(dd, bd, seq0, t::Int32, xs)
    l = Int32(1); nc = @f(dd, nc); d = @f(dd, d); n = @f(dd, n)
    s1 = sqof(seq0, l, 1); s2 = sqof(seq0, l, 2); s3 = sqof(seq0, l, 3); sS = sqof(seq0, 0, 16)
    # the PLE input (host-mapped) copied once into device memory (each word read over PCIe by one thread), then every
    # CTA's x load reads that: not 32 CTAs x 2560 words of zero-copy reads
    PLD = @f(dd, PLD)
    if PLD != U64P(0)
        i = gtid()
        while i < Int32(2560)
            llput(PLD, i, seq0 + UInt32(2), llget(@f(dd, PLEG), i, seq0 + UInt32(2))); i += gstride()
        end
    end
    xf = XF(1, PLD != U64P(0) ? PLD : @f(dd, PLEG), 2560, seq0 + UInt32(2))
    pk = @f(bd, ple_key); pv = @f(bd, ple_value)
    phase0!(xf, 2560, xs, NORL, (Job{0}(pk, pk.rows, EM(1, @f(dd, PK), s1)), Job{0}(pv, pv.rows, EM(1, @f(dd, PV), s1))), ctrof(dd, bd, 12))
    # partial sums (CTA ln < 4: lane ln; CTA 4: sum V^2) to every die: PPT[source die][16]
    c0 = blockIdx().x - Int32(1); tid = threadIdx().x - Int32(1)
    PK = @f(dd, PK); PV = @f(dd, PV); S = @f(dd, S); nk = @f(bd, ple_nk); nq = @f(bd, ple_nq)
    PPT = @f(dd, PPT)
    if c0 < Int32(5)
        a = 0.0f0; b = 0.0f0; dt = 0.0f0; c = tid
        while c < nc
            if c0 < Int32(4)
                k = llget(PK, c0 * nc + c, s1); s = llget(S, c0 * nc + c, sS)
                a = fma(k, k, a); b = fma(s, s, b); dt = fma(k * ldf(nk, c0 * nc + c), s * ldf(nq, c0 * nc + c), dt)
            else
                v = llget(PV, c, s1); a = fma(v, v, a)
            end
            c += blockDim().x
        end
        a = bsum(a); b = bsum(b); dt = bsum(dt)
        if tid == Int32(0)
            vals = c0 < Int32(4) ? ((c0, a), (Int32(4) + c0, b), (Int32(8) + c0, dt)) : ((Int32(12), a), (Int32(12), a), (Int32(12), a))
            for (o, v) in vals
                bput!(dd, PPT, Int32(16) * d + o, s2, v)
            end
        end
    end
    # finish on my channels
    ncw = @f(bd, ple_nc); cw = @f(bd, ple_conv); ring = @f(bd, pler)
    i = gtid()
    @inbounds while i < Int32(4) * nc
        ln = i ÷ nc; c = i % nc
        rK = 1.0f0 / sqrt(pls(PPT, n, ln, s2) / 2560.0f0 + EPS); rS = 1.0f0 / sqrt(pls(PPT, n, Int32(4) + ln, s2) / 2560.0f0 + EPS)
        s = rK * rS * pls(PPT, n, Int32(8) + ln, s2) * (1.0f0 / sqrt(2560.0f0))
        gt = sigm(sign(s) * sqrt(max(abs(s), 1.0f-6)))
        rG = 1.0f0 / sqrt(gt * gt * pls(PPT, n, Int32(12), s2) / 2560.0f0 + EPS)
        G = gt * llget(PV, c, s1)
        x = G * rG * ldf(ncw, i)
        n4 = Int32(4) * nc
        x9 = t >= Int32(9) ? ldf(ring, ((t - Int32(9)) & Int32(15)) * n4 + i) : 0.0f0
        x6 = t >= Int32(6) ? ldf(ring, ((t - Int32(6)) & Int32(15)) * n4 + i) : 0.0f0
        x3 = t >= Int32(3) ? ldf(ring, ((t - Int32(3)) & Int32(15)) * n4 + i) : 0.0f0
        y = ldf(cw, Int32(4) * i) * x9
        y = fma(ldf(cw, Int32(4) * i + Int32(1)), x6, y)
        y = fma(ldf(cw, Int32(4) * i + Int32(2)), x3, y)
        y = fma(ldf(cw, Int32(4) * i + Int32(3)), x, y)
        llput(S, i, s3, llget(S, i, sS) + (G + silu(y)))
        stf(ring, (t & Int32(15)) * n4 + i, x)
        i += gstride()
    end
end
@inline function pls(P::U64P, n::Int32, o::Int32, sq::UInt32)
    s = 0.0f0
    for q in Int32(0):n-Int32(1)
        s += llget(P, Int32(16) * q + o, sq)
    end
    s
end

# device-only (CUDA's overlay table, a host stub that errors): a host method holding PTX asm breaks juliac's x86 image
CUDA.CUDACore.@device_function @inline gtimer() = Base.llvmcall("""%t = call i64 asm sideeffect "mov.u64 \$0, %globaltimer;", "=l"()
    ret i64 %t""", UInt64, Tuple{})
const NTIM = 1664                     # stamps per CTA per token (NSTB per block, 48 blocks and the collapse, + 1)
const NSTB = 32                       # per block: 14 op ends, then sub-op points (SUBST)
# sub-op stamps: (offset from the block's first stamp, op index j of BLKOPS they fall in, name)
const SUBST = ((14, 1, "hc_down(attn): lane sums"), (15, 1, "hc_down(attn): x ready"), (16, 8, "hc_down(ffn): lane sums"),
    (17, 10, "router: x ready"), (18, 2, "hc_up(attn): rl + inj"), (19, 9, "hc_up(ffn): rl + inj"),
    (20, 11, "experts: router top 10"), (21, 11, "experts: gate/up phase"), (22, 12, "experts_down: idx/weights in"),
    (23, 12, "experts_down: phase"), (24, 8, "hc_down(ffn): x ready"), (25, 9, "hc_up(ffn): x ready"), (26, 3, "proj: x ready"),
    (27, 6, "out_proj: x ready"), (28, 11, "experts: gate/up x ready"), (29, 12, "experts_down: x ready"),
    (30, 4, "deltanet: first head's inputs in"), (31, 4, "deltanet: first head's state done"))
const OPNAMES = ["start", "ple", "hc_down(attn)", "hc_up+mix(attn)", "proj", "deltanet/attn_prep", "attn", "out_proj", "hc_write(attn)",
    "hc_down(ffn)", "hc_up+mix(ffn)", "router", "experts_gate_up", "experts_down", "hc_write(ffn)"]
@inline function stamp!(tim::Core.LLVMPtr{UInt64,1}, k::Int32)
    if tim != Core.LLVMPtr{UInt64,1}(0) && threadIdx().x == Int32(1)
        unsafe_store!(tim, gtimer(), (blockIdx().x - Int32(1)) * Int32(NTIM) + k + Int32(1), Val(8))
    end
end
CUDA.CUDACore.@device_function @inline smid() = Base.llvmcall("""%s = call i32 asm sideeffect "mov.u32 \$0, %smid;", "=r"()
    ret i32 %s""", UInt32, Tuple{})
# the CTA's SM in its last stamp slot (--prof: which CTAs share an SM)
@inline function stamp_sm!(tim::Core.LLVMPtr{UInt64,1})
    if tim != Core.LLVMPtr{UInt64,1}(0) && threadIdx().x == Int32(1)
        unsafe_store!(tim, UInt64(smid()), blockIdx().x * Int32(NTIM), Val(8))
    end
end
# the per-token program of one die
function k_token!(dd::Core.LLVMPtr{DD,1}, bds::Core.LLVMPtr{BD,1}, nblk::Int32, seq0::UInt32, t::Int32, xsmax::Int32, tim::Core.LLVMPtr{UInt64,1})
    # the dynamic shared memory: the abort header (16 bytes, see aborted), then the GEMV x buffer
    hd = @inbounds CuDynamicSharedArray(UInt64, 2)
    if threadIdx().x == Int32(1)
        @inbounds hd[1] = ABMAGIC; @inbounds hd[2] = reinterpret(UInt64, @f(dd, ABW))
    end
    xs = reinterpret(SPtr, pointer(hd)) + 16
    stop = CuStaticSharedArray(Int32, 1)
    stamp!(tim, Int32(0)); stamp_sm!(tim)
    for bi in Int32(1):nblk
        # a position the host stopped leaves at a block's start (every CTA, at the first block it sees the window)
        threadIdx().x == Int32(1) && (@inbounds stop[1] = Int32(aborted(seq0)))
        sync_threads()
        @inbounds stop[1] != Int32(0) && return
        bd = bds + (bi - Int32(1)) * Int32(sizeof(BD))
        k0 = Int32(1) + (bi - Int32(1)) * Int32(NSTB)
        l = @f(bd, l)
        sS = l == Int32(0) ? sqof(seq0, 0, 0) : sqof(seq0, l - Int32(1), 16)
        if l == Int32(0)
            op_sinit!(dd, seq0)
        end
        if @f(bd, ple) == Int32(1)
            op_ple!(dd, bd, seq0, t, xs); sync_threads()
            sS = sqof(seq0, 1, 3)
        end
        stamp!(tim, k0)
        if @f(bd, ple) == Int32(2)
            # the final collapse (after block 47, as "block 48"): an hc down/up of the norm-weighted S, its mix hz =
            # Z[0] on every die (the head's x)
            op_hcd!(dd, bd, l, 4, Val(:attn), sS, seq0, xs, tim, k0 + Int32(14)); sync_threads(); stamp!(tim, k0 + Int32(1))
            op_hcu!(dd, bd, l, 5, Val(:attn), sS, seq0, xs, tim, k0 + Int32(18)); sync_threads(); stamp!(tim, k0 + Int32(2))
            continue
        end
        op_hcd!(dd, bd, l, 4, Val(:attn), sS, seq0, xs, tim, k0 + Int32(14)); sync_threads(); stamp!(tim, k0 + Int32(1))
        op_hcu!(dd, bd, l, 5, Val(:attn), sS, seq0, xs, tim, k0 + Int32(18)); sync_threads(); stamp!(tim, k0 + Int32(2))
        if @f(bd, attn) == Int32(1)
            op_proj_attn!(dd, bd, l, seq0, xs); sync_threads(); stamp!(tim, k0 + Int32(3))
            if ATTNCOPY
                op_aprep_ac!(dd, bd, l, seq0, t); sync_threads(); stamp!(tim, k0 + Int32(4))
                op_attn_kc!(dd, bd, l, seq0, t, xs); sync_threads(); stamp!(tim, k0 + Int32(5))
            elseif KVCOPY
                op_aprep_kc!(dd, bd, l, seq0, t); sync_threads()
                op_aprep2_kc!(dd, bd, l, seq0, t); sync_threads(); stamp!(tim, k0 + Int32(4))
                op_attn_kc!(dd, bd, l, seq0, t, xs); sync_threads(); stamp!(tim, k0 + Int32(5))
            else
                op_aprep!(dd, bd, l, seq0, t); sync_threads()
                op_score!(dd, bd, l, seq0, t, xs); sync_threads(); stamp!(tim, k0 + Int32(4))
                op_attn!(dd, bd, l, seq0, t, xs); sync_threads(); stamp!(tim, k0 + Int32(5))
            end
        else
            op_proj_dn!(dd, bd, l, seq0, xs); sync_threads(); stamp!(tim, k0 + Int32(3))
            DNSPLIT ? op_dn_split!(dd, bd, l, seq0, t, xs) : op_dn!(dd, bd, l, seq0, t, xs); sync_threads(); stamp!(tim, k0 + Int32(4))
        end
        op_out!(dd, bd, l, seq0, xs); sync_threads(); stamp!(tim, k0 + Int32(6))
        n = @f(dd, n)
        ysp, ynp_, yw2 = ysrc(dd, n)
        op_hcw!(dd, bd, l, 10, ysp, ynp_, sqof(seq0, l, 9), l & Int32(1), sqof(seq0, l, HCAG ? 4 : 17), sS, seq0, false, t, yw2); sync_threads()
        stamp!(tim, k0 + Int32(7))
        sA = sqof(seq0, l, 10)
        op_hcd!(dd, bd, l, 11, Val(:ffn), sA, seq0, xs, tim, k0 + Int32(16)); sync_threads(); stamp!(tim, k0 + Int32(8))
        op_hcu!(dd, bd, l, 12, Val(:ffn), sA, seq0, xs, tim, k0 + Int32(19)); sync_threads(); stamp!(tim, k0 + Int32(9))
        op_rtp!(dd, bd, l, seq0, xs); sync_threads(); stamp!(tim, k0 + Int32(10))
        op_exp!(dd, bd, l, seq0, xs, tim, k0 + Int32(20)); sync_threads(); stamp!(tim, k0 + Int32(11))
        op_down!(dd, bd, l, seq0, xs, tim, k0 + Int32(22)); sync_threads(); stamp!(tim, k0 + Int32(12))
        op_hcw!(dd, bd, l, 16, @f(dd, OUTP), Int32(1), sqof(seq0, l, 15), Int32(2) + (l & Int32(1)), sqof(seq0, l, HCAG ? 11 : 19), sA, seq0, true, t)
        sync_threads(); stamp!(tim, k0 + Int32(13))
    end
    return
end
# a W2 vector of n values into plain floats
function k_w2f!(y::FPtr, x::U64P, n::Int32, sq::UInt32)
    i = gtid()
    while i < n
        stf(y, i, w2el(x, i, sq)); i += gstride()
    end
    return
end
# S (LL) of the last block into plain floats (for the head on the last die)
function k_ll2f!(y::FPtr, x::U64P, n::Int32, sq::UInt32)
    i = gtid()
    while i < n
        stf(y, i, llget(x, i, sq)); i += gstride()
    end
    return
end

# ---- Stage B1 host side ----
# the LL words of a die of a group of n (the same layout on every die of the group; ctx: the attention scores' keys)
function llsizes(n, ctx)
    nc = 2560 ÷ n; nkh = 16 ÷ n; nq = 24 ÷ n; F = 640 ÷ n; RO = hcro(n); ag = attn_geom(n)
    sort([:S => 4 * 4nc, :PLD => 4 * 2560, :DP => 4 * n * RO, :LS => 4 * n * 32, :T => 4 * 324, :Z => 4 * 2560, :RP => 2RPW,
        :QKV => 2 * 640nkh, :GATE => 2 * 384nkh, :AB => 2 * 6nkh, :ON => 2 * 384nkh, :YP => 2 * n * nc,
        :QG => 2 * ag.DH * ag.QH, :KK => 256 * ag.nkvl, :VV => 256 * ag.nkvl, :IX => 512 ÷ n + ag.IW, :QO => ag.DH * ag.QH,
        :NRM => 32 * ag.GS, :IXN => n, :SC => ag.GS * ag.QH * ctx, :AO => 256nq, :KVR => 8,
        :GU => 2 * 22F, :IDXW => 48, :H => 2 * 7040, :OUTP => 2nc, :PK => 4nc, :PV => nc, :PPT => 16n,
        :DPS => 4 * 1296, :YPS => 2 * 2560,
        :QF => 512nq, :QN => nq * ag.GS, :KF => 256 * ag.nkvl, :VF => 256 * ag.nkvl, :KN => ag.nkvl * ag.GS,
        :ZW => 4 * 1280, :HW => 2 * 3520, :YW => 2 * 1280, :DSUM => 4 * n * 324, :DAG => 4 * n * 162,
        :YCB => 2 * n * nc, :DCB => 4 * n * RO, :UIDX => 2 * UIDXW]; by = first)
end
const XSMAX = 8500                    # floats of the GEMV x buffer: xu(K) + K/16 for the largest K, the MoE down x (11 x 640)
xs_floats(n) = parse(Int, get(ENV, "FLASHNEXT_XSMAX", "8500"))
const B1THREADS = Ref(256)            # threads per CTA of k_token! (FLASHNEXT_B1_THREADS) and its register cap (FLASHNEXT_B1_MAXREGS)
const B1MAXREGS = Ref(128)
mutable struct B1Die
    rt::PDie
    ll::Dict{Symbol,U64P}
    keep::Vector{Any}
    dd::Any
    bd::Any
    nblk::Int
    S32::FPtr                         # plain copy of S for the host
    tim::Ptr{UInt64}                  # per-CTA op stamps (globaltimer ns), host-mapped: readable after a device fault
    red::FPtr; cnt::IPtr              # MoE down reduction buffers
    grid::Int                         # CTAs of k_token! on this die, and floats of its shared x buffer
    xsm::Int
    ctr::IPtr                         # phase group counters (32 per block)
end
mutable struct EngineB1
    g::GModel
    tk::Tok
    ple::PLE
    groups::Vector{Vector{B1Die}}
    ids::Vector{Int32}
    host::Any                         # pinned host memory: EMB (2560 LL words) then PLEG (2560)
    emb::Ptr{UInt64}
    pleg::Ptr{UInt64}
    ntok::Int
    kern::Dict{Int,Any}
    grid::Int
    tap::Any                          # DFlash2 tap buffer on the draft's die: 8 position slots of TAPSLOT floats
    draft::Any                        # the DFlash2 draft (load_draft), or nothing
    b4::Any                           # the B4 engine (load_b4!), or nothing
end
# the DFlash2 draft's die, on the last board (FLASHNEXT_DRAFT_DIE: 7 while die 6 is the fenced GPU-c767d250)
# (die 2: board 1 has the one Gen3 x16 uplink, and die 6 is ECC with about 400 MiB free beside another process; the
# fixture measured the same speed on both, 24.73 / 17.38 against 24.75 / 17.33 tok/s at SPECP 4 / 8)
const DRAFT_DIE = parse(Int, get(ENV, "FLASHNEXT_DRAFT_DIE", "2"))
const DRAFT_TAPS = (5, 15, 25, 35, 45)       # target blocks after which S feeds the draft (0-based)
mi(m::DevMat) = MatI(m)
function make_bd(B::PBlk, st::LState, pler::FPtr, hand::U64P, hls::Int, hoff::Int, k0::Int, tap::FPtr = NOF)
    BD(Int32(B.l), Int32(B.attn), Int32(B.l == 1), Int32(hls), Int32(hoff), Int32(k0),
        B.hcA_w, mi(B.hcA_dj), mi(B.hcA_dj2), mi(B.hcA_up), B.hcF_w, mi(B.hcF_dj), mi(B.hcF_dj2), mi(B.hcF_up),
        mi(B.qkv), mi(B.gate), mi(B.ab), B.ab_bias, B.conv_w, B.ssm_a, B.ssm_norm, mi(B.ssm_out),
        mi(B.wq), mi(B.wk), mi(B.wv), mi(B.wo), B.q_norm, B.k_norm, mi(B.idx_x), B.idx_qn, B.idx_kn,
        mi(B.router), mi(B.gu), mi(B.down), mi(B.sh_gate), mi(B.sh_up), mi(B.sh_down),
        mi(B.ple_key), mi(B.ple_value), B.ple_nk, B.ple_nq, B.ple_nc, B.ple_conv,
        st.conv, st.dn, st.kq, st.ks, st.vq, st.vs, st.pool, st.ring, pler, hand, tap, mi(B.router_sh), st.kf,
        mi(B.router_f), mi(B.wq_h), mi(B.wk_f), mi(B.wv_f), B.qn_f, B.kn_f)
end
# the die that collapses the final S (norm, hc down/up, mix) before the head: FLASHNEXT_COLLAPSE_DIE when it is a
# placed die (default 0, on board 0, the first board done with its share of a speculative round, where the early stop
# decides each position; off the last board: the collapse was its 0.25 ms a position, the stage that set a round's
# pipeline period), else the last die
function collapse_die(E0)
    dies = [rt.ord for G in E0.groups for rt in G]
    c = parse(Int, get(ENV, "FLASHNEXT_COLLAPSE_DIE", "0"))
    length(E0.groups[end]) > 1 && c in dies ? c : dies[end]
end
# the final collapse in k_token! (the placement ends with block 47: the last group's dies hold its shares)
inhead(E0) = E0.groups[end][1].head !== nothing && (E0.groups[end][1].head::DHead).down.rows > 0
inhead(E::EngineB1) = inhead_rt(E.groups[end][1].rt)
inhead_rt(rt::PDie) = rt.head !== nothing && (rt.head::DHead).down.rows > 0
hascoll(rt::PDie) = inhead_rt(rt)
const NOMI = MatI(NOMAT)
# the collapse as k_token!'s block "48": ple = 2, the hc attention fields its matrices
make_hbd(H::DHead, k0::Int) = BD(Int32(48), Int32(0), Int32(2), Int32(0), Int32(0), Int32(k0),
    H.norm, mi(H.down), mi(H.down2), mi(H.up), NOF, NOMI, NOMI, NOMI,
    NOMI, NOMI, NOMI, NOF, NOF, NOF, NOF, NOMI,
    NOMI, NOMI, NOMI, NOMI, NOF, NOF, NOMI, NOF, NOF,
    NOMI, NOMI, NOMI, NOMI, NOMI, NOMI,
    NOMI, NOMI, NOF, NOF, NOF, NOF,
    NOF, NOF, GPtr(0), GPtr(0), GPtr(0), GPtr(0), GPtr(0), NOF, NOF,
    U64P(0), NOF, NOMI, NOF, NOMI, NOMI, NOMI, NOMI, NOF, NOF)
# the collapse's own matrices on die ord (the head slices keep theirs on the last die)
function collapse_head(g::GModel, ord)
    device!(ord)
    mk(A) = DHead(put_f32n!(A, g, "output_hc_norm.weight"), put_mat!(A, g, "output_hc_down.weight"), put_mat!(A, g, "output_hc_up.weight"), NOMAT, 0)
    A = Arena(); mk(A)
    base = rawalloc(A.off)
    mk(Arena(base, A.off))
end
# (die, head) that collapse the final S
function collapse_target(E)
    haskey(KCACHE, :collapse) && return KCACHE[:collapse]::Tuple{B1Die,DHead}
    c = E.groups[end][end]; (c, c.rt.head::DHead)
end
function load_engine_b1(g::GModel, pl::PlacementB)
    E0 = load_engine_b(g, pl)
    # the token inputs (embedding, PLE) in 4 position slots: [EMB slot 0..3][PLEG slot 0..3], 2560 LL words each
    host = CUDA.alloc(CUDA.HostMemory, 8 * 2560 * 8, CUDA.MEMHOSTALLOC_DEVICEMAP | CUDA.MEMHOSTALLOC_PORTABLE)
    hp = convert(Ptr{UInt64}, host)
    ccall(:memset, Ptr{Cvoid}, (Ptr{UInt64}, Cint, Csize_t), hp, 0, 8 * 2560 * 8)
    groups = Vector{B1Die}[]
    for G in E0.groups
        DG = B1Die[]
        for rt in G
            device!(rt.ord)
            keep = Any[]; ll = Dict{Symbol,U64P}()
            # the die's LL words in one region (an early stop overwrites and zeroes them with one copy each), the same
            # layout on every die of the group
            szs = llsizes(rt.n, MAXCTX[])
            !inhead(E0) && length(E0.groups[end]) > 1 && rt.ord == collapse_die(E0) && push!(szs, :SF => 4 * 10240)
            reg = CUDA.zeros(UInt64, sum(last, szs)); push!(keep, reg)
            off = 0
            for (k, sz) in szs
                ll[k] = reinterpret(U64P, pointer(reg)) + 8 * off; off += sz
            end
            get!(() -> Dict{Int,Any}(), KCACHE, :llreg)[rt.ord] = (reinterpret(U64P, pointer(reg)), off)
            s32 = CUDA.zeros(Float32, 4 * (2560 ÷ rt.n)); push!(keep, s32)
            # (the final collapse as one more block of the program on the last group's dies)
            nb = length(rt.blks) + (hascoll(rt) ? 1 : 0)
            red = CUDA.zeros(Float32, 11 * 2560); cnt = CUDA.zeros(Int32, 80); ctr = CUDA.zeros(Int32, 32 * max(1, nb)); push!(keep, red, cnt, ctr)
            abw = CUDA.zeros(UInt32, 2); push!(keep, abw); ll[:ABW] = reinterpret(U64P, pointer(abw))
            tm = CUDA.alloc(CUDA.HostMemory, 8 * 64 * NTIM, CUDA.MEMHOSTALLOC_DEVICEMAP | CUDA.MEMHOSTALLOC_PORTABLE); push!(keep, tm)
            tp = convert(Ptr{UInt64}, tm); ccall(:memset, Ptr{Cvoid}, (Ptr{UInt64}, Cint, Csize_t), tp, 0, 8 * 64 * NTIM)
            # (SF, in the region above on the collapse's die when the last board is a pair: the full final S, both
            # dies' channels, handed to it whole in 4 position slots)
            push!(DG, B1Die(rt, ll, keep, nothing, nothing, nb, fptr(s32), tp, fptr(red), reinterpret(IPtr, pointer(cnt)), 0, xs_floats(rt.n), reinterpret(IPtr, pointer(ctr))))
        end
        push!(groups, DG)
    end
    # DFlash2's feature taps: S after blocks 5, 15, 25, 35, 45 (all 4 lanes) into 8 position slots on the draft's die
    device!(DRAFT_DIE); tapbuf = CUDA.zeros(Float32, 8 * TAPSLOT)
    tapof(l) = (k = findfirst(==(l), DRAFT_TAPS); k === nothing ? NOF : fptr(tapbuf) + 4 * (k - 1) * 4 * 2560)
    for (gi, DG) in enumerate(groups)
        # the group's LL region bases by die index, on each of its dies (the kernels' peer table)
        bases = UInt64[reinterpret(UInt64, llreg(D.rt.ord)[1]) for D in DG]
        for D in DG
            rt = D.rt; device!(rt.ord)
            pt = CuArray(bases); push!(D.keep, pt)
            n = rt.n; nc = 2560 ÷ n; ag = attn_geom(n); gk, sk = attn_gs(n, rt.d)
            # one DD and one BD array per position slot s = t & 3: S, the inputs and the handoff target at slot s (tokens
            # of a round run on the boards one after another, a board up to 3 positions ahead of the next)
            dds = DD[]; bds = BD[]
            for s in 0:3
                push!(dds, DD(Int32(n), Int32(rt.d), Int32(nc), Int32(16 ÷ n), Int32(24 ÷ n), Int32(640 ÷ n), Int32(hcro(n)), Int32(MAXCTX[]),
                    Int32(ag.GS), Int32(gk), Int32(sk), Int32(ag.DH), Int32(ag.QH), Int32(ag.nkvl), Int32(ag.IW), Int32(0),
                    D.ll[:S] + 8 * s * 4nc, D.ll[:DP], D.ll[:LS], D.ll[:T], D.ll[:Z], D.ll[:RP],
                    D.ll[:QKV], D.ll[:GATE], D.ll[:AB], D.ll[:ON], D.ll[:YP],
                    D.ll[:QG], D.ll[:KK], D.ll[:VV], D.ll[:IX], D.ll[:QO], D.ll[:NRM], D.ll[:IXN], D.ll[:SC], D.ll[:AO], D.ll[:KVR],
                    D.ll[:GU], D.ll[:IDXW], D.ll[:H], D.ll[:OUTP],
                    D.ll[:PK], D.ll[:PV], D.ll[:PPT],
                    reinterpret(U64P, hp + 8 * s * 2560), reinterpret(U64P, hp + 8 * (4 + s) * 2560), rt.rope,
                    D.red, D.cnt, PROF[] ? reinterpret(Core.LLVMPtr{UInt64,1}, D.tim) : Core.LLVMPtr{UInt64,1}(0), D.ctr,
                    D.ll[:PLD] + 8 * s * 2560, reinterpret(Core.LLVMPtr{UInt32,1}, D.ll[:ABW]),
                    llreg(rt.ord)[1], reinterpret(Core.LLVMPtr{UInt64,1}, pointer(pt)), D.ll[:DPS], D.ll[:YPS],
                    D.ll[:QF], D.ll[:QN], D.ll[:KF], D.ll[:VF], D.ll[:KN], D.ll[:ZW], D.ll[:HW], D.ll[:YW], D.ll[:DSUM], D.ll[:DAG],
                    D.ll[:YCB], D.ll[:DCB], Int32(0), Int32(0), D.ll[:UIDX]))
                # the handoff of the last block's S to the next board (same channels; a single die takes all)
                nxt = gi < length(groups) ? groups[gi+1] : nothing
                for (i, B) in enumerate(rt.blks)
                    hand = U64P(0); hls = 0; hoff = 0
                    if i == length(rt.blks) && nxt === nothing && n > 1 && !inhead(E0)
                        # the last board of two: the final S to the collapse's die whole
                        cdd = only(D2 for DG2 in groups for D2 in DG2 if D2.rt.ord == collapse_die(E0))
                        hls = 2560; hoff = nc * rt.d; hand = cdd.ll[:SF] + 8 * s * 4hls
                    elseif i == length(rt.blks) && nxt !== nothing
                        if length(nxt) == n
                            hls = nc; hoff = 0; hand = nxt[rt.d+1].ll[:S] + 8 * s * 4hls
                        else
                            length(nxt) == 1 || error("handoff from $n dies to $(length(nxt))")
                            hls = 2560; hoff = nc * rt.d; hand = nxt[1].ll[:S] + 8 * s * 4hls
                        end
                    end
                    push!(bds, make_bd(B, rt.st[i], rt.pler, hand, hls, hoff, 1 + (i - 1) * NSTB, tapof(B.l)))
                end
                hascoll(rt) && push!(bds, make_hbd(rt.head::DHead, 1 + length(rt.blks) * NSTB))
            end
            D.dd = CuArray(dds); D.bd = CuArray(bds)
        end
    end
    E = EngineB1(g, E0.tk, E0.ple, groups, Int32[], host, hp, hp + 4 * 2560 * 8, 0, Dict{Int,Any}(), 0, tapbuf, nothing, nothing)
    LASTB1[] = E
    t0 = time()
    for DG in groups, D in DG
        device!(D.rt.ord)
        k = getkernel(k_token!, KTOKEN_TT; maxregs = B1MAXREGS[], maxthreads = B1THREADS[])
        E.kern[D.rt.ord] = k
        occ = CUDA.active_blocks(k.fun, B1THREADS[]; shmem = b1shmem(D))
        D.grid = occ * NSM[]; E.grid = max(E.grid, D.grid)
        note(@sprintf("die %d: k_token! %d registers, %d CTAs/SM resident (x buffer %d floats)", D.rt.ord, CUDA.registers(k), occ, D.xsm))
    end
    note(@sprintf("k_token! compiled in %.1f s, grid %d x %d", time() - t0, E.grid, B1THREADS[]))
    check_combos(E)
    DFAST && !W2H && error("FLASHNEXT_DFAST needs the experts' h as W2 (FLASHNEXT_W2 with h)")
    HCAG && W2T && error("FLASHNEXT_W2 t keeps t in HCAG's buffer: not with FLASHNEXT_HCAG")
    cd = collapse_die(E0)
    if inhead(E0)
        delete!(KCACHE, :collapse)
        note("the final collapse in k_token!, split over the last group's $(length(E.groups[end])) dies")
    elseif cd != E.groups[end][end].rt.ord
        D = only(D for DG in E.groups for D in DG if D.rt.ord == cd)
        KCACHE[:collapse] = (D, collapse_head(g, cd))
        note("collapse of the final S on die $cd")
    else
        delete!(KCACHE, :collapse)
    end
    E
end
llword(sq::UInt32, v::Float32) = (UInt64(sq) << 32) | reinterpret(UInt32, v)
# defer: dies whose kernel is not launched now (launch_die! later, same seq0 and t)
function launch_token!(E::EngineB1, id::Integer; defer = Int[])
    t = length(E.ids)
    push!(E.ids, Int32(id))
    E.ntok += 1
    seq0 = UInt32(2048 * E.ntok)
    PROF[] && for DG in E.groups, D in DG
        ccall(:memset, Ptr{Cvoid}, (Ptr{UInt64}, Cint, Csize_t), D.tim, 0, 8 * 64 * NTIM)
    end
    # the kernels first: they wait for the embedding and PLE words (seq0 + 1, + 2) in host memory, written after
    ta = time()
    for DG in E.groups, D in DG
        D.nblk == 0 && continue                # a head-only die: its S is the handoff of the board before
        D.rt.ord in defer && continue
        device!(D.rt.ord)
        launch_die!(E, D, seq0, t)
    end
    tb = time()
    s = t & 3
    emb = embed(E.g, id)
    for i in 1:2560
        unsafe_store!(E.emb, llword(seq0 + UInt32(1), emb[i]), s * 2560 + i)
    end
    tc = time()
    if any(DG -> any(D -> any(B -> B.l == 1, D.rt.blks), DG), E.groups)
        gi = ple_input(E.ple, E.ids, t)
        for i in 1:2560
            unsafe_store!(E.pleg, llword(seq0 + UInt32(2), gi[i]), s * 2560 + i)
        end
    end
    STEPT[:kernel_launch] = get(STEPT, :kernel_launch, 0.0) + tb - ta; STEPT[:embed] = get(STEPT, :embed, 0.0) + tc - tb
    STEPT[:ple] = get(STEPT, :ple, 0.0) + time() - tc
    seq0, t
end
# die D's kernel for position t: the DD and BDs of slot t & 3
function launch_die!(E::EngineB1, D::B1Die, seq0::UInt32, t::Integer)
    device!(D.rt.ord); s = t & 3
    abort_settle!(D)
    (E.kern[D.rt.ord]::CUC.HostKernel{typeof(k_token!),KTOKEN_TT})(reinterpret(Core.LLVMPtr{DD,1}, pointer(D.dd::CuVector{DD})) + s * sizeof(DD),
        reinterpret(Core.LLVMPtr{BD,1}, pointer(D.bd::CuVector{BD})) + s * D.nblk * sizeof(BD),
        Int32(D.nblk), seq0, Int32(t), Int32(D.xsm), PROF[] ? reinterpret(Core.LLVMPtr{UInt64,1}, D.tim) : Core.LLVMPtr{UInt64,1}(0);
        threads = B1THREADS[], blocks = D.grid, shmem = b1shmem(D))
end
# k_token!'s dynamic shared memory: the abort header (16 bytes), then the x buffer
b1shmem(D) = 16 + 4 * D.xsm
# the last block of the placement (the last board may hold only the head: its S is the handoff of the board before)
lastblock(E::EngineB1) = maximum(B.l for DG in E.groups for D in DG for B in D.rt.blks)
# the final S of the last board (plain floats on the host)
function final_S(E::EngineB1, seq0, t)
    PROF[] ? watchdog(E) : SERVING[] && hangguard(E)
    DG = E.groups[end]
    lastl = lastblock(E)
    S = Vector{Float32}(undef, 10240)
    for D in DG
        rt = D.rt; device!(rt.ord); nc = 2560 ÷ rt.n
        launch(k_ll2f!, D.S32, D.ll[:S] + 8 * (t & 3) * 4nc, Int32(4nc), sqof(seq0, lastl, 16); threads = 256, blocks = cld(4nc, 256))
        part = dl(D.S32, 4nc)
        for l in 0:3
            S[l*2560+nc*rt.d+1:l*2560+nc*rt.d+nc] = part[l*nc+1:l*nc+nc]
        end
    end
    S
end
# host time of each part of a step (s, summed over steps): launch (inputs and kernels), kernels (until S is on the host), head
const STEPT = Dict{Symbol,Float64}(:launch => 0.0, :kernels => 0.0, :head => 0.0, :n => 0.0, :kernel_launch => 0.0, :embed => 0.0, :ple => 0.0)
function step!(E::EngineB1, id::Integer; logits = true)
    t0 = time()
    seq0, t = launch_token!(E, id)
    t1 = time()
    if logits && inhead(E)
        head_rows!(E, seq0; am = false)
        PROF[] ? watchdog(E) : SERVING[] && hangguard(E)
        r = Vector{Float32}(undef, 248320)
        for DG in E.groups, D in DG
            rt = D.rt; rt.head === nothing && continue
            device!(rt.ord); nr = Int(rt.head.out.rows)
            r[rt.head.r0+1:rt.head.r0+nr] = dl(rt.b[:logits], nr)
        end
        PROF[] && E.ntok > 1 && profile_acc!(E)
        STEPT[:launch] += t1 - t0; STEPT[:kernels] += time() - t1; STEPT[:n] += 1
        return r
    end
    S = final_S(E, seq0, t)
    t2 = time()
    PROF[] && E.ntok > 1 && profile_acc!(E)
    r = logits && E.groups[end][end].rt.head !== nothing ? head_logits([D.rt for DG in E.groups for D in DG], S) : S
    STEPT[:launch] += t1 - t0; STEPT[:kernels] += t2 - t1; STEPT[:head] += time() - t2; STEPT[:n] += 1
    r
end

# Greedy decoding without host round trips: the head is queued behind each die's token kernel. The last die (one die
# holding the collapse) turns its LL S into hz and writes hz as LL words (seq0 + 3) into every die; each die converts
# them, computes its rows of the logits and their argmax (value, row); the host compares 7 pairs.
function k_put_ll!(d::NTuple{8,U64P}, nd::Int32, x::FPtr, n::Int32, sq::UInt32)
    i = gtid()
    while i < n
        v = ldf(x, i)
        for k in 1:8
            k <= nd && llput(d[k], i, sq, v)
        end
        i += gstride()
    end
    return
end
# argmax of x[0:n) (the lowest index among equal maxima), its value and r0 + index into out[0:2); one CTA of 1024 threads
function k_argmax!(out::FPtr, x::FPtr, n::Int32, r0::Int32)
    tid = threadIdx().x - Int32(1); lane = tid % Int32(32); warp = tid ÷ Int32(32)
    sv = CuStaticSharedArray(Float32, 32); si = CuStaticSharedArray(Int32, 32)
    bv = -Inf32; bi = Int32(typemax(Int32))
    i = tid
    @inbounds while i < n
        v = ldf(x, i)
        v > bv && (bv = v; bi = i)
        i += blockDim().x
    end
    @inbounds begin
        for o in (16, 8, 4, 2, 1)
            ov = shfl_xor_sync(FULL, bv, o); oi = shfl_xor_sync(FULL, bi, o)
            (ov > bv || (ov == bv && oi < bi)) && (bv = ov; bi = oi)
        end
        lane == Int32(0) && (sv[warp+1] = bv; si[warp+1] = bi)
        sync_threads()
        if warp == Int32(0)
            bv = lane < blockDim().x ÷ Int32(32) ? sv[lane+1] : -Inf32; bi = lane < blockDim().x ÷ Int32(32) ? si[lane+1] : Int32(typemax(Int32))
            for o in (16, 8, 4, 2, 1)
                ov = shfl_xor_sync(FULL, bv, o); oi = shfl_xor_sync(FULL, bi, o)
                (ov > bv || (ov == bv && oi < bi)) && (bv = ov; bi = oi)
            end
            lane == Int32(0) && (stf(out, Int32(0), bv); stf(out, Int32(1), reinterpret(Float32, bi + r0)))
        end
    end
    return
end
# the full final S (LL, 10240 words) of position t on the head's die c: its own S on a last board of one die, else the
# handoff both dies write (SF)
fullS(c::B1Die, t) = (haskey(c.ll, :SF) ? c.ll[:SF] : c.ll[:S]) + 8 * (t & 3) * 10240
# the head after k_token!'s collapse: hz (Z[0] of block 48, LL) on each head die, its rows of the logits (and their
# argmax: am)
function head_rows!(E::EngineB1, seq0::UInt32; am = true, slot = 0)
    for DG in E.groups, D in DG
        rt = D.rt; rt.head === nothing && continue
        device!(rt.ord); bb = rt.b
        W2Z ? launch(k_w2f!, bb[:hz], D.ll[:ZW], Int32(2560), sqof(seq0, 48, 5); threads = 256, blocks = 10) :
             launch(k_ll2f!, bb[:hz], D.ll[:Z], Int32(2560), sqof(seq0, 48, 5); threads = 256, blocks = 10)
        gemv!(bb[:logits], rt.head.out, bb[:hz])
        am && launch(k_argmax!, bb[:am] + 8slot, bb[:logits], rt.head.out.rows, Int32(rt.head.r0); threads = 1024, blocks = 1)
    end
end
# the best (value, row) over the head dies' argmax slots 0 .. P-1 (after head_rows!)
function head_argmax(E::EngineB1, P::Int)
    outs = fill(Int32(0), P); best = fill(-Inf32, P)
    for DG in E.groups, D in DG
        D.rt.head === nothing && continue
        device!(D.rt.ord); r = dl(D.rt.b[:am], 2P)
        for p in 1:P
            v = r[2p-1]; i = reinterpret(Int32, r[2p])
            (v > best[p] || (v == best[p] && i < outs[p])) && (best[p] = v; outs[p] = i)
        end
    end
    outs
end
# B1 over one group with the collapse in k_token! (inhead): a round's positions one kernel each on every die, each
# position's head behind its kernel (Z of block 48 is the next position's too)
function verify_round_inhead!(E::EngineB1, toks::Vector{Int32}; early = nothing)
    B8ON && early === nothing && length(E.groups) == 1 && return round8!(engine_b8(E), toks)
    # (else one B1 token at a time: the positions share each group's LL buffers, and block 0 of position k + 1 writes
    # words (ZW at q 0, LS, ...) that block 48 or the head of position k may still read on a slower die, so every die
    # finishes position k before k + 1 is launched: hangguard waits for all of them)
    for (k, id) in enumerate(toks)
        sq = k == 1 && early !== nothing ? finish_anchor!(E, early) : first(launch_token!(E, id))
        head_rows!(E, sq; slot = k - 1)
        hangguard(E)
    end
    head_argmax(E, length(toks))
end
# the draft's stream waits for every die's work queued so far (its taps of the accepted positions are written by all
# dies, not only those of the last block: an event on each die, no host wait)
function draft_wait_dies!(E::EngineB1)
    evs = CuEvent[]
    for DG in E.groups, D in DG
        device!(D.rt.ord); e = pipe_ev(D.rt.ord, :dw, 0); CUDA.record(e, stream()); push!(evs, e)
    end
    device!(DRAFT_DIE)
    for e in evs
        CUC.cuStreamWaitEvent(stream(), e, 0)
    end
end
# ---- the B8 engine (host): one k_token8! per die a round of P <= P8MAX positions; its own LL region of P8MAX position
# copies (B1's layout), host input slots, MoE reduction buffers and phase counters; B1's weights, recurrent state and
# head slices (FLASHNEXT_B8=0: the round as P B1 tokens) ----
const B8ON = get(ENV, "FLASHNEXT_B8", "1") == "1"
mutable struct EngineB8
    E::EngineB1
    dies::Vector{B1Die}
    dd::Dict{Int,Any}; ll8::Dict{Int,Dict{Symbol,U64P}}; RW::Int
    keep::Vector{Any}
    emb::Ptr{UInt64}; pleg::Ptr{UInt64}
    grid::Dict{Int,Int}; xsm::Int
    tim::Dict{Int,Ptr{UInt64}}        # CTA 0's op-end stamps (host-mapped, FLASHNEXT_B8PROF=1): [16 block + op]
end
engine_b8(E::EngineB1) = get!(() -> load_b8!(E), KCACHE, :b8)::EngineB8
function load_b8!(E::EngineB1)
    t0 = time()
    (ROUTERCOPY && KVCOPY && !ATTNCOPY && length(E.groups) == 1 && inhead(E)) ||
        error("B8: one group with the collapse in k_token!, ROUTERCOPY and KVCOPY (not ATTNCOPY)")
    dies = E.groups[1]
    keep = Any[]
    host = CUDA.alloc(CUDA.HostMemory, 2 * P8MAX * 2560 * 8, CUDA.MEMHOSTALLOC_DEVICEMAP | CUDA.MEMHOSTALLOC_PORTABLE); push!(keep, host)
    hp = convert(Ptr{UInt64}, host); ccall(:memset, Ptr{Cvoid}, (Ptr{UInt64}, Cint, Csize_t), hp, 0, 2 * P8MAX * 2560 * 8)
    szs = llsizes(dies[1].rt.n, MAXCTX[]); RW = sum(last, szs)
    ll8 = Dict{Int,Dict{Symbol,U64P}}(); bases = UInt64[]
    for D in dies
        device!(D.rt.ord)
        reg = CUDA.zeros(UInt64, P8MAX * RW); push!(keep, reg)
        ll = Dict{Symbol,U64P}(); off = 0
        for (k, sz) in szs
            ll[k] = reinterpret(U64P, pointer(reg)) + 8 * off; off += sz
        end
        ll8[D.rt.ord] = ll; push!(bases, reinterpret(UInt64, pointer(reg)))
    end
    xsm = XSM8 + SC8                   # (floats: the x, then the LUT and the union tables)
    dd8 = Dict{Int,Any}(); grid = Dict{Int,Int}()
    for D in dies
        rt = D.rt; device!(rt.ord); ll = ll8[rt.ord]
        pt = CuArray(bases); push!(keep, pt)
        n = rt.n; nc = 2560 ÷ n; ag = attn_geom(n); gk, sk = attn_gs(n, rt.d)
        red = CUDA.zeros(Float32, 81 * P8MAX * 2nc); cnt = CUDA.zeros(Int32, 80); ctr = CUDA.zeros(Int32, CT8 * D.nblk); push!(keep, red, cnt, ctr)   # (2 nc: DFAST; CT8 counters a block: ctr8of)
        dd = DD(Int32(n), Int32(rt.d), Int32(nc), Int32(16 ÷ n), Int32(24 ÷ n), Int32(640 ÷ n), Int32(hcro(n)), Int32(MAXCTX[]),
            Int32(ag.GS), Int32(gk), Int32(sk), Int32(ag.DH), Int32(ag.QH), Int32(ag.nkvl), Int32(ag.IW), Int32(0),
            ll[:S], ll[:DP], ll[:LS], ll[:T], ll[:Z], ll[:RP],
            ll[:QKV], ll[:GATE], ll[:AB], ll[:ON], ll[:YP],
            ll[:QG], ll[:KK], ll[:VV], ll[:IX], ll[:QO], ll[:NRM], ll[:IXN], ll[:SC], ll[:AO], ll[:KVR],
            ll[:GU], ll[:IDXW], ll[:H], ll[:OUTP],
            ll[:PK], ll[:PV], ll[:PPT],
            reinterpret(U64P, hp), reinterpret(U64P, hp + 8 * P8MAX * 2560), rt.rope,
            fptr(red), reinterpret(IPtr, pointer(cnt)), Core.LLVMPtr{UInt64,1}(0), reinterpret(IPtr, pointer(ctr)),
            U64P(0), Core.LLVMPtr{UInt32,1}(0),
            reinterpret(U64P, bases[rt.d+1]), reinterpret(Core.LLVMPtr{UInt64,1}, pointer(pt)), ll[:DPS], ll[:YPS],
            ll[:QF], ll[:QN], ll[:KF], ll[:VF], ll[:KN], ll[:ZW], ll[:HW], ll[:YW], ll[:DSUM], ll[:DAG],
            ll[:YCB], ll[:DCB], Int32(RW), Int32(0), ll[:UIDX])
        a = CuArray([dd]); push!(keep, a); dd8[rt.ord] = a
        k = getkernel(k_token8!, KTOKEN8_TT; maxregs = 128, maxthreads = B4THREADS)
        occ = CUDA.active_blocks(k.fun, B4THREADS; shmem = 4 * xsm)
        mem = CUDA.memory(k)
        occ >= 1 || error("k_token8!: no CTA of $B4THREADS threads fits on die $(rt.ord): $(CUDA.registers(k)) registers, $(mem.shared) B static + $(4 * xsm) B dynamic shared, $(mem.local) B local")
        grid[rt.ord] = occ * NSM[]
        note(@sprintf("B8: die %d k_token8! %d registers, %d CTA(s)/SM, %d B static shared", rt.ord, CUDA.registers(k), occ, mem.shared))
    end
    note(@sprintf("B8: engine on %d dies in %.1f s (%.1f MB of LL words a die)", length(dies), time() - t0, P8MAX * RW * 8 / 1e6))
    tim = Dict{Int,Ptr{UInt64}}()
    for D in dies
        device!(D.rt.ord); tm = CUDA.alloc(CUDA.HostMemory, 8 * 16 * (D.nblk + 2), CUDA.MEMHOSTALLOC_DEVICEMAP | CUDA.MEMHOSTALLOC_PORTABLE); push!(keep, tm)
        tim[D.rt.ord] = convert(Ptr{UInt64}, tm)
    end
    EngineB8(E, dies, dd8, ll8, RW, keep, hp, hp + 8 * P8MAX * 2560, grid, xsm, tim)
end
# one round: tokens toks (P <= 4) at positions t .. t + P - 1 (t = length(E.ids)), one k_token8! per die, then each
# position's head; the target's argmax after each position
function round8!(E8::EngineB8, toks::Vector{Int32})
    E = E8.E; P = length(toks); t = length(E.ids)
    1 <= P <= P8MAX || error("round8!: 1 to $P8MAX positions")
    append!(E.ids, toks)
    E.ntok += 1; seq0 = UInt32(2048 * E.ntok)
    prof = get(ENV, "FLASHNEXT_B8PROF", "0") == "1"
    for D in E8.dies
        o = D.rt.ord; device!(o)
        k = getkernel(k_token8!, KTOKEN8_TT; maxregs = 128, maxthreads = B4THREADS)
        prof && ccall(:memset, Ptr{Cvoid}, (Ptr{UInt64}, Cint, Csize_t), E8.tim[o], 0, 8 * 16 * (D.nblk + 2))
        k(reinterpret(Core.LLVMPtr{DD,1}, pointer(E8.dd[o]::CuVector{DD})), reinterpret(Core.LLVMPtr{BD,1}, pointer(D.bd::CuVector{BD})),
            Int32(D.nblk), seq0, Int32(t), Int32(P), Int32(E8.xsm), prof ? reinterpret(Core.LLVMPtr{UInt64,1}, E8.tim[o]) : Core.LLVMPtr{UInt64,1}(0);
            threads = B4THREADS, blocks = E8.grid[o], shmem = 4 * E8.xsm)
    end
    hasple = any(DG -> any(D -> any(B -> B.l == 1, D.rt.blks), DG), E.groups)
    for p in 1:P
        emb = embed(E.g, toks[p])
        for i in 1:2560
            unsafe_store!(E8.emb, llword(seq0 + UInt32(1), emb[i]), (p - 1) * 2560 + i)
        end
        if hasple
            gi = ple_input(E.ple, E.ids, t + p - 1)
            for i in 1:2560
                unsafe_store!(E8.pleg, llword(seq0 + UInt32(2), gi[i]), (p - 1) * 2560 + i)
            end
        end
    end
    for p in 1:P, D in E8.dies
        rt = D.rt; rt.head === nothing && continue
        device!(rt.ord); bb = rt.b; ll = E8.ll8[rt.ord]
        W2Z ? launch(k_w2f!, bb[:hz], ll[:ZW] + 8 * (p - 1) * E8.RW, Int32(2560), sqof(seq0, 48, 5); threads = 256, blocks = 10) :
              launch(k_ll2f!, bb[:hz], ll[:Z] + 8 * (p - 1) * E8.RW, Int32(2560), sqof(seq0, 48, 5); threads = 256, blocks = 10)
        gemv!(bb[:logits], rt.head.out, bb[:hz])
        launch(k_argmax!, bb[:am] + 8 * (p - 1), bb[:logits], rt.head.out.rows, Int32(rt.head.r0); threads = 1024, blocks = 1)
    end
    SERVING[] && hangguard(E)
    r = head_argmax(E, P)
    prof && b8_acc!(E8, P)
    r
end
# FLASHNEXT_B8PROF=1: each op's time of a round on dies 0 and 6 (CTA 0's stamps, ms summed over the blocks), by P
const B8ACC = Dict{Tuple{Int,Int,Int},Float64}(); const B8N = Dict{Int,Int}()
function b8_acc!(E8::EngineB8, P::Int)
    for D in E8.dies
        D.rt.ord in (0, 6) || continue
        st = unsafe_wrap(Array, E8.tim[D.rt.ord], 16 * (D.nblk + 2))
        for bi in 1:length(D.rt.blks), k in 1:14
            a = st[16bi+k]; b = st[16bi+k+1]
            (a > 0 && b > 0 && b >= a) || continue
            B8ACC[(D.rt.ord, P, k)] = get(B8ACC, (D.rt.ord, P, k), 0.0) + (b - a) / 1e6
        end
    end
    B8N[P] = get(B8N, P, 0) + 1
end
function b8_report()
    for P in sort(collect(keys(B8N)))
        n = B8N[P]
        println(@sprintf("B8 rounds of %d position(s): %d; ms a round by op (die 0 / die 6):", P, n))
        for k in 1:14
            @printf("  %-16s %7.2f / %7.2f\n", B8OPS[k], get(B8ACC, (0, P, k), 0.0) / n, get(B8ACC, (6, P, k), 0.0) / n)
        end
    end
    empty!(B8ACC); empty!(B8N)
end
# --b8diff K: B1 and B8 (one position) on the prompt's first token over the first K blocks only (no head), then each
# die's LL buffers of block K - 1 compared bit for bit (the values: LL words' low 32 bits, W2 words' low 2 W2B bits;
# the seqs differ). Both start from the reset state and recompute position 0's states identically if they agree.
function b8diff(E::EngineB1, ids::Vector{Int32}, K::Int)
    E8 = engine_b8(E)
    dies = E8.dies
    words(p::U64P, o, n) = (w = Vector{UInt64}(undef, n); GC.@preserve w CUC.cuMemcpyDtoH_v2(pointer(w), reinterpret(CUC.CUdeviceptr, p + 8o), 8n); w)
    feed!(emb, pleg, seq0) = begin
        e = embed(E.g, ids[1])
        for i in 1:2560
            unsafe_store!(emb, llword(seq0 + UInt32(1), e[i]), i)
        end
        gi = ple_input(E.ple, Int32[ids[1]], 0)
        for i in 1:2560
            unsafe_store!(pleg, llword(seq0 + UInt32(2), gi[i]), i)
        end
    end
    n = dies[1].rt.n; nc = 2560 ÷ n; nkh = 16 ÷ n; F = 640 ÷ n; RO = hcro(n)
    l = K - 1; par = l & 1; qa = par; qf = 2 + par; yn = n ÷ 2 + 1
    # block l's slices (parity par; hc buffers by half q), in the block's order; (name, buffer, offset, words, kind)
    bufs = [("LSa", :LS, qa * n * 32, n * 32, :ll), ("DPSa", :DPS, qa * 1296, 1296, :ll), ("DPa", :DP, qa * n * RO, n * RO, :ll),
            ("Ta", :T, qa * 324, 324, :ll), ("ZWa", :ZW, qa * 1280, 1280, :w2), ("QKV", :QKV, par * 640nkh, 640nkh, :ll),
            ("ON", :ON, par * 384nkh, 384nkh, :ll), ("YPS", :YPS, par * 2560, 2560, :ll), ("YW", :YW, par * (yn * nc ÷ 2), yn * nc ÷ 2, :w2),
            ("LSf", :LS, qf * n * 32, n * 32, :ll), ("DPSf", :DPS, qf * 1296, 1296, :ll), ("DPf", :DP, qf * n * RO, n * RO, :ll),
            ("Tf", :T, qf * 324, 324, :ll), ("ZWf", :ZW, qf * 1280, 1280, :w2), ("RP", :RP, par * RPW, 513, :ll),
            ("GU", :GU, par * 22F, 22F, :ll), ("H", :H, par * 7040, 7040, :ll), ("HW", :HW, par * 3520, 3520, :w2),
            ("OUTP", :OUTP, par * nc, nc, :ll), ("S", :S, 0, 4nc, :ll)]
    vmask(kind) = kind == :w2 ? (UInt64(1) << (2W2B)) - UInt64(1) : UInt64(0xffffffff)
    reset_state!(E)
    E.ntok += 1; seq0 = UInt32(2048 * E.ntok)
    for D in dies
        device!(D.rt.ord)
        (E.kern[D.rt.ord]::CUC.HostKernel{typeof(k_token!),KTOKEN_TT})(reinterpret(Core.LLVMPtr{DD,1}, pointer(D.dd::CuVector{DD})),
            reinterpret(Core.LLVMPtr{BD,1}, pointer(D.bd::CuVector{BD})), Int32(K), seq0, Int32(0), Int32(D.xsm), Core.LLVMPtr{UInt64,1}(0);
            threads = B1THREADS[], blocks = D.grid, shmem = b1shmem(D))
    end
    feed!(E.emb, E.pleg, seq0)
    for D in dies
        device!(D.rt.ord); synchronize()
    end
    b1 = Dict((D.rt.ord, nm) => words(D.ll[bf], o, w) for D in dies for (nm, bf, o, w, _) in bufs if haskey(D.ll, bf))
    reset_state!(E)
    E.ntok += 1; seq0 = UInt32(2048 * E.ntok)
    for D in dies
        o = D.rt.ord; device!(o)
        k = getkernel(k_token8!, KTOKEN8_TT; maxregs = 128, maxthreads = B4THREADS)
        k(reinterpret(Core.LLVMPtr{DD,1}, pointer(E8.dd[o]::CuVector{DD})), reinterpret(Core.LLVMPtr{BD,1}, pointer(D.bd::CuVector{BD})),
            Int32(K), seq0, Int32(0), Int32(1), Int32(E8.xsm), Core.LLVMPtr{UInt64,1}(0); threads = B4THREADS, blocks = E8.grid[o], shmem = 4 * E8.xsm)
    end
    feed!(E8.emb, E8.pleg, seq0)
    for D in dies
        device!(D.rt.ord); synchronize()
    end
    # B8 once more on the same input: does it reproduce its own router logits and weights (a race would not)
    if get(ENV, "FLASHNEXT_B8DIFF_TWICE", "0") == "1"
        o0 = dies[1].rt.ord
        u1 = words(E8.ll8[o0][:UIDX], UIDXW * par, UIDXW); rp1 = words(E8.ll8[o0][:RP], par * RPW, 513)
        reset_state!(E); E.ntok += 1; seq0b = UInt32(2048 * E.ntok)
        for D in dies
            o = D.rt.ord; device!(o)
            k = getkernel(k_token8!, KTOKEN8_TT; maxregs = 128, maxthreads = B4THREADS)
            k(reinterpret(Core.LLVMPtr{DD,1}, pointer(E8.dd[o]::CuVector{DD})), reinterpret(Core.LLVMPtr{BD,1}, pointer(D.bd::CuVector{BD})),
                Int32(K), seq0b, Int32(0), Int32(1), Int32(E8.xsm), Core.LLVMPtr{UInt64,1}(0); threads = B4THREADS, blocks = E8.grid[o], shmem = 4 * E8.xsm)
        end
        feed!(E8.emb, E8.pleg, seq0b)
        for D in dies
            device!(D.rt.ord); synchronize()
        end
        u2 = words(E8.ll8[o0][:UIDX], UIDXW * par, UIDXW); rp2 = words(E8.ll8[o0][:RP], par * RPW, 513)
        @printf("b8diff: B8 twice on die %d: RP values %s, UIDX values %s\n", o0, all(i -> rp1[i] % UInt32 == rp2[i] % UInt32, 1:513) ? "equal" : "DIFFER",
            all(i -> u1[i] % UInt32 == u2[i] % UInt32, 1:UIDXW) ? "equal" : "DIFFER")
    end
    @printf("b8diff: B1 against B8 (1 position), token %d, blocks 0-%d; words whose values differ (first index), by die:\n", ids[1], l)
    bad = 0
    for (nm, bf, o, w, kind) in bufs
        line = @sprintf("  %-5s", nm); any_ = false
        for D in dies
            haskey(D.ll, bf) || continue
            a = b1[(D.rt.ord, nm)]; b = words(E8.ll8[D.rt.ord][bf], o, w); m = vmask(kind)
            # (words neither kernel wrote this token: seq 0 or another block's; compared too, but zero in both)
            d = findall(i -> (a[i] & m) != (b[i] & m), eachindex(a))
            fv(x) = kind == :w2 ? (uf24((x >> W2B) & W2FM), uf24(x & W2FM)) : (reinterpret(Float32, x % UInt32),)
            va = collect(Iterators.flatten(fv.(a))); vb = collect(Iterators.flatten(fv.(b)))
            rel = maximum(abs.(va .- vb); init = 0.0f0) / max(maximum(abs, va; init = 0.0f0), 1.0f-30)
            line *= isempty(d) ? "   ." : @sprintf(" %4d@%d(%.0e)", length(d), d[1] - 1, rel)
            any_ |= !isempty(d); bad += length(d)
        end
        println(line, any_ ? "" : "  (equal)")
    end
    # the MoE down's per-expert terms (RED: B1 [slot][rows], B8 [slot][position][rows]; position 0's union order is
    # its rank order), on the dies that compute it
    nr = dfast(n) ? 2nc : nc
    line = "  RED  "
    for D in dies
        dfast(n) && D.rt.d >= n ÷ 2 && (line *= "   -"; continue)
        fl(p, cnt) = (w = Vector{Float32}(undef, cnt); GC.@preserve w CUC.cuMemcpyDtoH_v2(pointer(w), reinterpret(CUC.CUdeviceptr, p), 4cnt); w)
        a = fl(D.red, 11nr); b = fl(reinterpret(FPtr, E8.dd[D.rt.ord] isa CuArray ? Array(E8.dd[D.rt.ord])[1].RED : FPtr(0)), 81 * P8MAX * nr)
        nd = 0; first = -1
        for k in 0:10, r in 0:nr-1
            if reinterpret(UInt32, a[k*nr+r+1]) != reinterpret(UInt32, b[(P8MAX*k)*nr+r+1])
                nd += 1; first < 0 && (first = k * nr + r)
            end
        end
        line *= nd == 0 ? "   ." : @sprintf(" %4d@%d", nd, first)
        if D.rt.ord == dies[1].rt.ord && nd > 0      # (die 0: a few terms, and the experts' weights of both)
            println(@sprintf("  die %d RED[k][0] B1: ", D.rt.ord), join([@sprintf("%.9g", a[k*nr+1]) for k in 0:10], " "))
            println(@sprintf("  die %d RED[k][0] B8: ", D.rt.ord), join([@sprintf("%.9g", b[P8MAX*k*nr+1]) for k in 0:10], " "))
            w1 = words(D.ll[:IDXW], 24 * par, 21); w8 = words(E8.ll8[D.rt.ord][:UIDX], UIDXW * par, UIDXW)
            println("  B1 ids/weights: ", join([reinterpret(Int32, x % UInt32) for x in w1[1:10]], " "), " | ", join([@sprintf("%.9g", reinterpret(Float32, x % UInt32)) for x in w1[11:21]], " "))
            U8 = reinterpret(Int32, w8[1] % UInt32)
            println("  B8 U $U8 ids/weights: ", join([reinterpret(Int32, x % UInt32) for x in w8[2:1+U8]], " "), " | ",
                join([@sprintf("%.9g", reinterpret(Float32, w8[UW0+P8MAX*u+1] % UInt32)) for u in 0:U8], " "))
        end
    end
    println(line)
    reset_state!(E)
    println("b8diff: ", bad == 0 ? "PASS (bit-identical)" : "$bad words differ")
    bad == 0
end
function step_greedy!(E::EngineB1, id::Integer)
    dies = [D for DG in E.groups for D in DG]
    if inhead(E)
        t0 = time()
        seq0, t = launch_token!(E, id)
        t1 = time()
        head_rows!(E, seq0)
        PROF[] ? watchdog(E) : SERVING[] && hangguard(E)
        best = -Inf32; bi = Int32(0)
        for D in dies
            D.rt.head === nothing && continue
            device!(D.rt.ord); r = dl(D.rt.b[:am], 2)
            v = r[1]; i = reinterpret(Int32, r[2])
            (v > best || (v == best && i < bi)) && (best = v; bi = i)
        end
        PROF[] && E.ntok > 1 && profile_acc!(E)
        STEPT[:launch] += t1 - t0; STEPT[:kernels] += time() - t1; STEPT[:n] += 1
        return bi
    end
    c, H = collapse_target(E)
    ((length(E.groups[end]) == 1 || haskey(c.ll, :SF)) && H !== nothing && H.down.rows > 0) || return Int32(argmax(step!(E, id)) - 1)
    t0 = time()
    seq0, t = launch_token!(E, id)
    t1 = time()
    sqh = seq0 + UInt32(3)
    device!(c.rt.ord); b = c.rt.b
    launch(k_ll2f!, b[:Sfull], fullS(c, t), Int32(10240), sqof(seq0, lastblock(E), 16); threads = 256, blocks = 40)
    launch(k_rmsnorm!, b[:xh], b[:Sfull], H.norm, Int32(2560), Int32(2560), EPS; threads = 256, blocks = 4)
    gemv!(b[:down], H.down, b[:xh])
    launch(k_hc_silu!, b[:ht], b[:down], Int32(320); threads = 320, blocks = 1)
    gemv!(b[:hup], H.up, b[:ht])
    launch(k_hc_mix!, b[:hz], b[:hup], b[:xh]; threads = 256, blocks = 10)
    others = [D for D in dies if D !== c]          # hz to every other head die (LL)
    dst = ntuple(k -> k <= length(others) ? reinterpret(U64P, others[k].rt.b[:hzll]) : U64P(0), 8)
    launch(k_put_ll!, dst, Int32(length(others)), b[:hz], Int32(2560), sqh; threads = 256, blocks = 10)
    for D in dies
        rt = D.rt; device!(rt.ord); bb = rt.b
        D === c || launch(k_ll2f!, bb[:hz], reinterpret(U64P, bb[:hzll]), Int32(2560), sqh; threads = 256, blocks = 10)
        gemv!(bb[:logits], rt.head.out, bb[:hz])
        launch(k_argmax!, bb[:am], bb[:logits], rt.head.out.rows, Int32(rt.head.r0); threads = 1024, blocks = 1)
    end
    PROF[] ? watchdog(E) : SERVING[] && hangguard(E)
    best = -Inf32; bi = Int32(0)
    for D in dies
        device!(D.rt.ord); r = dl(D.rt.b[:am], 2)
        v = r[1]; i = reinterpret(Int32, r[2])
        (v > best || (v == best && i < bi)) && (best = v; bi = i)
    end
    STEPT[:launch] += t1 - t0; STEPT[:kernels] += time() - t1; STEPT[:n] += 1
    bi
end
# decode an LL buffer (n words from offset o) on the host
function lldl(p::U64P, o, n)
    w = Vector{UInt64}(undef, n)
    GC.@preserve w CUC.cuMemcpyDtoH_v2(pointer(w), reinterpret(CUC.CUdeviceptr, p + 8o), 8n)
    Float32[reinterpret(Float32, x % UInt32) for x in w], UInt32[(x >> 32) % UInt32 for x in w]
end
# one token through B1 and through B0 (same weights, the first group only), every intermediate compared
function debug_b1(E::EngineB1, id)
    D = E.groups[1][1]; rt = D.rt; G0 = [d.rt for d in E.groups[1]]
    seq0, t = launch_token!(E, id)
    for DG in E.groups, DD_ in DG
        device!(DD_.rt.ord); synchronize()
    end
    l = rt.blks[1].l; par = l & 1; nc = 2560 ÷ rt.n; nkh = 16 ÷ rt.n
    b1 = Dict{Symbol,Any}()
    b1[:dp] = lldl(D.ll[:DP], par * 2600, 1300 * rt.n); b1[:z] = lldl(D.ll[:Z], par * 2560, 2560)
    b1[:qkv] = lldl(D.ll[:QKV], par * 640nkh, 640nkh); b1[:on] = lldl(D.ll[:ON], par * 384nkh, 384nkh)
    b1[:yp] = lldl(D.ll[:YP], par * 2 * nc, nc * rt.n); b1[:inj] = lldl(D.ll[:INJ], par * 8, 8)
    # B0 on the same input (block 0 only)
    S = repeat(embed(E.g, id), 4)
    putS!(G0, S)
    block_fwd_b!(G0, 1, t)
    device!(rt.ord)
    cmp(nm, a, b) = @printf("  %-4s n=%5d  max|b0| %.3e  max|diff| %.3e  seq-mismatch %d  nan %d\n", nm, length(b), maximum(abs, b; init = 0.0f0),
        maximum(abs.(a[1][1:length(b)] .- b); init = 0.0f0), count(s -> s == 0, a[2]), count(isnan, a[1]))
    cmp("dp", b1[:dp], dl(rt.b[:pt], 1300 * rt.n))
    cmp("z", b1[:z], dl(rt.b[:z], 2560))
    cmp("qkv", b1[:qkv], dl(rt.b[:qkv], 640nkh))
    cmp("on", b1[:on], dl(rt.b[:on], 384nkh))
    println("  (yp and S of B0 are after the whole block; B1 is after its last op)")
    println("  seqs: dp ", unique(b1[:dp][2])[1:min(3, end)], " expected ", sqof(seq0, l, 4), "; z ", unique(b1[:z][2])[1:min(3, end)], " expected ", sqof(seq0, l, 5))
end
const PROF = Ref(false)
const LASTB1 = Ref{Any}(nothing)
# with --prof: wait for every die's kernel, and if one stops making progress (no new stamp for 10 s) report where
# without stamps (the server): every die's stream done within 10 s, else the kernels hang; the server then exits,
# as its dies stay busy (a new server starts clean)
const SERVING = Ref(false)
function hangguard(E)
    t = time()
    while !all(D -> (device!(D.rt.ord); CUDA.isdone(stream())), (D for DG in E.groups for D in DG))
        if time() - t > 10
            busy = [D.rt.ord for DG in E.groups for D in DG if (device!(D.rt.ord); !CUDA.isdone(stream()))]
            println(stderr, "flashnext: B1 kernels not done after 10 s (hang) on dies ", join(busy, ", "), SERVING[] ? "; the server exits" : "")
            flush(stderr)
            SERVING[] && ccall(:_exit, Cvoid, (Cint,), 3)
            error("B1 kernel hang")
        end
    end
end
function watchdog(E)
    last = UInt64(0); t = time()
    while true
        all(D -> (device!(D.rt.ord); CUDA.isdone(stream())), (D for DG in E.groups for D in DG)) && return
        m = maximum(maximum(unsafe_wrap(Array, D.tim, NTIM * 64)) for DG in E.groups for D in DG)
        m != last && (last = m; t = time())
        if time() - t > 10
            println(stderr, "flashnext: B1 kernels made no progress for 10 s; last stamps:")
            fault_report(E)
            error("B1 kernel hang")
        end
        sleep(0.001)
    end
end
# after a device fault or hang (--prof): the latest op each CTA stamped, from the host-mapped stamps
function fault_report(E)
    for DG in E.groups, D in DG
        st = reshape(copy(unsafe_wrap(Array, D.tim, NTIM * 64)), NTIM, 64)[:, 1:D.grid]
        last = Dict{String,Vector{Int}}()
        for c in 1:D.grid
            k = all(iszero, st[:, c]) ? nothing : argmax(st[:, c])
            nm = k === nothing ? "none" : k == 1 ? "kernel start" : begin
                bi, o = divrem(k - 2, NSTB)
                l = bi + 1 <= length(D.rt.blks) ? D.rt.blks[bi+1].l : 48      # (48: the final collapse)
                o < 14 ? "block $l after $(BLKOPS[o+1])" : "block $l at $(SUBST[findfirst(s -> s[1] == o, SUBST)][3])"
            end
            push!(get!(last, nm, Int[]), c - 1)
        end
        for (nm, cs) in sort(collect(last); by = x -> x[2][1])
            println(stderr, "  die ", D.rt.ord, ": CTAs ", join(cs, ","), ": ", nm)
        end
    end
end
const BLKOPS = ["ple/start", "hc_down(attn)", "hc_up+mix(attn)", "proj", "deltanet/attn_prep", "attn", "out_proj", "hc_write(attn)",
    "hc_down(ffn)", "hc_up+mix(ffn)", "router", "experts_gate_up", "experts_down", "hc_write(ffn)"]
# weight bytes one die reads in each op of a block
mbytes(m::DevMat, rows = m.rows) = m.rows == 0 ? 0 : Int(rows) * (Int(m.K) ÷ devE(m.t)) * (devP(m.t) + devH(m.t))
function opbytes(B::PBlk)
    F = 640 ÷ B.n
    Dict("hc_down(attn)" => mbytes(B.hcA_dj) + mbytes(B.hcA_dj2), "hc_up+mix(attn)" => mbytes(B.hcA_up),
        "proj" => B.attn ? mbytes(B.wq) + mbytes(B.wk) + mbytes(B.wv) + mbytes(B.idx_x) : mbytes(B.qkv) + mbytes(B.gate) + mbytes(B.ab),
        "out_proj" => B.attn ? mbytes(B.wo) : mbytes(B.ssm_out),
        "hc_down(ffn)" => mbytes(B.hcF_dj) + mbytes(B.hcF_dj2), "hc_up+mix(ffn)" => mbytes(B.hcF_up),
        "router" => mbytes(B.router) + mbytes(B.router_sh) + mbytes(B.sh_gate) + mbytes(B.sh_up), "experts_gate_up" => mbytes(B.gu, 20F),
        "experts_down" => mbytes(B.down, 10 * (2560 ÷ B.n)) + mbytes(B.sh_down))
end
# critical-path time of every op from the per-CTA stamps: op k ends when its last CTA does. Also the weight bytes behind
# it (GB/s against the critical path) and the spread of the CTAs' end times. Accumulated over the tokens after the first
# (profile_acc!, from step! with --prof); profile_b1 prints the means.
const PACC = Dict{Symbol,Any}()
addto!(a::Dict, b::Dict) = (for (k, v) in b; a[k] = get(a, k, 0.0) + v; end; a)
function profile_acc!(E::EngineB1)
    tot = Dict{String,Float64}(); byt = Dict{String,Float64}(); spr = Dict{String,Float64}()
    subm = Dict{String,Float64}(); subx = Dict{String,Float64}(); die = Dict{Int,Float64}(); wait = Dict{Int,Float64}()
    for DG in E.groups, D in DG
        st = reshape(copy(unsafe_wrap(Array, D.tim, NTIM * 64)), NTIM, 64)[:, 1:D.grid]
        T0 = minimum(st[1, :])
        prev = T0; first_wait = 0.0
        for bi in 1:length(D.rt.blks)
            ob = opbytes(D.rt.blks[bi])
            # sub-op points: time since the CTA's own start of the op (mean and max over CTAs)
            for (o, j, nm) in SUBST
                a = st[1+(bi-1)*NSTB+j, :]; b = st[1+(bi-1)*NSTB+o+1, :]
                all(iszero, a) && (a = st[1+(bi-1)*NSTB+j-1, :])     # DeltaNet blocks skip the attention stamp
                ok = (a .> 0) .& (b .> a)
                any(ok) || continue
                dts = (b[ok] .- a[ok]) ./ 1e3
                subm[nm] = get(subm, nm, 0.0) + sum(dts) / length(dts); subx[nm] = get(subx, nm, 0.0) + maximum(dts)
            end
            for j in 0:13
                k = 1 + (bi - 1) * NSTB + j
                col = st[k+1, :]
                all(iszero, col) && continue
                Tk = maximum(col)
                dt = (Tk - prev) / 1e3
                bi == 1 && j == 0 && (first_wait = dt)
                nm = BLKOPS[j+1]
                tot[nm] = get(tot, nm, 0.0) + dt
                byt[nm] = get(byt, nm, 0.0) + get(ob, nm, 0)
                spr[nm] = get(spr, nm, 0.0) + (Tk - minimum(col)) / 1e3
                prev = Tk
            end
        end
        die[D.rt.ord] = (prev - T0) / 1e6; wait[D.rt.ord] = first_wait / 1e3
    end
    get!(PACC, :n, 0); PACC[:n] += 1
    for (k, v) in ((:tot, tot), (:byt, byt), (:spr, spr), (:subm, subm), (:subx, subx), (:die, die), (:wait, wait))
        addto!(get!(PACC, k, Dict{Any,Float64}()), v)
    end
end
# one block of the first group, last token: per op and die, when its CTAs ended (min-max, us from the block's start on
# die 0), and when they had their x (the "x ready" sub-stamps, max)
function block_timeline(E::EngineB1, bi::Int)
    DG = E.groups[1]
    sts = [reshape(copy(unsafe_wrap(Array, D.tim, NTIM * 64)), NTIM, 64)[:, 1:D.grid] for D in DG]
    t0 = minimum(sts[1][1+(bi-1)*NSTB+1, :])
    xr = Dict(j => o for (o, j, nm) in SUBST if occursin("x ready", nm))
    @printf("block %d timeline (us from its start on die %d):\n", DG[1].rt.blks[bi].l, DG[1].rt.ord)
    for j in 1:13
        line = @sprintf("  %-20s", BLKOPS[j+1])
        for (D, st) in zip(DG, sts)
            col = st[1+(bi-1)*NSTB+j+1, :]
            all(iszero, col) && (line *= "          -          "; continue)
            xc = haskey(xr, j) ? st[1+(bi-1)*NSTB+xr[j]+1, :] : UInt64[0]
            x = maximum(xc); xm = minimum(xc)
            line *= @sprintf("  die %d end %7.1f-%7.1f x %7.1f-%7.1f", D.rt.ord, (Int(minimum(col)) - Int(t0)) / 1e3, (Int(maximum(col)) - Int(t0)) / 1e3,
                xm == 0 ? NaN : (Int(xm) - Int(t0)) / 1e3, x == 0 ? NaN : (Int(x) - Int(t0)) / 1e3)
        end
        println(line)
    end
end
# one block on one die, per op and CTA (us from the block's start): when each CTA started the op (its previous op's
# end), had its x (the "x ready" stamps), and ended; the wait for x and the work after it as median and max with the
# CTA of the max, so a slow op's tail can be told apart from a late input (FLASHNEXT_CTAPROF=1 with --prof)
function cta_waits(E::EngineB1, bi::Int, D::B1Die)
    st = reshape(copy(unsafe_wrap(Array, D.tim, NTIM * 64)), NTIM, 64)[:, 1:D.grid]
    row(k) = Int.(st[1+(bi-1)*NSTB+k+1, :])
    t0 = minimum(row(0))
    xr = Dict(j => o for (o, j, nm) in SUBST if occursin("x ready", nm) || occursin("inputs in", nm))
    us(v) = v / 1e3
    med(v) = sort(v)[cld(length(v), 2)]
    @printf("block %d on die %d, %d CTAs (us from its start): op, start min-max, wait for x median/max (CTA), work median/max (CTA), end min-max\n",
        D.rt.blks[bi].l, D.rt.ord, D.grid)
    for j in 1:13
        e = row(j); all(iszero, e) && continue
        s = row(j - 1); all(iszero, s) && (s = row(j - 2))     # DeltaNet blocks skip the attention stamp
        x = haskey(xr, j) ? row(xr[j]) : s
        any(iszero, x) && (x = s)
        w = x .- s; k = e .- x
        @printf("  %-20s %7.1f-%7.1f  wait %6.1f/%6.1f (%2d)  work %6.1f/%6.1f (%2d)  end %7.1f-%7.1f\n", BLKOPS[j+1],
            us(minimum(s) - t0), us(maximum(s) - t0), us(med(w)), us(maximum(w)), argmax(w) - 1, us(med(k)), us(maximum(k)),
            argmax(k) - 1, us(minimum(e) - t0), us(maximum(e) - t0))
        # FLASHNEXT_CTADUMP=op: that op's start, x and end of every CTA (and its SM), by start
        if get(ENV, "FLASHNEXT_CTADUMP", "") == BLKOPS[j+1]
            sm = Int.(st[NTIM, :])
            for c in sortperm(s)
                @printf("    CTA %2d  SM %2d  start %7.1f  x %7.1f  end %7.1f\n", c - 1, sm[c], us(s[c] - t0), us(x[c] - t0), us(e[c] - t0))
            end
        end
    end
end
# per die, each block's time: from the previous block's last CTA end (the first block: its first CTA's start, after any
# wait for input) to its own last CTA end (us, the last token)
function block_durations(E::EngineB1)
    for DG in E.groups, D in DG
        st = reshape(copy(unsafe_wrap(Array, D.tim, NTIM * 64)), NTIM, 64)[:, 1:D.grid]
        nb = length(D.rt.blks)
        ends = [maximum(st[1+(bi-1)*NSTB+13+1, :]) for bi in 1:nb]
        t0 = minimum(st[1+1, :])
        d = [(Int(ends[bi]) - Int(bi == 1 ? t0 : ends[bi-1])) / 1e3 for bi in 1:nb]
        @printf("die %d blocks %d-%d: %s us (sum %.0f, mean of 2.. %.0f)\n", D.rt.ord, D.rt.blks[1].l, D.rt.blks[end].l,
            join([@sprintf("%.0f", x) for x in d], " "), sum(d), sum(d[2:end]) / max(1, length(d) - 1))
    end
end
# die clocks: one round trip a sample, die a (die 0) stamps t1, writes word 0 of b's box, b stamps tb on seeing it and
# answers in a's box, a stamps t2; tb - (t1 + t2) / 2 is b's globaltimer minus a's (to half the round trip)
function k_ping!(mine::U64P, other::U64P, out::Core.LLVMPtr{UInt64,1}, rounds::Int32, initiator::Int32)
    threadIdx().x == Int32(1) || return
    for r in Int32(1):rounds
        if initiator == Int32(1)
            t1 = gtimer(); llput(other, Int32(0), r % UInt32, 0.0f0); llget(mine, Int32(0), r % UInt32); t2 = gtimer()
            unsafe_store!(out, t1, 2r - Int32(1), Val(8)); unsafe_store!(out, t2, 2r, Val(8))
        else
            llget(mine, Int32(0), r % UInt32); tb = gtimer(); llput(other, Int32(0), r % UInt32, 0.0f0)
            unsafe_store!(out, tb, r, Val(8))
        end
    end
    return
end
# each die's globaltimer minus die 0's (ns), from the round trip of least time of 50; measured at every report (the
# dies' clocks drift apart by about 100 us within minutes, so a cached offset spoils later skew reports)
function clock_offsets(E::EngineB1)
    dies = [D.rt.ord for DG in E.groups for D in DG]; a = dies[1]; R = 50
    off = Dict{Int,Float64}(a => 0.0)
    for b in dies[2:end]
        device!(a); ba = CUDA.zeros(UInt64, 1); ha = CUDA.alloc(CUDA.HostMemory, 16R, CUDA.MEMHOSTALLOC_DEVICEMAP)
        device!(b); bb = CUDA.zeros(UInt64, 1); hb = CUDA.alloc(CUDA.HostMemory, 8R, CUDA.MEMHOSTALLOC_DEVICEMAP)
        pa = reinterpret(U64P, pointer(ba)); pb = reinterpret(U64P, pointer(bb))
        device!(b); launch(k_ping!, pb, pa, reinterpret(Core.LLVMPtr{UInt64,1}, convert(Ptr{UInt64}, hb)), Int32(R), Int32(0); threads = 32, blocks = 1)
        device!(a); launch(k_ping!, pa, pb, reinterpret(Core.LLVMPtr{UInt64,1}, convert(Ptr{UInt64}, ha)), Int32(R), Int32(1); threads = 32, blocks = 1)
        device!(a); synchronize(); device!(b); synchronize()
        ta = unsafe_wrap(Array, convert(Ptr{UInt64}, ha), 2R); tb = unsafe_wrap(Array, convert(Ptr{UInt64}, hb), R)
        r = argmin([Int(ta[2i]) - Int(ta[2i-1]) for i in 1:R])
        off[b] = Float64(Int(tb[r]) - (Int(ta[2r-1]) + Int(ta[2r])) / 2)
        note(@sprintf("clock: die %d - die %d = %.1f us (round trip %.1f us)", b, a, off[b] / 1e3, (Int(ta[2r]) - Int(ta[2r-1])) / 1e3))
        CUDA.free(ha); CUDA.free(hb)
    end
    KCACHE[:clkoff] = off
end
# the last token's ops on one clock (die 0's): per op, how far apart the dies end it (mean over the blocks after the
# first, us) and which die ends it last how often; the x-ready points likewise
function skew_report(E::EngineB1)
    off = clock_offsets(E)
    DG = [D for G in E.groups for D in G]
    sts = [reshape(copy(unsafe_wrap(Array, D.tim, NTIM * 64)), NTIM, 64)[:, 1:D.grid] for D in DG]
    nb = minimum(length(D.rt.blks) for D in DG)
    xr = Dict(j => o for (o, j, nm) in SUBST if occursin("x ready", nm))
    println("dies' skew, last token, blocks 2..$nb (die 0's clock): op, end spread mean/max us, last die (count); x ready spread")
    for j in 1:13
        spr = Float64[]; last = zeros(Int, length(DG)); xsp = Float64[]
        for bi in 2:nb
            e = [maximum(st[1+(bi-1)*NSTB+j+1, :]) for st in sts]
            any(iszero, e) && continue
            ec = [Float64(e[k]) - off[DG[k].rt.ord] for k in eachindex(DG)]
            push!(spr, (maximum(ec) - minimum(ec)) / 1e3); last[argmax(ec)] += 1
            if haskey(xr, j)
                x = [maximum(st[1+(bi-1)*NSTB+xr[j]+1, :]) for st in sts]
                all(>(0), x) && (xc = [Float64(x[k]) - off[DG[k].rt.ord] for k in eachindex(DG)]; push!(xsp, (maximum(xc) - minimum(xc)) / 1e3))
            end
        end
        isempty(spr) && continue
        @printf("  %-20s %6.1f / %6.1f  last: %s%s\n", BLKOPS[j+1], sum(spr) / length(spr), maximum(spr),
            join(["$(DG[k].rt.ord):$(last[k])" for k in eachindex(DG) if last[k] > 0], " "),
            isempty(xsp) ? "" : @sprintf("   x ready spread %.1f", sum(xsp) / length(xsp)))
    end
    # each die's own work after its x is ready (median CTA's end - x ready, mean over blocks): slow compute vs late input
    println("each die's work after x ready (us: median CTA, mean over blocks 2..$nb), by op:")
    for j in 1:13
        haskey(xr, j) || continue
        w = Float64[]
        for st in sts
            a = Float64[]
            for bi in 2:nb
                e = st[1+(bi-1)*NSTB+j+1, :]; x = st[1+(bi-1)*NSTB+xr[j]+1, :]
                ok = (e .> 0) .& (x .> 0) .& (e .>= x)
                any(ok) && push!(a, sort((e[ok] .- x[ok]) ./ 1e3)[cld(count(ok), 2)])
            end
            push!(w, isempty(a) ? NaN : sum(a) / length(a))
        end
        @printf("  %-20s %s\n", BLKOPS[j+1], join([@sprintf("%d:%5.1f", DG[k].rt.ord, w[k]) for k in eachindex(DG)], "  "))
    end
end
function profile_b1(E::EngineB1; emit = true)
    skew_report(E)
    block_timeline(E, 5)
    get(ENV, "FLASHNEXT_CTAPROF", "0") == "1" && for D in E.groups[1]     # both dies of board 0 (each on its own clock)
        cta_waits(E, 4, D); cta_waits(E, 5, D)
    end
    block_durations(E)
    haskey(PACC, :n) || profile_acc!(E)
    n = PACC[:n]; avg(k) = Dict(a => b / n for (a, b) in PACC[k])
    tot, byt, spr, subm, subx, die, wait = avg(:tot), avg(:byt), avg(:spr), avg(:subm), avg(:subx), avg(:die), avg(:wait)
    @printf("B1 profile: means over %d token(s)\n", n)
    for o in sort(collect(keys(die)))
        @printf("die %d: kernel %.3f ms (first op incl. waiting for input %.3f ms)\n", o, die[o], wait[o])
        emit && metric("b1.die_ms", "ms", die[o]; die = o, tokens = n)
    end
    for (nm, us) in sort(collect(tot); by = x -> -x[2])
        @printf("  %-20s %9.1f us %8.1f MB %6.1f GB/s  CTA spread %8.1f us (sums over dies)\n", nm, us, byt[nm] / 1e6, byt[nm] / (us * 1e3), spr[nm])
        emit && metric("b1.op_us", "us", us; op = nm, mb = byt[nm] / 1e6, spread_us = spr[nm], tokens = n)
        emit && byt[nm] > 0 && metric("b1.op_GBs", "GB/s", byt[nm] / (us * 1e3); op = nm, tokens = n)
    end
    for (_, _, nm) in SUBST
        haskey(subm, nm) || continue
        @printf("  %-32s since op start: mean %8.1f us  max %8.1f us (sums over blocks and dies)\n", nm, subm[nm], subx[nm])
        emit && metric("b1.subop_us", "us", subx[nm]; point = nm, mean_us = subm[nm], tokens = n)
    end
end
# B1 against the Float64 reference over the placed blocks (one comparison per token at the end of the range)
function run_check_b1(E::EngineB1, ids::Vector{Int32})
    g = E.g
    layers = [B.l for DG in E.groups for B in DG[1].rt.blks]
    R = Dict(l => RefState() for l in layers)
    worst = 0.0
    for (t0, id) in enumerate(ids)
        t = t0 - 1
        tt = time()
        Sg = step!(E, id; logits = false)
        dt = time() - tt
        S0 = Float64.(repeat(embed(g, id), 4))
        Sr = copy(S0)
        for l in layers
            Sr = ref_block!(g, l, Sr, R[l], t, E.ids, E.ple)
        end
        e = relerr(Sg .- S0, Sr .- S0)
        worst = max(worst, e)
        @printf("check B1 t=%d blocks %d..%d  |S - S0| rel err %.3e  (token %.2f ms)\n", t, first(layers), last(layers), e, dt * 1e3)
        PROF[] && t0 == length(ids) && profile_b1(E)
        metric("check.b1_relerr", "ratio", e; position = t)
    end
    ok = worst < 1e-3
    @printf("check B1: worst relative error %.3e: %s\n", worst, ok ? "PASS" : "FAIL")
    ok
end
# ---------------------------------------------------------------------------------------------
# CPU reference (Float64, straight from the math; for --check)
# ---------------------------------------------------------------------------------------------
function ref_mv(g::GModel, name::String, x::AbstractVector; r0 = 0, n = -1)
    t = g.tensors[name]; K = t.dims[1]
    n < 0 && (n = nrows(t))
    y = zeros(Float64, n)
    # the rows in chunks over the threads (on one, a --check token's reference took about 30 s; the server needs
    # JULIA_NUM_THREADS for more than one)
    nc = Threads.nthreads(); ch = cld(n, nc)
    Threads.@threads for c in 1:nc
        w = Vector{Float32}(undef, K)
        for r in (c - 1) * ch + 1:min(n, c * ch)
            dequant_row!(w, t.typ, tptr(g, t) + (r0 + r - 1) * rowbytes(t), K)
            s = 0.0
            @inbounds for k in 1:K
                s += Float64(w[k]) * x[k]
            end
            y[r] = s
        end
    end
    y
end
rsig(x) = 1 / (1 + exp(-x))
rsilu(x) = x * rsig(x)
rrms(x, w, eps = 1e-6) = x ./ sqrt(sum(abs2, x) / length(x) + eps) .* w
mutable struct RefState
    conv::Matrix{Float64}      # 10240 x 4 (x(t) in column t mod 4)
    dn::Array{Float64,3}       # 128 (value) x 128 (key) x 48
    K::Vector{Matrix{Float64}} # per position: 256 x 2 (normed, roped)
    V::Vector{Matrix{Float64}}
    iraw::Vector{Vector{Float64}}
    pool::Vector{Vector{Float64}}
    plering::Dict{Int,Vector{Float64}}
end
RefState() = RefState(zeros(10240, 4), zeros(128, 128, 48), Matrix{Float64}[], Matrix{Float64}[], Vector{Float64}[], Vector{Float64}[], Dict{Int,Vector{Float64}}())
function rrope(x::AbstractVector, t)
    y = copy(x)
    for p in 0:31
        th = 1.0e7^(-2.0 * p / 64.0); a = t * th
        y[p+1] = x[p+1] * cos(a) - x[p+33] * sin(a)
        y[p+33] = x[p+33] * cos(a) + x[p+1] * sin(a)
    end
    y
end
function ref_hc(g, p, S, which)
    nw = tensor_f32(g, p * "hc_$(which)_norm.weight")
    xh = vcat([rrms(S[2560l+1:2560l+2560], nw[2560l+1:2560l+2560]) for l in 0:3]...)
    d = ref_mv(g, p * "hc_$(which)_down.weight", xh)
    inj = ref_mv(g, p * "hc_$(which)_inject.weight", xh)
    tt = rsilu.(d ./ 4)
    up = ref_mv(g, p * "hc_$(which)_up.weight", tt)
    z = [sum(rsig(up[2560l+c]) * xh[2560l+c] for l in 0:3) / 4 for c in 1:2560]
    z, inj
end
ref_write!(S, y, inj) = (for l in 0:3, c in 1:2560; S[2560l+c] += 2 * rsig(inj[l+1] / 4) * y[c]; end; S)
function ref_block!(g::GModel, l::Int, S::Vector{Float64}, R::RefState, t::Int, ids::Vector{Int32}, ple::PLE)
    p = "blk.$l."
    if l == 1
        gin = Float64.(ple_input(ple, ids, t))
        K = ref_mv(g, p * "ple_key.weight", gin); V = ref_mv(g, p * "ple_value.weight", gin)
        nk, nq, nc = tensor_f32(g, p * "ple_norm_key.weight"), tensor_f32(g, p * "ple_norm_query.weight"), tensor_f32(g, p * "ple_norm_conv.weight")
        cw = tensor_f32(g, p * "ple_conv1d.weight")
        X = zeros(10240); G = zeros(10240)
        for ln in 0:3
            r = 2560ln+1:2560ln+2560
            kn = rrms(K[r], nk[r]); qn = rrms(S[r], nq[r])
            s = sum(kn .* qn) / sqrt(2560)
            gt = rsig(sign(s) * sqrt(max(abs(s), 1e-6)))
            G[r] = gt .* V
            X[r] = rrms(G[r], nc[r])
        end
        R.plering[t] = X
        for ch in 1:10240
            y = 0.0
            for k in 0:3
                tp = t - (3 - k) * 3
                xv = tp < 0 ? 0.0 : R.plering[tp][ch]
                y += cw[4(ch-1)+k+1] * xv
            end
            S[ch] += G[ch] + rsilu(y)
        end
    end
    z, inj = ref_hc(g, p, S, "attn")
    if haskey(g.tensors, p * "ssm_a")
        qkv = ref_mv(g, p * "attn_qkv.weight", z); gate = ref_mv(g, p * "attn_gate.weight", z)
        al = ref_mv(g, p * "ssm_alpha.weight", z) .+ tensor_f32(g, p * "ssm_dt.bias"); be = ref_mv(g, p * "ssm_beta.weight", z)
        cw = tensor_f32(g, p * "ssm_conv1d.weight")
        R.conv[:, t%4+1] = qkv
        u = [rsilu(sum(cw[4(c-1)+k+1] * (t - (3 - k) >= 0 ? R.conv[c, (t-(3-k))%4+1] : 0.0) for k in 0:3)) for c in 1:10240]
        ssa = tensor_f32(g, p * "ssm_a"); nw = tensor_f32(g, p * "ssm_norm.weight")
        on = zeros(6144)
        for h in 0:47
            kh = h % 16
            q = u[128kh+1:128kh+128]; k = u[2048+128kh+1:2048+128kh+128]; v = u[4096+128h+1:4096+128h+128]
            q = q ./ sqrt(sum(abs2, q) + 1e-6); k = k ./ sqrt(sum(abs2, k) + 1e-6)
            sp = al[h+1] > 20 ? al[h+1] : log1p(exp(al[h+1]))
            a = exp(ssa[h+1] * sp); bt = rsig(be[h+1])
            Sh = view(R.dn, :, :, h + 1)
            Sh .*= a
            e = v .- Sh * k
            Sh .+= (bt .* e) * k'
            o = (Sh * q) ./ sqrt(128)
            on[128h+1:128h+128] = o ./ sqrt(sum(abs2, o) / 128 + 1e-6) .* nw .* rsig.(gate[128h+1:128h+128])
        end
        y = ref_mv(g, p * "ssm_out.weight", on)
    else
        qg = ref_mv(g, p * "attn_q.weight", z); kk = ref_mv(g, p * "attn_k.weight", z); vv = ref_mv(g, p * "attn_v.weight", z)
        qn, kn = tensor_f32(g, p * "attn_q_norm.weight"), tensor_f32(g, p * "attn_k_norm.weight")
        Kt = hcat([rrope(rrms(kk[256j+1:256j+256], kn), t) for j in 0:1]...)
        if KVQ_REF[]      # the cache's int8 + fp16-scale rounding, to separate it from kernel error
            Kt = reshape(qdq32(vec(Kt)), 256, 2); vv = qdq32(vv)
        end
        push!(R.K, Kt); push!(R.V, reshape(vv, 256, 2))
        ao = zeros(6144)
        for h in 0:23
            q = rrope(rrms(qg[512h+1:512h+256], qn), t)
            kv = h ÷ 12
            s = [dot64(q, R.K[j][:, kv+1]) / 16 for j in 1:t+1]
            m = maximum(s); pr = exp.(s .- m); pr ./= sum(pr)
            o = sum(pr[j] .* R.V[j][:, kv+1] for j in 1:t+1)
            ao[256h+1:256h+256] = o .* rsig.(qg[512h+257:512h+512])
        end
        y = ref_mv(g, p * "attn_output.weight", ao)
    end
    ref_write!(S, y, inj)
    z, inj = ref_hc(g, p, S, "ffn")
    r = ref_mv(g, p * "ffn_gate_inp.weight", z); sh = ref_mv(g, p * "ffn_gate_inp_shexp.weight", z)[1]
    top = sortperm(r; rev = true)[1:10]
    w = exp.(r[top] .- r[top[1]]); w ./= sum(w)
    out = zeros(2560)
    for (j, e) in enumerate(top)
        ge = ref_mv(g, p * "ffn_gate_exps.weight", z; r0 = 640(e - 1), n = 640)
        ue = ref_mv(g, p * "ffn_up_exps.weight", z; r0 = 640(e - 1), n = 640)
        out .+= w[j] .* ref_mv(g, p * "ffn_down_exps.weight", rsilu.(ge) .* ue; r0 = 2560(e - 1), n = 2560)
    end
    gs = ref_mv(g, p * "ffn_gate_shexp.weight", z); us = ref_mv(g, p * "ffn_up_shexp.weight", z)
    out .+= rsig(sh) .* ref_mv(g, p * "ffn_down_shexp.weight", rsilu.(gs) .* us)
    ref_write!(S, out, inj)
    S
end
dot64(a, b) = sum(a[i] * b[i] for i in eachindex(a))
const KVQ_REF = Ref(true)
function qdq32(x::Vector{Float64})
    y = similar(x)
    for g in 0:length(x)÷32-1
        r = 32g+1:32g+32
        m = max(Float32(maximum(abs, x[r])), 7.888609f-31)
        s = Float64(Float16(m / 127.0f0))
        y[r] = [clamp(round(Float32(v) * (127.0f0 / m)), -127.0f0, 127.0f0) * s for v in x[r]]
    end
    y
end

# ---------------------------------------------------------------------------------------------
# Loading, checks, generation
# ---------------------------------------------------------------------------------------------
# ---- the weight holder: fills the arenas once and keeps them (and the n-gram table) resident; engines attach by CUDA IPC
const HOLD_DIR = "/tmp/flashnext-hold"
# bump when a device layout changes (wide rows for the projections: slower, 83 -> 94 us IQ4_XS, 43 -> 70 us Q3_K;
# 9, 10: every line split over the group without copies: router rows, expert/shared down rows, attention by dims)
const LAYOUT_VERSION = OWIDE ? 12 : 11     # (11, 12: + the copy trials' router and attention matrices)
placement_key(pl::PlacementA) = string("A;", join(pl.dies, ","), ";", join(pl.layers, ","), ";", pl.head, ";v", LAYOUT_VERSION)
function ipc_handle(p::CuPtr)
    h = Ref{CUC.CUipcMemHandle}()
    CUC.cuIpcGetMemHandle(h, reinterpret(CUC.CUdeviceptr, p))
    UInt8[reinterpret(UInt8, c) for c in h[].reserved]
end
function ipc_open(b::Vector{UInt8})
    h = CUC.CUipcMemHandle(ntuple(i -> reinterpret(Cchar, b[i]), 64))
    r = Ref{CUC.CUdeviceptr}()
    CUC.cuIpcOpenMemHandle_v2(r, h, 1)       # CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS
    reinterpret(CuPtr{UInt8}, r[])
end
alive(pid) = ccall(:kill, Cint, (Cint, Cint), pid, 0) == 0 || Libc.errno() == Libc.EPERM
function read_manifest()
    f = joinpath(HOLD_DIR, "manifest")
    isfile(f) || return nothing
    m = Dict{String,Any}("dies" => Dict{Int,Tuple{Int,Vector{UInt8}}}())
    for ln in eachline(f)
        w = split(ln)
        isempty(w) && continue
        if w[1] == "die"
            m["dies"][parse(Int, w[2])] = (parse(Int, w[3]), hex2bytes(w[4]))
        else
            m[w[1]] = join(w[2:end], " ")
        end
    end
    m
end
die_builds(pl::PlacementA) = [(ord, A -> build_die!(A, g_hold[], pl, i)) for (i, ord) in enumerate(pl.dies)]
die_builds(pl) = [(ord, A -> build_die!(A, g_hold[], pl, i, k)) for (i, gr) in enumerate(pl.groups) for (k, ord) in enumerate(gr)]
const g_hold = Ref{GModel}()
# outlive the gpuq job that starts us: close our copy of its queue lock (fd 9, only when it is that file; the
# parent's flock stays until the parent closes its own descriptor). Not fd 4: gpuq since nm-z/gpuq#87 passes no
# resource-usage pipe there, so fd 4 is the runtime's own (libuv's), and closing it left the event loop spinning:
# the holder at 100 % CPU before its first fill, the server hung after one request.
function gpuq_detach()
    haskey(ENV, "GPUQ_LOCK_FD") || return
    p = "/proc/self/fd/9"
    islink(p) && readlink(p) == get(ENV, "GPUQ_LOCK", "/tmp/flash-sass-engine-validation.lock") && ccall(:close, Cint, (Cint,), 9)
end
function hold(g::GModel, pl)
    gpuq_detach()
    mkpath(HOLD_DIR)
    lines = ["key " * placement_key(pl), "pid $(getpid())"]
    g_hold[] = g
    for (ord, build) in die_builds(pl)
        device!(ord)
        A = Arena(); build(A); sz = A.off
        base = rawalloc(sz)
        t0 = time()
        build(Arena(base, sz))
        note(@sprintf("hold: die %d filled, %.2f GB in %.1f s", ord, sz / 1e9, time() - t0))
        push!(lines, "die $ord $sz $(bytes2hex(ipc_handle(base)))")
    end
    for t in host_tables(g)
        t0 = time()
        # no transparent huge pages here (MADV_NOHUGEPAGE): with THP "always", locked pages of this mapping were seen
        # leaving it after mlock returned
        madv(tptr(g, t), t.nbytes, 15)
        r = ccall(:mlock, Cint, (Ptr{Cvoid}, Csize_t), tptr(g, t), t.nbytes)
        note(@sprintf("hold: %s (%.2f GB) %s in %.1f s, %.2f GB resident", t.name, t.nbytes / 1e9, r == 0 ? "locked in RAM" : "NOT locked (errno $(Libc.errno()))",
            time() - t0, resident(tptr(g, t), t.nbytes) / 1e9))
    end
    tmp = joinpath(HOLD_DIR, "manifest.tmp")
    write(tmp, join(lines, "\n") * "\n")
    mv(tmp, joinpath(HOLD_DIR, "manifest"); force = true)
    note("hold: ready, pid $(getpid())")
    # locked pages of the shared file mapping were seen to leave it; locking again faults them back (nearly free when
    # all are present)
    while true
        sleep(120)
        for t in host_tables(g)
            ccall(:mlock, Cint, (Ptr{Cvoid}, Csize_t), tptr(g, t), t.nbytes)
        end
    end
end
# ---- the resident test server (--serve): the B1 engine stays attached to the holder's weights with its host code
# compiled and its kernels loaded, so a test is one request of seconds. A request is one line on the FIFO serve.req:
# the path of a response FIFO, then the arguments, tab separated (NAME=VALUE arguments set environment variables for
# the request). Everything the request prints goes into the response FIFO, ending with "flashnext-serve: status N".
const SERVE_REQ = joinpath(HOLD_DIR, "serve.req")
function serve(E::EngineB1)
    gpuq_detach()                           # outlive the gpuq job that starts us
    SERVING[] = true
    ispath(SERVE_REQ) && rm(SERVE_REQ)
    run(`mkfifo $SERVE_REQ`)
    note("serve: ready, pid $(getpid())")
    while true
        for line in eachline(SERVE_REQ)
            parts = split(line, '\t')
            isempty(parts[1]) && continue
            # a client gone before the response ends (its timeout) fails the writes or the close: noted, and the server
            # goes on (it had exited on "close: Illegal seek" after a 60 s client timeout)
            try open(String(parts[1]), "w") do io
                redirect_stdout(io) do
                    redirect_stderr(io) do
                        st = try
                            # through invokelatest: the newest definitions after a --reload
                            length(parts) >= 2 && parts[2] == "--reload" ? reload_host!(E) : Base.invokelatest(serve_one, E, String.(parts[2:end]))
                        catch e
                            println(stderr, "flashnext: error: ", first(sprint(showerror, e), 2000))
                            for fr in stacktrace(catch_backtrace())[1:min(end, 12)]
                                println(stderr, "  ", fr)
                            end
                            1
                        end
                        println("flashnext-serve: status ", st)
                    end
                end
            end
            catch e
                note("serve: response to ", parts[1], " lost: ", first(sprint(showerror, e), 300))
            end
        end
    end
end
# --reload: this file included again into the running server (changed methods take effect; the engine, its device
# state and the kernel module are kept; a changed struct needs a restart). The module's globals are new bindings,
# so the ones the engine depends on are carried over, and the kernel module is loaded again (the draft's graphs too).
const SRCFILE = @__FILE__
function reload_host!(E)
    t0 = time()
    st = (KMODE[], KPATH[], NSM[], B1THREADS[], B1MAXREGS[], PROF[], (KMODS, KCACHE, KEEPMODS, KSTAT[]))
    Base.include(@__MODULE__, SRCFILE)
    Base.invokelatest(restore_host!, E, st)
    note(@sprintf("reload: %s in %.1f s", SRCFILE, time() - t0))
    0
end
# --combos: the (format, lanes) of every placed block matrix, by the phase mode that runs it (1: gathered expert
# gate/up; 3: MoE down tasks; 4: hc down; 0: the rest) — the lists the dispatchers are generated from
function placed_combos(E::EngineB1)
    s = Dict(m => Set{Tuple{Int,Int}}() for m in (0, 1, 3, 4))
    for DG in E.groups, D in DG, B in D.rt.blks, f in fieldnames(PBlk)
        m = getfield(B, f)
        (m isa DevMat && m.rows > 0) || continue
        mode = f == :gu ? 1 : f in (:down, :sh_down) ? 3 : f in (:hcA_dj, :hcA_dj2, :hcF_dj, :hcF_dj2) ? 4 : 0
        push!(s[mode], (Int(m.t), Int(m.L)))
    end
    s
end
function print_combos(E::EngineB1)
    for (mode, c) in sort(collect(placed_combos(E)); by = first)
        println("mode $mode: (", join(["(T_$(tname(t)), $L)" for (t, L) in sort(collect(c))], ", "), ")")
    end
    0
end
function restore_host!(E, st)
    KMODE[], KPATH[], NSM[], B1THREADS[], B1MAXREGS[], PROF[], old = st
    push!(KEEPMODS, old[1:3])                  # the old bindings' modules, kernels and kept objects stay alive
    KSTAT[] = old[4]
    for (k, v) in old[2]                       # the device buffers cached there (not the kernels)
        k isa Tuple && k[1] in (:k, :occ4) && continue
        KCACHE[k] = v
    end
    SERVING[] = true
    KPATH[] === nothing || reload_kernels!(E, KPATH[])
end
function serve_one(E::EngineB1, a::Vector{String})
    # garbage from the last request collected while the dies are idle (a CUDA object finalized mid-token waits for the
    # device, which waits for launches the host has not made)
    GC.gc(true)
    env = Pair{String,String}[]; rest = String[]
    for x in a
        m = match(r"^([A-Z][A-Z0-9_]*)=(.*)$", x)
        m === nothing ? push!(rest, x) : push!(env, String(m[1]) => String(m[2]))
    end
    withenv(env...) do
        k = findfirst(==("--kernels"), rest)
        if k !== nothing                   # --kernels sass FILE: a new module (the same entries) from now on
            p = rest[k+2]; st = stat(p)
            (p == KPATH[] && KSTAT[] !== nothing && st.mtime == KSTAT[].mtime && st.size == KSTAT[].size) || reload_kernels!(E, p; kind = rest[k+1])
            deleteat!(rest, k:k+2)
            isempty(rest) && return 0
        end
        !isempty(rest) && rest[1] == "--combos" && return print_combos(E)
        # --tokenize TEXT | --tokenize-file PATH: the ids (raw text, special tokens parsed), as llama-tokenize prints them
        !isempty(rest) && rest[1] == "--tokenize" && (println(join(encode(E.tk, rest[2]), " ")); return 0)
        !isempty(rest) && rest[1] == "--tokenize-file" && (println(join(encode(E.tk, read(rest[2], String)), " ")); return 0)
        if !isempty(rest) && rest[1] == "--ple-check"   # the n-gram hash fixture: ids [42, 248046, 198] at position 2
            r = ple_rows(E.ple, Int32[42, 248046, 198], 2)
            ok = r[1] == 510813 && r[9] == 165588995
            println("ple rows: ", join(r, " "), "\nple check (rows[0] = 510813, rows[8] = 165588995): ", ok ? "PASS" : "FAIL")
            metric("check.ple_hash", "pass", ok ? 1 : 0)
            return ok ? 0 : 1
        end
        if !isempty(rest) && rest[1] == "--layouts"     # --layouts [BLOCK]: each matrix's row-group layout on die 0
            B = E.groups[1][1].rt.blks[length(rest) > 1 ? parse(Int, rest[2]) : 5]
            @printf("block %d on die %d: matrix, format, rows x K, lanes a row, rows a group, groups, KB a group (warps: %d)\n",
                B.l, E.groups[1][1].rt.ord, E.groups[1][1].grid * 8)
            for f in fieldnames(PBlk)
                m = getfield(B, f); (m isa DevMat && m.rows > 0) || continue
                R = 32 ÷ m.L; G = cld(Int(m.rows), R)
                @printf("  %-10s %-7s %6d x %5d  L %2d  R %2d  %5d groups  %6.1f KB\n", f, tname(m.t), m.rows, m.K, m.L, R, G,
                    devbytes_rows(m.t, Int(m.K), Int(m.rows), Int(m.L)) / G / 1024)
            end
            return 0
        end
        if !isempty(rest) && rest[1] == "--bench" && rest[2] == "sweep"     # --bench sweep [DIES]: transfers by die count
            bench_sweep(length(rest) > 2 ? parse.(Int, split(rest[3], ",")) : collect(0:7))
            return 0
        end
        if !isempty(rest) && rest[1] == "--bench"
            device!(parse(Int, get(ENV, "FLASHNEXT_BENCH_DIE", "0")))      # (its holder share stays; the bench's matrices beside it)
            NSM[] = CUDA.attribute(device(), CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
            rest[2] == "phase" ? bench_phase(E.g) : rest[2] == "gemv4" ? bench_gemv4(E.g) : rest[2] == "draft" ? bench_draft(E) : rest[2] == "phase4" ? bench_phase4(E.g) : error("--bench phase | gemv4 | draft | phase4")
            return 0
        end
        if !isempty(rest) && rest[1] == "--b8diff"      # --b8diff K [--ids ...]: B1 against B8 over blocks 0..K-1
            o = parse_args(rest[3:end])
            ids = haskey(o, "ids") ? o["ids"] : encode(E.tk, chat_prompt(get(o, "p", "The capital of France is")))
            return b8diff(E, ids, parse(Int, rest[2])) ? 0 : 1
        end
        o = parse_args(rest)
        reset_state!(E); set_prof!(E, get(o, "prof", false))
        get(o, "spec", "off") != "off" && E.draft === nothing && (E.draft = load_draft(E))
        prompt = get(o, "p", "The capital of France is")
        ids = haskey(o, "ids") ? o["ids"] : encode(E.tk, o["raw"] ? prompt : chat_prompt(prompt))
        get(o, "check", false) && return run_check_b1(E, ids) ? 0 : 1
        get(o, "b4", false) && get(o, "teacher", false) && haskey(o, "expect") && return b4_teacher_check(E, ids, o["expect"])
        get(o, "b4", false) && get(o, "debug", false) && return b4_debug(E, ids)
        run_request(E, E.tk, o, ids)
    end
end
# a new sequence: no tokens, DeltaNet conv and recurrent states and the PLE ring zero (attention reads only positions
# below t); seq numbers keep counting (E.ntok)
function reset_state!(E::EngineB1)
    empty!(E.ids)
    zero!(p, n) = CUC.cuMemsetD32_v2(reinterpret(CUC.CUdeviceptr, p), UInt32(0), n)
    for DG in E.groups, D in DG
        rt = D.rt; device!(rt.ord); nkh = 16 ÷ rt.n; nc = 2560 ÷ rt.n
        for (i, B) in enumerate(rt.blks)
            st = rt.st[i]
            if B.attn
                zero!(st.ring, max(512, SRN * attn_geom(rt.n).IW))
            else
                zero!(st.conv, SRN * 640nkh); zero!(st.dn, SRN * 3nkh * 16384)
            end
        end
        rt.pler != NOF && zero!(rt.pler, 16 * 4nc)
        synchronize()
    end
end
# the op stamps (--prof) on or off: each die's DD with or without its stamp pointer
function set_prof!(E::EngineB1, on::Bool)
    PROF[] = on
    for DG in E.groups, D in DG
        device!(D.rt.ord)
        dds = Array(D.dd)
        tim = on ? reinterpret(Core.LLVMPtr{UInt64,1}, D.tim) : Core.LLVMPtr{UInt64,1}(0)
        dds[1].tim == tim && continue
        D.dd = CuArray([DD((f == :tim ? tim : getfield(dd, f) for f in fieldnames(DD))...) for dd in dds])
    end
end
# a module file's bytes for CuModule: PTX text NUL-terminated (cuModuleLoadData reads PTX up to a NUL)
kread(kind, path) = (b = read(path); kind == "ptx" && (isempty(b) || b[end] != 0x00) && push!(b, 0x00); b)
function reload_kernels!(E::EngineB1, path; kind = "sass")
    t0 = time()
    # the old module and the draft's graphs (they hold its functions; captured again, after one direct use, from the
    # new one) released now, with every die idle: left to the GC, a finalizer's cuModuleUnload waits for the device,
    # and ran during a token's launches it deadlocked with the persistent kernels in flight
    # Nothing of the old module is unloaded or destroyed: unloaded explicitly, objects still tied to it crashed libcuda
    # later. The old modules, kernels and graphs stay referenced (KEEPMODS: about 3 MB of code a die per switch).
    for DG in E.groups, D in DG
        device!(D.rt.ord); synchronize()
    end
    if E.draft !== nothing
        push!(KEEPMODS, E.draft.gprop, collect(values(E.draft.gapp)))
        E.draft.gprop = nothing; empty!(E.draft.gapp); empty!(E.draft.uses)
    end
    push!(KEEPMODS, copy(KMODS), [kv for kv in KCACHE if kv[1] isa Tuple && kv[1][1] in (:k, :occ4)])
    # only the kernel entries: KCACHE also holds device buffers
    KMODE[] = kread(kind, path); KPATH[] = path; KSTAT[] = stat(path); empty!(KMODS)
    filter!(kv -> !(kv[1] isa Tuple && kv[1][1] in (:k, :occ4)), KCACHE)
    for DG in E.groups, D in DG
        device!(D.rt.ord)
        k = getkernel(k_token!, KTOKEN_TT; maxregs = B1MAXREGS[], maxthreads = B1THREADS[])
        # (a module's CTA size is its k_token!'s .maxntid, FLASHNEXT_B1_THREADS when it was built: modules of different
        # sizes alternate in one server)
        KMODE[] === nothing || (B1THREADS[] = min(1024, Int(CUDA.maxthreads(k))))
        E.kern[D.rt.ord] = k
        D.nblk == 0 && continue
        # the module's register cap decides the CTAs per SM: the grid follows
        D.grid = CUDA.active_blocks(k.fun, B1THREADS[]; shmem = b1shmem(D)) * NSM[]
        note(@sprintf("die %d: k_token! %d registers, grid %d", D.rt.ord, CUDA.registers(k), D.grid))
    end
    note(@sprintf("kernels from %s (%s, %d bytes): k_token! loaded on %d dies in %.1f s", path, kind, length(KMODE[]),
        sum(length, E.groups), time() - t0))
    metric("kernels.load_s", "s", time() - t0; kind)
end
# the tables the host reads every token: token embedding rows and the 16 n-gram rows
host_tables(g::GModel) = [g.tensors["token_embd.weight"], g.tensors["per_layer_token_embd.weight"]]
madv(p::Ptr, n::Integer, advice) = (a = UInt(p) & ~UInt(4095); ccall(:madvise, Cint, (Ptr{Cvoid}, Csize_t, Cint), Ptr{Cvoid}(a), UInt(p) + n - a, advice))
# bytes of [p, p + n) in RAM (mincore)
function resident(p::Ptr, n::Integer)
    a = UInt(p) & ~UInt(4095); len = UInt(p) + n - a
    v = zeros(UInt8, cld(len, 4096))
    ccall(:mincore, Cint, (Ptr{Cvoid}, Csize_t, Ptr{UInt8}), Ptr{Cvoid}(a), len, v)
    4096 * count(x -> x & 0x1 == 0x1, v)
end

# other compute processes on our dies (the fence that pins known-bad VRAM, and our holder, are allowed)
# our holder, whatever placement it holds (it is allowed on any die)
holder_pid(m) = m !== nothing && alive(parse(Int, m["pid"])) ? parse(Int, m["pid"]) : 0
function check_dies(dies, holder)
    uuid = Dict{String,Int}()
    for ln in readlines(`nvidia-smi --query-gpu=index,uuid --format=csv,noheader`)
        i, u = strip.(split(ln, ","))
        uuid[u] = parse(Int, i)
    end
    bad = String[]
    for ln in readlines(`nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name --format=csv,noheader`)
        w = strip.(split(ln, ","))
        length(w) < 3 && continue
        d = get(uuid, w[1], -1); pid = parse(Int, w[2])
        (d in dies && pid != getpid() && pid != holder && !occursin("fence_hold", w[3])) && push!(bad, "die $d: pid $pid $(w[3])")
    end
    isempty(bad) || error("other compute processes on the requested dies:\n  " * join(bad, "\n  "))
end

function load_engine(g::GModel, pl::PlacementA)
    m = read_manifest()
    holder = (m !== nothing && get(m, "key", "") == placement_key(pl) && alive(parse(Int, m["pid"]))) ? parse(Int, m["pid"]) : 0
    check_dies(pl.dies, holder_pid(m))
    dies = DieRT[]
    for (i, ord) in enumerate(pl.dies)
        device!(ord)
        NSM[] = CUDA.attribute(device(), CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
        A = Arena(); build_die!(A, g, pl, i)
        sz = A.off
        t0 = time()
        if holder != 0
            hs, hb = m["dies"][ord]
            hs == sz || error("holder arena on die $ord is $hs bytes, expected $sz")
            blks, head = build_die!(Arena(ipc_open(hb), sz; fill = false), g, pl, i)
        else
            blks, head = build_die!(Arena(rawalloc(sz), sz), g, pl, i)
        end
        dt = time() - t0
        note(@sprintf("die %d: blocks %s%s, %.2f GB %s in %.1f s", ord, string(pl.layers[i]), head === nothing ? "" : " + head", sz / 1e9,
            holder != 0 ? "attached from holder $holder" : "loaded", dt))
        metric("load.die_bytes", "bytes", sz; die = ord)
        metric("load.die_s", "s", dt; die = ord, attached = holder != 0)
        push!(dies, DieRT(ord, blks, head))
    end
    Engine(g, Tok(g), PLE(g), dies, Int32[], 0)
end

relerr(a, b) = sqrt(sum(abs2, a .- b) / max(sum(abs2, b), 1e-300))
# GPU blocks vs the Float64 reference: each reference block starts from the GPU's input, so every row of the
# report isolates one block at one position
function run_check(E::Engine, ids::Vector{Int32})
    g = E.g
    R = Dict(B.l => RefState() for rt in E.dies for B in rt.blks)
    worst = 0.0
    for (t0, id) in enumerate(ids)
        t = t0 - 1
        push!(E.ids, id)
        S = repeat(embed(g, id), 4)
        for rt in E.dies
            device!(rt.ord)
            ul!(rt.b[:S], S)
            any(B -> B.l == 1, rt.blks) && ul!(rt.b[:pleg], ple_input(E.ple, E.ids, t))
            for i in eachindex(rt.blks)
                l = rt.blks[i].l
                Sin = Float64.(dl(rt.b[:S], 10240))
                block_fwd!(rt, i, t)
                Sg = dl(rt.b[:S], 10240)
                Sr = ref_block!(g, l, copy(Sin), R[l], t, E.ids, E.ple)
                e = relerr(Sg .- Sin, Sr .- Sin)
                if !isfinite(e)
                    for (k, n) in ((:qg, 12288), (:k, 512), (:v, 512), (:qo, 6144), (:ao, 6144), (:y, 2560), (:pm, 24), (:pd, 24), (:po, 256), (:z, 2560))
                        v = dl(rt.b[k], n)
                        @printf("  %-3s nan %5d  max|x| %.3e  first %s\n", k, count(isnan, v), maximum(abs, filter(isfinite, v); init = 0.0f0), string(v[1:4]))
                    end
                end
                worst = max(worst, e)
                @printf("check t=%d blk=%2d %s  |dS| rel err %.3e  |S|=%.3e\n", t, l, rt.blks[i].attn ? "attn" : "gdn ", e, sqrt(sum(abs2, Sg)))
                metric("check.block_relerr", "ratio", e; position = t, block = l)
            end
            S = dl(rt.b[:S], 10240)
        end
    end
    ok = worst < 2e-3
    @printf("check: worst block-update relative error %.3e: %s\n", worst, ok ? "PASS" : "FAIL")
    metric("check.pass", "bool", ok; worst = worst)
    ok
end

splitmix(s::UInt64) = (s += 0x9e3779b97f4a7c15; z = s; z = (z ⊻ (z >> 30)) * 0xbf58476d1ce4e5b9; z = (z ⊻ (z >> 27)) * 0x94d049bb133111eb; (s, z ⊻ (z >> 31)))
# llama.cpp order: top-k, top-p (on the temperature-1 probabilities), temperature, draw
function sample(logits::Vector{Float32}, st::Base.RefValue{UInt64}; k = 20, topp = 0.95f0, temp = 1.0f0)
    cand = partialsortperm(logits, 1:k; rev = true)
    v = logits[cand]
    e = exp.(v .- v[1]); tot = sum(e)
    keep = 0; c = 0.0f0
    for i in 1:k
        c >= topp * tot && break
        keep = i; c += e[i]
    end
    w = exp.((v[1:keep] .- v[1]) ./ temp)
    st[], z = splitmix(st[])
    u = Float32((z >> 11) * 2.0^-53) * sum(w)
    acc = 0.0f0
    for i in 1:keep
        acc += w[i]
        acc > u && return Int32(cand[i] - 1)
    end
    Int32(cand[keep] - 1)
end

function generate(E, ids::Vector{Int32}, n::Int; greedy = true, seed = 0x5eed, echo = true, spec = "off")
    st = Ref(UInt64(seed))
    # spec shadow: the draft proposes at the start of every round a speculative decode would run (anchor = the round's
    # first input), its context fed after every target step; the rounds' accepted counts (matched drafts + 1)
    Dr = spec == "shadow" && E isa EngineB1 && greedy ? E.draft : nothing
    Dr !== nothing && (Dr.context = 0; empty!(Dr.ms))
    tp = time()
    if E isa EngineB1                  # the prompt pipelined over the boards (and the draft's context with it)
        empty!(pipe_used)
        for id in ids[1:end-1]
            launch_pipelined!(E, id; Dr)
        end
        for DG in E.groups, D in DG; device!(D.rt.ord); synchronize(); end
        device!(DRAFT_DIE); synchronize()
    else
        for id in ids[1:end-1]
            step!(E, id; logits = false)
        end
    end
    dev = greedy && E isa EngineB1                # argmax on the devices
    logits = dev ? Float32[] : step!(E, ids[end])
    nxt = dev ? step_greedy!(E, ids[end]) : Int32(0)
    Dr !== nothing && draft_append!(Dr, length(ids) - 1, 1)
    tpp = time() - tp
    out = Int32[]
    for k in keys(STEPT); STEPT[k] = 0.0; end              # decode steps only
    pos = length(ids); rs = pos; hist = zeros(Int, 4)
    props = Dr === nothing ? Int32[] : draft_propose(Dr, E, nxt)
    td = time()
    for i in 1:n
        dev || (nxt = greedy ? Int32(argmax(logits) - 1) : sample(logits, st))
        push!(out, nxt)
        echo && !hidden(E.tk, nxt) && (write(stdout, piece(E.tk, nxt)); flush(stdout))
        (nxt == E.tk.eos || i == n) && break
        dev ? (nxt = step_greedy!(E, nxt)) : (logits = step!(E, nxt))
        if Dr !== nothing
            draft_append!(Dr, pos, 1)
            j = pos - rs + 1                      # this output is the round's j-th: drafts 1..3, then the bonus
            if !(j <= 3 && props[j+1] == nxt)
                hist[j <= 3 ? j : 4] += 1         # j - 1 drafts matched (3 at j = 4): j accepted
                rs = pos + 1; props = draft_propose(Dr, E, nxt)
            end
        end
        pos += 1
    end
    tdd = time() - td
    echo && println()
    if Dr !== nothing && sum(hist) > 0
        r = sum(hist); acc = sum(k * hist[k] for k in 1:4) / r; k = 1e3 / max(1, Dr.ms[:n])
        @printf("draft (shadow): %d rounds, accepted per round 1/2/3/4: %s, mean %.3f; per proposal %.2f ms (backbone %.2f, head %.2f, select %.2f), per append %.2f ms\n",
            r, join(hist, "/"), acc, (Dr.ms[:backbone] + Dr.ms[:head] + Dr.ms[:select]) * k, Dr.ms[:backbone] * k, Dr.ms[:head] * k, Dr.ms[:select] * k,
            Dr.ms[:append] * 1e3 / max(1, pos))
        metric("draft.accepted_mean", "tokens", acc; rounds = r, h1 = hist[1], h2 = hist[2], h3 = hist[3], h4 = hist[4])
        metric("draft.propose_ms", "ms", (Dr.ms[:backbone] + Dr.ms[:head] + Dr.ms[:select]) * k)
    end
    pp = length(ids) / tpp; tg = length(out) > 1 ? (length(out) - 1) / tdd : 0.0
    @printf("prompt %d tokens in %.2f s (%.2f tok/s); generated %d tokens in %.2f s (%.2f tok/s)\n", length(ids), tpp, pp, length(out), tdd, tg)
    if E isa EngineB1 && STEPT[:n] > 0
        k = 1e3 / STEPT[:n]
        @printf("per step (ms, mean over %d): launch %.2f (kernel launches %.2f, embedding %.2f, PLE input %.2f), kernels until S %.2f, head %.2f\n",
            STEPT[:n], STEPT[:launch] * k, STEPT[:kernel_launch] * k, STEPT[:embed] * k, STEPT[:ple] * k, STEPT[:kernels] * k, STEPT[:head] * k)
        for p in (:launch, :kernels, :head, :kernel_launch, :embed, :ple)
            metric("step.host_ms", "ms", STEPT[p] * k; part = String(p))
        end
    end
    metric("pp.tokens_s", "tokens/s", pp; tokens = length(ids))
    metric("tg.tokens_s", "tokens/s", tg; tokens = length(out), reference_flash_sass = 40.3, practical = 192.0, theoretical = 329.0)
    out
end

# teacher forcing on an output oracle: at each output position our argmax against the oracle's id, with the logit
# margins (where they differ: ours - oracle's; elsewhere: first - second), to tell near ties from real differences
function oracle_margins(E, ids::Vector{Int32}, ex::Vector{Int32})
    for id in ids[1:end-1]
        step!(E, id; logits = false)
    end
    lg = step!(E, ids[end])
    diff = 0; close = Tuple{Int,Float32}[]
    for (i, e) in enumerate(ex)
        a = Int32(argmax(lg) - 1)
        if a != e
            diff += 1
            @printf("output %3d: ours %d (%.4f) vs oracle %d (%.4f): margin %.4f\n", i, a, lg[a+1], e, lg[e+1], lg[a+1] - lg[e+1])
            metric("check.oracle_margin", "logit", lg[a+1] - lg[e+1]; output = i)
        else
            s = partialsort(lg, 2; rev = true)
            push!(close, (i, lg[a+1] - s))
        end
        i == length(ex) && break
        lg = step!(E, e)
    end
    sort!(close; by = last)
    println("oracle teacher forcing: $diff of $(length(ex)) argmax differ; smallest first-second margins where equal: ",
        join([@sprintf("%d:%.4f", i, m) for (i, m) in close[1:min(5, end)]], " "))
    metric("check.oracle_argmax_differ", "positions", diff; compared = length(ex))
end
function teacher(E, ids::Vector{Int32}, n::Int)
    preds = Int32[]
    am(id) = E isa EngineB1 ? step_greedy!(E, id) : Int32(argmax(step!(E, id)) - 1)
    for id in ids
        push!(preds, am(id))
    end
    for _ in 1:n
        push!(preds, am(preds[end]))
        preds[end] == E.tk.eos && break
    end
    preds
end

function parse_args(a)
    o = Dict{String,Any}("dies" => [0, 1, 2, 3, 4, 5, 6], "blocks" => [7, 7, 6, 7, 7, 7, 7], "layers" => 48, "n" => 64,
        "ctx" => 4096, "greedy" => true, "seed" => 0x5eed, "raw" => false)
    i = 1
    while i <= length(a)
        x = a[i]
        if x in ("--dies", "--blocks", "--bblocks")
            o[x[3:end]] = parse.(Int, split(a[i+1], ",")); i += 1
        elseif x in ("--stage", "--groups", "--spec")
            o[x[3:end]] = a[i+1]; i += 1
        elseif x in ("--layers", "-n", "--ctx")
            o[lstrip(x, '-')] = parse(Int, a[i+1]); i += 1
        elseif x == "--seed"
            o["seed"] = parse(UInt64, a[i+1]); i += 1
        elseif x in ("-p", "--tokenize")
            o[lstrip(x, '-')] = a[i+1]; i += 1
        elseif x in ("--ids", "--expect")          # token ids (spaces or commas): the prompt as is, an output oracle
            o[x[3:end]] = parse.(Int32, split(a[i+1], (' ', ','); keepempty = false)); i += 1
        elseif x == "--sample"
            o["greedy"] = false
        elseif x == "--greedy"
            o["greedy"] = true
        elseif startswith(x, "--")
            o[x[3:end]] = true
        else
            error("unknown argument $x")
        end
        i += 1
    end
    o
end

# ---------------------------------------------------------------------------------------------
# Kernel instances and device-free PTX generation (sm_52), for --ptx and the .ptx/.sass paths
# ---------------------------------------------------------------------------------------------
# maxregs / maxthreads > 0: the entry's launch bounds (.maxnreg, .maxntid), so ptxas allocates registers for that CTA size
function ptx_config(; ptx = v"7.8", maxregs = 0, maxthreads = 0)
    tgt = GPUC.PTXCompilerTarget(; cap = v"5.2", ptx, debuginfo = false, maxregs = maxregs > 0 ? maxregs : nothing,
        maxthreads = maxthreads > 0 ? maxthreads : nothing)
    par = CUC.CUDACompilerParams(; sm = CUC.SMVersion(v"5.2"), ptx)
    GPUC.CompilerConfig(tgt, par; kernel = true, name = nothing, always_inline = false)
end
function kernel_ptx(f, tt)
    job = GPUC.CompilerJob(GPUC.methodinstance(typeof(f), tt), ptx_config())
    GPUC.JuliaContext() do ctx
        asm, meta = GPUC.compile(:asm, job)
        asm
    end
end
const GEMV_INST = [(T_Q2_0, 8), (T_Q2_0, 2), (T_Q2_0, 4), (T_BF16, 32), (T_BF16, 16), (T_BF16, 2), (T_Q4_K, 2), (T_Q4_K, 8),
    (T_Q5_K, 2), (T_Q5_K, 8), (T_Q6_K, 2), (T_Q6_K, 8), (T_Q3_K, 2), (T_Q3_K, 8), (T_IQ4_XS, 2), (T_IQ4_XS, 8),
    (T_Q4_0, 4), (T_Q5_0, 4), (T_Q8_0, 4), (T_IQ4_NL, 4)]
const F, G_, I_ = FPtr, GPtr, IPtr
kernel_list() = vcat([("gemv_$(tname(t))_L$(L)_m0", k_gemv!, gemv_tt(t, L, 0)) for (t, L) in GEMV_INST],
    [("gemv_Q2_0_L8_m1", k_gemv!, gemv_tt(T_Q2_0, 8, 1)), ("gemv_Q2_0_L2_m3", k_gemv!, gemv_tt(T_Q2_0, 2, 3))],
    [("gemv_$(tname(t))_L$(L)_m2", k_gemv!, gemv_tt(t, L, 2)) for (t, L) in ((T_Q2_0, 2), (T_Q4_0, 4), (T_Q5_0, 4), (T_Q8_0, 4), (T_IQ4_NL, 4))],
    [("rmsnorm", k_rmsnorm!, Tuple{F,F,F,Int32,Int32,Float32}), ("hc_silu", k_hc_silu!, Tuple{F,F,Int32}),
     ("hc_mix", k_hc_mix!, Tuple{F,F,F}), ("hc_write", k_hc_write!, Tuple{F,F,F}), ("conv1d", k_conv1d!, Tuple{F,F,F,F,Int32,Int32}),
     ("deltanet", k_deltanet!, Tuple{F,F,F,F,F,F,F,F,Float32,Int32}),
     ("attn_prep", k_attn_prep!, Tuple{F,G_,G_,G_,G_,F,F,F,F,F,F,Int32,Float32,Int32,Int32}),
     ("idx_prep", k_idx_prep!, Tuple{F,F,G_,F,F,F,F,F,Int32,Float32}), ("idx_score", k_idx_score!, Tuple{F,F,G_,Int32}),
     ("idx_select", k_idx_select!, Tuple{I_,F,Int32,Int32}),
     ("attn_part", k_attn_part!, Tuple{F,F,F,F,G_,G_,G_,G_,I_,Int32,Int32,Int32,Int32,Int32}), ("attn_merge", k_attn_merge!, Tuple{F,F,F,F,F,Int32}),
     ("router", k_router!, Tuple{I_,F,F}), ("swiglu", k_swiglu!, Tuple{F,F,Int32,Int32,Int32}), ("ple", k_ple!, Tuple{F,F,F,F,F,F,F,F,Int32,Float32})])
# ---- Stage D: the B1 engine's kernels in one PTX module (flashnext.ptx) and one cubin (flashnext.sass) ----
const KTOKEN_TT = Tuple{Core.LLVMPtr{DD,1},Core.LLVMPtr{BD,1},Int32,UInt32,Int32,Int32,Core.LLVMPtr{UInt64,1}}
# (function, argument types, register cap or 0, CTA size or 0) of every kernel the B1 engine and its head launch
function b1_kernels()
    mr = parse(Int, get(ENV, "FLASHNEXT_B1_MAXREGS", "128")); nt = parse(Int, get(ENV, "FLASHNEXT_B1_THREADS", "256"))
    ks = Tuple{Any,Type,Int,Int}[(k_token!, KTOKEN_TT, mr, nt), (k_ll2f!, Tuple{F,U64P,Int32,UInt32}, 0, 0), (k_w2f!, Tuple{F,U64P,Int32,UInt32}, 0, 0), (k_ping!, Tuple{U64P,U64P,Core.LLVMPtr{UInt64,1},Int32,Int32}, 0, 0),
    (k_rmsnorm!, Tuple{F,F,F,Int32,Int32,Float32}, 0, 0),
    (k_hc_silu!, Tuple{F,F,Int32}, 0, 0), (k_hc_mix!, Tuple{F,F,F}, 0, 0), (k_put_ll!, Tuple{NTuple{8,U64P},Int32,F,Int32,UInt32}, 0, 0),
    (k_argmax!, Tuple{F,F,Int32,Int32}, 0, 0), (k_gemv!, gemv_tt(T_BF16, 32, 0), 0, 0), (k_gemv!, gemv_tt(T_BF16, 2, 0), 0, 0), (k_gemv!, gemv_tt(T_Q5_K, 2, 0), 0, 0),
    (k_phasebench!, PHASEBENCH_TT, mr, nt), (k_gemv4bench!, Tuple{MatI,Int32,FPtr,FPtr,Int32}, mr, 256),
    # DFlash2
    (k_gemv4!, GEMV4_TT, 128, 256), (k_dconv!, Tuple{F,F,F,F,Int32,Int32}, 0, 0), (k_drope!, Tuple{F,F,Int32,I_,Int32}, 0, 0),
    (k_dcache!, Tuple{H16P,H16P,F,F,I_,Int32}, 0, 0), (k_dattn!, Tuple{F,F,F,F,H16P,H16P,I_,Int32}, 0, 0),
    (k_dswiglu!, Tuple{F,F,Int32}, 0, 0), (k_dswiglu2!, Tuple{F,F,Int32,Int32}, 0, 0), (k_dadd!, Tuple{F,F,Int32}, 0, 0), (k_dfeat!, Tuple{F,F,I_,Int32}, 0, 0),
    (k_top16!, Tuple{F,F,Int32,Int32,Int32,Int32}, 0, 0), (k_token8!, KTOKEN8_TT, 128, B4THREADS), (k_token4!, KTOKEN4_TT, 128, B4THREADS), (k_phase4bench!, PHASE4BENCH_TT, 128, B4THREADS)]
    # (k_token4!, B4's, only with FLASHNEXT_B4K=1: B4 is not ported to the tensor-split layout)
    get(ENV, "FLASHNEXT_B4K", "0") == "1" || filter!(k -> !(k[1] in (k_token4!, k_phase4bench!)), ks)
    get(ENV, "FLASHNEXT_BENCHK", "0") == "1" ? ks : filter(k -> !(k[1] in bench_kernels()), ks)
end
# the microbenchmark kernels: in the module only when built with FLASHNEXT_BENCHK=1 (about 38 s of PTX and 16 s of
# ptxas), else compiled by CUDA.jl at their first launch (getkernel)
bench_kernels() = (k_phasebench!, k_phase4bench!, k_gemv4bench!)
const PHASEBENCH_TT = Tuple{MatI,Int32,U64P,U64P,Int32,Int32,Int32,IPtr,MatI,MatI,Core.LLVMPtr{UInt64,1}}
kentry(f, tt) = GPUC.mangle_sig(Tuple{typeof(f), tt.parameters...})
const LLVMJ = lmod("LLVM")
kjob(k) = GPUC.CompilerJob(GPUC.methodinstance(typeof(k[1]), k[2]), ptx_config(; maxregs = k[3], maxthreads = k[4]))
# --kernel-bc DIR I,J,...: kernels I, J, ... of b1_kernels() to LLVM bitcode DIR/kI.bc (a worker of module_ptx)
function kernel_bc(dir, which::Vector{Int})
    ks = b1_kernels()
    GPUC.JuliaContext() do ctx
        for i in which
            t0 = time()
            ir, _ = GPUC.compile(:llvm, kjob(ks[i]))
            open(io -> write(io, ir), joinpath(dir, "k$i.bc"), "w")
            note(@sprintf("  %s to LLVM in %.1f s", nameof(ks[i][1]), time() - t0))
        end
    end
    0
end
# the kernels with a phase program (most of the compile time), spread first over the worker processes
heavy_kernels() = (k_token!, k_token8!, k_token4!, k_phasebench!, k_phase4bench!)
# every kernel to LLVM (each job with its own launch bounds), linked into one module, one PTX text. The slowest
# kernel (k_token4!, about 56 s) compiles here, where this file is already loaded; the rest compile meanwhile in
# FLASHNEXT_KWORKERS worker processes (julia runs one thread) as bitcode, linked here (a worker spends about 28 s
# loading this file, off the critical path this way)
function module_ptx(dir)
    ks = b1_kernels()
    slow = something(findfirst(k -> k[1] === k_token8!, ks), findfirst(k -> k[1] === k_token4!, ks), findfirst(k -> k[1] === k_token!, ks))
    k1 = findfirst(k -> k[3] == 0 && k[4] == 0, ks)        # (a kernel without launch bounds: the job emitting the PTX)
    rest = [i for i in eachindex(ks) if i != slow && i != k1]
    order = vcat(filter(i -> ks[i][1] in heavy_kernels(), rest), filter(i -> !(ks[i][1] in heavy_kernels()), rest))
    nw = clamp(parse(Int, get(ENV, "FLASHNEXT_KWORKERS", "3")), 0, length(order))
    parts = [order[w:max(1, nw):end] for w in 1:nw]
    cmd(p) = `$(Base.julia_cmd()) --project=$(Base.active_project()) -e "include(\"$SRCFILE\"); exit(main([\"--kernel-bc\", \"$dir\", \"$(join(p, ","))\"]))"`
    procs = [run(pipeline(cmd(p); stdout, stderr); wait = false) for p in parts if !isempty(p)]
    heavy = nw == 0 ? Int[] : order
    here = nw == 0 ? collect(eachindex(ks)) : [k1; slow]
    GPUC.JuliaContext() do ctx
        mod = nothing; job1 = nothing; rel = nothing
        for i in here
            ir, meta = GPUC.compile(:llvm, kjob(ks[i]))
            if mod === nothing
                mod = ir; job1 = kjob(ks[i]); rel = meta.relocations
            else
                LLVMJ.link!(mod, ir)
            end
        end
        foreach(wait, procs)
        all(success, procs) || error("module_ptx: a kernel worker failed")
        nw > 0 && for i in heavy
            LLVMJ.link!(mod, parse(LLVMJ.Module, read(joinpath(dir, "k$i.bc"))))
            rm(joinpath(dir, "k$i.bc"))
        end
        asm, _ = GPUC.emit_asm(job1, mod, rel, LLVMJ.CodeGenFileType.Assembly)
        asm
    end
end
function build_kernels(dir)
    t0 = time()
    # every plain f32 mul / add / sub as .rn: ptxas (or the driver's JIT) may fuse a contractable mul and add into an
    # fma, choosing per kernel; the explicit fma calls are unchanged
    asm = replace(module_ptx(dir), r"(?m)^(\s*(?:@!?%p\d+\s+)?)(mul|add|sub)\.f32\b" => s"\1\2.rn.f32")   # (not atom.add)
    p = joinpath(dir, "flashnext.ptx"); write(p, asm)
    note(@sprintf("wrote %s (%d kernels, %.1f MB) in %.1f s", p, length(b1_kernels()), length(asm) / 1e6, time() - t0))
    # a raw-pointer unsafe_store! without Val(4) is split into byte stores; of a float register's 4 bytes (st.shared.b8
    # of the register itself, then of its shifts) the low one arrived as 0 (the router weights' low 8 mantissa bits in
    # k_token8!), so a module with any is reported
    nb = count("st.shared.b8", asm)
    nb > 0 && note(@sprintf("%d byte stores to shared memory (an unaligned raw-pointer store: give it Val(4))", nb))
    ptxas = get(ENV, "PTXAS", "ptxas")
    c = joinpath(dir, "flashnext.sass")
    t1 = time()
    run(`$ptxas -arch=sm_52 -O3 $p -o $c`)
    note(@sprintf("wrote %s (%.1f MB) in %.1f s", c, filesize(c) / 1e6, time() - t1))
    0
end
# --kernels ptx FILE | sass FILE: the kernels come from that module (the driver JIT-compiles a .ptx once per die)
const KMODE = Ref{Any}(nothing)            # nothing: JIT through CUDA.jl; else the module's bytes
const KPATH = Ref{Any}(nothing)            # the module's file
const KSTAT = Ref{Any}(nothing)            # its stat when loaded (a request naming the same unchanged file reloads nothing)
const KEEPMODS = Any[]                     # replaced modules, kernels and draft graphs, kept alive (see reload_kernels!)
const KMODS = Dict{Any,Any}()
function getkernel(f, tt; maxregs = 0, maxthreads = 0)
    get!(KCACHE, (:k, f, tt, CUDA.device())) do
        kw = (; (maxregs > 0 ? (; maxregs) : (;))..., (maxthreads > 0 ? (; maxthreads) : (;))...)
        if KMODE[] === nothing
            cufunction(f, tt; kw...)
        else
            mod = get!(() -> CuModule(KMODE[]), KMODS, CUDA.device())
            fun = try
                CuFunction(mod, kentry(f, tt))
            catch e
                (e isa CUDA.CuError && f in bench_kernels()) || rethrow()
                note("$(nameof(f)) is not in the module (built without FLASHNEXT_BENCHK=1): compiling it with CUDA.jl")
                return cufunction(f, tt; kw...)
            end
            CUC.HostKernel{typeof(f),tt}(f, fun, CUC.KernelState(CUC.create_exceptions!(mod), UInt32(0)))
        end
    end
end
function ptx_all(dir)
    bad = 0
    for (name, f, tt) in kernel_list()
        try
            s = kernel_ptx(f, tt)
            write(joinpath(dir, name * ".ptx"), s)
        catch e
            bad += 1
            println("ERROR ", name, ": ", first(sprint(showerror, e), 4000))
        end
    end
    bad
end

# ---------------------------------------------------------------------------------------------
# benches and checks
# ---------------------------------------------------------------------------------------------
function upload_tensor(g::GModel, t::GTensor; rows = nrows(t), L = haskey(ENV, "FLASHNEXT_BENCH_L") ? parse(Int, ENV["FLASHNEXT_BENCH_L"]) :
                       lanes_for(t.typ, t.dims[1]; wide = get(ENV, "FLASHNEXT_BENCH_WIDE", "") == "1"))
    K = t.dims[1]
    buf = zeros(UInt8, devbytes_rows(t.typ, K, rows, L))
    GC.@preserve buf repack_il!(pointer(buf), pointer(buf) + devpay(t.typ, K, rows, L), t.typ, K, 0, tptr(g, t), rows, rowbytes(t), L)
    d = CuArray(buf)
    d, DevMat(t.typ, rows, K, pointer(d), L)
end

# the B1 GEMV phase alone, in k_token!'s launch shape (grid x 256, 4 XSMAX bytes of shared memory, its register cap):
# reps phases of one matrix on an LL x, back to back
function k_phasebench!(m::MatI, rows::Int32, x::U64P, y::U64P, xsmax::Int32, reps::Int32, mode::Int32, idxg::IPtr, m2::MatI, m3::MatI,
                       dbg::Core.LLVMPtr{UInt64,1})
    xs = pointer(CuDynamicSharedArray(Float32, (Int(xsmax),)))
    ixs = CuStaticSharedArray(Int32, 16); wvs = CuStaticSharedArray(Float32, 16)
    tid = threadIdx().x - Int32(1)
    tid < Int32(16) && (ixs[tid+1] = unsafe_load(idxg, tid + 1, Val(4)); wvs[tid+1] = 0.1f0)
    sync_threads()
    for r in Int32(1):reps
        cb = idxg + 4 * (Int32(17) + r)                 # rep r's group counter: idxg[18 + r]
        xf = XF(1, x, m.K, UInt32(1))
        if mode == Int32(12)           # the x load alone
            xload!(xs, xs + 4 * xu(m.K), xf, m.K, NORL)
            sync_threads()
            continue
        end
        if mode == Int32(4)
            phase0!(xf, m.K, xs, NORL, (Job{4}(m, rows, EM(3, y, UInt32(r); nc = m.K ÷ Int32(4))),), cb)
        elseif mode == Int32(1)        # the experts phase: 10 gathered experts (2 x 320 rows each) and the shared gate/up
            cphase!(xf, m.K, xs, NORL, pointer(ixs), pointer(wvs), NORED, (Job{1}(m, rows, EM(5, y, UInt32(r)); rpe = rows ÷ Int32(10)),
                Job{0}(m2, m2.rows, EM(1, y, UInt32(r); o1 = rows)), Job{0}(m3, m3.rows, EM(1, y, UInt32(r); o1 = rows + m2.rows))), cb)
        elseif mode == Int32(5)        # the gathered experts alone
            cphase!(xf, m.K, xs, NORL, pointer(ixs), pointer(wvs), NORED, (Job{1}(m, rows, EM(5, y, UInt32(r)); rpe = rows ÷ Int32(10)),), cb)
        elseif mode == Int32(7)        # the gathered experts as two jobs of the same kind
            h = rows ÷ Int32(2)
            cphase!(xf, m.K, xs, NORL, pointer(ixs), pointer(wvs), NORED, (Job{1}(m, h, EM(5, y, UInt32(r)); rpe = rows ÷ Int32(10)),
                Job{1}(m, h, EM(5, y, UInt32(r); o1 = h ÷ Int32(2)); rpe = rows ÷ Int32(10))), cb)
        elseif mode == Int32(8)        # the gathered experts and 16 rows of the shared up
            cphase!(xf, m.K, xs, NORL, pointer(ixs), pointer(wvs), NORED, (Job{0}(m3, Int32(16), EM(1, y, UInt32(r); o1 = rows)),
                Job{1}(m, rows, EM(5, y, UInt32(r)); rpe = rows ÷ Int32(10))), cb)
        elseif mode == Int32(9)        # two matrices of different formats in one phase (qkv and gate of a DeltaNet block)
            cphase!(xf, m.K, xs, NORL, pointer(ixs), pointer(wvs), NORED, (Job{0}(m, rows, EM(1, y, UInt32(r))),
                Job{0}(m2, m2.rows, EM(1, y, UInt32(r); o1 = rows))), cb)
        elseif mode == Int32(11)       # hc up with its mix epilogue (S at y + 8 * 8192 with seq 1, weights there too as floats)
            Sp = y + 8 * 8192
            cphase!(xf, m.K, xs, NORL, NOSI, NOSF, NORED, (Job{0}(m, rows, EM(4, y, UInt32(r); nc = rows ÷ Int32(4), S = Sp, sS = UInt32(1),
                w = reinterpret(FPtr, Sp), zoff = Int32(0))),), cb)
        elseif mode == Int32(6)        # the shared gate/up alone
            cphase!(xf, m.K, xs, NORL, pointer(ixs), pointer(wvs), NORED, (Job{0}(m2, m2.rows, EM(1, y, UInt32(r); o1 = rows)),
                Job{0}(m3, m3.rows, EM(1, y, UInt32(r); o1 = rows + m2.rows))), cb)
        else
            phase0!(xf, m.K, xs, NORL, (Job{0}(m, rows, EM(1, y, UInt32(r))),), cb)
        end
    end
    return
end

# ---- B4: four positions per weight read (DFlash2's verify and its draft) ----
# x of the 4 positions interleaved in shared memory: float4 xs4[xu4(k)] = (x_0[k], x_1[k], x_2[k], x_3[k]), xu4(k) =
# k + k ÷ 32, so the lanes of a quarter-warp reading blocks 32 or 64 elements apart hit distinct banks (blocks 256
# apart share them: 2 ways for L = 2); 16-sums x164[g] likewise. K = 2560: 48.6 KB.
@inline xu4(k::Int32) = k + (k >> Int32(5))
xu4(k::Integer) = k + (k >> 5)
@inline xo4(c::Int) = c + c ÷ 32
@inline x4at(xs4::SPtr, i::Int32) = sx4(xs4, Int32(4) * i)          # the float4 at xu4 index i
@inline fma4(f::Float32, x::V4F, a::NTuple{4,Float32}) = (fma(f, x[1].value, a[1]), fma(f, x[2].value, a[2]), fma(f, x[3].value, a[3]), fma(f, x[4].value, a[4]))
@inline add4(a::NTuple{4,Float32}, b::NTuple{4,Float32}) = (a[1] + b[1], a[2] + b[2], a[3] + b[3], a[4] + b[4])
@inline prmtf(q4::UInt32, e::Int) = reinterpret(Float32, CUDA.byte_perm(q4, 0x3f000000, UInt32(0x7054 | (e << 8))))
const Z4 = (0.0f0, 0.0f0, 0.0f0, 0.0f0)
@inline function bcomp4(::Val{T_BF16}, r, xs4::SPtr, x164::SPtr, xb4::Int32, g16::Int32)
    a = Z4; b = Z4
    Base.Cartesian.@nexprs 4 c -> begin
        v = r[c]
        Base.Cartesian.@nexprs 4 q -> begin
            e = 8(c - 1) + 2(q - 1)
            a = fma4(reinterpret(Float32, v[q] << 16), x4at(xs4, xb4 + Int32(e)), a)
            b = fma4(reinterpret(Float32, v[q] & 0xffff0000), x4at(xs4, xb4 + Int32(e + 1)), b)
        end
    end
    add4(a, b)
end
# s + sum_b f_b x4[xq + 4 b] (byte b of q4 is element 4 b of a run: the 2-bit layouts)
@inline function dotp44s(q4::UInt32, xs4::SPtr, xq::Int32, s::NTuple{4,Float32})
    Base.Cartesian.@nexprs 4 e -> (s = fma4(prmtf(q4, e - 1), x4at(xs4, xq + Int32(4(e - 1))), s))
    s
end
# Q2_0: 64 codes of 2 bits (element m at bits 2m of the 16 bytes), d (q - 1); codes at bits 5-6 of the bytes:
# f = 0.5 + q / 8, sum (q - 1) x = 8 sum f x - 5 X
@inline function bcomp4(::Val{T_Q2_0}, r, xs4::SPtr, x164::SPtr, xb4::Int32, g16::Int32)
    v, h = r; d = hf(h)
    s = Z4
    Base.Cartesian.@nexprs 4 w -> begin
        xw = xb4 + Int32(xo4(16(w - 1)))
        Base.Cartesian.@nexprs 4 k -> (s = dotp44s(bitrotate(v[w], 5 - 2(k - 1)) & 0x60606060, xs4, xw + Int32(k - 1), s))
    end
    X = add4v(add4v(x4at(x164, g16), x4at(x164, g16 + Int32(1))), add4v(x4at(x164, g16 + Int32(2)), x4at(x164, g16 + Int32(3))))
    Base.Cartesian.@ntuple 4 p -> d * fma(8.0f0, s[p], -5.0f0 * X[p].value)
end
# Q4_0: nibbles (element c low, c + 16 high of byte c), d (q - 8): f = 0.5 + q / 32, sum (q - 8) x = 32 sum f x - 24 X
@inline function bcomp4(::Val{T_Q4_0}, r, xs4::SPtr, x164::SPtr, xb4::Int32, g16::Int32)
    v, h = r; d = hf(h)
    s = Z4
    Base.Cartesian.@nexprs 4 w -> begin
        s = dotp44(bitrotate(v[w], 3) & 0x78787878, xs4, xb4 + Int32(4(w - 1)), s)
        s = dotp44(bitrotate(v[w], -1) & 0x78787878, xs4, xb4 + Int32(16 + 4(w - 1)), s)
    end
    X = add4v(x4at(x164, g16), x4at(x164, g16 + Int32(1)))
    Base.Cartesian.@ntuple 4 p -> d * fma(32.0f0, s[p], -24.0f0 * X[p].value)
end
# Q8_0: bytes c = q + 128 (sign flipped) as integers (8-bit codes: the byte permute's codes must leave bit 7 clear,
# it lands in the exponent): sum q x = sum c x - 128 X
@inline function bcomp4(::Val{T_Q8_0}, r, xs4::SPtr, x164::SPtr, xb4::Int32, g16::Int32)
    v1, v2, h = r; d = hf(h)
    s = Z4
    Base.Cartesian.@nexprs 4 w -> begin
        a_w = v1[w] ⊻ 0x80808080; b_w = v2[w] ⊻ 0x80808080
        Base.Cartesian.@nexprs 4 e -> begin
            s = fma4(u2f((a_w >> (8(e - 1))) & 0xff), x4at(xs4, xb4 + Int32(4(w - 1) + e - 1)), s)
            s = fma4(u2f((b_w >> (8(e - 1))) & 0xff), x4at(xs4, xb4 + Int32(16 + 4(w - 1) + e - 1)), s)
        end
    end
    X = add4v(x4at(x164, g16), x4at(x164, g16 + Int32(1)))
    Base.Cartesian.@ntuple 4 p -> d * fma(-128.0f0, X[p].value, s[p])
end
# the LUT formats (IQ4_NL, IQ4_XS): each nibble's value from the shared table, 4 positions per value
@inline function dot4b4lut4(w::UInt32, sh, xs4::SPtr, xq::Int32, s::NTuple{4,Float32}, lut::SPtr)
    Base.Cartesian.@nexprs 4 e -> (s = fma4(sld(lut, Int32((w >> (8(e - 1) + sh)) & 0xf)), x4at(xs4, xq + Int32(e - 1)), s))
    s
end
@inline function bcomp4(::Val{T_IQ4_NL}, r, xs4::SPtr, x164::SPtr, xb4::Int32, g16::Int32, lut::SPtr)
    v, h = r; d = hf(h)
    s = Z4
    Base.Cartesian.@nexprs 4 w -> begin
        s = dot4b4lut4(v[w], 0, xs4, xb4 + Int32(4(w - 1)), s, lut)
        s = dot4b4lut4(v[w], 4, xs4, xb4 + Int32(16 + 4(w - 1)), s, lut)
    end
    Base.Cartesian.@ntuple 4 p -> d * s[p]
end
# Q5_0: 4 low bits + the high bit from qh (element j: bit j, element j + 16: bit j + 16), d (q - 16)
@inline function bcomp4(::Val{T_Q5_0}, r, xs4::SPtr, x164::SPtr, xb4::Int32, g16::Int32)
    v, h = r
    d = hf(UInt16(h & 0xffff)); qh = UInt32(h >> 32)
    s = Z4
    Base.Cartesian.@nexprs 4 w -> begin
        Base.Cartesian.@nexprs 4 e -> begin
            j = 4(w - 1) + e - 1
            s = fma4(u2f(((v[w] >> (8(e - 1))) & 0xf) | (((qh >> j) & 0x1) << 4)), x4at(xs4, xb4 + Int32(j)), s)
            s = fma4(u2f(((v[w] >> (8(e - 1) + 4)) & 0xf) | (((qh >> (j + 16)) & 0x1) << 4)), x4at(xs4, xb4 + Int32(16 + j)), s)
        end
    end
    X = add4v(x4at(x164, g16), x4at(x164, g16 + Int32(1)))
    Base.Cartesian.@ntuple 4 p -> d * (s[p] - 16.0f0 * X[p].value)
end
@inline function bdot4(::Val{T_IQ4_XS}, pay::GPtr, hdr::GPtr, xs4::SPtr, x164::SPtr, xb4::Int32, g16::Int32, lut::SPtr)
    h = ldu64(hdr, Int32(0))
    d = hf(UInt16(h & 0xffff)); sh = UInt32((h >> 16) & 0xffff); sl = UInt32(h >> 32)
    acc = Z4
    p = Int32(0)
    cur = (ld16(pay, Int32(0)), ld16(pay, Int32(CHS)))
    @nounroll while p < Int32(4)               # pairs of 32-element runs, the next pair's chunks loaded first
        xq = xb4 + Int32(66) * p                 # xu4 of element 64 p
        nxt = p < Int32(3) ? (ld16(pay, (Int32(2) * p + Int32(2)) * Int32(CHS)), ld16(pay, (Int32(2) * p + Int32(3)) * Int32(CHS))) : cur
        Base.Cartesian.@nexprs 2 k -> begin
            ib = Int32(2) * p + Int32(k - 1)
            q = cur[k]
            s = Z4
            Base.Cartesian.@nexprs 4 w -> begin
                s = dot4b4lut4(q[w], 0, xs4, xq + Int32(xo4(32(k - 1) + 4(w - 1))), s, lut)
                s = dot4b4lut4(q[w], 4, xs4, xq + Int32(xo4(32(k - 1) + 16 + 4(w - 1))), s, lut)
            end
            ls = u2f(((sl >> (Int32(4) * ib)) & 0xf) | (((sh >> (Int32(2) * ib)) & 0x3) << 4)) - 32.0f0
            acc = Base.Cartesian.@ntuple 4 pp -> fma(ls, s[pp], acc[pp])
        end
        cur = nxt
        p += Int32(1)
    end
    Base.Cartesian.@ntuple 4 pp -> d * acc[pp]
end
# one block of 4 positions, any format: the split formats' loads then arithmetic, the others' bdot4
@inline function blk4(::Val{T}, pay::GPtr, hdr::GPtr, xs4::SPtr, x164::SPtr, xb4::Int32, g16::Int32, lut::SPtr) where {T}
    T == T_IQ4_NL && return bcomp4(Val(T), bload(Val(T), pay, hdr), xs4, x164, xb4, g16, lut)
    T == T_IQ4_XS && return bdot4(Val(T), pay, hdr, xs4, x164, xb4, g16, lut)
    issplit(T) && return bcomp4(Val(T), bload(Val(T), pay, hdr), xs4, x164, xb4, g16)
    bdot4(Val(T), pay, hdr, xs4, x164, xb4, g16)
end
@inline function bdot4(::Val{T_Q3_K}, pay::GPtr, hdr::GPtr, xs4::SPtr, x164::SPtr, xb4::Int32, g16::Int32)
    hd = ld16(hdr, Int32(0))
    d = hf(hd[4])
    km1, km2 = 0x03030303, 0x0f0f0f0f
    a0, a1, tmp = hd[1], hd[2], hd[3]
    sc = ((a0 & km2) | ((tmp & km1) << 4), (a1 & km2) | (((tmp >> 2) & km1) << 4),
          ((a0 >> 4) & km2) | (((tmp >> 4) & km1) << 4), ((a1 >> 4) & km2) | (((tmp >> 6) & km1) << 4))
    hm1 = pld(pay, (0)); hm2 = pld(pay, (16))
    acc = Z4
    h = Int32(0)
    cur = (ld16(pay, Int32(2) * Int32(CHS)), ld16(pay, Int32(3) * Int32(CHS)))
    @nounroll while h < Int32(2)
        nxt = h == Int32(0) ? (ld16(pay, Int32(4) * Int32(CHS)), ld16(pay, Int32(5) * Int32(CHS))) : cur
        qa, qb = cur
        xq = xb4 + Int32(132) * h                # xu4 of element 128 h
        rh = (Int32(6) - Int32(4) * h) & Int32(31)
        Base.Cartesian.@nexprs 4 j -> begin
            sa = Z4; sb = Z4
            rj = (rh - Int32(j - 1)) & Int32(31)
            Base.Cartesian.@nexprs 4 w -> begin
                qa4 = (bitrotate(qa[w], 4 - 2(j - 1)) & 0x30303030) | (bitrotate(hm1[w], rj) & 0x40404040)
                qb4 = (bitrotate(qb[w], 4 - 2(j - 1)) & 0x30303030) | (bitrotate(hm2[w], rj) & 0x40404040)
                Base.Cartesian.@nexprs 4 e -> begin
                    sa = fma4(prmtf(qa4, e - 1), x4at(xs4, xq + Int32(xo4(32(j - 1) + 4(w - 1) + e - 1))), sa)
                    sb = fma4(prmtf(qb4, e - 1), x4at(xs4, xq + Int32(xo4(32(j - 1) + 16 + 4(w - 1) + e - 1))), sb)
                end
            end
            g = Int32(8) * h + Int32(2(j - 1))
            Xa = x4at(x164, g16 + g); Xb = x4at(x164, g16 + g + Int32(1))
            ca = sbyter(sc, g) - 32.0f0; cb = sbyter(sc, g + Int32(1)) - 32.0f0
            acc = Base.Cartesian.@ntuple 4 p -> fma(ca, fma(16.0f0, sa[p], -12.0f0 * Xa[p].value), fma(cb, fma(16.0f0, sb[p], -12.0f0 * Xb[p].value), acc[p]))
        end
        cur = nxt
        h += Int32(1)
    end
    (d * acc[1], d * acc[2], d * acc[3], d * acc[4])
end
# s + sum_e f_e x4[xq + o_e] for the 4 bytes of q4 (codes as in dotp4), elements at xu4 offsets o .. o + 3
@inline function dotp44(q4::UInt32, xs4::SPtr, xq::Int32, s::NTuple{4,Float32})
    Base.Cartesian.@nexprs 4 e -> (s = fma4(prmtf(q4, e - 1), x4at(xs4, xq + Int32(e - 1)), s))
    s
end
@inline sc4(c::Float32, a::NTuple{4,Float32}, X::V4F, k::Float32, acc::NTuple{4,Float32}) =
    Base.Cartesian.@ntuple 4 p -> fma(c, fma(k, a[p], X[p].value), acc[p])
@inline function bdot4(::Val{T_Q4_K}, pay::GPtr, hdr::GPtr, xs4::SPtr, x164::SPtr, xb4::Int32, g16::Int32)
    h = ld16(hdr, Int32(0))
    d = hf(h[1]); dm = hf(h[1] >> 16)
    acc = Z4; macc = Z4
    i = Int32(0)
    @nounroll while i < Int32(4)               # 64-element runs
        q1 = ld16(pay, Int32(2) * i * Int32(CHS)); q2 = ld16(pay, (Int32(2) * i + Int32(1)) * Int32(CHS))
        xq = xb4 + Int32(66) * i                 # xu4 of element 64 i
        a = Z4; b = Z4
        Base.Cartesian.@nexprs 4 w -> begin
            a = dotp44(bitrotate(q1[w], 3) & 0x78787878, xs4, xq + Int32(xo4(4(w - 1))), a)
            a = dotp44(bitrotate(q2[w], 3) & 0x78787878, xs4, xq + Int32(xo4(16 + 4(w - 1))), a)
            b = dotp44(bitrotate(q1[w], -1) & 0x78787878, xs4, xq + Int32(xo4(32 + 4(w - 1))), b)
            b = dotp44(bitrotate(q2[w], -1) & 0x78787878, xs4, xq + Int32(xo4(48 + 4(w - 1))), b)
        end
        X1 = add4v(x4at(x164, g16 + Int32(4) * i), x4at(x164, g16 + Int32(4) * i + Int32(1)))
        X2 = add4v(x4at(x164, g16 + Int32(4) * i + Int32(2)), x4at(x164, g16 + Int32(4) * i + Int32(3)))
        sc1, m1 = kscale(Int32(2) * i, h[2], h[3], h[4]); sc2, m2 = kscale(Int32(2) * i + Int32(1), h[2], h[3], h[4])
        # sum q x = 32 sum f x - 16 X
        acc = sc4(u2f(sc1), a, scl4(-16.0f0, X1), 32.0f0, acc); acc = sc4(u2f(sc2), b, scl4(-16.0f0, X2), 32.0f0, acc)
        macc = Base.Cartesian.@ntuple 4 p -> fma(u2f(m1), X1[p].value, fma(u2f(m2), X2[p].value, macc[p]))
        i += Int32(1)
    end
    Base.Cartesian.@ntuple 4 p -> d * acc[p] - dm * macc[p]
end
# two rows' 4-element runs on one x load: the 4-position x (a float4 per element from shared memory) was the limit,
# one LDS.128 per weight (8 weights a clock per SM, ~85 GB/s of Q4_K); two rows on it halve that
@inline function dotp44x2(qa::UInt32, qb::UInt32, xs4::SPtr, xq::Int32, sa::NTuple{4,Float32}, sb::NTuple{4,Float32})
    Base.Cartesian.@nexprs 4 e -> begin
        xv = x4at(xs4, xq + Int32(e - 1))
        sa = fma4(prmtf(qa, e - 1), xv, sa); sb = fma4(prmtf(qb, e - 1), xv, sb)
    end
    sa, sb
end
@inline function bdot4x2(::Val{T_Q4_K}, pa::GPtr, ha::GPtr, pb::GPtr, hb::GPtr, xs4::SPtr, x164::SPtr, xb4::Int32, g16::Int32)
    hA = ld16(ha, Int32(0)); hB = ld16(hb, Int32(0))
    dA = hf(hA[1]); dmA = hf(hA[1] >> 16); dB = hf(hB[1]); dmB = hf(hB[1] >> 16)
    accA = Z4; maccA = Z4; accB = Z4; maccB = Z4
    i = Int32(0)
    @nounroll while i < Int32(4)
        q1A = ld16(pa, Int32(2) * i * Int32(CHS)); q2A = ld16(pa, (Int32(2) * i + Int32(1)) * Int32(CHS))
        q1B = ld16(pb, Int32(2) * i * Int32(CHS)); q2B = ld16(pb, (Int32(2) * i + Int32(1)) * Int32(CHS))
        xq = xb4 + Int32(66) * i
        aA = Z4; bA = Z4; aB = Z4; bB = Z4
        Base.Cartesian.@nexprs 4 w -> begin
            aA, aB = dotp44x2(bitrotate(q1A[w], 3) & 0x78787878, bitrotate(q1B[w], 3) & 0x78787878, xs4, xq + Int32(xo4(4(w - 1))), aA, aB)
            aA, aB = dotp44x2(bitrotate(q2A[w], 3) & 0x78787878, bitrotate(q2B[w], 3) & 0x78787878, xs4, xq + Int32(xo4(16 + 4(w - 1))), aA, aB)
            bA, bB = dotp44x2(bitrotate(q1A[w], -1) & 0x78787878, bitrotate(q1B[w], -1) & 0x78787878, xs4, xq + Int32(xo4(32 + 4(w - 1))), bA, bB)
            bA, bB = dotp44x2(bitrotate(q2A[w], -1) & 0x78787878, bitrotate(q2B[w], -1) & 0x78787878, xs4, xq + Int32(xo4(48 + 4(w - 1))), bA, bB)
        end
        X1 = add4v(x4at(x164, g16 + Int32(4) * i), x4at(x164, g16 + Int32(4) * i + Int32(1)))
        X2 = add4v(x4at(x164, g16 + Int32(4) * i + Int32(2)), x4at(x164, g16 + Int32(4) * i + Int32(3)))
        s1A, m1A = kscale(Int32(2) * i, hA[2], hA[3], hA[4]); s2A, m2A = kscale(Int32(2) * i + Int32(1), hA[2], hA[3], hA[4])
        s1B, m1B = kscale(Int32(2) * i, hB[2], hB[3], hB[4]); s2B, m2B = kscale(Int32(2) * i + Int32(1), hB[2], hB[3], hB[4])
        accA = sc4(u2f(s1A), aA, scl4(-16.0f0, X1), 32.0f0, accA); accA = sc4(u2f(s2A), bA, scl4(-16.0f0, X2), 32.0f0, accA)
        accB = sc4(u2f(s1B), aB, scl4(-16.0f0, X1), 32.0f0, accB); accB = sc4(u2f(s2B), bB, scl4(-16.0f0, X2), 32.0f0, accB)
        maccA = Base.Cartesian.@ntuple 4 p -> fma(u2f(m1A), X1[p].value, fma(u2f(m2A), X2[p].value, maccA[p]))
        maccB = Base.Cartesian.@ntuple 4 p -> fma(u2f(m1B), X1[p].value, fma(u2f(m2B), X2[p].value, maccB[p]))
        i += Int32(1)
    end
    (Base.Cartesian.@ntuple 4 p -> dA * accA[p] - dmA * maccA[p]), (Base.Cartesian.@ntuple 4 p -> dB * accB[p] - dmB * maccB[p])
end
@inline function bdot4x2(::Val{T_Q5_K}, pa::GPtr, ha::GPtr, pb::GPtr, hb::GPtr, xs4::SPtr, x164::SPtr, xb4::Int32, g16::Int32)
    hA = ld16(ha, Int32(0)); hB = ld16(hb, Int32(0))
    dA = hf(hA[1]); dmA = hf(hA[1] >> 16); dB = hf(hB[1]); dmB = hf(hB[1] >> 16)
    qaA = pld(pa, (0)); qbA = pld(pa, (16)); qaB = pld(pb, (0)); qbB = pld(pb, (16))
    accA = Z4; maccA = Z4; accB = Z4; maccB = Z4
    i = Int32(0)
    @nounroll while i < Int32(4)
        q1A = ld16(pa, (Int32(2) + Int32(2) * i) * Int32(CHS)); q2A = ld16(pa, (Int32(3) + Int32(2) * i) * Int32(CHS))
        q1B = ld16(pb, (Int32(2) + Int32(2) * i) * Int32(CHS)); q2B = ld16(pb, (Int32(3) + Int32(2) * i) * Int32(CHS))
        xq = xb4 + Int32(66) * i
        ra = (Int32(6) - Int32(2) * i) & Int32(31); rb = (Int32(5) - Int32(2) * i) & Int32(31)
        aA = Z4; bA = Z4; aB = Z4; bB = Z4
        Base.Cartesian.@nexprs 4 w -> begin
            aA, aB = dotp44x2((bitrotate(q1A[w], 2) & 0x3c3c3c3c) | (bitrotate(qaA[w], ra) & 0x40404040),
                              (bitrotate(q1B[w], 2) & 0x3c3c3c3c) | (bitrotate(qaB[w], ra) & 0x40404040), xs4, xq + Int32(xo4(4(w - 1))), aA, aB)
            aA, aB = dotp44x2((bitrotate(q2A[w], 2) & 0x3c3c3c3c) | (bitrotate(qbA[w], ra) & 0x40404040),
                              (bitrotate(q2B[w], 2) & 0x3c3c3c3c) | (bitrotate(qbB[w], ra) & 0x40404040), xs4, xq + Int32(xo4(16 + 4(w - 1))), aA, aB)
            bA, bB = dotp44x2((bitrotate(q1A[w], -2) & 0x3c3c3c3c) | (bitrotate(qaA[w], rb) & 0x40404040),
                              (bitrotate(q1B[w], -2) & 0x3c3c3c3c) | (bitrotate(qaB[w], rb) & 0x40404040), xs4, xq + Int32(xo4(32 + 4(w - 1))), bA, bB)
            bA, bB = dotp44x2((bitrotate(q2A[w], -2) & 0x3c3c3c3c) | (bitrotate(qbA[w], rb) & 0x40404040),
                              (bitrotate(q2B[w], -2) & 0x3c3c3c3c) | (bitrotate(qbB[w], rb) & 0x40404040), xs4, xq + Int32(xo4(48 + 4(w - 1))), bA, bB)
        end
        X1 = add4v(x4at(x164, g16 + Int32(4) * i), x4at(x164, g16 + Int32(4) * i + Int32(1)))
        X2 = add4v(x4at(x164, g16 + Int32(4) * i + Int32(2)), x4at(x164, g16 + Int32(4) * i + Int32(3)))
        s1A, m1A = kscale(Int32(2) * i, hA[2], hA[3], hA[4]); s2A, m2A = kscale(Int32(2) * i + Int32(1), hA[2], hA[3], hA[4])
        s1B, m1B = kscale(Int32(2) * i, hB[2], hB[3], hB[4]); s2B, m2B = kscale(Int32(2) * i + Int32(1), hB[2], hB[3], hB[4])
        accA = sc4(u2f(s1A), aA, scl4(-32.0f0, X1), 64.0f0, accA); accA = sc4(u2f(s2A), bA, scl4(-32.0f0, X2), 64.0f0, accA)
        accB = sc4(u2f(s1B), aB, scl4(-32.0f0, X1), 64.0f0, accB); accB = sc4(u2f(s2B), bB, scl4(-32.0f0, X2), 64.0f0, accB)
        maccA = Base.Cartesian.@ntuple 4 p -> fma(u2f(m1A), X1[p].value, fma(u2f(m2A), X2[p].value, maccA[p]))
        maccB = Base.Cartesian.@ntuple 4 p -> fma(u2f(m1B), X1[p].value, fma(u2f(m2B), X2[p].value, maccB[p]))
        i += Int32(1)
    end
    (Base.Cartesian.@ntuple 4 p -> dA * accA[p] - dmA * maccA[p]), (Base.Cartesian.@ntuple 4 p -> dB * accB[p] - dmB * maccB[p])
end
@inline function bdot4(::Val{T_Q5_K}, pay::GPtr, hdr::GPtr, xs4::SPtr, x164::SPtr, xb4::Int32, g16::Int32)
    h = ld16(hdr, Int32(0))
    d = hf(h[1]); dm = hf(h[1] >> 16)
    qa = pld(pay, (0)); qb = pld(pay, (16))
    acc = Z4; macc = Z4
    i = Int32(0)
    @nounroll while i < Int32(4)
        q1 = ld16(pay, (Int32(2) + Int32(2) * i) * Int32(CHS)); q2 = ld16(pay, (Int32(3) + Int32(2) * i) * Int32(CHS))
        xq = xb4 + Int32(66) * i
        ra = (Int32(6) - Int32(2) * i) & Int32(31); rb = (Int32(5) - Int32(2) * i) & Int32(31)
        a = Z4; b = Z4
        Base.Cartesian.@nexprs 4 w -> begin
            a = dotp44((bitrotate(q1[w], 2) & 0x3c3c3c3c) | (bitrotate(qa[w], ra) & 0x40404040), xs4, xq + Int32(xo4(4(w - 1))), a)
            a = dotp44((bitrotate(q2[w], 2) & 0x3c3c3c3c) | (bitrotate(qb[w], ra) & 0x40404040), xs4, xq + Int32(xo4(16 + 4(w - 1))), a)
            b = dotp44((bitrotate(q1[w], -2) & 0x3c3c3c3c) | (bitrotate(qa[w], rb) & 0x40404040), xs4, xq + Int32(xo4(32 + 4(w - 1))), b)
            b = dotp44((bitrotate(q2[w], -2) & 0x3c3c3c3c) | (bitrotate(qb[w], rb) & 0x40404040), xs4, xq + Int32(xo4(48 + 4(w - 1))), b)
        end
        X1 = add4v(x4at(x164, g16 + Int32(4) * i), x4at(x164, g16 + Int32(4) * i + Int32(1)))
        X2 = add4v(x4at(x164, g16 + Int32(4) * i + Int32(2)), x4at(x164, g16 + Int32(4) * i + Int32(3)))
        sc1, m1 = kscale(Int32(2) * i, h[2], h[3], h[4]); sc2, m2 = kscale(Int32(2) * i + Int32(1), h[2], h[3], h[4])
        # sum q x = 64 sum f x - 32 X
        acc = sc4(u2f(sc1), a, scl4(-32.0f0, X1), 64.0f0, acc); acc = sc4(u2f(sc2), b, scl4(-32.0f0, X2), 64.0f0, acc)
        macc = Base.Cartesian.@ntuple 4 p -> fma(u2f(m1), X1[p].value, fma(u2f(m2), X2[p].value, macc[p]))
        i += Int32(1)
    end
    Base.Cartesian.@ntuple 4 p -> d * acc[p] - dm * macc[p]
end
@inline function bdot4(::Val{T_Q6_K}, pay::GPtr, hdr::GPtr, xs4::SPtr, x164::SPtr, xb4::Int32, g16::Int32)
    d = hf(ldu16(hdr, Int32(0)))
    sc = pld(pay, (192))
    acc = Z4
    q = Int32(0)
    @inline q6c(q) = (ld16(pay, (Int32(4) * (q >> 1) + (q & Int32(1))) * Int32(CHS)), ld16(pay, (Int32(4) * (q >> 1) + Int32(2) + (q & Int32(1))) * Int32(CHS)),
        ld16(pay, (Int32(8) + Int32(2) * (q >> 1) + (q & Int32(1))) * Int32(CHS)))
    cur = q6c(q)
    @nounroll while q < Int32(4)
        n = q >> 1; hh = q & Int32(1)
        nxt = q < Int32(3) ? q6c(q + Int32(1)) : cur
        l0, l1, qh = cur
        s1 = Z4; s2 = Z4; s3 = Z4; s4 = Z4
        xq = xb4 + Int32(132) * n + Int32(16) * hh
        Base.Cartesian.@nexprs 4 w -> begin
            o = 4(w - 1)
            s1 = dotp44((bitrotate(l0[w], 1) & 0x1e1e1e1e) | (bitrotate(qh[w], 5) & 0x60606060), xs4, xq + Int32(xo4(o)), s1)
            s2 = dotp44((bitrotate(l1[w], 1) & 0x1e1e1e1e) | (bitrotate(qh[w], 3) & 0x60606060), xs4, xq + Int32(xo4(o + 32)), s2)
            s3 = dotp44((bitrotate(l0[w], -3) & 0x1e1e1e1e) | (bitrotate(qh[w], 1) & 0x60606060), xs4, xq + Int32(xo4(o + 64)), s3)
            s4 = dotp44((bitrotate(l1[w], -3) & 0x1e1e1e1e) | (bitrotate(qh[w], -1) & 0x60606060), xs4, xq + Int32(xo4(o + 96)), s4)
        end
        g = Int32(8) * n + hh
        # sum (q - 32) x = 128 sum f x - 96 X
        acc = sc4(sbyter(sc, g), s1, scl4(-96.0f0, x4at(x164, g16 + g)), 128.0f0, acc)
        acc = sc4(sbyter(sc, g + Int32(2)), s2, scl4(-96.0f0, x4at(x164, g16 + g + Int32(2))), 128.0f0, acc)
        acc = sc4(sbyter(sc, g + Int32(4)), s3, scl4(-96.0f0, x4at(x164, g16 + g + Int32(4))), 128.0f0, acc)
        acc = sc4(sbyter(sc, g + Int32(6)), s4, scl4(-96.0f0, x4at(x164, g16 + g + Int32(6))), 128.0f0, acc)
        cur = nxt
        q += Int32(1)
    end
    Base.Cartesian.@ntuple 4 p -> d * acc[p]
end
@inline add4v(a::V4F, b::V4F) = (VecElement(a[1].value + b[1].value), VecElement(a[2].value + b[2].value), VecElement(a[3].value + b[3].value), VecElement(a[4].value + b[4].value))
@inline scl4(c::Float32, a::V4F) = (VecElement(c * a[1].value), VecElement(c * a[2].value), VecElement(c * a[3].value), VecElement(c * a[4].value))
@inline function wsum4(a::NTuple{4,Float32}, ::Val{L}) where {L}
    (wsum(a[1], Val(L)), wsum(a[2], Val(L)), wsum(a[3], Val(L)), wsum(a[4], Val(L)))
end
# one row group of m (rows of 4 positions) into y[4 r + p]
@inline function grp4!(::Val{T}, ::Val{L}, m::MatI, g::Int32, rows::Int32, y::FPtr, xs4::SPtr, x164::SPtr) where {T,L}
    E = Int32(devE(T)); H = Int32(devH(T)); NC = Int32(devP(T) ÷ 16); R = Int32(32 ÷ L)
    NI = (m.K ÷ E) ÷ Int32(L)
    lane = (threadIdx().x - Int32(1)) % Int32(32); rho = lane ÷ Int32(L); j = lane % Int32(L)
    acc = Z4
    gi = g * NI
    if T == T_BF16
        rr = bload(Val(T), m.pay + (gi * NC * Int32(32) + lane) * Int32(16), m.hdr + (gi * Int32(32) + lane) * H)
        i = Int32(0)
        @nounroll while i < NI
            nx = i + Int32(1) < NI ? bload(Val(T), m.pay + ((gi + Int32(1)) * NC * Int32(32) + lane) * Int32(16), m.hdr + ((gi + Int32(1)) * Int32(32) + lane) * H) : rr
            k0 = (j + Int32(L) * i) * E
            acc = add4(acc, bcomp4(Val(T), rr, xs4, x164, xu4(k0), k0 >> 4))
            rr = nx; gi += Int32(1); i += Int32(1)
        end
    else
        @nounroll for i in Int32(0):NI-Int32(1)
            k0 = (j + Int32(L) * i) * E
            acc = add4(acc, bdot4(Val(T), m.pay + ((gi + i) * NC * Int32(32) + lane) * Int32(16), m.hdr + ((gi + i) * Int32(32) + lane) * H, xs4, x164, xu4(k0), k0 >> 4))
        end
    end
    acc = wsum4(acc, Val(L))
    r = g * R + rho
    if j == Int32(0) && rho < R && r < rows
        Base.Cartesian.@nexprs 4 p -> stf(y, Int32(4) * r + Int32(p - 1), acc[p])
    end
end
# the B4 phase alone: x (4 x K floats, position-major) into xs4 and its 16-sums, then a static split of the row groups;
# reps times
function k_gemv4bench!(m::MatI, rows::Int32, x::FPtr, y::FPtr, reps::Int32)
    K = m.K
    xs4 = pointer(CuDynamicSharedArray(Float32, (Int(4 * (xu4(K) + K ÷ Int32(16))),)))
    x164 = xs4 + 16 * xu4(K)
    tid = threadIdx().x - Int32(1); nt = blockDim().x; nw = nt ÷ Int32(32); warp = tid ÷ Int32(32)
    for _ in Int32(1):reps
        k = tid
        while k < K
            Base.Cartesian.@nexprs 4 p -> unsafe_store!(xs4 + 4 * (Int32(4) * xu4(k) + Int32(p - 1)), ldf(x, Int32(p - 1) * K + k), 1, Val(4))
            k += nt
        end
        sync_threads()
        gq = tid
        while gq < K ÷ Int32(16)
            s = Z4
            for c in Int32(0):Int32(15)
                v = x4at(xs4, xu4(Int32(16) * gq + c))
                s = add4(s, (v[1].value, v[2].value, v[3].value, v[4].value))
            end
            Base.Cartesian.@nexprs 4 p -> unsafe_store!(x164 + 4 * (Int32(4) * gq + Int32(p - 1)), s[p], 1, Val(4))
            gq += nt
        end
        sync_threads()
        W = gridDim().x * nw; w = (blockIdx().x - Int32(1)) * nw + warp
        R = Int32(32) ÷ m.L
        G = (rows + R - Int32(1)) ÷ R
        g = (w * G) ÷ W; hi = ((w + Int32(1)) * G) ÷ W
        while g < hi
            if m.t == Int32(T_Q3_K) && m.L == Int32(2)
                grp4!(Val(T_Q3_K), Val(2), m, g, rows, y, xs4, x164)
            elseif m.t == Int32(T_BF16) && m.L == Int32(16)
                grp4!(Val(T_BF16), Val(16), m, g, rows, y, xs4, x164)
            elseif m.t == Int32(T_BF16) && m.L == Int32(2)
                grp4!(Val(T_BF16), Val(2), m, g, rows, y, xs4, x164)
            end
            g += Int32(1)
        end
        sync_threads()
    end
    return
end
# y[p ldy + r] (+)= sum_k m[r][k] x[p ldx + xoff + k] for the P <= 4 positions (rows of 4 positions per weight read):
# the draft's matrices and the target head's candidates. x into xs4 with its 16-sums, then a static split of the
# row groups over the warps of the grid.
# (row group, K chunk) of ks chunks: the chunk's blocks' partial sums added to y atomically (zeroed by the host first
# unless accumulating); for matrices of fewer row groups than warps
@inline function grp4k!(::Val{T}, ::Val{L}, m::MatI, g::Int32, c::Int32, ks::Int32, rows::Int32, y::FPtr, ldy::Int32, P::Int32,
                        xs4::SPtr, x164::SPtr) where {T,L}
    E = Int32(devE(T)); H = Int32(devH(T)); NC = Int32(devP(T) ÷ 16); R = Int32(32 ÷ L)
    NI = (m.K ÷ E) ÷ Int32(L)
    lane = (threadIdx().x - Int32(1)) % Int32(32); rho = lane ÷ Int32(L); j = lane % Int32(L)
    acc = Z4
    gi = g * NI
    @nounroll for i in (c * NI) ÷ ks:((c + Int32(1)) * NI) ÷ ks - Int32(1)
        k0 = (j + Int32(L) * i) * E
        acc = add4(acc, bdot4(Val(T), m.pay + ((gi + i) * NC * Int32(32) + lane) * Int32(16), m.hdr + ((gi + i) * Int32(32) + lane) * H, xs4, x164, xu4(k0), k0 >> 4))
    end
    acc = wsum4(acc, Val(L))
    r = g * R + rho
    if j == Int32(0) && rho < R && r < rows
        Base.Cartesian.@nexprs 4 p -> (Int32(p - 1) < P && CUDA.atomic_add!(y + 4 * (Int32(p - 1) * ldy + r), acc[p]))
    end
end
function k_gemv4!(m::MatI, rows::Int32, x::FPtr, ldx::Int32, xoff::Int32, y::FPtr, ldy::Int32, accum::Int32, P::Int32, ks::Int32)
    K = m.K
    xs4 = pointer(CuDynamicSharedArray(Float32, (Int(4 * (xu4(K) + K ÷ Int32(16))),)))
    x164 = xs4 + 16 * xu4(K)
    tid = threadIdx().x - Int32(1); nt = blockDim().x; nw = nt ÷ Int32(32); warp = tid ÷ Int32(32)
    @inbounds begin
        k = tid
        while k < K
            Base.Cartesian.@nexprs 4 p -> unsafe_store!(xs4 + 4 * (Int32(4) * xu4(k) + Int32(p - 1)),
                Int32(p - 1) < P ? ldf(x, Int32(p - 1) * ldx + xoff + k) : 0.0f0, 1, Val(4))
            k += nt
        end
        sync_threads()
        gq = tid
        while gq < K ÷ Int32(16)
            s = Z4
            for c in Int32(0):Int32(15)
                v = x4at(xs4, xu4(Int32(16) * gq + c))
                s = add4(s, (v[1].value, v[2].value, v[3].value, v[4].value))
            end
            Base.Cartesian.@nexprs 4 p -> unsafe_store!(x164 + 4 * (Int32(4) * gq + Int32(p - 1)), s[p], 1, Val(4))
            gq += nt
        end
        sync_threads()
        W = gridDim().x * nw; w = (blockIdx().x - Int32(1)) * nw + warp
        R = Int32(32) ÷ m.L
        G = (rows + R - Int32(1)) ÷ R
        if ks > Int32(1)
            NT = G * ks
            t = (w * NT) ÷ W; thi = ((w + Int32(1)) * NT) ÷ W
            while t < thi
                gg = t ÷ ks; c = t - gg * ks
                if m.t == Int32(T_Q4_K) && m.L == Int32(2)
                    grp4k!(Val(T_Q4_K), Val(2), m, gg, c, ks, rows, y, ldy, P, xs4, x164)
                elseif m.t == Int32(T_Q4_K) && m.L == Int32(8)
                    grp4k!(Val(T_Q4_K), Val(8), m, gg, c, ks, rows, y, ldy, P, xs4, x164)
                elseif m.t == Int32(T_Q6_K) && m.L == Int32(2)
                    grp4k!(Val(T_Q6_K), Val(2), m, gg, c, ks, rows, y, ldy, P, xs4, x164)
                elseif m.t == Int32(T_Q6_K) && m.L == Int32(8)
                    grp4k!(Val(T_Q6_K), Val(8), m, gg, c, ks, rows, y, ldy, P, xs4, x164)
                elseif m.t == Int32(T_Q5_K) && m.L == Int32(2)
                    grp4k!(Val(T_Q5_K), Val(2), m, gg, c, ks, rows, y, ldy, P, xs4, x164)
                else
                    devtrap()
                end
                t += Int32(1)
            end
            return
        end
        g = (w * G) ÷ W; hi = ((w + Int32(1)) * G) ÷ W
        if GEMV4X2 && m.t == Int32(T_Q5_K) && m.L == Int32(2)    # the target head's slices
            while g + Int32(1) < hi
                grp4y2!(Val(T_Q5_K), Val(2), m, g, rows, y, ldy, accum, P, xs4, x164)
                g += Int32(2)
            end
        end
        if GEMV4X2 && m.t == Int32(T_Q4_K)     # pairs of groups first (bdot4x2), an odd last group alone below
            while g + Int32(1) < hi
                if m.L == Int32(2)
                    grp4y2!(Val(T_Q4_K), Val(2), m, g, rows, y, ldy, accum, P, xs4, x164)
                elseif m.L == Int32(10)
                    grp4y2!(Val(T_Q4_K), Val(10), m, g, rows, y, ldy, accum, P, xs4, x164)
                elseif m.L == Int32(6)
                    grp4y2!(Val(T_Q4_K), Val(6), m, g, rows, y, ldy, accum, P, xs4, x164)
                else
                    break
                end
                g += Int32(2)
            end
        end
        while g < hi
            if m.t == Int32(T_Q4_K) && m.L == Int32(10)
                grp4y!(Val(T_Q4_K), Val(10), m, g, rows, y, ldy, accum, P, xs4, x164)
            elseif m.t == Int32(T_Q4_K) && m.L == Int32(6)
                grp4y!(Val(T_Q4_K), Val(6), m, g, rows, y, ldy, accum, P, xs4, x164)
            elseif m.t == Int32(T_Q6_K) && m.L == Int32(10)
                grp4y!(Val(T_Q6_K), Val(10), m, g, rows, y, ldy, accum, P, xs4, x164)
            elseif m.t == Int32(T_Q6_K) && m.L == Int32(6)
                grp4y!(Val(T_Q6_K), Val(6), m, g, rows, y, ldy, accum, P, xs4, x164)
            elseif m.t == Int32(T_Q4_K) && m.L == Int32(2)
                grp4y!(Val(T_Q4_K), Val(2), m, g, rows, y, ldy, accum, P, xs4, x164)
            elseif m.t == Int32(T_Q4_K) && m.L == Int32(8)
                grp4y!(Val(T_Q4_K), Val(8), m, g, rows, y, ldy, accum, P, xs4, x164)
            elseif m.t == Int32(T_Q6_K) && m.L == Int32(2)
                grp4y!(Val(T_Q6_K), Val(2), m, g, rows, y, ldy, accum, P, xs4, x164)
            elseif m.t == Int32(T_Q6_K) && m.L == Int32(8)
                grp4y!(Val(T_Q6_K), Val(8), m, g, rows, y, ldy, accum, P, xs4, x164)
            elseif m.t == Int32(T_Q5_K) && m.L == Int32(2)
                grp4y!(Val(T_Q5_K), Val(2), m, g, rows, y, ldy, accum, P, xs4, x164)
            else
                devtrap()                  # a layout outside GEMV4_COMBOS (load_draft checks)
            end
            g += Int32(1)
        end
    end
    return
end
# row groups g and g + 1 together (Q4_K): each lane's two rows share every x load (bdot4x2)
@inline function grp4y2!(::Val{T}, ::Val{L}, m::MatI, g::Int32, rows::Int32, y::FPtr, ldy::Int32, accum::Int32, P::Int32,
                         xs4::SPtr, x164::SPtr) where {T,L}
    E = Int32(devE(T)); H = Int32(devH(T)); NC = Int32(devP(T) ÷ 16); R = Int32(32 ÷ L)
    NI = (m.K ÷ E) ÷ Int32(L)
    lane = (threadIdx().x - Int32(1)) % Int32(32); rho = lane ÷ Int32(L); j = lane % Int32(L)
    accA = Z4; accB = Z4
    ga = g * NI; gb = (g + Int32(1)) * NI
    @nounroll for i in Int32(0):NI-Int32(1)
        k0 = (j + Int32(L) * i) * E
        a, b = bdot4x2(Val(T), m.pay + ((ga + i) * NC * Int32(32) + lane) * Int32(16), m.hdr + ((ga + i) * Int32(32) + lane) * H,
                       m.pay + ((gb + i) * NC * Int32(32) + lane) * Int32(16), m.hdr + ((gb + i) * Int32(32) + lane) * H, xs4, x164, xu4(k0), k0 >> 4)
        accA = add4(accA, a); accB = add4(accB, b)
    end
    accA = wsum4(accA, Val(L)); accB = wsum4(accB, Val(L))
    for (gg, acc) in ((g, accA), (g + Int32(1), accB))
        r = gg * R + rho
        if j == Int32(0) && rho < R && r < rows
            Base.Cartesian.@nexprs 4 p -> if Int32(p - 1) < P
                o = Int32(p - 1) * ldy + r
                stf(y, o, accum != Int32(0) ? ldf(y, o) + acc[p] : acc[p])
            end
        end
    end
end
const GEMV4X2 = get(ENV, "FLASHNEXT_GEMV4X2", "1") == "1"
@inline function grp4y!(::Val{T}, ::Val{L}, m::MatI, g::Int32, rows::Int32, y::FPtr, ldy::Int32, accum::Int32, P::Int32,
                        xs4::SPtr, x164::SPtr) where {T,L}
    E = Int32(devE(T)); H = Int32(devH(T)); NC = Int32(devP(T) ÷ 16); R = Int32(32 ÷ L)
    NI = (m.K ÷ E) ÷ Int32(L)
    lane = (threadIdx().x - Int32(1)) % Int32(32); rho = lane ÷ Int32(L); j = lane % Int32(L)
    acc = Z4
    gi = g * NI
    @nounroll for i in Int32(0):NI-Int32(1)
        k0 = (j + Int32(L) * i) * E
        acc = add4(acc, bdot4(Val(T), m.pay + ((gi + i) * NC * Int32(32) + lane) * Int32(16), m.hdr + ((gi + i) * Int32(32) + lane) * H, xs4, x164, xu4(k0), k0 >> 4))
    end
    acc = wsum4(acc, Val(L))
    r = g * R + rho
    if j == Int32(0) && rho < R && r < rows
        Base.Cartesian.@nexprs 4 p -> if Int32(p - 1) < P
            o = Int32(p - 1) * ldy + r
            stf(y, o, accum != Int32(0) ? ldf(y, o) + acc[p] : acc[p])
        end
    end
end
# ---- B4 phases: the target's pass over P <= 4 positions (DFlash2's verify, and a 4-token prefill) ----
# Every LL buffer of the B4 engine holds 4 position slots; position p's words of a vector start p ps words after
# position 0's. x of a phase: the 4 positions interleaved in xs4 (xu4); a row group's weights are read once for all 4.
const NORL4 = (NORL, NORL, NORL, NORL)
# B8's rounds: up to P8MAX positions (the draft's block), x in runs of up to 4 slots; per-position lane factors
const P8MAX = 8
const RL8 = NTuple{P8MAX,NTuple{4,Float32}}
const NORL8 = ntuple(_ -> NORL, P8MAX)
struct XF4
    xf::XF; ps::Int32; ps2::Int32     # position strides (words) of p1 and p2
end
XF4(xf::XF, ps, ps2 = 0) = XF4(xf, Int32(ps), Int32(ps2))
struct EM4
    em::EM; ps::Int32; pS::Int32      # position strides of d1/d2 (both) and of the hc mix's S
end
struct Job4{MODE}
    j::Job{MODE}; e4::EM4
end
Job4{M}(m::MatI, rows, em::EM, ps; pS = 0, xo = 0, eo = 0, ns = 1, rpe = 0) where {M} =
    Job4{M}(Job{M}(m, rows, em; xo, eo, ns, rpe), EM4(em, Int32(ps), Int32(pS)))
@inline xps(xf::XF4, p::Int32) = (xf.xf.p1 + 8 * p * xf.ps, xf.xf.p2 + 8 * p * xf.ps2)
# x[k] of position p (kinds as XF); the raw words first (xraw4), then their values (xfix4), so a thread's loads of the
# 4 positions are in flight together
@inline function xsrc4(xf::XF4, k::Int32, p::Int32)
    x = xf.xf; p1, p2 = xps(xf, p)
    x.kind == Int32(8) && return (p1, k >> 1, x.s1)          # W2: element k in word k ÷ 2
    x.kind == Int32(1) ? (k < x.n1 ? (p1, k, x.s1) : (p2, k - x.n1, x.s2)) :
    x.kind == Int32(5) ? (k < x.n1 ? (p1, k, x.s1) : (p2, k - x.n1, x.s2)) : (p1, k + x.n2, x.s1)
end
@inline function xval4(xf::XF4, k::Int32, p::Int32, w::UInt64, src, rl::NTuple{4,Float32})
    x = xf.xf; pp, i, s = src
    if x.kind == Int32(8)
        return w2fix(pp, k, s, w)
    elseif x.kind == Int32(1) || x.kind == Int32(2)
        v = llfix(pp, i, s, w)
        return x.kind == Int32(2) ? v * ldf(x.w, i) : v
    elseif x.kind == Int32(5)
        k < x.n1 && return llfix(pp, i, s, w)
        j = k - x.n1
        return silu(llget(pp, j, s)) * llget(pp, x.F + j, s)
    elseif x.kind == Int32(4)
        p1, _ = xps(xf, p)
        e = k ÷ x.F + x.n2; j = k % x.F
        return silu(llget(p1, Int32(2) * x.F * e + j, x.s1)) * llget(p1, Int32(2) * x.F * e + x.F + j, x.s1)
    else                                 # 3: hc t from the down partials of position p
        p1, _ = xps(xf, p)
        sm = 0.0f0
        for l in Int32(0):Int32(3)
            ps_ = 0.0f0
            for q in Int32(0):x.n1-Int32(1)
                ps_ += llget(p1, q * Int32(1300) + Int32(324) * l + k, x.s1)
            end
            sm = fma(rl4(rl, l), ps_, sm)
        end
        return silu(sm * 0.25f0)
    end
end
# x[0:Kx) of P positions into xs4 (zeros past P) and the 16-sums; kind 2 also returns each position's sums of S^2
# per lane (of the lanes this x covers: lane = (k + n2) / nc)
@inline function xload4!(xs4::SPtr, x164::SPtr, xf::XF4, Kx::Int32, rl::NTuple{4,NTuple{4,Float32}}, P::Int32)
    tid = threadIdx().x - Int32(1); nt = blockDim().x
    x = xf.xf
    ls = ntuple(_ -> Z4, Val(4))
    k = tid
    @inbounds while k < Kx
        srcs = Base.Cartesian.@ntuple 4 p -> xsrc4(xf, k, Int32(p - 1))
        raw = Base.Cartesian.@ntuple 4 p -> (Int32(p - 1) < P && (x.kind == Int32(1) || x.kind == Int32(2) || x.kind == Int32(8) || (x.kind == Int32(5) && k < x.n1)) ?
            llraw(srcs[p][1], srcs[p][2]) : UInt64(0))
        if x.kind == Int32(2)            # S w, and S^2 into its lane's sum
            ln = (k + x.n2) ÷ x.nc
            sv = Base.Cartesian.@ntuple 4 p -> (Int32(p - 1) < P ? llfix(srcs[p][1], srcs[p][2], srcs[p][3], raw[p]) : 0.0f0)
            wk = ldf(x.w, k + x.n2)
            Base.Cartesian.@nexprs 4 p -> unsafe_store!(xs4 + 4 * (Int32(4) * xu4(k) + Int32(p - 1)), sv[p] * wk, 1, Val(4))
            ls = Base.Cartesian.@ntuple 4 p -> begin
                v2 = sv[p] * sv[p]
                (ls[p][1] + (ln == Int32(0) ? v2 : 0.0f0), ls[p][2] + (ln == Int32(1) ? v2 : 0.0f0),
                 ls[p][3] + (ln == Int32(2) ? v2 : 0.0f0), ls[p][4] + (ln == Int32(3) ? v2 : 0.0f0))
            end
        else
            vals = Base.Cartesian.@ntuple 4 p -> (Int32(p - 1) < P ? xval4(xf, k, Int32(p - 1), raw[p], srcs[p], rl[p]) : 0.0f0)
            Base.Cartesian.@nexprs 4 p -> unsafe_store!(xs4 + 4 * (Int32(4) * xu4(k) + Int32(p - 1)), vals[p], 1, Val(4))
        end
        k += nt
    end
    sync_threads()
    gq = tid
    @inbounds while gq < Kx ÷ Int32(16)
        s = Z4
        for c in Int32(0):Int32(15)
            v = x4at(xs4, xu4(Int32(16) * gq + c))
            s = add4(s, (v[1].value, v[2].value, v[3].value, v[4].value))
        end
        Base.Cartesian.@nexprs 4 p -> unsafe_store!(x164 + 4 * (Int32(4) * gq + Int32(p - 1)), s[p], 1, Val(4))
        gq += nt
    end
    ls
end
# the B4 phase: x of P positions, then the jobs' row groups taken from the phase counter (as cphase!)
@inline function cphase4!(xf::XF4, Kx::Int32, xs4::SPtr, rl::NTuple{4,NTuple{4,Float32}}, idx::SIP, wsc::SPtr, red::Red, js::Tuple, ctr::IPtr, P::Int32)
    tid = threadIdx().x - Int32(1); nt = blockDim().x; nw = nt ÷ Int32(32); warp = tid ÷ Int32(32); lane = tid % Int32(32)
    x164 = xs4 + 16 * xu4(Kx)
    luta = CuStaticSharedArray(Float32, 16)
    @inbounds begin
        tid < Int32(16) && (luta[tid+1] = kv4val(tid))
        W = gridDim().x * nw
        ls = xload4!(xs4, x164, xf, Kx, rl, P)
        sync_threads()
        G = jobs4_groups(js)
        v = shfl_sync(FULL, lane == Int32(0) ? ctrnext(ctr) : Int32(0), Int32(1))
        while v < G
            nx = shfl_sync(FULL, lane == Int32(0) ? ctrnext(ctr) : Int32(0), Int32(1))
            run4_one!(js, v, idx, wsc, red, xs4, x164, pointer(luta), rl, P, Kx)
            v = nx
        end
        if lane == Int32(0) && v == G + W - Int32(1)
            UA.store!(ctr, Int32(0), UA.monotonic, UA.device); threadfence()
        end
        sync_threads()
        return ls
    end
end
@inline jobs4_groups(::Tuple{}) = Int32(0)
@inline jobs4_groups(js::Tuple) = jobshape(first(js).j)[1] + jobs4_groups(Base.tail(js))
@inline run4_one!(::Tuple{}, g::Int32, idx, wsc, red, xs4, x164, lut, rl, P, Kx) = nothing
@inline function run4_one!(js::Tuple, g::Int32, idx, wsc, red, xs4, x164, lut, rl, P, Kx)
    j = first(js); G = jobshape(j.j)[1]
    g < G ? jobgrp4!(j, g, idx, wsc, red, xs4, x164, lut, rl, P, Kx) : run4_one!(Base.tail(js), g - G, idx, wsc, red, xs4, x164, lut, rl, P, Kx)
end
# position p of an output (kinds 1, 2 as put_owner!)
@inline function put_owner4!(e4::EM4, r::Int32, v::Float32, p::Int32)
    em = e4.em; o = 8 * p * e4.ps
    if em.kind == Int32(1)
        llput(em.d1 + o, em.o1 + r, em.sq, v)
        em.d2 != U64P(0) && llput(em.d2 + o, em.o1 + r, em.sq, v)
    elseif r >= em.lo && r < em.lo + em.nc
        llput(em.d1 + o, em.o1 + r - em.lo, em.sq, v)
    else
        llput(em.d2 + o, em.o1 + r - em.plo, em.sq, v)
    end
end
# one row group (MODE 0, 1, 4) or (row group, expert) task (MODE 3) of 4 positions. MODE 4 takes the blocks of its
# rows whose elements fall in this pass's x: [xo, xo + Kx) of the matrix's K (xo = the pass's first element).
@inline function grp4!(::Val{T}, ::Val{L}, ::Val{MODE}, jb::Job4{MODE}, g::Int32, idx::SIP, wsc::SPtr, red::Red,
                       xs4::SPtr, x164::SPtr, lut::SPtr, rl::NTuple{4,NTuple{4,Float32}}, P::Int32, Kx::Int32) where {T,L,MODE}
    j_ = jb.j; e4 = jb.e4; em = e4.em; m = j_.m; rows = j_.rows; xo = j_.xo; eo = j_.eo; ns = j_.ns; rpe = j_.rpe
    E = Int32(devE(T)); H = Int32(devH(T)); NC = Int32(devP(T) ÷ 16); R = Int32(32 ÷ L)
    nb = m.K ÷ E; NI = nb ÷ Int32(L)
    lane = (threadIdx().x - Int32(1)) % Int32(32)
    rho = lane ÷ Int32(L); j = lane % Int32(L)
    @inbounds if MODE == 4
        r = g * R + rho
        bseg = em.nc ÷ E
        # the pass's two segments (Kx = 2 nc): accumulators (segment, position)
        sA = xo ÷ em.nc
        a = Z4; b = Z4
        gi = g * NI
        for i in Int32(0):NI-Int32(1)
            k0 = (j + Int32(L) * i) * E
            if k0 >= xo && k0 < xo + Kx
                v = blk4(Val(T), m.pay + ((gi + i) * NC * Int32(32) + lane) * Int32(16), m.hdr + ((gi + i) * Int32(32) + lane) * H, xs4, x164, xu4(k0 - xo), (k0 - xo) >> 4, lut)
                (j + Int32(L) * i) ÷ bseg == sA ? (a = add4(a, v)) : (b = add4(b, v))
            end
        end
        a = wsum4(a, Val(L)); b = wsum4(b, Val(L))
        if j == Int32(0) && rho < R && r < rows
            Base.Cartesian.@nexprs 4 p -> if Int32(p - 1) < P
                o = 8 * Int32(p - 1) * e4.ps
                llput(em.d1 + o, em.o1 + Int32(324) * (sA + em.lo) + r, em.sq, a[p])
                em.d2 != U64P(0) && llput(em.d2 + o, em.o1 + Int32(324) * (sA + em.lo) + r, em.sq, a[p])
                llput(em.d1 + o, em.o1 + Int32(324) * (sA + Int32(1) + em.lo) + r, em.sq, b[p])
                em.d2 != U64P(0) && llput(em.d2 + o, em.o1 + Int32(324) * (sA + Int32(1) + em.lo) + r, em.sq, b[p])
            end
        end
    elseif MODE == 2                   # rows, the blocks of the pass's K window [xo, xo + Kx) (x = that window)
        acc = Z4; gi = g * NI
        for i in Int32(0):NI-Int32(1)
            k0 = (j + Int32(L) * i) * E
            if k0 >= xo && k0 < xo + Kx
                acc = add4(acc, blk4(Val(T), m.pay + ((gi + i) * NC * Int32(32) + lane) * Int32(16), m.hdr + ((gi + i) * Int32(32) + lane) * H, xs4, x164, xu4(k0 - xo), (k0 - xo) >> 4, lut))
            end
        end
        acc = wsum4(acc, Val(L))
        r = g * R + rho
        if j == Int32(0) && rho < R && r < rows
            Base.Cartesian.@nexprs 4 p -> (Int32(p - 1) < P && put_owner4!(e4, r, acc[p], Int32(p - 1)))
        end
    elseif MODE == 3
        gr = g ÷ ns; e = g % ns
        pg = rpe == Int32(0) ? gr : (unsafe_load(idx, e + 1, Val(4)) * rpe + gr * R) ÷ R
        acc = Z4; gi = pg * NI
        for i in Int32(0):NI-Int32(1)
            k0 = (j + Int32(L) * i) * E
            acc = add4(acc, blk4(Val(T), m.pay + ((gi + i) * NC * Int32(32) + lane) * Int32(16), m.hdr + ((gi + i) * Int32(32) + lane) * H, xs4, x164, xu4(xo + e * m.K + k0), (xo + e * m.K + k0) >> 4, lut))
        end
        acc = wsum4(acc, Val(L))
        r = gr * R + rho
        if j == Int32(0) && rho < R && r < rows
            Base.Cartesian.@nexprs 4 p -> stf(red.P, ((eo + e) * Int32(4) + Int32(p - 1)) * red.stride + r, unsafe_load(wsc, (eo + e) * Int32(4) + Int32(p), Val(4)) * acc[p])
        end
        threadfence()
        sync_warp()
        blk = (gr * R) ÷ Int32(32)
        old = lane == Int32(0) ? CUDA.atomic_add!(red.cnt + 4blk, Int32(1)) : Int32(0)
        old = shfl_sync(FULL, old, Int32(1))
        if old == red.target - Int32(1)
            threadfence()
            rr = blk * Int32(32) + lane
            for p in Int32(0):P-Int32(1)
                y = 0.0f0
                for s in Int32(0):red.ns-Int32(1)
                    y += ldvolf(red.P, (s * Int32(4) + p) * red.stride + rr)
                end
                rr < rows && put_owner4!(e4, rr, y, p)
            end
            lane == Int32(0) && UA.store!(red.cnt + 4blk, Int32(0), UA.monotonic, UA.device)
            threadfence()
        end
    else
        pg = MODE == 1 ? (unsafe_load(idx, (g * R) ÷ rpe + 1, Val(4)) * rpe + (g * R) % rpe) ÷ R : g
        acc = Z4; gi = pg * NI
        for i in Int32(0):NI-Int32(1)
            k0 = xo + (j + Int32(L) * i) * E
            acc = add4(acc, blk4(Val(T), m.pay + ((gi + i) * NC * Int32(32) + lane) * Int32(16), m.hdr + ((gi + i) * Int32(32) + lane) * H, xs4, x164, xu4(k0), k0 >> 4, lut))
        end
        acc = wsum4(acc, Val(L))
        r = g * R + rho
        if em.kind == Int32(4)
            l = r & Int32(3); c = r >> 2
            live = j == Int32(0) && rho < R && r < rows
            Base.Cartesian.@nexprs 4 p -> begin
                tt = live && Int32(p - 1) < P ? sigm(acc[p]) * llget(em.S + 8 * Int32(p - 1) * e4.pS, l * em.nc + c, em.sS) * rl4(rl[p], l) * ldf(em.w, l * em.nc + c) : 0.0f0
                tt += shfl_down_sync(FULL, tt, Int32(L)); tt += shfl_down_sync(FULL, tt, Int32(2L))
                if live && l == Int32(0) && Int32(p - 1) < P
                    o = 8 * Int32(p - 1) * e4.ps
                    llput(em.d1 + o, em.zoff + c, em.sq, tt * 0.25f0)
                    em.d2 != U64P(0) && llput(em.d2 + o, em.zoff + c, em.sq, tt * 0.25f0)
                end
            end
        elseif em.kind == Int32(5)
            Base.Cartesian.@nexprs 4 p -> begin
                u = shfl_sync(FULL, acc[p], (rho + Int32(1)) * Int32(L) + Int32(1))
                if j == Int32(0) && rho < R && rho % Int32(2) == Int32(0) && r < rows && Int32(p - 1) < P
                    if em.nc > Int32(0)
                        # B8: pair q = r ÷ 2 is inner channel q mod nc of union expert q ÷ nc; position p's slot of
                        # that expert (wsc[4 u + p], -1: not its expert) picks the word: slot zoff + o1 + q mod nc
                        q = r ÷ Int32(2); ue = q ÷ em.nc
                        ks = reinterpret(Int32, unsafe_load(wsc, Int32(4) * ue + Int32(p), Val(4)))
                        ks >= Int32(0) && llput(em.d1 + 8 * Int32(p - 1) * e4.ps, ks * em.zoff + em.o1 + (q - ue * em.nc), em.sq, silu(acc[p]) * u)
                    else
                        llput(em.d1 + 8 * Int32(p - 1) * e4.ps, em.o1 + r ÷ Int32(2), em.sq, silu(acc[p]) * u)
                    end
                end
            end
        elseif j == Int32(0) && rho < R && r < rows
            Base.Cartesian.@nexprs 4 p -> (Int32(p - 1) < P && put_owner4!(e4, r, acc[p], Int32(p - 1)))
        end
    end
    return
end
for (name, mode, combos) in ((:gdg40, 0, COMBOS), (:gdg41, 1, ((T_Q2_0, 8),)), (:gdg42, 2, COMBOS), (:gdg43, 3, COMBOS3), (:gdg44, 4, ((T_BF16, 32), (T_BF16, 8))))
    body = Expr(:block)
    for (t, L) in combos
        push!(body.args, :(jb.j.m.t == Int32($t) && jb.j.m.L == Int32($L) && return grp4!(Val($t), Val($L), Val($mode), jb, g, idx, wsc, red, xs4, x164, lut, rl, P, Kx)))
    end
    @eval @noinline function $name(jb::Job4{$mode}, g::Int32, idx::SIP, wsc::SPtr, red::Red, xs4::SPtr, x164::SPtr, lut::SPtr,
                                   rl::NTuple{4,NTuple{4,Float32}}, P::Int32, Kx::Int32)
        $body
        devtrap()
        return nothing
    end
end
@inline function jobgrp4!(jb::Job4{M}, g::Int32, idx, wsc, red, xs4, x164, lut, rl, P, Kx) where {M}
    M == 0 ? gdg40(jb, g, idx, wsc, red, xs4, x164, lut, rl, P, Kx) : M == 1 ? gdg41(jb, g, idx, wsc, red, xs4, x164, lut, rl, P, Kx) :
    M == 2 ? gdg42(jb, g, idx, wsc, red, xs4, x164, lut, rl, P, Kx) : M == 3 ? gdg43(jb, g, idx, wsc, red, xs4, x164, lut, rl, P, Kx) :
    gdg44(jb, g, idx, wsc, red, xs4, x164, lut, rl, P, Kx)
end
# ---- B8's GEMV in B1's arithmetic: each position's x in B1's layout (xu, its 16-sums after xu(Kx)), a slot of XS8(Kx)
# floats each, and every block of every position through B1's blkdot / bcomp, its rows' sums as B1's (wsum, HCSPLIT's
# segments, the MoE down's slots in rank order): B8's outputs equal B1's bit for bit. The x slots of 4 positions do not
# fit for K = 2560 (4 x 3080 floats), so a phase takes its positions in runs of npr8(Kx) (3 there, else 4), each run
# with its own group counter (32 words apart) and its own read of the weights.
@inline XS8(Kx::Int32) = (xu(Kx) + (Kx >> Int32(4)) + Int32(3)) & ~Int32(3)
@inline npr8(Kx::Int32) = clamp(Int32(XSM8) ÷ XS8(Kx), Int32(1), Int32(4))
@inline xslot(xs::SPtr, XS::Int32, s::Int32) = xs + 4 * (s * XS)
# position p's view of a 4-position x (XF4: its buffers' position strides)
@inline xfpos(xf::XF4, p::Int32) = (x = xf.xf; XF(x.kind, x.p1 + 8 * p * xf.ps, x.n1, x.s1, x.p2 + 8 * p * xf.ps2, x.n2, x.s2, x.w, x.F, x.nc, x.dst, x.sd))
# positions p0 .. p0 + ns - 1 of x into slots 0 .. ns - 1 (B1's xload! each; the caller syncs)
@inline function xload8!(xs::SPtr, XS::Int32, xf::XF4, Kx::Int32, rl::RL8, p0::Int32, ns::Int32)
    for s in Int32(0):ns-Int32(1)
        p = p0 + s; xp = xslot(xs, XS, s)
        xload!(xp, xp + 4 * xu(Kx), xfpos(xf, p), Kx, @inbounds rl[p+Int32(1)])
    end
end
# the routed down's x of a chunk: union experts u0 .. u0 + Kx / 640 - 1 of positions p0 .., each position's h of its
# slot of the expert (0 where it did not select it: its weight is 0), through B1's xput! (its 16-sums)
@inline function xload8d!(xs::SPtr, XS::Int32, dd, H::U64P, HW::U64P, slt::SPtr, u0::Int32, Kx::Int32, p0::Int32, ns::Int32, sq::UInt32)
    tid = threadIdx().x - Int32(1); nt = blockDim().x
    for s in Int32(0):ns-Int32(1)
        p = p0 + s; xp = xslot(xs, XS, s); x16 = xp + 4 * xu(Kx)
        k = tid
        @inbounds while k < Kx
            u = u0 + k ÷ Int32(640); o = k - (k ÷ Int32(640)) * Int32(640)
            ks = reinterpret(Int32, unsafe_load(slt, Int32(P8MAX) * u + p + Int32(1), Val(4)))
            v = ks >= Int32(0) ? (W2H ? w2el(pp(dd, HW, p), Int32(640) * ks + o, sq) : llget(pp(dd, H, p), Int32(640) * ks + o, sq)) : 0.0f0
            xput!(xp, x16, k, v, true)
            k += nt
        end
    end
end
const DEC8 = get(ENV, "FLASHNEXT_DEC8", "0") == "1"
# B1's Q2_0 bcomp (Q2PRMT) for slots 0 .. ns - 1, each word's 16 codes decoded once for all the slots: every slot's
# arithmetic is B1's, in its order (X, then per word its 16 fmas, then d (8 S - 5 X)); only the decode is shared
# (k_token8! had 4.0 times k_token!'s PRMTs, one decode a slot)
@inline function bcomp8(::Val{T_Q2_0}, r, xs::SPtr, XS::Int32, o16::Int32, xb::Int32, g16::Int32, ns::Int32)
    v, h = r; d = hf(h)
    X = Base.Cartesian.@ntuple 4 s -> (Int32(s - 1) < ns ? (x16 = xslot(xs, XS, Int32(s - 1)) + o16;
        (sld(x16, g16) + sld(x16, g16 + Int32(1))) + (sld(x16, g16 + Int32(2)) + sld(x16, g16 + Int32(3)))) : 0.0f0)
    Base.Cartesian.@nexprs 4 w -> begin
        Base.Cartesian.@nexprs 4 k -> begin
            q4_k = bitrotate(v[w], 5 - 2(k - 1)) & 0x60606060
            fa_k = prmtf(q4_k, 0); fb_k = prmtf(q4_k, 1); fc_k = prmtf(q4_k, 2); fd_k = prmtf(q4_k, 3)
        end
        sw_w = Base.Cartesian.@ntuple 4 s -> if Int32(s - 1) < ns
            xp = xslot(xs, XS, Int32(s - 1))
            xa = sxo(xp, xb, 16(w - 1)); xbv = sxo(xp, xb, 16(w - 1) + 4); xc = sxo(xp, xb, 16(w - 1) + 8); xd = sxo(xp, xb, 16(w - 1) + 12)
            a = 0.0f0
            Base.Cartesian.@nexprs 4 k -> begin
                a = fma(fa_k, xa[k].value, a); a = fma(fb_k, xbv[k].value, a)
                a = fma(fc_k, xc[k].value, a); a = fma(fd_k, xd[k].value, a)
            end
            a
        else
            0.0f0
        end
    end
    Base.Cartesian.@ntuple 4 s -> (Int32(s - 1) < ns ? d * fma(8.0f0, (sw_1[s] + sw_2[s]) + (sw_3[s] + sw_4[s]), -5.0f0 * X[s]) : 0.0f0)
end
# B1's rowdot (the loads of block i + 1 issued before block i is computed, for the split formats) for ns slots
@inline function rowdot8(::Val{T}, ::Val{L}, m::MatI, pg::Int32, lane::Int32, j::Int32, xs::SPtr, XS::Int32, Kx::Int32, x0::Int32, lut::SPtr,
                         ns::Int32) where {T,L}
    NI = (m.K ÷ Int32(devE(T))) ÷ Int32(L)
    o16 = 4 * xu(Kx)
    acc = Z4
    if issplit(T)
        E = Int32(devE(T)); H = Int32(devH(T)); NC = Int32(devP(T) ÷ 16)
        gi = pg * NI
        r = bload(Val(T), m.pay + (gi * NC * Int32(32) + lane) * Int32(16), m.hdr + (gi * Int32(32) + lane) * H)
        i = Int32(0)
        @nounroll while i < NI
            nx = i + Int32(1) < NI ? bload(Val(T), m.pay + ((gi + Int32(1)) * NC * Int32(32) + lane) * Int32(16), m.hdr + ((gi + Int32(1)) * Int32(32) + lane) * H) : r
            k0 = x0 + (j + Int32(L) * i) * E
            if T == T_Q2_0 && Q2PRMT && DEC8
                b = bcomp8(Val(T), r, xs, XS, Int32(o16), xu(k0), k0 >> 4, ns)
                acc = Base.Cartesian.@ntuple 4 s -> (Int32(s - 1) < ns ? acc[s] + b[s] : 0.0f0)
            else
                acc = Base.Cartesian.@ntuple 4 s -> (Int32(s - 1) < ns ? acc[s] + bcomp(Val(T), r, xslot(xs, XS, Int32(s - 1)), xslot(xs, XS, Int32(s - 1)) + o16, xu(k0), k0 >> 4, lut) : 0.0f0)
            end
            r = nx; gi += Int32(1); i += Int32(1)
        end
        return acc
    end
    @nounroll for i in Int32(0):NI-Int32(1)
        acc = Base.Cartesian.@ntuple 4 s -> (Int32(s - 1) < ns ? acc[s] + blkdot(Val(T), Val(L), m, pg, NI, i, lane, j, xslot(xs, XS, Int32(s - 1)), xslot(xs, XS, Int32(s - 1)) + o16, x0, lut) : 0.0f0)
    end
    acc
end
@inline wsum8(a::NTuple{4,Float32}, ::Val{L}) where {L} = Base.Cartesian.@ntuple 4 s -> wsum(a[s], Val(L))
# one row group (MODE 0, 1), HCSPLIT task (MODE 4) or (row group, expert) task (MODE 3) of slots 0 .. ns - 1 (positions
# p0 ..), as B1's grp!
@inline function grp8!(::Val{T}, ::Val{L}, ::Val{MODE}, jb::Job4{MODE}, g::Int32, idx::SIP, wsc::SPtr, red::Red,
                       xs::SPtr, XS::Int32, Kx::Int32, lut::SPtr, rl::RL8, p0::Int32, ns::Int32, P::Int32) where {T,L,MODE}
    j_ = jb.j; e4 = jb.e4; em = e4.em; m = j_.m; rows = j_.rows; xo = j_.xo; eo = j_.eo; nsj = j_.ns; rpe = j_.rpe
    PM = Int32(P8MAX)                  # (the position stride of the union tables and the MoE down's terms)
    E = Int32(devE(T)); R = Int32(32 ÷ L)
    nb = m.K ÷ E; NI = nb ÷ Int32(L)
    lane = (threadIdx().x - Int32(1)) % Int32(32)
    rho = lane ÷ Int32(L); j = lane % Int32(L)
    o16 = 4 * xu(Kx)
    @inbounds if MODE == 4 && !(HCSPLIT && L == 32)
        # B1's row group of 4 segments (8 dies: L = 8, 5 blocks a lane): each segment's blocks in order, then wsum
        r = g * R + rho
        bseg = em.nc ÷ E; nseg = nb ÷ bseg
        a1 = Z4; a2 = Z4; a3 = Z4; a4 = Z4
        for i in Int32(0):NI-Int32(1)
            sgi = (j + Int32(L) * i) ÷ bseg
            v = Base.Cartesian.@ntuple 4 s -> (Int32(s - 1) < ns ? (xp = xslot(xs, XS, Int32(s - 1)); blkdot(Val(T), Val(L), m, g, NI, i, lane, j, xp, xp + o16, xo, lut)) : 0.0f0)
            a1 = Base.Cartesian.@ntuple 4 s -> a1[s] + (sgi == Int32(0) ? v[s] : 0.0f0); a2 = Base.Cartesian.@ntuple 4 s -> a2[s] + (sgi == Int32(1) ? v[s] : 0.0f0)
            a3 = Base.Cartesian.@ntuple 4 s -> a3[s] + (sgi == Int32(2) ? v[s] : 0.0f0); a4 = Base.Cartesian.@ntuple 4 s -> a4[s] + (sgi == Int32(3) ? v[s] : 0.0f0)
        end
        a1 = wsum8(a1, Val(L)); a2 = wsum8(a2, Val(L)); a3 = wsum8(a3, Val(L)); a4 = wsum8(a4, Val(L))
        if j == Int32(0) && rho < R && r < rows
            Base.Cartesian.@nexprs 4 s -> if Int32(s - 1) < ns
                o = 8 * (p0 + Int32(s - 1)) * e4.ps
                for (sg, av) in ((Int32(0), a1[s]), (Int32(1), a2[s]), (Int32(2), a3[s]), (Int32(3), a4[s]))
                    if sg < nseg
                        llput(em.d1 + o, em.o1 + Int32(324) * (sg + em.lo) + r, em.sq, av)
                        em.d2 != U64P(0) && llput(em.d2 + o, em.o1 + Int32(324) * (sg + em.lo) + r, em.sq, av)
                    end
                end
            end
        end
    elseif MODE == 4                       # B1's HCSPLIT task (row r, segment sg)
        nseg = m.K ÷ em.nc; r = g ÷ nseg; sg = g - r * nseg
        bseg = em.nc ÷ E; b0 = sg * bseg
        a = Z4; u = lane
        while u < bseg
            bA = b0 + u; bB = bA + Int32(32)
            a = Base.Cartesian.@ntuple 4 s -> if Int32(s - 1) < ns
                xp = xslot(xs, XS, Int32(s - 1))
                vA = blkdot(Val(T), Val(L), m, r, NI, bA >> 5, bA & Int32(31), bA & Int32(31), xp, xp + o16, xo, lut)
                vB = u + Int32(32) < bseg ? blkdot(Val(T), Val(L), m, r, NI, bB >> 5, bB & Int32(31), bB & Int32(31), xp, xp + o16, xo, lut) : 0.0f0
                a[s] + (vA + vB)
            else
                0.0f0
            end
            u += Int32(64)
        end
        a = wsum8(a, Val(L))
        if lane == Int32(0) && r < rows
            Base.Cartesian.@nexprs 4 s -> if Int32(s - 1) < ns
                o = 8 * (p0 + Int32(s - 1)) * e4.ps
                llput(em.d1 + o, em.o1 + Int32(324) * (sg + em.lo) + r, em.sq, a[s])
                em.d2 != U64P(0) && llput(em.d2 + o, em.o1 + Int32(324) * (sg + em.lo) + r, em.sq, a[s])
            end
        end
    elseif MODE == 3
        gr = g ÷ nsj; e = g % nsj
        pg = rpe == Int32(0) ? gr : (unsafe_load(idx, e + 1, Val(4)) * rpe + gr * R) ÷ R
        acc = wsum8(rowdot8(Val(T), Val(L), m, pg, lane, j, xs, XS, Kx, xo + e * m.K, lut, ns), Val(L))
        r = gr * R + rho
        if j == Int32(0) && rho < R && r < rows
            Base.Cartesian.@nexprs 4 s -> if Int32(s - 1) < ns
                p = p0 + Int32(s - 1)
                stf(red.P, ((eo + e) * PM + p) * red.stride + r, unsafe_load(wsc, (eo + e) * PM + p + Int32(1), Val(4)) * acc[s])
            end
        end
        threadfence()
        sync_warp()
        blk = (gr * R) ÷ Int32(32)
        old = lane == Int32(0) ? CUDA.atomic_add!(red.cnt + 4blk, Int32(1)) : Int32(0)
        old = shfl_sync(FULL, old, Int32(1))
        if old == red.target - Int32(1)
            threadfence()
            rr = blk * Int32(32) + lane
            for p in Int32(0):P-Int32(1)
                # B1's sum: the 10 routed experts in the position's rank order, then the shared expert (slot ns - 1)
                y = 0.0f0
                if red.ord != Core.LLVMPtr{Int32,3}(0)
                    for k in Int32(0):Int32(9)
                        y += ldvolf(red.P, (unsafe_load(red.ord, Int32(10) * p + k + Int32(1), Val(4)) * PM + p) * red.stride + rr)
                    end
                    y += ldvolf(red.P, ((red.ns - Int32(1)) * PM + p) * red.stride + rr)
                else
                    for s in Int32(0):red.ns-Int32(1)
                        y += ldvolf(red.P, (s * PM + p) * red.stride + rr)
                    end
                end
                rr < rows && put_owner4!(e4, rr, y, p)
            end
            lane == Int32(0) && UA.store!(red.cnt + 4blk, Int32(0), UA.monotonic, UA.device)
            threadfence()
        end
    else
        pg = MODE == 1 ? (unsafe_load(idx, (g * R) ÷ rpe + 1, Val(4)) * rpe + (g * R) % rpe) ÷ R : g
        acc = wsum8(rowdot8(Val(T), Val(L), m, pg, lane, j, xs, XS, Kx, xo, lut, ns), Val(L))
        r = g * R + rho
        if em.kind == Int32(4)
            l = r & Int32(3); c = r >> 2
            live = j == Int32(0) && rho < R && r < rows
            Base.Cartesian.@nexprs 4 s -> begin
                p = p0 + Int32(s - 1); rp = rl[min(p, PM - Int32(1))+Int32(1)]
                tt = live && Int32(s - 1) < ns ? sigm(acc[s]) * llget(em.S + 8 * p * e4.pS, l * em.nc + c, em.sS) * rl4(rp, l) * ldf(em.w, l * em.nc + c) : 0.0f0
                tt += shfl_down_sync(FULL, tt, Int32(L)); tt += shfl_down_sync(FULL, tt, Int32(2L))
                if live && l == Int32(0) && Int32(s - 1) < ns
                    o = 8 * p * e4.ps
                    llput(em.d1 + o, em.zoff + c, em.sq, tt * 0.25f0)
                    em.d2 != U64P(0) && llput(em.d2 + o, em.zoff + c, em.sq, tt * 0.25f0)
                end
            end
        elseif em.kind == Int32(5)
            Base.Cartesian.@nexprs 4 s -> begin
                u = shfl_sync(FULL, acc[s], (rho + Int32(1)) * Int32(L) + Int32(1))
                p = p0 + Int32(s - 1)
                if j == Int32(0) && rho < R && rho % Int32(2) == Int32(0) && r < rows && Int32(s - 1) < ns
                    if em.nc > Int32(0)
                        q = r ÷ Int32(2); ue = q ÷ em.nc
                        ks = reinterpret(Int32, unsafe_load(wsc, PM * ue + p + Int32(1), Val(4)))
                        ks >= Int32(0) && llput(em.d1 + 8 * p * e4.ps, ks * em.zoff + em.o1 + (q - ue * em.nc), em.sq, silu(acc[s]) * u)
                    else
                        llput(em.d1 + 8 * p * e4.ps, em.o1 + r ÷ Int32(2), em.sq, silu(acc[s]) * u)
                    end
                end
            end
        elseif j == Int32(0) && rho < R && r < rows
            Base.Cartesian.@nexprs 4 s -> (Int32(s - 1) < ns && put_owner4!(e4, r, acc[s], p0 + Int32(s - 1)))
        end
    end
    return
end
for (name, mode, combos) in ((:gdg80, 0, COMBOS), (:gdg81, 1, ((T_Q2_0, 8),)), (:gdg83, 3, COMBOS3), (:gdg84, 4, ((T_BF16, 32), (T_BF16, 8))))
    body = Expr(:block)
    for (t, L) in combos
        push!(body.args, :(jb.j.m.t == Int32($t) && jb.j.m.L == Int32($L) && return grp8!(Val($t), Val($L), Val($mode), jb, g, idx, wsc, red, xs, XS, Kx, lut, rl, p0, ns, P)))
    end
    @eval @noinline function $name(jb::Job4{$mode}, g::Int32, idx::SIP, wsc::SPtr, red::Red, xs::SPtr, XS::Int32, Kx::Int32, lut::SPtr,
                                   rl::RL8, p0::Int32, ns::Int32, P::Int32)
        $body
        devtrap()
        return nothing
    end
end
@inline function jobgrp8!(jb::Job4{M}, g::Int32, idx, wsc, red, xs, XS, Kx, lut, rl, p0, ns, P) where {M}
    M == 0 ? gdg80(jb, g, idx, wsc, red, xs, XS, Kx, lut, rl, p0, ns, P) : M == 1 ? gdg81(jb, g, idx, wsc, red, xs, XS, Kx, lut, rl, p0, ns, P) :
    M == 3 ? gdg83(jb, g, idx, wsc, red, xs, XS, Kx, lut, rl, p0, ns, P) : gdg84(jb, g, idx, wsc, red, xs, XS, Kx, lut, rl, p0, ns, P)
end
@inline run8_one!(::Tuple{}, g::Int32, idx, wsc, red, xs, XS, Kx, lut, rl, p0, ns, P) = nothing
@inline function run8_one!(js::Tuple, g::Int32, idx, wsc, red, xs, XS, Kx, lut, rl, p0, ns, P)
    j = first(js); G = jobshape(j.j)[1]
    g < G ? jobgrp8!(j, g, idx, wsc, red, xs, XS, Kx, lut, rl, p0, ns, P) : run8_one!(Base.tail(js), g - G, idx, wsc, red, xs, XS, Kx, lut, rl, p0, ns, P)
end
# the groups of one run (positions p0 .. p0 + ns - 1, x loaded) from counter ctr
@inline function grun8!(js::Tuple, ctr::IPtr, idx, wsc, red, xs, XS, Kx, lut, rl, p0, ns, P)
    lane = (threadIdx().x - Int32(1)) % Int32(32)
    W = gridDim().x * (blockDim().x ÷ Int32(32))
    G = jobs4_groups(js)
    v = shfl_sync(FULL, lane == Int32(0) ? ctrnext(ctr) : Int32(0), Int32(1))
    while v < G
        nx = shfl_sync(FULL, lane == Int32(0) ? ctrnext(ctr) : Int32(0), Int32(1))
        run8_one!(js, v, idx, wsc, red, xs, XS, Kx, lut, rl, p0, ns, P)
        v = nx
    end
    if lane == Int32(0) && v == G + W - Int32(1)
        UA.store!(ctr, Int32(0), UA.monotonic, UA.device); threadfence()
    end
end
# B8's phase counters: CT8 a block. A phase's own s < 64, its run r at s + 64 r (r < 3: 8 positions in runs of 3);
# the MoE down's (chunk c, run r) at 192 + 32 r + c (runs of 4, c < 27)
const CT8 = 256
@inline ctr8of(dd, bd, s) = @f(dd, CTR) + 4 * (Int32(CT8) * ((@f(bd, k0) - Int32(1)) ÷ Int32(NSTB)) + Int32(s))
# ---- the B4 ops (pairs of dies only: n = 2). Position strides of the B4 LL buffers (words): ----
@inline pS4(dd) = Int32(4) * @f(dd, nc)
@inline pYP4(dd) = Int32(8) * @f(dd, nc)                 # out_proj partials: [parity][pass][die][nc]
@inline pGU4(dd) = Int32(84) * @f(dd, F)                  # [parity][40 routed F | shared gate F | shared up F]
const PZ4 = 5120; const PDP4 = 5200; const PINJ4 = 16; const PRP4 = 2052; const PLS4 = 256; const PIDXW4 = 416
const B4THREADS = 512                 # one CTA of 512 per SM: the 4-position x (43.8 KB) leaves no room for a second
@inline ctr4of(dd, bd, s) = @f(dd, CTR) + 4 * (Int32(32) * ((@f(bd, k0) - Int32(1)) ÷ Int32(NSTB)) + Int32(s))
function op4_sinit!(dd, seq0, P::Int32)
    nc = @f(dd, nc); d = @f(dd, d); n4 = Int32(4) * nc
    i = gtid()
    while i < P * n4
        p = i ÷ n4; k = i % n4
        llput(@f(dd, S) + 8 * p * n4, k, sqof(seq0, 0, 0), llget(@f(dd, EMB) + 8 * p * Int32(2560), nc * d + k % nc, seq0 + UInt32(1)))
        i += gstride()
    end
end
function op4_ple!(dd, bd, seq0, t::Int32, xs, P::Int32)
    l = Int32(1); nc = @f(dd, nc); d = @f(dd, d); n = @f(dd, n); n4 = Int32(4) * nc
    s1 = sqof(seq0, l, 1); s2 = sqof(seq0, l, 2); s3 = sqof(seq0, l, 3); sS = sqof(seq0, 0, 16)
    pk = @f(bd, ple_key); pv = @f(bd, ple_value)
    cphase4!(XF4(XF(1, @f(dd, PLEG), 2560, seq0 + UInt32(2)), 2560), Int32(2560), xs, NORL4, NOSI, NOSF, NORED,
        (Job4{0}(pk, pk.rows, EM(1, @f(dd, PK), s1), n4), Job4{0}(pv, pv.rows, EM(1, @f(dd, PV), s1), nc)), ctr4of(dd, bd, 17), P)
    b = blockIdx().x - Int32(1); tid = threadIdx().x - Int32(1)
    nk = @f(bd, ple_nk); nq = @f(bd, ple_nq)
    while b < Int32(5) * P              # partial sums of position p = b / 5 (CTA c0 < 4: lane c0; 4: sum V^2) to both dies
        p = b ÷ Int32(5); c0 = b % Int32(5)
        PK = @f(dd, PK) + 8 * p * n4; PV = @f(dd, PV) + 8 * p * nc; S = @f(dd, S) + 8 * p * n4
        PPT = @f(dd, PPT) + 8 * p * Int32(32); PPTp = @f(dd, PPTp); PPTp != U64P(0) && (PPTp += 8 * p * Int32(32))
        a = 0.0f0; bb = 0.0f0; dt = 0.0f0; c = tid
        while c < nc
            if c0 < Int32(4)
                k = llget(PK, c0 * nc + c, s1); s = llget(S, c0 * nc + c, sS)
                a = fma(k, k, a); bb = fma(s, s, bb); dt = fma(k * ldf(nk, c0 * nc + c), s * ldf(nq, c0 * nc + c), dt)
            else
                v = llget(PV, c, s1); a = fma(v, v, a)
            end
            c += blockDim().x
        end
        a = bsum(a); bb = bsum(bb); dt = bsum(dt)
        if tid == Int32(0)
            vals = c0 < Int32(4) ? ((c0, a), (Int32(4) + c0, bb), (Int32(8) + c0, dt)) : ((Int32(12), a), (Int32(12), a), (Int32(12), a))
            for (o, v) in vals
                llput(PPT, Int32(16) * d + o, s2, v)
                PPTp != U64P(0) && llput(PPTp, Int32(16) * d + o, s2, v)
            end
        end
        b += gridDim().x
    end
    # finish on my channels, positions in order (the dilated conv reads positions 3, 6, 9 back: a ring of 16)
    ncw = @f(bd, ple_nc); cw = @f(bd, ple_conv); ring = @f(bd, pler)
    i = gtid()
    @inbounds while i < n4
        ln = i ÷ nc; c = i % nc
        for p in Int32(0):P-Int32(1)
            tp = t + p
            PPT = @f(dd, PPT) + 8 * p * Int32(32); S = @f(dd, S) + 8 * p * n4
            rK = 1.0f0 / sqrt(pls(PPT, n, ln, s2) / 2560.0f0 + EPS); rS = 1.0f0 / sqrt(pls(PPT, n, Int32(4) + ln, s2) / 2560.0f0 + EPS)
            s = rK * rS * pls(PPT, n, Int32(8) + ln, s2) * (1.0f0 / sqrt(2560.0f0))
            gt = sigm(sign(s) * sqrt(max(abs(s), 1.0f-6)))
            rG = 1.0f0 / sqrt(gt * gt * pls(PPT, n, Int32(12), s2) / 2560.0f0 + EPS)
            G = gt * llget(@f(dd, PV) + 8 * p * nc, c, s1)
            x = G * rG * ldf(ncw, i)
            x9 = tp >= Int32(9) ? ldf(ring, ((tp - Int32(9)) & Int32(15)) * n4 + i) : 0.0f0
            x6 = tp >= Int32(6) ? ldf(ring, ((tp - Int32(6)) & Int32(15)) * n4 + i) : 0.0f0
            x3 = tp >= Int32(3) ? ldf(ring, ((tp - Int32(3)) & Int32(15)) * n4 + i) : 0.0f0
            y = ldf(cw, Int32(4) * i) * x9
            y = fma(ldf(cw, Int32(4) * i + Int32(1)), x6, y)
            y = fma(ldf(cw, Int32(4) * i + Int32(2)), x3, y)
            y = fma(ldf(cw, Int32(4) * i + Int32(3)), x, y)
            llput(S, i, s3, llget(S, i, sS) + (G + silu(y)))
            stf(ring, (tp & Int32(15)) * n4 + i, x)
        end
        i += gstride()
    end
end
# hc down of P positions: two passes of two lanes each (x = 2 nc), each pass's segments its own partials
function op4_hcd!(dd, bd, l, slot, which::Val{W}, sS, seq0, xs, P::Int32) where {W}
    nc = @f(dd, nc); d = @f(dd, d); par = l & Int32(1)
    m = W == :attn ? @f(bd, hcA_dj) : @f(bd, hcF_dj)
    w = W == :attn ? @f(bd, hcA_w) : @f(bd, hcF_w)
    sq = sqof(seq0, l, slot)
    DP = @f(dd, DP) + 8 * (par * Int32(2600)); DPp = @f(dd, DPp)
    DPp != U64P(0) && (DPp += 8 * (par * Int32(2600)))
    m.rows == Int32(0) && return
    em = EM(3, DP, sq; d2 = DPp, o1 = 1300d, nc = nc, lo = 0)
    lsA = cphase4!(XF4(XF(2, @f(dd, S), Int32(2) * nc, sS; w = w, nc = nc), pS4(dd)), Int32(2) * nc, xs, NORL4, NOSI, NOSF, NORED,
        (Job4{4}(m, m.rows, em, PDP4; xo = 0),), ctr4of(dd, bd, W == :ffn ? 6 : 0), P)
    lsB = cphase4!(XF4(XF(2, @f(dd, S), Int32(2) * nc, sS; w = w, n2 = Int32(2) * nc, nc = nc), pS4(dd)), Int32(2) * nc, xs, NORL4, NOSI, NOSF, NORED,
        (Job4{4}(m, m.rows, em, PDP4; xo = Int32(2) * nc),), ctr4of(dd, bd, W == :ffn ? 7 : 1), P)
    if blockIdx().x == Int32(1)        # per-warp lane sums (16 warps): [parity][die][16 warps][4 lanes] per position
        tid = threadIdx().x - Int32(1); lane = tid % Int32(32); warp = tid ÷ Int32(32)
        LS = @f(dd, LS); LSp = @f(dd, LSp); o = (par * Int32(2) + d) * Int32(64) + Int32(4) * warp
        Base.Cartesian.@nexprs 4 p -> if Int32(p - 1) < P
            Base.Cartesian.@nexprs 4 ln -> begin
                s_ln = wsum(lsA[p][ln] + lsB[p][ln], Val(32))
                if lane == Int32(0)
                    llput(LS + 8 * Int32(p - 1) * Int32(PLS4), o + Int32(ln - 1), sq, s_ln)
                    LSp != U64P(0) && llput(LSp + 8 * Int32(p - 1) * Int32(PLS4), o + Int32(ln - 1), sq, s_ln)
                end
            end
        end
    end
end
# r_l of one position from 16 warps' lane sums of both dies: lane j loads words j, j + 32 of each die's 64
@inline function hc_rl4(p::U64P, sq::UInt32, eps::Float32)
    lane = (threadIdx().x - Int32(1)) % Int32(32)
    w = (llraw(p, lane), llraw(p, Int32(32) + lane), llraw(p, Int32(64) + lane), llraw(p, Int32(96) + lane))
    v = (llfix(p, lane, sq, w[1]) + llfix(p, Int32(32) + lane, sq, w[2])) + (llfix(p, Int32(64) + lane, sq, w[3]) + llfix(p, Int32(96) + lane, sq, w[4]))
    v += shfl_xor_sync(FULL, v, 4); v += shfl_xor_sync(FULL, v, 8); v += shfl_xor_sync(FULL, v, 16)
    Base.Cartesian.@ntuple 4 l -> (1.0f0 / sqrt(shfl_sync(FULL, v, Int32(l)) / 2560.0f0 + eps))
end
function op4_hcu!(dd, bd, l, slot, which::Val{W}, sS, seq0, xs, P::Int32) where {W}
    nc = @f(dd, nc); d = @f(dd, d); n = @f(dd, n); par = l & Int32(1)
    m = W == :attn ? @f(bd, hcA_up) : @f(bd, hcF_up)
    w = W == :attn ? @f(bd, hcA_w) : @f(bd, hcF_w)
    sq = sqof(seq0, l, slot); sqp = sqof(seq0, l, slot - 1)
    LS = @f(dd, LS) + 8 * (par * Int32(128))
    rl = Base.Cartesian.@ntuple 4 p -> (Int32(p - 1) < P ? hc_rl4(LS + 8 * Int32(p - 1) * Int32(PLS4), sqp, EPS) : NORL)
    tid = threadIdx().x - Int32(1)
    if blockIdx().x == Int32(1) && tid < Int32(8) * P
        p = tid ÷ Int32(8); i = tid % Int32(8)
        DP = @f(dd, DP) + 8 * (p * Int32(PDP4) + par * Int32(2600))
        rlp = p == Int32(0) ? rl[1] : p == Int32(1) ? rl[2] : p == Int32(2) ? rl[3] : rl[4]
        if i < Int32(4)
            s = 0.0f0
            Base.Cartesian.@nexprs 4 ln -> begin
                ps_ln = llget(DP, Int32(324 * (ln - 1) + 320) + i, sqp) + llget(DP, Int32(1300 + 324 * (ln - 1) + 320) + i, sqp)
                s = fma(rlp[ln], ps_ln, s)
            end
            llput(@f(dd, INJ) + 8 * p * Int32(PINJ4), Int32(8) * par + i, sq, s)
        else
            llput(@f(dd, INJ) + 8 * p * Int32(PINJ4), Int32(8) * par + i, sq, rl4(rlp, i - Int32(4)))
        end
    end
    Z = @f(dd, Z) + 8 * (par * Int32(2560)); Zp = @f(dd, Zp)
    Zp != U64P(0) && (Zp += 8 * (par * Int32(2560)))
    em = EM(4, Z, sq; d2 = Zp, nc = nc, S = @f(dd, S), sS = sS, w = w, zoff = nc * d)
    cphase4!(XF4(XF(3, @f(dd, DP) + 8 * (par * Int32(2600)), n, sqp), PDP4), Int32(320), xs, rl, NOSI, NOSF, NORED,
        (Job4{0}(m, m.rows, em, PZ4; pS = pS4(dd)),), ctr4of(dd, bd, W == :ffn ? 8 : 2), P)
end
function op4_proj_dn!(dd, bd, l, seq0, xs, P::Int32)
    par = l & Int32(1); nkh = @f(dd, nkh)
    sz = sqof(seq0, l, 5); sq = sqof(seq0, l, 6)
    qkv = @f(bd, qkv); gate = @f(bd, gate); ab = @f(bd, ab)
    cphase4!(XF4(XF(1, @f(dd, Z) + 8 * (par * Int32(2560)), 2560, sz), PZ4), Int32(2560), xs, NORL4, NOSI, NOSF, NORED,
        (Job4{0}(qkv, qkv.rows, EM(1, @f(dd, QKV) + 8 * (par * Int32(640) * nkh), sq), Int32(1280) * nkh),
         Job4{0}(gate, gate.rows, EM(1, @f(dd, GATE) + 8 * (par * Int32(384) * nkh), sq), Int32(768) * nkh),
         Job4{0}(ab, ab.rows, EM(1, @f(dd, AB) + 8 * (par * Int32(6) * nkh), sq), Int32(12) * nkh)), ctr4of(dd, bd, 3), P)
end
function op4_proj_attn!(dd, bd, l, seq0, xs, P::Int32)
    par = l & Int32(1); nq = @f(dd, nq); nkv = @f(dd, nkv)
    sz = sqof(seq0, l, 5); sq = sqof(seq0, l, 6)
    wq = @f(bd, wq); wk = @f(bd, wk); wv = @f(bd, wv); ix = @f(bd, idx_x)
    cphase4!(XF4(XF(1, @f(dd, Z) + 8 * (par * Int32(2560)), 2560, sz), PZ4), Int32(2560), xs, NORL4, NOSI, NOSF, NORED,
        (Job4{0}(wq, wq.rows, EM(1, @f(dd, QG), sq), Int32(512) * nq), Job4{0}(wk, wk.rows, EM(1, @f(dd, KK), sq), Int32(256) * nkv),
         Job4{0}(wv, wv.rows, EM(1, @f(dd, VV), sq), Int32(256) * nkv), Job4{0}(ix, ix.rows, EM(1, @f(dd, IX), sq), Int32(640))), ctr4of(dd, bd, 3), P)
end
# DeltaNet of P positions in order per value head h; the recurrent state in 4 copies: the round's base is copy bank,
# position p reads copy (bank + p) & 3 and writes (bank + p + 1) & 3; the conv inputs in a ring of 8 positions
function op4_dn!(dd, bd, l, seq0, t::Int32, P::Int32, bank::Int32, xs::SPtr)
    nkh = @f(dd, nkh); par = l & Int32(1)
    h = blockIdx().x - Int32(1)
    n = Int32(640) * nkh; cps = Int32(3) * nkh * Int32(16384)
    us = CuDeviceArray{Float32,1,AS.Shared}(xs, (384,))     # the x buffer is free between phases (static shared is full)
    while h < Int32(3) * nkh
        sq_in = sqof(seq0, l, 6); sq = sqof(seq0, l, 7)
        cw = @f(bd, conv_w); st = @f(bd, conv); ST = @f(bd, dn)
        tid = threadIdx().x - Int32(1)
        kh = h % nkh
        for p in Int32(0):P-Int32(1)
            tp = t + p
            QKV = @f(dd, QKV) + 8 * (p * Int32(1280) * nkh + par * Int32(640) * nkh)
            GATE = @f(dd, GATE) + 8 * (p * Int32(768) * nkh + par * Int32(384) * nkh)
            AB = @f(dd, AB) + 8 * (p * Int32(12) * nkh + par * Int32(6) * nkh)
            c = tid
            @inbounds while c < Int32(384)
                ch = c < Int32(128) ? Int32(128) * kh + c : c < Int32(256) ? Int32(128) * nkh + Int32(128) * kh + (c - Int32(128)) : Int32(256) * nkh + Int32(128) * h + (c - Int32(256))
                xc = llget(QKV, ch, sq_in)
                # the 3 previous inputs: this round's from their LL words (the ring slot of a q/k channel is another
                # CTA's), earlier rounds' from the ring
                Q0 = @f(dd, QKV) + 8 * (par * Int32(640) * nkh)
                x1 = p >= Int32(1) ? llget(Q0 + 8 * (p - Int32(1)) * Int32(1280) * nkh, ch, sq_in) : ldf(st, ((tp - Int32(1)) & Int32(7)) * n + ch)
                x2 = p >= Int32(2) ? llget(Q0 + 8 * (p - Int32(2)) * Int32(1280) * nkh, ch, sq_in) : ldf(st, ((tp - Int32(2)) & Int32(7)) * n + ch)
                x3 = p >= Int32(3) ? llget(Q0 + 8 * (p - Int32(3)) * Int32(1280) * nkh, ch, sq_in) : ldf(st, ((tp - Int32(3)) & Int32(7)) * n + ch)
                y = ldf(cw, Int32(4) * ch) * x3
                y = fma(ldf(cw, Int32(4) * ch + Int32(1)), x2, y)
                y = fma(ldf(cw, Int32(4) * ch + Int32(2)), x1, y)
                y = fma(ldf(cw, Int32(4) * ch + Int32(3)), xc, y)
                us[c+1] = silu(y)
                (c >= Int32(256) || h < nkh) && stf(st, (tp & Int32(7)) * n + ch, xc)
                c += blockDim().x
            end
            sync_threads()
            dn_core2!(ST + 4 * (((bank + p) & Int32(3)) * cps), ST + 4 * (((bank + p + Int32(1)) & Int32(3)) * cps), us, xs + 4 * 384, AB, GATE,
                @f(bd, ab_bias), @f(bd, ssm_a), @f(bd, ssm_norm), h, nkh, sq_in, @f(dd, ON) + 8 * (p * Int32(768) * nkh + par * Int32(384) * nkh), sq)
            sync_threads()
            # the q/k channels of key head kh: the conv state slot must hold xc before the next position's conv reads it
            threadfence_block()
        end
        h += gridDim().x
    end
end
# dn_core! with the state read from STs and written to STd
@inline function dn_core2!(STs::FPtr, STd::FPtr, us, scr::SPtr, AB::U64P, GATE::U64P, abb::FPtr, ssa::FPtr, nw::FPtr, h::Int32, nkh::Int32, sq_in::UInt32, ON::U64P, sq::UInt32)
    tid = threadIdx().x - Int32(1)
    qs = CuDeviceArray{Float32,1,AS.Shared}(scr, (128,)); ks = CuDeviceArray{Float32,1,AS.Shared}(scr + 512, (128,))
    vs = CuDeviceArray{Float32,1,AS.Shared}(scr + 1024, (128,)); os = CuDeviceArray{Float32,1,AS.Shared}(scr + 1536, (128,))
    @inbounds begin
        qv = tid < Int32(128) ? us[tid+1] : 0.0f0
        kv = tid < Int32(128) ? us[tid+129] : 0.0f0
        sq2 = bsum(qv * qv); sk = bsum(kv * kv)
        if tid < Int32(128)
            qs[tid+1] = qv / sqrt(sq2 + EPS)
            ks[tid+1] = kv / sqrt(sk + EPS)
            vs[tid+1] = us[tid+257]
        end
        sync_threads()
        a = llget(AB, h, sq_in) + ldf(abb, h); b = llget(AB, Int32(3) * nkh + h, sq_in) + ldf(abb, Int32(3) * nkh + h)
        alpha = exp(ldf(ssa, h) * softplus(a)); beta = sigm(b)
        if tid < Int32(256)
            c0 = Int32(4) * (tid ÷ Int32(8)); i0 = Int32(16) * (tid % Int32(8))
            off = 4 * (Int32(16384) * h + Int32(128) * c0 + i0)      # (row c0 + cc is 512 bytes further per cc)
            if DNL4
                # the four rows' loads issued together, ahead of their dot products: the op is latency-bound, and a
                # row's load cannot move above the previous row's store (STs and STd may alias, as far as LLVM knows)
                sv0 = ld16f(STs + off); sv1 = ld16f(STs + off + 512); sv2 = ld16f(STs + off + 1024); sv3 = ld16f(STs + off + 1536)
                dn_row2!(STd + off, sv0, ks, qs, vs, os, i0, c0, alpha, beta, tid)
                dn_row2!(STd + off + 512, sv1, ks, qs, vs, os, i0, c0 + Int32(1), alpha, beta, tid)
                dn_row2!(STd + off + 1024, sv2, ks, qs, vs, os, i0, c0 + Int32(2), alpha, beta, tid)
                dn_row2!(STd + off + 1536, sv3, ks, qs, vs, os, i0, c0 + Int32(3), alpha, beta, tid)
            else
                for cc in Int32(0):Int32(3)
                    dn_row2!(STd + off + 512 * cc, ld16f(STs + off + 512 * cc), ks, qs, vs, os, i0, c0 + cc, alpha, beta, tid)
                end
            end
        end
        sync_threads()
        ov = tid < Int32(128) ? os[tid+1] : 0.0f0
        ss = bsum(ov * ov)
        if tid < Int32(128)
            r = 1.0f0 / sqrt(ss / 128.0f0 + EPS)
            llput(ON, Int32(128) * h + tid, sq, ov * r * ldf(nw, tid) * sigm(llget(GATE, Int32(128) * h + tid, sq_in)))
        end
    end
end
# one state row of dn_core2!: this thread's 16 of row c (already loaded as sv) against k, the delta-rule update written
# to dst, and the row's readout against q into os[c]
@inline function dn_row2!(dst::FPtr, sv, ks, qs, vs, os, i0::Int32, c::Int32, alpha::Float32, beta::Float32, tid::Int32)
    @inbounds begin
        r = 0.0f0
        for k in 1:16
            r = fma(ks[i0+k], sv[k], r)
        end
        r = wsum(r, Val(8)) * alpha
        be = beta * (vs[c+1] - r)
        nv = dn_update(sv, ks, i0, be, alpha)
        st16f(dst, nv)
        o = 0.0f0
        for k in 1:16
            o = fma(qs[i0+k], nv[k], o)
        end
        o = wsum(o, Val(8)) * 0.088388347648318440550f0
        (tid % Int32(8)) == Int32(0) && (os[c+1] = o)
    end
    nothing
end
# FLASHNEXT_DNL4=1 (read when the kernels compile): dn_core2! loads a thread's four state rows before computing them
const DNL4 = get(ENV, "FLASHNEXT_DNL4", "0") == "1"
function op4_aprep!(dd, bd, l, seq0, t::Int32, P::Int32, xs::SPtr)
    nq = @f(dd, nq); nkv = @f(dd, nkv)
    sq_in = sqof(seq0, l, 6); sq = sqof(seq0, l, 7)
    hb = blockIdx().x - Int32(1); d = threadIdx().x - Int32(1)
    sh = CuDeviceArray{Float32,1,AS.Shared}(xs, (256,))
    while hb < nq + nkv + Int32(1)
        for p in Int32(0):P-Int32(1)
            tp = t + p
            cs = @f(dd, rope) + 4 * (Int32(64) * tp); cs3 = @f(dd, rope) + 4 * (Int32(64) * max(tp - Int32(3), Int32(0)))
            if hb < nq
                x = d < Int32(256) ? llget(@f(dd, QG) + 8 * p * Int32(512) * nq, Int32(512) * hb + d, sq_in) : 0.0f0
                y = norm_rope256x(x, d, @f(bd, q_norm), cs, sh)
                d < Int32(256) && llput(@f(dd, QO) + 8 * p * Int32(256) * nq, Int32(256) * hb + d, sq, y)
            elseif hb < nq + nkv
                kv = hb - nq
                x = d < Int32(256) ? llget(@f(dd, KK) + 8 * p * Int32(256) * nkv, Int32(256) * kv + d, sq_in) : 0.0f0
                y = norm_rope256x(x, d, @f(bd, k_norm), cs, sh)
                row = tp * nkv + kv
                if d < Int32(256)
                    kvq32(y, @f(bd, kq), row * Int32(256) + d, @f(bd, ks), row * Int32(8) + d ÷ Int32(32))
                    kvq32(llget(@f(dd, VV) + 8 * p * Int32(256) * nkv, Int32(256) * kv + d, sq_in), @f(bd, vq), row * Int32(256) + d, @f(bd, vs), row * Int32(8) + d ÷ Int32(32))
                end
            else
                ring = @f(bd, ring)
                d < Int32(128) && stf(ring, (tp & Int32(3)) * Int32(128) + d, llget(@f(dd, IX) + 8 * p * Int32(640), Int32(512) + d, sq_in))
                sync_threads()
                x = d < Int32(128) ? (((ldf(ring, d) + ldf(ring, Int32(128) + d)) + ldf(ring, Int32(256) + d)) + ldf(ring, Int32(384) + d)) * 0.25f0 : 0.0f0
                ss = bsum(x * x)
                y = x / sqrt(ss / 128.0f0 + EPS) * (d < Int32(128) ? ldf(@f(bd, idx_kn), d) : 0.0f0)
                d < Int32(256) && (sh[d+1] = y)
                sync_threads()
                if d < Int32(64)
                    pp = d % Int32(32)
                    cc = ldf(cs3, pp); s = ldf(cs3, Int32(32) + pp)
                    y = d < Int32(32) ? fma(-sh[d+33], s, sh[d+1] * cc) : fma(sh[d-31], s, sh[d+1] * cc)
                end
                if d < Int32(128) && (tp & Int32(3)) == Int32(3)
                    unsafe_store!(reinterpret(Core.LLVMPtr{Float16,1}, @f(bd, pool)), Float16(y), (tp ÷ Int32(4)) * Int32(128) + d + Int32(1), Val(2))
                end
            end
            sync_threads()
        end
        if hb >= nq && hb < nq + nkv
            threadfence()
            sync_threads()
            d == Int32(0) && llput(@f(dd, KVR), hb - nq, sq, 1.0f0)
        end
        hb += gridDim().x
    end
end
# attention of (head h, position p) over keys 0 .. t + p (the round's keys are in the cache: op4_aprep!)
function op4_attn!(dd, bd, l, seq0, t::Int32, xs::SPtr, P::Int32)
    nq = @f(dd, nq); nkv = @f(dd, nkv)
    sq_in = sqof(seq0, l, 7); sq = sqof(seq0, l, 8); sqg = sqof(seq0, l, 6)
    hp = blockIdx().x - Int32(1)
    tid = threadIdx().x - Int32(1); lane = tid % Int32(32); w = tid ÷ Int32(32); nw = blockDim().x ÷ Int32(32)
    msh = CuStaticSharedArray(Float32, 16); dsh = CuStaticSharedArray(Float32, 16)
    osh = CuDeviceArray{Float32,2,AS.Shared}(xs, (256, 16))
    kq = @f(bd, kq); ks = @f(bd, ks); vq = @f(bd, vq); vs = @f(bd, vs)
    @inbounds while hp < nq * P
        h = hp % nq; p = hp ÷ nq; tp = t + p
        kv = h ÷ Int32(12)
        llget(@f(dd, KVR), kv, sq_in)
        q = ldq8(@f(dd, QO) + 8 * p * Int32(256) * nq, h, lane, sq_in)
        m = -1.0f30; dsum = 0.0f0
        acc = ntuple(i -> 0.0f0, Val(8))
        j = w
        while j <= tp
            row = j * nkv + kv
            s = 0.0f0
            for i in 1:8
                dd_ = Int32(8) * lane + Int32(i - 1)
                s = fma(q[i], kvd(kq, row * Int32(256) + dd_, ks, row * Int32(8) + dd_ ÷ Int32(32)), s)
            end
            s = wsum(s, Val(32)) * 0.0625f0
            mn = max(m, s)
            sc = exp(m - mn); pr = exp(s - mn)
            dsum = fma(dsum, sc, pr)
            acc = vacc(acc, pr, sc, vq, vs, row, lane)
            m = mn
            j += nw
        end
        if lane == Int32(0)
            msh[w+1] = m; dsh[w+1] = dsum
        end
        for i in 1:8
            osh[Int32(8)*lane+Int32(i), w+1] = acc[i]
        end
        sync_threads()
        if tid < Int32(256)
            M = msh[1]
            for ww in Int32(2):nw
                M = max(M, msh[ww])
            end
            o = 0.0f0; D = 0.0f0
            for ww in Int32(1):nw
                e = exp(msh[ww] - M)
                o = fma(osh[tid+1, ww], e, o); D = fma(dsh[ww], e, D)
            end
            llput(@f(dd, AO) + 8 * p * Int32(256) * nq, Int32(256) * h + tid, sq,
                o / D * sigm(llget(@f(dd, QG) + 8 * p * Int32(512) * nq, Int32(512) * h + Int32(256) + tid, sqg)))
        end
        sync_threads()
        hp += gridDim().x
    end
end
# out_proj of P positions: K (3072 per die) in two windows; each window's row partials to their own YP slot
function op4_out!(dd, bd, l, seq0, xs, P::Int32)
    nc = @f(dd, nc); d = @f(dd, d); nkh = @f(dd, nkh); nq = @f(dd, nq); par = l & Int32(1)
    attn = @f(bd, attn) == Int32(1)
    sq = sqof(seq0, l, 9)
    m = attn ? @f(bd, wo) : @f(bd, ssm_out)
    Kw = m.K ÷ Int32(2)
    YP = @f(dd, YP); YPp = @f(dd, YPp)
    for pass in Int32(0):Int32(1)
        xf = attn ? XF4(XF(1, @f(dd, AO) + 8 * (pass * Kw), Kw, sqof(seq0, l, 8)), Int32(256) * nq) :
                    XF4(XF(1, @f(dd, ON) + 8 * (par * Int32(384) * nkh + pass * Kw), Kw, sqof(seq0, l, 7)), Int32(768) * nkh)
        em = EM(2, YP, sq; d2 = YPp, o1 = ((par * Int32(2) + pass) * Int32(2) + d) * nc, lo = nc * d, plo = nc * (Int32(1) - d), nc = nc)
        cphase4!(xf, Kw, xs, NORL4, NOSI, NOSF, NORED, (Job4{2}(m, m.rows, em, pYP4(dd); xo = pass * Kw),), ctr4of(dd, bd, 4 + pass), P)
    end
end
# S[p] += 2 sigmoid(inj_l / 4) (sum of the ns partial slots), each position; the handoff and the DFlash2 taps too
function op4_hcw!(dd, bd, l, slot, src::U64P, pst::Int32, ns::Int32, ssrc::UInt32, sinj::UInt32, sS::UInt32, seq0, last::Bool, t::Int32, P::Int32)
    nc = @f(dd, nc); par = l & Int32(1); n4 = Int32(4) * nc
    sq = sqof(seq0, l, slot)
    hand = last ? @f(bd, hand) : U64P(0); hls = @f(bd, hand_ls); hoff = @f(bd, hand_off)
    tap = last ? @f(bd, tap) : NOF
    i = gtid()
    @inbounds while i < P * n4
        p = i ÷ n4; k = i % n4
        ln = k ÷ nc; c = k % nc
        y = 0.0f0
        for q in Int32(0):ns-Int32(1)
            y += llget(src + 8 * p * pst, (par * ns + q) * nc + c, ssrc)
        end
        S = @f(dd, S) + 8 * p * n4
        v = fma(2.0f0 * sigm(0.25f0 * llget(@f(dd, INJ) + 8 * p * Int32(PINJ4), Int32(8) * par + ln, sinj)), y, llget(S, k, sS))
        llput(S, k, sq, v)
        hand != U64P(0) && llput(hand + 8 * p * Int32(4) * hls, ln * hls + hoff + c, sq, v)
        tap != NOF && stf(tap, ((t + p) & Int32(7)) * Int32(TAPSLOT) + nc * @f(dd, d) + ln * Int32(2560) + c, v)
        i += gstride()
    end
end
function op4_rtp!(dd, bd, l, seq0, xs, P::Int32)
    nc = @f(dd, nc); d = @f(dd, d); par = l & Int32(1); F = @f(dd, F)
    sq = sqof(seq0, l, 13); sqe = sqof(seq0, l, 14)
    RP = @f(dd, RP); RPp = @f(dd, RPp)
    o = (par * Int32(2) + d) * Int32(513)
    GU = @f(dd, GU) + 8 * (par * Int32(42) * F)
    shg = @f(bd, sh_gate); shu = @f(bd, sh_up)
    cphase4!(XF4(XF(1, @f(dd, Z) + 8 * (par * Int32(2560)), 2560, sqof(seq0, l, 12)), PZ4), Int32(2560), xs, NORL4, NOSI, NOSF, NORED,
        (Job4{0}(@f(bd, router), 513, EM(1, RP + 8o, sq; d2 = RPp == U64P(0) ? RPp : RPp + 8o), PRP4; xo = nc * d),
         Job4{0}(shg, F, EM(1, GU, sqe; o1 = 40F), pGU4(dd)), Job4{0}(shu, F, EM(1, GU, sqe; o1 = 41F), pGU4(dd))), ctr4of(dd, bd, 9), P)
end
# the router top 10 of one position by warp w (lane holds 16 values): ix[1:10], wv[1:11] (11: the shared expert's gate)
@inline function router_top10w!(ix, wv, RP::U64P, np::Int32, sq::UInt32)
    lane = (threadIdx().x - Int32(1)) % Int32(32)
    @inbounds begin
        vals = rsum16(RP, np, lane, sq)
        sh = 0.0f0
        for q in Int32(0):np-Int32(1)
            sh += llget(RP, q * Int32(513) + Int32(512), sq)
        end
        taken = UInt32(0)
        m0 = 0.0f0
        for j in 1:10
            bv = -Inf32; bi = Int32(-1)
            for k in 1:16
                if (taken >> (k - 1)) & 0x1 == 0x0 && vals[k] > bv
                    bv = vals[k]; bi = Int32(16) * lane + Int32(k - 1)
                end
            end
            for o in (16, 8, 4, 2, 1)
                ov = shfl_xor_sync(FULL, bv, o); oi = shfl_xor_sync(FULL, bi, o)
                if ov > bv || (ov == bv && oi >= Int32(0) && (bi < Int32(0) || oi < bi))
                    bv = ov; bi = oi
                end
            end
            bi ÷ Int32(16) == lane && (taken |= UInt32(1) << (bi % Int32(16)))
            j == 1 && (m0 = bv)
            if lane == Int32(0)
                unsafe_store!(ix, bi, j, Val(4)); unsafe_store!(wv, exp(bv - m0), j, Val(4))
            end
        end
        if lane == Int32(0)
            den = 0.0f0
            for j in 1:10
                den += unsafe_load(wv, j, Val(4))
            end
            for j in 1:10
                unsafe_store!(wv, unsafe_load(wv, j, Val(4)) / den, j, Val(4))
            end
            unsafe_store!(wv, sigm(sh), 11, Val(4))
        end
    end
end
# every CTA: the 4 positions' top 10 (warp p), their union U (first appearance, positions in order) with each
# position's weight per expert (0: not selected; slot U: the shared expert's gate); CTA 0 publishes them for
# op4_down! in IDXW ([0] = U, [1 + u] = id, [41 + 4 u + p] = weight); then gate/up of the U experts for 4 positions
function op4_exp!(dd, bd, l, seq0, xs, P::Int32)
    F = @f(dd, F); n = @f(dd, n); par = l & Int32(1)
    sq = sqof(seq0, l, 14)
    ix4 = CuStaticSharedArray(Int32, 64); wv4 = CuStaticSharedArray(Float32, 64)
    uix = CuStaticSharedArray(Int32, 48); uw = CuStaticSharedArray(Float32, 168); un = CuStaticSharedArray(Int32, 1)
    tid = threadIdx().x - Int32(1); warp = tid ÷ Int32(32)
    if warp < P
        router_top10w!(pointer(ix4) + 4 * Int32(16) * warp, pointer(wv4) + 4 * Int32(16) * warp,
            @f(dd, RP) + 8 * (warp * Int32(PRP4) + par * Int32(1026)), n, sqof(seq0, l, 13))
    end
    sync_threads()
    @inbounds if tid == Int32(0)
        U = Int32(0)
        for p in Int32(0):P-Int32(1), k in Int32(0):Int32(9)
            e = ix4[Int32(16)*p+k+Int32(1)]
            f = Int32(-1)
            for u in Int32(0):U-Int32(1)
                uix[u+1] == e && (f = u)
            end
            if f < Int32(0)
                uix[U+1] = e
                for q in Int32(0):Int32(3)
                    uw[Int32(4)*U+q+Int32(1)] = 0.0f0
                end
                f = U; U += Int32(1)
            end
            uw[Int32(4)*f+p+Int32(1)] = wv4[Int32(16)*p+k+Int32(1)]
        end
        for q in Int32(0):Int32(3)
            uw[Int32(4)*U+q+Int32(1)] = q < P ? wv4[Int32(16)*q+Int32(11)] : 0.0f0
        end
        un[1] = U
    end
    sync_threads()
    U = un[1]
    if blockIdx().x == Int32(1)
        IDXW = @f(dd, IDXW) + 8 * (par * Int32(208))
        tid == Int32(0) && llput(IDXW, Int32(0), sq, reinterpret(Float32, U))
        tid < U && llput(IDXW, Int32(1) + tid, sq, reinterpret(Float32, uix[tid+1]))
        tid < Int32(4) * (U + Int32(1)) && llput(IDXW, Int32(41) + tid, sq, uw[tid+1])
    end
    gu = @f(bd, gu)
    cphase4!(XF4(XF(1, @f(dd, Z) + 8 * (par * Int32(2560)), 2560, sqof(seq0, l, 12)), PZ4), Int32(2560), xs, NORL4, pointer(uix), NOSF, NORED,
        (Job4{1}(gu, Int32(2) * F * U, EM(5, @f(dd, GU) + 8 * (par * Int32(42) * F), sq), pGU4(dd); rpe = Int32(2) * F),), ctr4of(dd, bd, 10), P)
end
# routed (U experts, 8 per phase: x = 8 F) and shared down: (row group, expert) tasks, the 32-row blocks summed by
# the task completing them over all U + 1 slots and 4 positions
function op4_down!(dd, bd, l, seq0, xs, P::Int32)
    nc = @f(dd, nc); d = @f(dd, d); F = @f(dd, F); par = l & Int32(1)
    sqi = sqof(seq0, l, 14); sq = sqof(seq0, l, 15)
    GU = @f(dd, GU) + 8 * (par * Int32(42) * F)
    uix = CuStaticSharedArray(Int32, 48); uw = CuStaticSharedArray(Float32, 168); un = CuStaticSharedArray(Int32, 1)
    tid = threadIdx().x - Int32(1)
    IDXW = @f(dd, IDXW) + 8 * (par * Int32(208))
    tid == Int32(0) && (un[1] = reinterpret(Int32, llget(IDXW, Int32(0), sqi)))
    sync_threads()
    U = un[1]
    tid < U && (uix[tid+1] = reinterpret(Int32, llget(IDXW, Int32(1) + tid, sqi)))
    tid < Int32(4) * (U + Int32(1)) && (uw[tid+1] = llget(IDXW, Int32(41) + tid, sqi))
    sync_threads()
    OUTP = @f(dd, OUTP); OUTPp = @f(dd, OUTPp)
    em = EM(2, OUTP, sq; d2 = OUTPp, o1 = (par * Int32(2) + d) * nc, lo = nc * d, plo = nc * (Int32(1) - d), nc = nc)
    dn = @f(bd, down); shd = @f(bd, sh_down)
    red = Red(@f(dd, RED), @f(dd, CNT), U * dn.L + shd.L, U + Int32(1), Int32(2560))
    c = Int32(0)
    while Int32(8) * c < U
        ne = min(Int32(8), U - Int32(8) * c)
        cphase4!(XF4(XF(1, GU + 8 * (Int32(8) * c * F), ne * F, sqi), pGU4(dd)), ne * F, xs, NORL4, pointer(uix) + 4 * (Int32(8) * c), pointer(uw), red,
            (Job4{3}(dn, 2560, em, pS4(dd); ns = ne, eo = Int32(8) * c, rpe = 2560),), ctr4of(dd, bd, 11 + c), P)
        c += Int32(1)
    end
    cphase4!(XF4(XF(4, GU, F, sqi; n2 = 20, F = F), pGU4(dd)), F, xs, NORL4, NOSI, pointer(uw), red,
        (Job4{3}(shd, 2560, em, pS4(dd); eo = U, ns = 1, rpe = 0),), ctr4of(dd, bd, 16), P)
end
# the per-round program of one die: P <= 4 positions t .. t + P - 1 (B4: n = 2 dies per board)
function k_token4!(dd::Core.LLVMPtr{DD4,1}, bds::Core.LLVMPtr{BD,1}, nblk::Int32, seq0::UInt32, t::Int32, P::Int32, bank::Int32, xsmax::Int32,
                   tim::Core.LLVMPtr{UInt64,1})
    xs = pointer(CuDynamicSharedArray(Float32, (Int(xsmax),)))
    stamp4!(tim, Int32(0))
    for bi in Int32(1):nblk
        bd = bds + (bi - Int32(1)) * Int32(sizeof(BD))
        k0 = Int32(16) * bi
        l = @f(bd, l)
        sS = l == Int32(0) ? sqof(seq0, 0, 0) : sqof(seq0, l - Int32(1), 16)
        l == Int32(0) && op4_sinit!(dd, seq0, P)
        if @f(bd, ple) == Int32(1)
            op4_ple!(dd, bd, seq0, t, xs, P); sync_threads()
            sS = sqof(seq0, 1, 3)
        end
        stamp4!(tim, k0)
        op4_hcd!(dd, bd, l, 4, Val(:attn), sS, seq0, xs, P); sync_threads(); stamp4!(tim, k0 + Int32(1))
        op4_hcu!(dd, bd, l, 5, Val(:attn), sS, seq0, xs, P); sync_threads(); stamp4!(tim, k0 + Int32(2))
        if @f(bd, attn) == Int32(1)
            op4_proj_attn!(dd, bd, l, seq0, xs, P); sync_threads(); stamp4!(tim, k0 + Int32(3))
            op4_aprep!(dd, bd, l, seq0, t, P, xs); sync_threads(); stamp4!(tim, k0 + Int32(4))
            op4_attn!(dd, bd, l, seq0, t, xs, P); sync_threads(); stamp4!(tim, k0 + Int32(5))
        else
            op4_proj_dn!(dd, bd, l, seq0, xs, P); sync_threads(); stamp4!(tim, k0 + Int32(3))
            op4_dn!(dd, bd, l, seq0, t, P, bank, xs); sync_threads(); stamp4!(tim, k0 + Int32(4)); stamp4!(tim, k0 + Int32(5))
        end
        op4_out!(dd, bd, l, seq0, xs, P); sync_threads(); stamp4!(tim, k0 + Int32(6))
        op4_hcw!(dd, bd, l, 10, @f(dd, YP), pYP4(dd), Int32(4), sqof(seq0, l, 9), sqof(seq0, l, 5), sS, seq0, false, t, P); sync_threads(); stamp4!(tim, k0 + Int32(7))
        sA = sqof(seq0, l, 10)
        op4_hcd!(dd, bd, l, 11, Val(:ffn), sA, seq0, xs, P); sync_threads(); stamp4!(tim, k0 + Int32(8))
        op4_hcu!(dd, bd, l, 12, Val(:ffn), sA, seq0, xs, P); sync_threads(); stamp4!(tim, k0 + Int32(9))
        op4_rtp!(dd, bd, l, seq0, xs, P); sync_threads(); stamp4!(tim, k0 + Int32(10))
        op4_exp!(dd, bd, l, seq0, xs, P); sync_threads(); stamp4!(tim, k0 + Int32(11))
        op4_down!(dd, bd, l, seq0, xs, P); sync_threads(); stamp4!(tim, k0 + Int32(12))
        op4_hcw!(dd, bd, l, 16, @f(dd, OUTP), pS4(dd), Int32(2), sqof(seq0, l, 15), sqof(seq0, l, 12), sA, seq0, true, t, P); sync_threads(); stamp4!(tim, k0 + Int32(13))
    end
    return
end
# CTA 0's op ends (--prof): tim[16 block + op]
@inline function stamp4!(tim::Core.LLVMPtr{UInt64,1}, k::Int32)
    if tim != Core.LLVMPtr{UInt64,1}(0) && threadIdx().x == Int32(1) && blockIdx().x == Int32(1)
        unsafe_store!(tim, gtimer(), k + Int32(1), Val(8))
    end
end
const KTOKEN4_TT = Tuple{Core.LLVMPtr{DD4,1},Core.LLVMPtr{BD,1},Int32,UInt32,Int32,Int32,Int32,Int32,Core.LLVMPtr{UInt64,1}}

# ---- B8: B4's rounds of P <= 4 positions t .. t + P - 1 on the tensor-split layout (one group of n dies, n even):
# each position its own copy of the LL region (position p at + p PSTR words), B1's weights and recurrent state (DeltaNet
# copies (t + p) & 3, conv ring 8, PLE ring 16, KV cache rows by position: a round's rejected positions are
# overwritten when redone), the exchanges per position as B1's, the GEMV phases of all P positions together
# (cphase4!: each weight read once a round). The MoE: each position's 10 experts' h to every die (its 11 slots as
# B1's), the gate/up and down rows of the round's union of experts read once (B4). Needs ROUTERCOPY and KVCOPY. ----
@inline pp(dd, X::U64P, p) = X + 8 * (Int32(p) * @f(dd, PSTR))
# the dynamic shared memory: the x slots of a run (XSM8 floats: 3 positions at K = 2560, 4 at K <= 1920), then the
# IQ4 LUT and the round's tables (once: each inlined static array would be its own shared memory, and 512 threads
# leave 48 KB for all of it). Per position p < P8MAX: its top 10 (ix4, wv4: 16 apart); the union U <= 10 P8MAX of
# experts (uix), each one's weight and slot per position (uw, slt: [P8MAX u + p]); ord ([10 p + rank] -> union slot);
# a run's experts (rix) and their slot rows (rsl: [P8MAX k + p])
const XSM8 = 3 * 3080; const SC8 = 16 + 2004
@inline b8lut(xs::SPtr) = xs + 4 * XSM8
@inline function b8tabs(xs::SPtr)
    b = xs + 4 * (XSM8 + 16)
    (CuDeviceArray{Int32,1,AS.Shared}(reinterpret(SIP, b), (16P8MAX,)), CuDeviceArray{Float32,1,AS.Shared}(b + 4 * 128, (16P8MAX,)),
     CuDeviceArray{Int32,1,AS.Shared}(reinterpret(SIP, b + 4 * 256), (88,)), CuDeviceArray{Float32,1,AS.Shared}(b + 4 * 344, (81P8MAX,)),
     CuDeviceArray{Float32,1,AS.Shared}(b + 4 * 992, (80P8MAX,)), CuDeviceArray{Int32,1,AS.Shared}(reinterpret(SIP, b + 4 * 1632), (4,)),
     CuDeviceArray{Int32,1,AS.Shared}(reinterpret(SIP, b + 4 * 1636), (10P8MAX,)),
     CuDeviceArray{Int32,1,AS.Shared}(reinterpret(SIP, b + 4 * 1716), (32,)), CuDeviceArray{Float32,1,AS.Shared}(b + 4 * 1748, (32P8MAX,)))
end
# UIDX (global, a parity's UIDXW words): [0] U, [1 + u] expert id, [UW0 + P8MAX u + p] weight (u = U: the shared
# expert's gate), [US0 + P8MAX u + p] slot (the position's rank of the expert, -1 if not its)
const UW0 = 81; const US0 = UW0 + 81P8MAX; const UIDXW = US0 + 80P8MAX + 7
function op8_sinit!(dd, seq0, P::Int32)
    nc = @f(dd, nc); d = @f(dd, d); n4 = Int32(4) * nc
    i = gtid()
    while i < P * n4
        p = i ÷ n4; k = i - p * n4
        llput(pp(dd, @f(dd, S), p), k, sqof(seq0, 0, 0), llget(@f(dd, EMB) + 8 * p * Int32(2560), nc * d + k % nc, seq0 + UInt32(1)))
        i += gstride()
    end
end
function op8_ple!(dd, bd, seq0, t::Int32, xs, P::Int32)
    l = Int32(1); nc = @f(dd, nc); d = @f(dd, d); n = @f(dd, n); n4 = Int32(4) * nc; PS = @f(dd, PSTR)
    s1 = sqof(seq0, l, 1); s2 = sqof(seq0, l, 2); s3 = sqof(seq0, l, 3); sS = sqof(seq0, 0, 16)
    pk = @f(bd, ple_key); pv = @f(bd, ple_value)
    cphase8!(XF4(XF(1, @f(dd, PLEG), 2560, seq0 + UInt32(2)), 2560), Int32(2560), xs, NORL8, NOSI, NOSF, NORED,
        (Job4{0}(pk, pk.rows, EM(1, @f(dd, PK), s1), PS), Job4{0}(pv, pv.rows, EM(1, @f(dd, PV), s1), PS)), ctr8of(dd, bd, 31), P, b8lut(xs))
    b = blockIdx().x - Int32(1); tid = threadIdx().x - Int32(1)
    nk = @f(bd, ple_nk); nq = @f(bd, ple_nq)
    while b < Int32(5) * P              # partial sums of position p = b / 5 (CTA c0 < 4: lane c0; 4: sum V^2) to every die
        p = b ÷ Int32(5); c0 = b % Int32(5)
        PK = pp(dd, @f(dd, PK), p); PV = pp(dd, @f(dd, PV), p); S = pp(dd, @f(dd, S), p); PPT = pp(dd, @f(dd, PPT), p)
        a = 0.0f0; bb = 0.0f0; dt = 0.0f0; c = tid
        while tid < Int32(256) && c < nc       # (B1's partition: its 256 threads' strides; the sums are B1's)
            if c0 < Int32(4)
                k = llget(PK, c0 * nc + c, s1); s = llget(S, c0 * nc + c, sS)
                a = fma(k, k, a); bb = fma(s, s, bb); dt = fma(k * ldf(nk, c0 * nc + c), s * ldf(nq, c0 * nc + c), dt)
            else
                v = llget(PV, c, s1); a = fma(v, v, a)
            end
            c += Int32(256)
        end
        a = bsum(a); bb = bsum(bb); dt = bsum(dt)
        if tid == Int32(0)
            vals = c0 < Int32(4) ? ((c0, a), (Int32(4) + c0, bb), (Int32(8) + c0, dt)) : ((Int32(12), a), (Int32(12), a), (Int32(12), a))
            for (o, v) in vals
                bput!(dd, PPT, Int32(16) * d + o, s2, v)
            end
        end
        b += gridDim().x
    end
    ncw = @f(bd, ple_nc); cw = @f(bd, ple_conv); ring = @f(bd, pler)
    i = gtid()
    @inbounds while i < n4
        ln = i ÷ nc; c = i % nc
        for p in Int32(0):P-Int32(1)
            tp = t + p
            PPT = pp(dd, @f(dd, PPT), p); S = pp(dd, @f(dd, S), p)
            rK = 1.0f0 / sqrt(pls(PPT, n, ln, s2) / 2560.0f0 + EPS); rS = 1.0f0 / sqrt(pls(PPT, n, Int32(4) + ln, s2) / 2560.0f0 + EPS)
            s = rK * rS * pls(PPT, n, Int32(8) + ln, s2) * (1.0f0 / sqrt(2560.0f0))
            gt = sigm(sign(s) * sqrt(max(abs(s), 1.0f-6)))
            rG = 1.0f0 / sqrt(gt * gt * pls(PPT, n, Int32(12), s2) / 2560.0f0 + EPS)
            G = gt * llget(pp(dd, @f(dd, PV), p), c, s1)
            x = G * rG * ldf(ncw, i)
            x9 = tp >= Int32(9) ? ldf(ring, ((tp - Int32(9)) & Int32(15)) * n4 + i) : 0.0f0
            x6 = tp >= Int32(6) ? ldf(ring, ((tp - Int32(6)) & Int32(15)) * n4 + i) : 0.0f0
            x3 = tp >= Int32(3) ? ldf(ring, ((tp - Int32(3)) & Int32(15)) * n4 + i) : 0.0f0
            y = ldf(cw, Int32(4) * i) * x9
            y = fma(ldf(cw, Int32(4) * i + Int32(1)), x6, y)
            y = fma(ldf(cw, Int32(4) * i + Int32(2)), x3, y)
            y = fma(ldf(cw, Int32(4) * i + Int32(3)), x, y)
            llput(S, i, s3, llget(S, i, sS) + (G + silu(y)))
            stf(ring, (tp & Int32(15)) * n4 + i, x)
        end
        i += gstride()
    end
end
# hc down of P positions: each position's lane sums first, then two passes of two lanes (B4: x = 2 nc), then each
# position's lane-summed partials to their owners
function op8_hcd!(dd, bd, l, slot, which::Val{W}, sS, seq0, xs, P::Int32) where {W}
    nc = @f(dd, nc); d = @f(dd, d); n = @f(dd, n); RO = @f(dd, RO); q = hcq(l, W); PS = @f(dd, PSTR)
    m = W == :attn ? @f(bd, hcA_dj) : @f(bd, hcF_dj)
    w = W == :attn ? @f(bd, hcA_w) : @f(bd, hcF_w)
    sq = sqof(seq0, l, slot)
    m.rows == Int32(0) && return
    LSq = @f(dd, LS) + 8 * (q * n * Int32(32)); o = d * Int32(32)
    if blockIdx().x == Int32(1)
        tid = threadIdx().x - Int32(1); lane = tid % Int32(32); warp = tid ÷ Int32(32); nwp = blockDim().x ÷ Int32(32)
        for p in Int32(0):P-Int32(1)
            S = pp(dd, @f(dd, S), p)
            a1 = 0.0f0; a2 = 0.0f0; a3 = 0.0f0; a4 = 0.0f0
            # (B1's partition: its 256 threads' strides, the first 8 warps here, so the sums are B1's bit for bit)
            i = tid
            while tid < Int32(256) && i < Int32(4) * nc
                v = llget(S, i, sS); v2 = v * v; ln = i ÷ nc
                a1 += ln == Int32(0) ? v2 : 0.0f0; a2 += ln == Int32(1) ? v2 : 0.0f0
                a3 += ln == Int32(2) ? v2 : 0.0f0; a4 += ln == Int32(3) ? v2 : 0.0f0
                i += Int32(256)
            end
            a1 = wsum(a1, Val(32)); a2 = wsum(a2, Val(32)); a3 = wsum(a3, Val(32)); a4 = wsum(a4, Val(32))
            # 16 warps of 512 threads into the 8 warps' words of B1 (warp w and w + 8 summed by the reader's shuffles:
            # word 4 (w mod 8) + l holds both): the first 8 warps write, the others add theirs through shared memory
            sh = CuDeviceArray{Float32,1,AS.Shared}(xs, (64,))     # (xs is free before the phases)
            if warp >= Int32(8) && lane == Int32(0)
                sh[Int32(4)*(warp-Int32(8))+Int32(1)] = a1; sh[Int32(4)*(warp-Int32(8))+Int32(2)] = a2
                sh[Int32(4)*(warp-Int32(8))+Int32(3)] = a3; sh[Int32(4)*(warp-Int32(8))+Int32(4)] = a4
            end
            sync_threads()
            # then B1's sum of the 8 warps' in warp order (sh[32 ..]), 4 words a die (LSW)
            if warp < Int32(8) && lane == Int32(0)
                hi = nwp > Int32(8)
                sh[Int32(32)+Int32(4)*warp+Int32(1)] = a1 + (hi ? sh[Int32(4)*warp+Int32(1)] : 0.0f0)
                sh[Int32(32)+Int32(4)*warp+Int32(2)] = a2 + (hi ? sh[Int32(4)*warp+Int32(2)] : 0.0f0)
                sh[Int32(32)+Int32(4)*warp+Int32(3)] = a3 + (hi ? sh[Int32(4)*warp+Int32(3)] : 0.0f0)
                sh[Int32(32)+Int32(4)*warp+Int32(4)] = a4 + (hi ? sh[Int32(4)*warp+Int32(4)] : 0.0f0)
            end
            sync_threads()
            if tid < Int32(4)
                s = 0.0f0
                for w8 in Int32(0):Int32(7)
                    s += sh[Int32(32)+Int32(4)*w8+tid+Int32(1)]
                end
                llput(pp(dd, LSq, p), o + tid, sq, s)
            end
            sync_threads()
        end
    end
    for p in Int32(0):P-Int32(1)
        fwd_bcast!(dd, pp(dd, LSq, p), Int32(0), Int32(32), Int32(LSW), Int32(LSW), sq, Int32(4) * p)
    end
    DPS = @f(dd, DPS) + 8 * (q * Int32(1296))
    em = EM(3, DPS, sq; o1 = 0, nc = nc, lo = 0)
    # (one phase over all 4 nc as B1's, its HCSPLIT tasks)
    cphase8!(XF4(XF(2, @f(dd, S), Int32(4) * nc, sS; w = w, nc = nc), PS), Int32(4) * nc, xs, NORL8, NOSI, NOSF, NORED,
        (Job4{4}(m, m.rows, em, PS; xo = 0),), ctr8of(dd, bd, W == :ffn ? 5 : 0), P, b8lut(xs))
    DP = @f(dd, DP) + 8 * (q * n * RO)
    for p in Int32(0):P-Int32(1)
        if rhier(n)
            DSUM = pp(dd, @f(dd, DSUM) + 8 * (q * n * Int32(324) + Int32(324) * d), p)
            fwd_dp!(dd, pp(dd, DPS, p), DSUM, pp(dd, LSq, p), m.rows, RO, sq, Int32(16) + Int32(48) * p, true, false)
            w2t(n) ? fwd_red_hier_w2!(dd, DSUM, pp(dd, DP, p), pp(dd, @f(dd, DCB) + 8 * (q * n * RO), p), m.rows, RO, sq, Int32(32) + Int32(48) * p) :
                     fwd_red_hier!(dd, DSUM, pp(dd, DP, p), pp(dd, @f(dd, DCB) + 8 * (q * n * RO), p), m.rows, RO, sq, Int32(32) + Int32(48) * p)
        else
            fwd_dp!(dd, pp(dd, DPS, p), pp(dd, DP, p), pp(dd, LSq, p), m.rows, RO, sq, Int32(16) + Int32(48) * p)
        end
    end
end
# my rows of the hc down output of P positions (op_hct!), item = (position, row) over the CTAs' warp 0
function op8_hct!(dd, l, slot, which::Val{W}, seq0, rows::Int32, P::Int32) where {W}
    n = @f(dd, n); d = @f(dd, d); RO = @f(dd, RO); q = hcq(l, W)
    sqp = sqof(seq0, l, W == :ffn ? 11 : 4); sq = sqof(seq0, l, slot)
    tid = threadIdx().x - Int32(1); lane = tid % Int32(32)
    r0 = RO * d; nr = min(rows, r0 + RO) - r0
    (tid >= Int32(32) || nr <= Int32(0)) && return
    DP = @f(dd, DP) + 8 * (q * n * RO); T = @f(dd, T) + 8 * (q * Int32(324))
    it = blockIdx().x - Int32(1)
    while it < P * nr
        p = it ÷ nr; b = it - p * nr
        v = lane < ynp(n) ? (w2t(n) ? w2el(pp(dd, DP, p), lane * RO + b, sqp) : llget(pp(dd, DP, p), lane * RO + b, sqp)) : 0.0f0
        v = wsum(v, Val(32))
        r = r0 + b
        lane == Int32(0) && llput(pp(dd, T, p), r, sq, r < Int32(320) ? silu(v * 0.25f0) : v)
        it += gridDim().x
    end
end
@inline zx8(dd, q, sq) = W2Z ? XF4(XF(8, @f(dd, ZW) + 8 * (q * Int32(1280)), 2560, sq), @f(dd, PSTR)) :
                               XF4(XF(1, @f(dd, Z) + 8 * (q * Int32(2560)), 2560, sq), @f(dd, PSTR))
function op8_hcu!(dd, bd, l, slot, which::Val{W}, sS, seq0, xs, P::Int32) where {W}
    nc = @f(dd, nc); d = @f(dd, d); n = @f(dd, n); q = hcq(l, W); PS = @f(dd, PSTR)
    m = W == :attn ? @f(bd, hcA_up) : @f(bd, hcF_up)
    w = W == :attn ? @f(bd, hcA_w) : @f(bd, hcF_w)
    st = W == :ffn ? Int32(19) : Int32(17)
    rows = (W == :attn ? @f(bd, hcA_dj) : @f(bd, hcF_dj)).rows
    op8_hct!(dd, l, st, which, seq0, rows, P)
    RO = @f(dd, RO); T = @f(dd, T) + 8 * (q * Int32(324))
    for p in Int32(0):P-Int32(1)
        w2t(n) ? fwd_bcast_w2!(dd, pp(dd, T, p), pp(dd, tw(dd, q), p), Int32(0), RO, RO, sqof(seq0, l, st), Int32(64) + Int32(8) * p, false, rows - RO * (n - Int32(1))) :
                 fwd_bcast!(dd, pp(dd, T, p), Int32(0), RO, RO, rows - RO * (n - Int32(1)), sqof(seq0, l, st), Int32(64) + Int32(8) * p)
    end
    sq = sqof(seq0, l, slot); sqp = sqof(seq0, l, W == :ffn ? 11 : 4)
    LSq = @f(dd, LS) + 8 * (q * n * Int32(32))
    rl = Base.Cartesian.@ntuple 8 p -> (Int32(p - 1) < P ? hc_rln(pp(dd, LSq, Int32(p - 1)), n, sqp, EPS) : NORL)
    Z = @f(dd, Z) + 8 * (q * Int32(2560))
    em = EM(4, Z, sq; nc = nc, S = @f(dd, S), sS = sS, w = w, zoff = nc * d)
    cphase8!(XF4(w2t(n) ? XF(8, tw(dd, q), 320, sqof(seq0, l, st)) : XF(1, T, 320, sqof(seq0, l, st)), PS), Int32(320), xs, rl, NOSI, NOSF, NORED, (Job4{0}(m, m.rows, em, PS; pS = PS),),
        ctr8of(dd, bd, W == :ffn ? 7 : 2), P, b8lut(xs))
    for p in Int32(0):P-Int32(1)
        W2Z ? fwd_bcast_w2!(dd, pp(dd, Z, p), pp(dd, @f(dd, ZW) + 8 * (q * Int32(1280)), p), Int32(0), nc, nc, sq, Int32(16) * p) :
              fwd_bcast!(dd, pp(dd, Z, p), Int32(0), nc, nc, nc, sq, Int32(16) * p)
    end
end
function op8_proj_dn!(dd, bd, l, seq0, xs, P::Int32)
    par = l & Int32(1); nkh = @f(dd, nkh); PS = @f(dd, PSTR)
    sz = sqof(seq0, l, 5); sq = sqof(seq0, l, 6)
    qkv = @f(bd, qkv); gate = @f(bd, gate); ab = @f(bd, ab)
    cphase8!(zx8(dd, par, sz), Int32(2560), xs, NORL8, NOSI, NOSF, NORED,
        (Job4{0}(qkv, qkv.rows, EM(1, @f(dd, QKV) + 8 * (par * Int32(640) * nkh), sq), PS),
         Job4{0}(gate, gate.rows, EM(1, @f(dd, GATE) + 8 * (par * Int32(384) * nkh), sq), PS),
         Job4{0}(ab, ab.rows, EM(1, @f(dd, AB) + 8 * (par * Int32(6) * nkh), sq), PS)), ctr8of(dd, bd, 3), P, b8lut(xs))
end
function op8_proj_attn!(dd, bd, l, seq0, xs, P::Int32)
    par = l & Int32(1); PS = @f(dd, PSTR)
    sz = sqof(seq0, l, 5); sq = sqof(seq0, l, 6)
    wq = @f(bd, wq); wk = @f(bd, wk); wv = @f(bd, wv); ix = @f(bd, idx_x)
    cphase8!(zx8(dd, par, sz), Int32(2560), xs, NORL8, NOSI, NOSF, NORED,
        (Job4{0}(wq, wq.rows, EM(1, @f(dd, QG), sq), PS), Job4{0}(wk, wk.rows, EM(1, @f(dd, KK), sq), PS),
         Job4{0}(wv, wv.rows, EM(1, @f(dd, VV), sq), PS), Job4{0}(ix, ix.rows, EM(1, @f(dd, IX), sq), PS)), ctr8of(dd, bd, 3), P, b8lut(xs))
end
# DeltaNet of P positions in order per value head h (op4_dn! with B1's state copies: position p reads (t + p) mod SRN)
function op8_dn!(dd, bd, l, seq0, t::Int32, P::Int32, xs::SPtr)
    nkh = @f(dd, nkh); par = l & Int32(1)
    h = blockIdx().x - Int32(1)
    n = Int32(640) * nkh; cps = Int32(3) * nkh * Int32(16384)
    us = CuDeviceArray{Float32,1,AS.Shared}(xs, (384,))
    Q0 = @f(dd, QKV) + 8 * (par * Int32(640) * nkh)
    while h < Int32(3) * nkh
        sq_in = sqof(seq0, l, 6); sq = sqof(seq0, l, 7)
        cw = @f(bd, conv_w); st = @f(bd, conv); ST = @f(bd, dn)
        tid = threadIdx().x - Int32(1)
        kh = h % nkh
        for p in Int32(0):P-Int32(1)
            tp = t + p
            QKV = pp(dd, Q0, p)
            GATE = pp(dd, @f(dd, GATE) + 8 * (par * Int32(384) * nkh), p)
            AB = pp(dd, @f(dd, AB) + 8 * (par * Int32(6) * nkh), p)
            c = tid
            @inbounds while c < Int32(384)
                ch = c < Int32(128) ? Int32(128) * kh + c : c < Int32(256) ? Int32(128) * nkh + Int32(128) * kh + (c - Int32(128)) : Int32(256) * nkh + Int32(128) * h + (c - Int32(256))
                xc = llget(QKV, ch, sq_in)
                # the 3 previous inputs: this round's from their LL words (the ring slot of a q/k channel is another
                # CTA's), earlier rounds' from the ring
                x1 = p >= Int32(1) ? llget(pp(dd, Q0, p - Int32(1)), ch, sq_in) : ldf(st, ((tp - Int32(1)) & SRM) * n + ch)
                x2 = p >= Int32(2) ? llget(pp(dd, Q0, p - Int32(2)), ch, sq_in) : ldf(st, ((tp - Int32(2)) & SRM) * n + ch)
                x3 = p >= Int32(3) ? llget(pp(dd, Q0, p - Int32(3)), ch, sq_in) : ldf(st, ((tp - Int32(3)) & SRM) * n + ch)
                y = ldf(cw, Int32(4) * ch) * x3
                y = fma(ldf(cw, Int32(4) * ch + Int32(1)), x2, y)
                y = fma(ldf(cw, Int32(4) * ch + Int32(2)), x1, y)
                y = fma(ldf(cw, Int32(4) * ch + Int32(3)), xc, y)
                us[c+1] = silu(y)
                (c >= Int32(256) || h < nkh) && stf(st, (tp & SRM) * n + ch, xc)
                c += blockDim().x
            end
            sync_threads()
            dn_core2!(ST + 4 * ((tp & SRM) * cps), ST + 4 * (((tp + Int32(1)) & SRM) * cps), us, xs + 4 * 384, AB, GATE,
                @f(bd, ab_bias), @f(bd, ssm_a), @f(bd, ssm_norm), h, nkh, sq_in, pp(dd, @f(dd, ON) + 8 * (par * Int32(384) * nkh), p), sq)
            sync_threads()
            threadfence_block()
        end
        h += gridDim().x
    end
end
# attention of P positions with the KV cache copied on the KV head's dies (KVCOPY): op_aprep_kc!, op_aprep2_kc!,
# op_attn_kc! of each position (the indexer and the cache rows in position order)
function op8_aprep_kc!(dd, bd, l, seq0, t::Int32, P::Int32)
    QH = @f(dd, QH); nkvl = @f(dd, nkvl); DH = @f(dd, DH); s = @f(dd, s); IW = @f(dd, IW); nh = @f(dd, n); nq = @f(dd, nq); GS = @f(dd, GS)
    sq_in = sqof(seq0, l, 6); sq = sqof(seq0, l, 7)
    hb = blockIdx().x - Int32(1); i = threadIdx().x - Int32(1)
    @inbounds while hb < QH + Int32(2)
        for p in Int32(0):P-Int32(1)
            QG = pp(dd, @f(dd, QG), p)
            if hb < QH
                x = i < DH ? llget(QG, Int32(2) * DH * hb + i, sq_in) : 0.0f0
                ss = bsum(x * x)
                hg = nh == Int32(1) ? hb : Int32(12) * @f(dd, g) + hb; ow = hg ÷ nq; hl = hg - ow * nq
                QF = onpeer(dd, pp(dd, @f(dd, QF), p), ow)
                if i < DH
                    llput(QF, Int32(512) * hl + DH * s + i, sq, x * ldf(@f(bd, q_norm), i))
                    llput(QF, Int32(512) * hl + Int32(256) + DH * s + i, sq, llget(QG, Int32(2) * DH * hb + DH + i, sq_in))
                end
                i == Int32(0) && llput(onpeer(dd, pp(dd, @f(dd, QN), p), ow), GS * hl + s, sq, ss)
            elseif hb == QH
                for kv in Int32(0):nkvl-Int32(1)
                    x = i < DH ? llget(pp(dd, @f(dd, KK), p), DH * kv + i, sq_in) : 0.0f0
                    ss = bsum(x * x)
                    if i < DH
                        gput!(dd, pp(dd, @f(dd, KF), p), Int32(256) * kv + DH * s + i, sq, x * ldf(@f(bd, k_norm), i))
                        gput!(dd, pp(dd, @f(dd, VF), p), Int32(256) * kv + DH * s + i, sq, llget(pp(dd, @f(dd, VV), p), DH * kv + i, sq_in))
                    end
                    i == Int32(0) && gput!(dd, pp(dd, @f(dd, KN), p), GS * kv + s, sq, ss)
                end
            else
                tp = t + p
                ring = @f(bd, ring)
                xr = i < IW ? llget(pp(dd, @f(dd, IX), p), Int32(512) ÷ nh + i, sq_in) : 0.0f0
                i < IW && stf(ring, (tp & SRM) * IW + i, xr)
                x = i < IW ? ipool8(dd, ring, IW, i, t, p, sq_in) : 0.0f0
                ss = bsum(x * x)
                i == Int32(0) && bput!(dd, pp(dd, @f(dd, IXN), p), @f(dd, d), sq, ss)
            end
            sync_threads()
        end
        hb += gridDim().x
    end
end
function op8_aprep2_kc!(dd, bd, l, seq0, t::Int32, P::Int32, xs::SPtr)
    nq = @f(dd, nq); nkvl = @f(dd, nkvl); GS = @f(dd, GS)
    sq_in = sqof(seq0, l, 7); sq = sqof(seq0, l, 18)
    hb = blockIdx().x - Int32(1); d = threadIdx().x - Int32(1)
    sh = CuDeviceArray{Float32,1,AS.Shared}(xs, (256,))
    @inbounds while hb < (nkvl + nq) * P
        p = hb ÷ (nkvl + nq); hh = hb - p * (nkvl + nq); tp = t + p
        cs = @f(dd, rope) + 4 * (Int32(64) * tp)
        if hh < nkvl
            kv = hh
            ss = 0.0f0
            for e in Int32(0):GS-Int32(1)
                ss += llget(pp(dd, @f(dd, KN), p), GS * kv + e, sq_in)
            end
            x = d < Int32(256) ? llget(pp(dd, @f(dd, KF), p), Int32(256) * kv + d, sq_in) / sqrt(ss / 256.0f0 + EPS) : 0.0f0
            y = rope256(x, d, cs, sh)
            row = tp * nkvl + kv
            if d < Int32(256)
                kvq32(y, @f(bd, kq), row * Int32(256) + d, @f(bd, ks), row * Int32(8) + d ÷ Int32(32))
                kvq32(llget(pp(dd, @f(dd, VF), p), Int32(256) * kv + d, sq_in), @f(bd, vq), row * Int32(256) + d, @f(bd, vs), row * Int32(8) + d ÷ Int32(32))
            end
            threadfence()
            sync_threads()
            d == Int32(0) && llput(pp(dd, @f(dd, KVR), p), kv, sq, 1.0f0)
        else
            hl = hh - nkvl
            ss = 0.0f0
            for e in Int32(0):GS-Int32(1)
                ss += llget(pp(dd, @f(dd, QN), p), GS * hl + e, sq_in)
            end
            x = d < Int32(256) ? llget(pp(dd, @f(dd, QF), p), Int32(512) * hl + d, sq_in) / sqrt(ss / 256.0f0 + EPS) : 0.0f0
            y = rope256(x, d, cs, sh)
            d < Int32(256) && llput(pp(dd, @f(dd, QO), p), Int32(256) * hl + d, sq, y)
        end
        sync_threads()
        hb += gridDim().x
    end
end
# attention of (head h, position p) over keys 0 .. t + p (the round's earlier positions' rows: KVR of each); then
# each position's indexer pool (every 4th position)
function op8_attn_kc!(dd, bd, l, seq0, t::Int32, xs::SPtr, P::Int32)
    nq = @f(dd, nq); nkvl = @f(dd, nkvl)
    sq_in = sqof(seq0, l, 18); sq = sqof(seq0, l, 8); sqg = sqof(seq0, l, 7)
    it = blockIdx().x - Int32(1)
    # (B1's partition of the keys over its 8 warps, and its combine of them: the first 8 warps here; B1's bits)
    tid = threadIdx().x - Int32(1); lane = tid % Int32(32); w = tid ÷ Int32(32); nw = Int32(8)
    msh = CuStaticSharedArray(Float32, 16); dsh = CuStaticSharedArray(Float32, 16)
    osh = CuDeviceArray{Float32,2,AS.Shared}(xs, (256, 16))
    kq = @f(bd, kq); ks = @f(bd, ks); vq = @f(bd, vq); vs = @f(bd, vs)
    hg0 = @f(dd, n) == Int32(1) ? Int32(0) : nq * @f(dd, d)
    @inbounds while it < (nq + Int32(1)) * P
        p = it ÷ (nq + Int32(1)); h = it - p * (nq + Int32(1)); tp = t + p
        if h < nq
            kv = nkvl == Int32(1) ? Int32(0) : (hg0 + h) ÷ Int32(12)
            for q in Int32(0):p
                llget(pp(dd, @f(dd, KVR), q), kv, sq_in)
            end
            q8 = ldq8(pp(dd, @f(dd, QO), p), h, lane, sq_in)
            m = -1.0f30; dsum = 0.0f0
            acc = ntuple(i -> 0.0f0, Val(8))
            j = w < nw ? w : tp + Int32(1)
            while j <= tp
                row = j * nkvl + kv
                s = 0.0f0
                for i in 1:8
                    dd_ = Int32(8) * lane + Int32(i - 1)
                    s = fma(q8[i], kvd(kq, row * Int32(256) + dd_, ks, row * Int32(8) + dd_ ÷ Int32(32)), s)
                end
                s = wsum(s, Val(32)) * 0.0625f0
                mn = max(m, s)
                sc = exp(m - mn); pr = exp(s - mn)
                dsum = fma(dsum, sc, pr)
                acc = vacc(acc, pr, sc, vq, vs, row, lane)
                m = mn
                j += nw
            end
            if lane == Int32(0)
                msh[w+1] = m; dsh[w+1] = dsum
            end
            for i in 1:8
                osh[Int32(8)*lane+Int32(i), w+1] = acc[i]
            end
            sync_threads()
            if tid < Int32(256)
                M = msh[1]
                for ww in Int32(2):nw
                    M = max(M, msh[ww])
                end
                o = 0.0f0; D = 0.0f0
                for ww in Int32(1):nw
                    e = exp(msh[ww] - M)
                    o = fma(osh[tid+1, ww], e, o); D = fma(dsh[ww], e, D)
                end
                llput(pp(dd, @f(dd, AO), p), Int32(256) * h + tid, sq, o / D * sigm(llget(pp(dd, @f(dd, QF), p), Int32(512) * h + Int32(256) + tid, sqg)))
            end
        elseif (tp & Int32(3)) == Int32(3)
            ipool_fin8!(dd, bd, l, seq0, tp, p, xs + 4 * 4096)
        end
        sync_threads()
        it += gridDim().x
    end
end
# the pooled indexer key of dim i at position t + p: the ring's 4 slots in order, this round's positions (t ..
# t + p) from their LL words (another CTA wrote their ring slots), earlier ones from the ring
@inline function ipool8(dd, ring::FPtr, IW::Int32, i::Int32, t::Int32, p::Int32, sq::UInt32)
    tp = t + p; nh = @f(dd, n)
    r = Base.Cartesian.@ntuple 4 k -> begin
        tq_k = tp - ((tp - Int32(k - 1)) & Int32(3))
        tq_k >= t ? llget(pp(dd, @f(dd, IX), tq_k - t), Int32(512) ÷ nh + i, sq) : ldf(ring, (tq_k & SRM) * IW + i)
    end
    (((r[1] + r[2]) + r[3]) + r[4]) * 0.25f0
end
function ipool_fin8!(dd, bd, l, seq0, tp::Int32, p::Int32, scr::SPtr)
    nh = @f(dd, n); IW = @f(dd, IW); d = @f(dd, d); tid = threadIdx().x - Int32(1)
    sq_n = sqof(seq0, l, 7); sqg = sqof(seq0, l, 6)
    sh = CuDeviceArray{Float32,1,AS.Shared}(scr, (256,))
    @inbounds begin
        x = tid < IW ? ipool8(dd, @f(bd, ring), IW, tid, tp - p, p, sqg) : 0.0f0
        ss = 0.0f0
        for e in Int32(0):nh-Int32(1)
            ss += llget(pp(dd, @f(dd, IXN), p), e, sq_n)
        end
        y = tid < IW ? x / sqrt(ss / 128.0f0 + EPS) * ldf(@f(bd, idx_kn), tid) : 0.0f0
        tid < Int32(256) && (sh[tid+1] = y)
        sync_threads()
        wr = nh == Int32(1) ? Int32(32) : Int32(64) ÷ nh
        if (nh == Int32(1) || d < nh ÷ Int32(2)) && tid < Int32(2) * wr
            cs3 = @f(dd, rope) + 4 * (Int32(64) * max(tp - Int32(3), Int32(0)))
            pq = (nh == Int32(1) ? Int32(0) : wr * d) + tid % wr
            cc = ldf(cs3, pq); sn = ldf(cs3, Int32(32) + pq)
            y = tid < wr ? fma(-sh[tid+wr+1], sn, sh[tid+1] * cc) : fma(sh[tid-wr+1], sn, sh[tid+1] * cc)
        end
        tid < IW && unsafe_store!(reinterpret(Core.LLVMPtr{Float16,1}, @f(bd, pool)), Float16(y), (tp ÷ Int32(4)) * IW + tid + Int32(1), Val(2))
    end
end
function op8_out!(dd, bd, l, seq0, xs, P::Int32)
    nc = @f(dd, nc); n = @f(dd, n); nkh = @f(dd, nkh); nq = @f(dd, nq); par = l & Int32(1); PS = @f(dd, PSTR)
    attn = @f(bd, attn) == Int32(1)
    sq = sqof(seq0, l, 9)
    xf = attn ? XF4(XF(1, @f(dd, AO), 256nq, sqof(seq0, l, 8)), PS) : XF4(XF(1, @f(dd, ON) + 8 * (par * Int32(384) * nkh), 384nkh, sqof(seq0, l, 7)), PS)
    m = attn ? @f(bd, wo) : @f(bd, ssm_out)
    YPS = @f(dd, YPS) + 8 * (par * Int32(2560))
    cphase8!(xf, m.K, xs, NORL8, NOSI, NOSF, NORED, (Job4{0}(m, m.rows, EM(1, YPS, sq), PS),), ctr8of(dd, bd, 4), P, b8lut(xs))
    for p in Int32(0):P-Int32(1)
        if w2r(n)
            fwd_red_hier_w2!(dd, pp(dd, YPS, p), pp(dd, @f(dd, YW) + 8 * (par * ((ynp(n) * nc) ÷ Int32(2))), p), pp(dd, @f(dd, YCB) + 8 * (par * n * nc), p),
                Int32(2560), nc, sq, Int32(40) * p)
        elseif rhier(n)
            fwd_red_hier!(dd, pp(dd, YPS, p), pp(dd, @f(dd, YP) + 8 * (par * ynp(n) * nc), p), pp(dd, @f(dd, YCB) + 8 * (par * n * nc), p),
                Int32(2560), nc, sq, Int32(40) * p)
        else
            fwd_owners!(dd, pp(dd, YPS, p), pp(dd, @f(dd, YP) + 8 * (par * n * nc), p), Int32(2560), Int32(1), Int32(2560), nc, sq, Int32(40) * p)
        end
    end
end
# S[i] += 2 sigmoid(inj_l / 4) (sum of np slots of nc) of each position; the DFlash2 taps (slot (t + p) & 7)
function op8_hcw!(dd, bd, l, slot, src::U64P, np::Int32, ssrc::UInt32, q::Int32, sinj::UInt32, sS::UInt32, seq0, last::Bool, t::Int32, P::Int32,
                  w2::Bool = false)
    nc = @f(dd, nc); par = l & Int32(1); n4 = Int32(4) * nc
    sq = sqof(seq0, l, slot)
    tap = last ? @f(bd, tap) : NOF
    i = gtid()
    @inbounds while i < P * n4
        p = i ÷ n4; k = i - p * n4
        ln = k ÷ nc; c = k % nc
        sp = pp(dd, src, p)
        y = 0.0f0
        for r in Int32(0):np-Int32(1)
            y += w2 ? w2el(sp, (par * np + r) * nc + c, ssrc) : llget(sp, (par * np + r) * nc + c, ssrc)
        end
        S = pp(dd, @f(dd, S), p)
        inj = w2t(@f(dd, n)) ? w2el(pp(dd, tw(dd, q), p), Int32(320) + ln, sinj) : llget(pp(dd, @f(dd, T) + 8 * (q * Int32(324) + Int32(320)), p), ln, sinj)
        v = fma(2.0f0 * sigm(0.25f0 * inj), y, llget(S, k, sS))
        llput(S, k, sq, v)
        tap != NOF && stf(tap, ((t + p) & Int32(7)) * Int32(TAPSLOT) + nc * @f(dd, d) + ln * Int32(2560) + c, v)
        i += gstride()
    end
end
function op8_rtp!(dd, bd, l, seq0, xs, P::Int32)
    par = l & Int32(1); F = @f(dd, F); PS = @f(dd, PSTR)
    sq = sqof(seq0, l, 13); sqe = sqof(seq0, l, 14)
    RP = @f(dd, RP) + 8 * (par * Int32(RPW)); GU = @f(dd, GU) + 8 * (par * Int32(22) * F)
    rf = @f(bd, router_f); shg = @f(bd, sh_gate); shu = @f(bd, sh_up)
    if ROUTERPAIR                      # (as op_rtp!: my board's half, to me and my partner)
        d = @f(dd, d); rf = @f(bd, router); rsh = @f(bd, router_sh); RPq = onpeer(dd, RP, d ⊻ Int32(1))
        cphase8!(zx8(dd, Int32(2) + par, sqof(seq0, l, 12)), Int32(2560), xs, NORL8, NOSI, NOSF, NORED,
            (Job4{0}(rf, Int32(256), EM(1, RP, sq; d2 = RPq, o1 = Int32(256) * (d & Int32(1))), PS),
             Job4{0}(rsh, min(rsh.rows, Int32(1)), EM(1, RP, sq; d2 = RPq, o1 = 512), PS),
             Job4{0}(shg, F, EM(1, GU, sqe; o1 = 20F), PS), Job4{0}(shu, F, EM(1, GU, sqe; o1 = 21F), PS)),
            ctr8of(dd, bd, 8), P, b8lut(xs))
        return
    end
    cphase8!(zx8(dd, Int32(2) + par, sqof(seq0, l, 12)), Int32(2560), xs, NORL8, NOSI, NOSF, NORED,
        (Job4{0}(rf, rf.rows, EM(1, RP, sq), PS), Job4{0}(shg, F, EM(1, GU, sqe; o1 = 20F), PS), Job4{0}(shu, F, EM(1, GU, sqe; o1 = 21F), PS)),
        ctr8of(dd, bd, 8), P, b8lut(xs))
end
# each position's top 10 (warp p), the round's union U of experts (first appearance, positions in order) with each
# position's weight and slot per expert (B4's, and the slot: the position's own order 0..9, -1 if not its expert);
# CTA 0 publishes them in UIDX (UW0, US0) and each position's shared expert h; then the gate/up of the U experts for
# P positions, h into each selecting position's slot
function op8_exp!(dd, bd, l, seq0, xs, P::Int32)
    F = @f(dd, F); n = @f(dd, n); d = @f(dd, d); par = l & Int32(1); PS = @f(dd, PSTR); PM = Int32(P8MAX)
    sq = sqof(seq0, l, 14)
    ix4, wv4, uix, uw, slt, un, _, rix, rsl = b8tabs(xs)
    tid = threadIdx().x - Int32(1); warp = tid ÷ Int32(32); nt = blockDim().x
    if warp < P
        router_top10w!(pointer(ix4) + 4 * Int32(16) * warp, pointer(wv4) + 4 * Int32(16) * warp,
            pp(dd, @f(dd, RP) + 8 * (par * Int32(RPW)), warp), Int32(1), sqof(seq0, l, 13))
    end
    sync_threads()
    @inbounds if tid == Int32(0)
        U = Int32(0)
        for p in Int32(0):P-Int32(1), k in Int32(0):Int32(9)
            e = ix4[Int32(16)*p+k+Int32(1)]
            f = Int32(-1)
            for u in Int32(0):U-Int32(1)
                uix[u+1] == e && (f = u)
            end
            if f < Int32(0)
                uix[U+1] = e
                for q in Int32(0):PM-Int32(1)
                    uw[PM*U+q+Int32(1)] = 0.0f0; slt[PM*U+q+Int32(1)] = reinterpret(Float32, Int32(-1))
                end
                f = U; U += Int32(1)
            end
            uw[PM*f+p+Int32(1)] = wv4[Int32(16)*p+k+Int32(1)]
            slt[PM*f+p+Int32(1)] = reinterpret(Float32, k)
        end
        for q in Int32(0):PM-Int32(1)
            uw[PM*U+q+Int32(1)] = q < P ? wv4[Int32(16)*q+Int32(11)] : 0.0f0
        end
        un[1] = U
    end
    sync_threads()
    U = un[1]
    H = @f(dd, H) + 8 * (par * Int32(7040))
    if blockIdx().x == Int32(1)
        UIDX = @f(dd, UIDX) + 8 * (par * Int32(UIDXW))
        tid == Int32(0) && llput(UIDX, Int32(0), sq, reinterpret(Float32, U))
        i = tid
        @inbounds while i < PM * (U + Int32(1))
            i < U && llput(UIDX, Int32(1) + i, sq, reinterpret(Float32, uix[i+1]))
            llput(UIDX, Int32(UW0) + i, sq, uw[i+1])
            i < PM * U && llput(UIDX, Int32(US0) + i, sq, slt[i+1])
            i += nt
        end
        i = tid
        while i < P * F                # the shared expert's h of each position (its gate/up ran in op8_rtp!)
            p = i ÷ F; j = i - p * F
            GU = pp(dd, @f(dd, GU) + 8 * (par * Int32(22) * F), p)
            llput(pp(dd, H, p), Int32(6400) + F * d + j, sq, silu(llget(GU, Int32(20) * F + j, sq)) * llget(GU, Int32(21) * F + j, sq))
            i += blockDim().x
        end
    end
    gu = @f(bd, gu); xf = zx8(dd, Int32(2) + par, sqof(seq0, l, 12)); ctr = ctr8of(dd, bd, 10)
    em = EM(5, H, sq; o1 = F * d, nc = F, zoff = 640)
    np = npr8(Int32(2560))             # (the positions a run at K = 2560)
    if P <= np
        cphase8!(xf, Int32(2560), xs, NORL8, pointer(uix), pointer(slt), NORED,
            (Job4{1}(gu, Int32(2) * F * U, em, PS; rpe = Int32(2) * F),), ctr, P, b8lut(xs))
    else
        # runs of np positions, each over the union experts its own positions selected (rix, union order; their slot
        # rows rsl), so an expert only another run selected is not read for it
        XS = XS8(Int32(2560)); p0 = Int32(0); run = Int32(0)
        @inbounds while p0 < P
            ns = min(np, P - p0)
            if tid == Int32(0)
                nr = Int32(0)
                for u in Int32(0):U-Int32(1)
                    sel = false
                    for q in p0:p0+ns-Int32(1)
                        sel |= reinterpret(Int32, slt[PM*u+q+Int32(1)]) >= Int32(0)
                    end
                    if sel
                        rix[nr+1] = uix[u+1]
                        for q in Int32(0):PM-Int32(1)
                            rsl[PM*nr+q+Int32(1)] = slt[PM*u+q+Int32(1)]
                        end
                        nr += Int32(1)
                    end
                end
                un[2] = nr
            end
            xload8!(xs, XS, xf, Int32(2560), NORL8, p0, ns)
            sync_threads()
            grun8!((Job4{1}(gu, Int32(2) * F * un[2], em, PS; rpe = Int32(2) * F),), ctr + 4 * (Int32(64) * run), pointer(rix), pointer(rsl),
                NORED, xs, XS, Int32(2560), b8lut(xs), NORL8, p0, ns, P)
            sync_threads()
            p0 += ns; run += Int32(1)
        end
    end
    HW = @f(dd, HW) + 8 * (par * Int32(3520))
    nch = (F + Int32(31)) ÷ Int32(32) * (n ÷ Int32(2) + Int32(1))
    for p in Int32(0):P-Int32(1), es in Int32(0):Int32(10)
        W2H ? fwd_bcast_w2!(dd, pp(dd, H, p), pp(dd, HW, p), Int32(640) * es, F, F, sq, (Int32(11) * p + es) * nch, dfast(n)) :
              fwd_bcast!(dd, pp(dd, H, p), Int32(640) * es, F, F, F, sq, (Int32(11) * p + es) * nch)
    end
end
# a B8 phase: positions in runs of npr8(Kx), each its x (B1's layout) and its groups (counter ctr + 64 words a run)
@inline function cphase8!(xf::XF4, Kx::Int32, xs::SPtr, rl::RL8, idx::SIP, wsc::SPtr, red::Red, js::Tuple, ctr::IPtr, P::Int32, lut::SPtr)
    XS = XS8(Kx); np = npr8(Kx)
    p0 = Int32(0); run = Int32(0)
    @inbounds while p0 < P
        ns = min(np, P - p0)
        xload8!(xs, XS, xf, Kx, rl, p0, ns)
        sync_threads()
        grun8!(js, ctr + 4 * (Int32(64) * run), idx, wsc, red, xs, XS, Kx, lut, rl, p0, ns, P)
        sync_threads()
        p0 += ns; run += Int32(1)
    end
end
# routed (the U experts, 4 a pass: x = 4 x 640) and shared down of P positions, my nc output rows: (row group, expert)
# tasks, each 32-row block summed by the task completing it over all U + 1 slots (B4), into OUTP of each position
function op8_down!(dd, bd, l, seq0, xs, P::Int32)
    nc = @f(dd, nc); par = l & Int32(1); PS = @f(dd, PSTR); n = @f(dd, n); d = @f(dd, d)
    sqi = sqof(seq0, l, 14); sq = sqof(seq0, l, 15)
    df = dfast(n)                      # (DFAST: as op_down!)
    df && d >= n ÷ Int32(2) && return
    _, _, uix, uw, slt, un, ord = b8tabs(xs)
    tid = threadIdx().x - Int32(1); nt = blockDim().x; PM = Int32(P8MAX)
    UIDX = @f(dd, UIDX) + 8 * (par * Int32(UIDXW))
    tid == Int32(0) && (un[1] = reinterpret(Int32, llget(UIDX, Int32(0), sqi)))
    sync_threads()
    U = un[1]
    i = tid
    @inbounds while i < PM * (U + Int32(1))
        i < U && (uix[i+1] = reinterpret(Int32, llget(UIDX, Int32(1) + i, sqi)))
        uw[i+1] = llget(UIDX, Int32(UW0) + i, sqi)
        i < PM * U && (slt[i+1] = llget(UIDX, Int32(US0) + i, sqi))
        i += nt
    end
    sync_threads()
    # each position's experts' union slots in its rank order (the down's sum in B1's order)
    i = tid
    @inbounds while i < PM * U
        u = i ÷ PM; p = i - u * PM; ks = reinterpret(Int32, slt[i+1])
        ks >= Int32(0) && (ord[Int32(10)*p+ks+Int32(1)] = u)
        i += nt
    end
    sync_threads()
    OUTP = @f(dd, OUTP) + 8 * (par * nc)
    em = df ? EM(2, OUTP, sq; d2 = reinterpret(U64P, peerb(dd, d + n ÷ Int32(2)) + (reinterpret(UInt64, OUTP) - reinterpret(UInt64, @f(dd, REG)))),
                 o1 = 0, lo = 0, plo = nc, nc = nc) : EM(1, OUTP, sq)
    nr = df ? Int32(2) * nc : nc
    dn = @f(bd, down); shd = @f(bd, sh_down)
    # (every (row group, expert) task runs once a run of positions, routed and shared alike: runs of RD = 4 x slots)
    RD = Int32(4); nrun = (P + RD - Int32(1)) ÷ RD
    red = Red(@f(dd, RED), @f(dd, CNT), (U * dn.L + shd.L) * nrun, U + Int32(1), nr, reinterpret(Core.LLVMPtr{Int32,3}, pointer(ord)))
    H = @f(dd, H) + 8 * (par * Int32(7040)); HW = @f(dd, HW) + 8 * (par * Int32(3520))
    # union experts 3 a chunk (x 3 x 640 a position: 4 positions' slots fit), positions in runs of RD: the (chunk, run)
    # units run side by side, not one phase each: CTA b starts at unit b mod NU and takes its groups, one a warp, nw at
    # a time from the unit's counter (the next batch fetched while the current one runs), then the next unit's, until
    # it has seen every unit; it loads a unit's x only when it gets groups there (13 phases a block for U = 40 were
    # 15.5 of a round's 93 ms)
    C = (U + Int32(2)) ÷ Int32(3); NU = C * nrun
    vb = CuStaticSharedArray(Int32, 2)
    warp = tid ÷ Int32(32); nw = blockDim().x ÷ Int32(32)
    w = (blockIdx().x - Int32(1)) % NU
    for _ in Int32(1):NU
        r = w ÷ C; c = w - r * C; p0 = RD * r; ns = min(RD, P - p0)
        ne = min(Int32(3), U - Int32(3) * c); Kx = ne * Int32(640); XS = XS8(Kx)
        js = (Job4{3}(dn, nr, em, PS; ns = ne, eo = Int32(3) * c, rpe = dn.rows ÷ Int32(512)),)
        G = jobs4_groups(js); ctr = ctr8of(dd, bd, Int32(192) + Int32(32) * r + c); idx = pointer(uix) + 4 * (Int32(3) * c)
        tid == Int32(0) && (vb[1] = UA.add!(ctr, nw, UA.monotonic, UA.device))
        sync_threads()
        v = vb[1]; s = Int32(2); loaded = false
        while v < G
            if !loaded
                xload8d!(xs, XS, dd, H, HW, pointer(slt), Int32(3) * c, Kx, p0, ns, sqi)
                sync_threads(); loaded = true
            end
            tid == Int32(0) && (vb[s] = UA.add!(ctr, nw, UA.monotonic, UA.device))
            g = v + warp
            g < G && run8_one!(js, g, idx, pointer(uw), red, xs, XS, Kx, b8lut(xs), NORL8, p0, ns, P)
            sync_threads()
            v = vb[s]; s = Int32(3) - s
        end
        # (every CTA's one fetch past the end: the last of them zeroes the counter)
        tid == Int32(0) && v == nw * ((G + nw - Int32(1)) ÷ nw + gridDim().x - Int32(1)) && (UA.store!(ctr, Int32(0), UA.monotonic, UA.device); threadfence())
        sync_threads()                 # (every thread has read v before the next unit's fetch rewrites vb)
        w = w + Int32(1) == NU ? Int32(0) : w + Int32(1)
    end
    xs8 = W2H ? XF4(XF(8, HW + 8 * Int32(3200), 640, sqi), PS) : XF4(XF(1, H + 8 * Int32(6400), 640, sqi), PS)
    cphase8!(xs8, Int32(640), xs, NORL8, NOSI, pointer(uw), red, (Job4{3}(shd, nr, em, PS; eo = U, ns = 1, rpe = 0),), ctr8of(dd, bd, 30), P, b8lut(xs))
end
# the per-round program of one die: P <= 4 positions t .. t + P - 1
function k_token8!(dd::Core.LLVMPtr{DD,1}, bds::Core.LLVMPtr{BD,1}, nblk::Int32, seq0::UInt32, t::Int32, P::Int32, xsmax::Int32,
                   tim::Core.LLVMPtr{UInt64,1})
    xs = pointer(CuDynamicSharedArray(Float32, (Int(xsmax),)))
    threadIdx().x <= Int32(16) && unsafe_store!(b8lut(xs), kv4val(threadIdx().x - Int32(1)), threadIdx().x, Val(4))
    sync_threads()
    for bi in Int32(1):nblk
        bd = bds + (bi - Int32(1)) * Int32(sizeof(BD))
        l = @f(bd, l)
        sS = l == Int32(0) ? sqof(seq0, 0, 0) : sqof(seq0, l - Int32(1), 16)
        k0 = Int32(16) * bi
        stamp4!(tim, k0)
        l == Int32(0) && op8_sinit!(dd, seq0, P)
        if @f(bd, ple) == Int32(1)
            op8_ple!(dd, bd, seq0, t, xs, P); sync_threads()
            sS = sqof(seq0, 1, 3)
        end
        stamp4!(tim, k0 + Int32(1))
        op8_hcd!(dd, bd, l, 4, Val(:attn), sS, seq0, xs, P); sync_threads(); stamp4!(tim, k0 + Int32(2))
        op8_hcu!(dd, bd, l, 5, Val(:attn), sS, seq0, xs, P); sync_threads(); stamp4!(tim, k0 + Int32(3))
        @f(bd, ple) == Int32(2) && continue          # the final collapse: hz = Z[0] of block 48
        if @f(bd, attn) == Int32(1)
            op8_proj_attn!(dd, bd, l, seq0, xs, P); sync_threads(); stamp4!(tim, k0 + Int32(4))
            op8_aprep_kc!(dd, bd, l, seq0, t, P); sync_threads()
            op8_aprep2_kc!(dd, bd, l, seq0, t, P, xs); sync_threads(); stamp4!(tim, k0 + Int32(5))
            op8_attn_kc!(dd, bd, l, seq0, t, xs, P); sync_threads(); stamp4!(tim, k0 + Int32(6))
        else
            op8_proj_dn!(dd, bd, l, seq0, xs, P); sync_threads(); stamp4!(tim, k0 + Int32(4))
            op8_dn!(dd, bd, l, seq0, t, P, xs); sync_threads(); stamp4!(tim, k0 + Int32(5)); stamp4!(tim, k0 + Int32(6))
        end
        op8_out!(dd, bd, l, seq0, xs, P); sync_threads(); stamp4!(tim, k0 + Int32(7))
        n = @f(dd, n)
        op8_hcw!(dd, bd, l, 10, w2r(n) ? @f(dd, YW) : @f(dd, YP), ynp(n), sqof(seq0, l, 9), l & Int32(1), sqof(seq0, l, 17), sS, seq0, false, t, P, w2r(n))
        sync_threads()
        stamp4!(tim, k0 + Int32(8))
        sA = sqof(seq0, l, 10)
        op8_hcd!(dd, bd, l, 11, Val(:ffn), sA, seq0, xs, P); sync_threads(); stamp4!(tim, k0 + Int32(9))
        op8_hcu!(dd, bd, l, 12, Val(:ffn), sA, seq0, xs, P); sync_threads(); stamp4!(tim, k0 + Int32(10))
        op8_rtp!(dd, bd, l, seq0, xs, P); sync_threads(); stamp4!(tim, k0 + Int32(11))
        op8_exp!(dd, bd, l, seq0, xs, P); sync_threads(); stamp4!(tim, k0 + Int32(12))
        op8_down!(dd, bd, l, seq0, xs, P); sync_threads(); stamp4!(tim, k0 + Int32(13))
        op8_hcw!(dd, bd, l, 16, @f(dd, OUTP), Int32(1), sqof(seq0, l, 15), Int32(2) + (l & Int32(1)), sqof(seq0, l, 19), sA, seq0, true, t, P)
        sync_threads(); stamp4!(tim, k0 + Int32(14))
    end
    return
end
const B8OPS = ["start/ple", "hc_down(attn)", "hc_up(attn)", "proj", "aprep/deltanet", "attn", "out_proj", "hc_write(attn)",
    "hc_down(ffn)", "hc_up(ffn)", "router", "experts", "experts_down", "hc_write(ffn)"]
const KTOKEN8_TT = Tuple{Core.LLVMPtr{DD,1},Core.LLVMPtr{BD,1},Int32,UInt32,Int32,Int32,Int32,Core.LLVMPtr{UInt64,1}}
# one B4 phase of matrix m on 4 positions of x (LL, seq 1; position stride K) into y (LL, stride rows), as k_token4! runs it
function k_phase4bench!(m::MatI, rows::Int32, x::U64P, y::U64P, xsmax::Int32, ctr::IPtr)
    xs = pointer(CuDynamicSharedArray(Float32, (Int(xsmax),)))
    cphase4!(XF4(XF(1, x, m.K, UInt32(1)), m.K), m.K, xs, NORL4, NOSI, NOSF, NORED, (Job4{0}(m, rows, EM(1, y, UInt32(1)), rows),), ctr, Int32(4))
    return
end
const PHASE4BENCH_TT = Tuple{MatI,Int32,U64P,U64P,Int32,IPtr}
# --bench phase4: the B4 phase of every format (and lane count) of the target against the CPU, 4 positions
function bench_phase4(g::GModel)
    k = getkernel(k_phase4bench!, PHASE4BENCH_TT; maxregs = 128, maxthreads = B4THREADS)
    xsm = 4 * (xu4(2560) + 160)
    ctr = CUDA.zeros(Int32, 1)
    cases = [("blk.0.ffn_gate_exps.weight", 640, false), ("blk.0.attn_qkv.weight", 512, false), ("blk.4.attn_gate.weight", 512, false),
        ("blk.0.attn_gate.weight", 512, false), ("blk.3.attn_v.weight", 512, false), ("blk.5.ffn_up_shexp.weight", 320, false),
        ("blk.0.ffn_gate_inp.weight", 512, false), ("blk.3.ffn_down_shexp.weight", 512, false), ("blk.0.ffn_down_shexp.weight", 512, false),
        ("blk.10.ffn_down_shexp.weight", 512, false), ("blk.3.attn_k.weight", 512, true), ("blk.0.ffn_gate_inp.weight", 513, true),
        ("blk.5.ffn_up_shexp.weight", 320, true), ("blk.3.attn_v.weight", 512, true), ("output_hc_up.weight", 512, false),
        ("blk.14.ffn_down_shexp.weight", 512, false)]
    for (name, rows, wide) in cases
        haskey(g.tensors, name) || continue
        t = g.tensors[name]; K = t.dims[1]
        K <= 2560 || continue
        rows = min(rows, nrows(t))
        d, W = upload_tensor(g, t; rows, L = lanes_for(t.typ, K; wide))
        xh = Float32[sin(0.37f0 * i + 0.9f0 * p) for i in 1:K, p in 0:3]
        x = CuArray(vec([llword(UInt32(1), xh[i, p]) for i in 1:K, p in 1:4]))
        y = CUDA.zeros(UInt64, 4 * rows)
        k(MatI(W), Int32(rows), reinterpret(U64P, pointer(x)), reinterpret(U64P, pointer(y)), Int32(xsm), reinterpret(IPtr, pointer(ctr));
            threads = B4THREADS, blocks = NSM[], shmem = 4 * xsm)
        synchronize()
        yv = Array(y); worst = 0.0; row = zeros(Float32, K); bad = 0
        for r in 0:rows-1
            dequant_row!(row, t.typ, tptr(g, t) + r * rowbytes(t), K)
            for p in 1:4
                ref = sum(Float64(row[i]) * xh[i, p] for i in 1:K); sc = sum(abs(Float64(row[i]) * xh[i, p]) for i in 1:K)
                w = yv[(p-1)*rows+r+1]
                got = (w >> 32) % UInt32 == 1 ? Float64(reinterpret(Float32, w % UInt32)) : NaN
                e = abs(got - ref) / max(sc, 1e-30)
                (isnan(e) || e > 1e-5) && (bad += 1)
                worst = isnan(e) ? Inf : max(worst, e)
            end
        end
        @printf("phase4 %-6s K=%-5d rows=%-4d L=%-2d: worst error %.2e of sum |w x|, %d of %d outputs off (%s)\n", tname(t.typ), K, rows, W.L, worst, bad, 4rows, name)
        metric("phase4.check_relerr", "ratio", worst; format = tname(t.typ), K = K, L = Int(W.L))
        CUDA.unsafe_free!(d)
    end
end
# B4 against 4 x B1 rows on the CPU, and its time
function bench_gemv4(g::GModel)
    NSM[] = CUDA.attribute(device(), CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
    k = getkernel(k_gemv4bench!, Tuple{MatI,Int32,FPtr,FPtr,Int32}; maxregs = B1MAXREGS[], maxthreads = 256)
    for (name, rows) in (("blk.4.attn_gate.weight", 3072), ("blk.0.ffn_gate_inp.weight", 512), ("output_hc_up.weight", 5120))
        t = g.tensors[name]; K = t.dims[1]
        d, W = upload_tensor(g, t; rows)
        m = MatI(W)
        sh = 4 * 4 * (xu4(K) + K ÷ 16)
        occ = CUDA.active_blocks(k.fun, 256; shmem = sh)
        grid = occ * NSM[]
        xh = Float32[sin(0.37f0 * i + 0.9f0 * p) for i in 1:K, p in 0:3]
        x = CuArray(vec(xh)); y = CUDA.zeros(Float32, 4 * rows)
        run(n) = k(m, Int32(rows), fptr(x), fptr(y), Int32(n); threads = 256, blocks = grid, shmem = sh)
        run(1); synchronize()
        yv = Array(y); worst = 0.0; row = zeros(Float32, K)
        for r in 0:min(rows, 32)-1
            dequant_row!(row, t.typ, tptr(g, t) + r * rowbytes(t), K)
            for p in 1:4
                ref = sum(Float64(row[i]) * xh[i, p] for i in 1:K); scale = sum(abs(Float64(row[i]) * xh[i, p]) for i in 1:K)
                worst = max(worst, abs(yv[4r+p] - ref) / max(scale, 1e-30))
            end
        end
        reps = 50; best = Inf
        for _ in 1:3
            best = min(best, CUDA.@elapsed run(reps))
        end
        bytes = mbytes(W, rows)
        @printf("B4 %-6s K=%-5d rows=%-5d L=%-2d %d CTAs/SM: %.1f us per 4 positions (%.1f GB/s); worst error %.2e (%s)\n",
            tname(t.typ), K, rows, m.L, occ, best / reps * 1e6, bytes * reps / best / 1e9, worst, name)
        metric("b4.us", "us", best / reps * 1e6; format = tname(t.typ), K = K, rows = rows)
        CUDA.unsafe_free!(d)
    end
end
function bench_phase(g::GModel)
    NSM[] = CUDA.attribute(device(), CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
    cases = [("blk.0.ffn_gate_exps.weight", 6400, 0), ("blk.0.attn_qkv.weight", 5120, 0), ("blk.11.attn_output.weight", 2560, 0),
        ("output_hc_up.weight", 5120, 0), ("blk.0.ffn_gate_inp.weight", 512, 0), ("blk.5.ffn_up_shexp.weight", 640, 0),
        ("blk.10.ffn_down_shexp.weight", 2560, 0), ("blk.2.ssm_out.weight", 2560, 0)]
    append!(cases, [("blk.5.ffn_gate_exps.weight", 6400, 1), ("blk.5.ffn_gate_exps.weight", 6400, 5), ("blk.5.ffn_gate_exps.weight", 6400, 6), ("blk.5.ffn_gate_exps.weight", 6400, 7), ("blk.5.ffn_gate_exps.weight", 6400, 8),
        ("blk.0.ffn_gate_inp.weight", 2, 0), ("blk.5.ffn_up_shexp.weight", 16, 0), ("blk.4.attn_gate.weight", 3072, 0), ("blk.4.attn_qkv.weight", 5120, 0), ("blk.4.attn_qkv.weight", 5120, 9), ("blk.4.attn_qkv.weight", 5120, 10), ("blk.0.ffn_gate_exps.weight", 6400, 10), ("output_hc_up.weight", 5120, 10), ("output_hc_up.weight", 5120, 11), ("blk.5.ffn_up_shexp.weight", 640, 10), ("blk.5.ffn_up_shexp.weight", 160, 0), ("output_hc_up.weight", 16, 0),
        ("blk.0.attn_qkv.weight", 16, 12), ("blk.0.ffn_gate_inp.weight", 16, 12),    # 26, 27: the x load alone (K = 2560)
        ("blk.0.attn_gate.weight", 6144, 0), ("output.weight", 8192, 0)])               # 28 Q4_K, 29 Q5_K
    k = getkernel(k_phasebench!, PHASEBENCH_TT; maxregs = B1MAXREGS[], maxthreads = B1THREADS[])
    xsm = xs_floats(2)
    occ = CUDA.active_blocks(k.fun, B1THREADS[]; shmem = 4 * xsm)
    grid = occ * NSM[]
    @printf("phase bench: %d registers, %d CTAs/SM, grid %d x %d, x buffer %d floats\n", CUDA.registers(k), occ, grid, B1THREADS[], xsm)
    # expert ids, then (from element 19) one dynamic group counter per rep, as k_token! uses them
    idx = CuArray(vcat(Int32[3, 77, 150, 201, 260, 311, 377, 402, 455, 500, 0, 0, 0, 0, 0, 0, 0, 0], zeros(Int32, 64)))
    dbg = CUDA.zeros(UInt64, 34 * grid * 8)
    nom = MatI(NOMAT)
    sel = haskey(ENV, "FLASHNEXT_BENCH_CASES") ? parse.(Int, split(ENV["FLASHNEXT_BENCH_CASES"], ",")) : eachindex(cases)
    # FLASHNEXT_BENCH_DIV: each case's rows / DIV (a die's share of an n-way split)
    dv = parse(Int, get(ENV, "FLASHNEXT_BENCH_DIV", "1"))
    for (name, rows0, mode) in cases[sel]
        rows = rows0 > 16 ? rows0 ÷ dv : rows0
        haskey(g.tensors, name) || continue
        mode == 10 && continue         # (the per-warp timeline: it drove the per-group dispatch, which is gone)
        t = g.tensors[name]; K = t.dims[1]
        xu(K) + K ÷ 16 <= xsm || (println("phase ", name, ": x of ", K, " does not fit ", xsm, " floats, skipped"); continue)
        d, W = upload_tensor(g, t; rows = mode in (1, 5, 6, 7, 8) ? 640 * 512 : rows)
        m = MatI(W); m2 = m3 = nom; bytes = mbytes(W, rows); extra = Any[]
        mode == 6 && (bytes = 0)
        if mode in (1, 6, 8)
            for nm in ("blk.5.ffn_gate_shexp.weight", "blk.5.ffn_up_shexp.weight")
                d2, W2 = upload_tensor(g, g.tensors[nm]; rows = 320); push!(extra, d2)
                nm == "blk.5.ffn_gate_shexp.weight" ? (m2 = MatI(W2)) : (m3 = MatI(W2)); bytes += mbytes(W2)
            end
        end
        if mode == 9
            d2, W2 = upload_tensor(g, g.tensors["blk.4.attn_gate.weight"]; rows = 3072); push!(extra, d2)
            m2 = MatI(W2); bytes += mbytes(W2)
        end
        xw = CuArray([llword(UInt32(1), sin(0.37f0 * i)) for i in 1:K])
        y = CUDA.zeros(UInt64, 4 * rows + 4 * 324 + 4096)
        mode == 11 && (y = CuArray(vcat(zeros(UInt64, 8192), [llword(UInt32(1), 0.5f0) for _ in 1:8192])))
        reps = 50
        run(n) = k(m, Int32(rows), reinterpret(U64P, pointer(xw)), reinterpret(U64P, pointer(y)), Int32(xsm), Int32(n), Int32(mode),
            reinterpret(IPtr, pointer(idx)), m2, m3, reinterpret(Core.LLVMPtr{UInt64,1}, pointer(dbg)); threads = B1THREADS[], blocks = grid, shmem = 4 * xsm)
        run(1); synchronize()
        if mode == 0                   # the rows against a CPU Float64 dot of the dequantized rows
            yv = Array(y); xh = [Float64(sin(0.37f0 * i)) for i in 1:K]; worst = 0.0; row = zeros(Float32, K)
            for r in 0:min(rows, 64)-1
                dequant_row!(row, t.typ, tptr(g, t) + r * rowbytes(t), K)
                ref = sum(Float64(row[i]) * xh[i] for i in 1:K); scale = sum(abs(Float64(row[i]) * xh[i]) for i in 1:K)
                worst = max(worst, abs(Float64(reinterpret(Float32, yv[r+1] % UInt32)) - ref) / max(scale, 1e-30))
            end
            @printf("phase %-6s rows 0-%d against the CPU: worst error %.2e of sum |w x|\n", tname(t.typ), min(rows, 64) - 1, worst)
            metric("phase.check_relerr", "ratio", worst; format = tname(t.typ), K = K)
        end
        best = Inf
        for _ in 1:3
            best = min(best, CUDA.@elapsed run(reps))
        end
        @printf("phase %-6s K=%-5d rows=%-6d L=%-2d mode %d %6.2f MB  %7.1f us/phase  %6.1f GB/s  (%s)\n", tname(t.typ), K, rows, m.L, mode, bytes / 1e6,
            best / reps * 1e6, bytes * reps / best / 1e9, name)
        metric("phase.bandwidth", "GB/s", bytes * reps / best / 1e9; format = tname(t.typ), K = K, rows = rows, mode = mode)
        CUDA.unsafe_free!(d); foreach(CUDA.unsafe_free!, extra)
    end
end
function bench_gemv(g::GModel)
    picked = Dict{Tuple{Int,Int},String}()
    for n in g.order
        t = g.tensors[n]
        length(t.dims) >= 2 || continue
        (occursin("per_layer", n) || t.typ in (T_F32, T_F16) || t.dims[1] < 64) && continue
        haskey(picked, (t.typ, t.dims[1])) || (picked[(t.typ, t.dims[1])] = n)
    end
    for ((typ, K), n) in sort(collect(picked); by = first)
        t = g.tensors[n]
        rows = min(nrows(t), occursin("exps", n) ? 640 * 512 : nrows(t))
        tc = time()
        d, W = upload_tensor(g, t; rows)
        xh = Float32[sin(0.37f0 * i) + 0.1f0 * cos(1.3f0 * i) for i in 1:K]
        x = CuArray(xh); y = CUDA.zeros(Float32, rows)
        t0 = time()
        gemv!(fptr(y), W, fptr(x)); synchronize()
        tcomp = time() - t0
        best = Inf
        for _ in 1:5
            best = min(best, CUDA.@elapsed gemv!(fptr(y), W, fptr(x)))
        end
        yh = Array(y)
        maxerr = 0.0; maxref = 0.0
        for r in unique([1, 2, rows ÷ 2, rows - 1, rows])
            w = dequant_row(typ, tptr(g, t) + (r - 1) * rowbytes(t), K)
            ref = sum(Float64(w[k]) * Float64(xh[k]) for k in 1:K)
            maxerr = max(maxerr, abs(ref - yh[r])); maxref = max(maxref, abs(ref))
        end
        bytes = sizeof(d)
        L = lanes_for(typ, K)
        ks = cufunction(k_gemv_stream!, Tuple{Val{typ},Val{L},FPtr,GPtr,GPtr,Int32,Int32})
        sbest = Inf
        for _ in 1:4
            sbest = min(sbest, CUDA.@elapsed ks(Val(typ), Val(L), fptr(y), W.pay, W.hdr, W.rows, W.K; threads = 256, blocks = min(cld(Int(rows), 256 ÷ L), 8 * NSM[])))
        end
        @printf("     load-only pattern: %7.1f GB/s\n", bytes / sbest / 1e9)
        rel = maxerr / max(maxref, 1e-30)
        @printf("gemv %-6s K=%-6d rows=%-7d L=%-2d %8.1f MB %7.1f GB/s %8.3f ms  compile %.1fs  relerr %.2e %s  (%s)\n",
            tname(typ), K, rows, lanes_for(typ, K), bytes / 1e6, bytes / best / 1e9, best * 1e3, tcomp, rel, rel < 1e-4 ? "PASS" : "FAIL", n)
        metric("gemv.bandwidth", "GB/s", bytes / best / 1e9; format = tname(typ), K = K, rows = rows)
        metric("gemv.check", "pass", rel < 1e-4; format = tname(typ), K = K, relerr = rel)
        CUDA.unsafe_free!(d)
    end
end

# ---- cross-die communication bench: peer-access matrix, all-gather rounds over P2P and over host-mapped memory
@inline ld_sys(p::IPtr) = UA.load(p, UA.monotonic, UA.system)
@inline st_sys(p::IPtr, v::Int32) = UA.store!(p, v, UA.monotonic, UA.system)
@inline ld_dev(p::IPtr) = UA.load(p, UA.monotonic, UA.device)
# one CTA per die: write my n-float slice into every die's buffer, raise my flag there, wait for every die's flag here
function k_agbench!(me::Int32, nd::Int32, rbuf::NTuple{8,FPtr}, rflag::NTuple{8,IPtr}, myflag::IPtr, n::Int32, iters::Int32, out::Core.LLVMPtr{UInt64,1})
    tid = threadIdx().x - Int32(1)
    t0 = CUDA.clock(UInt64)
    for it in Int32(1):iters
        for d in 1:8
            if d <= nd
                p = rbuf[d]
                i = tid
                while i < n
                    unsafe_store!(p, Float32(it), me * n + i + Int32(1), Val(4)); i += Int32(256)
                end
            end
        end
        CUDA.threadfence_system()
        sync_threads()
        for d in 1:8
            (tid == Int32(d - 1) && d <= nd) && st_sys(rflag[d] + 4me, it)
        end
        if tid < nd
            while ld_sys(myflag + 4tid) < it
            end
        end
        sync_threads()
    end
    tid == Int32(0) && unsafe_store!(out, CUDA.clock(UInt64) - t0, 1, Val(8))
    return
end
# LL protocol: each 8-byte store is {seq (high 32), value (low 32)}; no fences; the receiver polls the words themselves.
# Two buffers alternate by round parity (a die can be at most one round ahead of any reader).
@inline ld_sys64(p::U64P, i) = UA.load(p + 8 * (i - 1), UA.monotonic, UA.system)
@inline st_vol64(p::U64P, i, v::UInt64) = UA.store!(p + 8 * (i - 1), v, UA.monotonic, UA.system)
function k_agll!(me::Int32, nd::Int32, rbuf::NTuple{8,U64P}, mybuf::U64P, n::Int32, iters::Int32, out::Core.LLVMPtr{UInt64,1}, sink::FPtr)
    tid = threadIdx().x - Int32(1)
    t0 = CUDA.clock(UInt64)
    acc = 0.0f0
    for it in Int32(1):iters
        par = (it & Int32(1)) * Int32(8) * n
        for d in 1:8
            if d <= nd
                p = rbuf[d]
                i = tid
                while i < n
                    st_vol64(p, par + me * n + i + Int32(1), (UInt64(it) << 32) | reinterpret(UInt32, Float32(i)))
                    i += Int32(256)
                end
            end
        end
        i = tid
        while i < nd * n
            v = ld_sys64(mybuf, par + i + Int32(1))
            while (v >> 32) % Int32 != it
                v = ld_sys64(mybuf, par + i + Int32(1))
            end
            acc += reinterpret(Float32, v % UInt32)
            i += Int32(256)
        end
        sync_threads()
    end
    tid == Int32(0) && unsafe_store!(out, CUDA.clock(UInt64) - t0, 1, Val(8))
    stf(sink, tid, acc)
    return
end
function bench_comm_ll(dies, n; modes = (:ll_p2p, :ll_host), quiet = false)
    nd = length(dies)
    iters = Int32(2000)
    pad(t, z) = ntuple(i -> i <= length(t) ? t[i] : z, 8)
    res = Dict{Symbol,Float64}()
    for mode in modes
        keep = []; outs = []; sinks = []
        if mode == :ll_p2p
            bufs = [(device!(d); CUDA.zeros(UInt64, 2 * 8 * n)) for d in dies]
            append!(keep, bufs)
            rb = pad(Tuple(reinterpret(U64P, pointer(b)) for b in bufs), U64P(0)); mb = [rb[k] for k in 1:nd]
        else
            sz = nd * 2 * 8 * n * 8
            hm = CUDA.alloc(CUDA.HostMemory, sz, CUDA.MEMHOSTALLOC_DEVICEMAP | CUDA.MEMHOSTALLOC_PORTABLE); push!(keep, hm)
            hp = convert(Ptr{UInt8}, hm); ccall(:memset, Ptr{Cvoid}, (Ptr{UInt8}, Cint, Csize_t), hp, 0, sz)
            rb = pad(Tuple(U64P(UInt(hp) + (k - 1) * 2 * 8 * n * 8) for k in 1:nd), U64P(0)); mb = [rb[k] for k in 1:nd]
        end
        for d in dies
            device!(d); push!(outs, CUDA.zeros(UInt64, 1)); push!(sinks, CUDA.zeros(Float32, 256))
        end
        for (k, d) in enumerate(dies)
            device!(d)
            @cuda threads = 256 k_agll!(Int32(k - 1), Int32(nd), rb, mb[k], Int32(n), iters, reinterpret(Core.LLVMPtr{UInt64,1}, pointer(outs[k])), fptr(sinks[k]))
        end
        for d in dies
            device!(d); synchronize()
        end
        mhz = CUDA.attribute(CuDevice(dies[1]), CUDA.DEVICE_ATTRIBUTE_CLOCK_RATE) / 1e3
        us = [Array(outs[k])[1] / iters / mhz for k in 1:nd]
        quiet || @printf("all-gather %-7s %d dies x %4d floats: %.2f us per round (max over dies; min %.2f)\n", mode, nd, n, maximum(us), minimum(us))
        metric("comm.allgather_us", "us", maximum(us); mode = mode, dies = nd, floats = n)
        res[mode] = maximum(us)
    end
    res
end
# --bench sweep: for the first 2, 3, ..., n of the dies, transfers one at a time and all at once. Single: each ordered
# pair of them alone (16 MB of 16-byte stores, GB/s: min, median, max over the pairs). Concurrent: a ring (k -> k + 1,
# all writers at once) and all-to-all (each die writes every other, in turn on its own stream, all dies at once),
# GB/s per writer and in total; and the LL all-gather of 16 and 384 floats a die (us a round).
function bench_sweep(dies)
    nd = length(dies)
    ctxs = [(device!(d); context()) for d in dies]
    for i in 1:nd, j in 1:nd
        i != j && (device!(dies[i]); try CUDA.enable_peer_access(ctxs[j]) catch end)
    end
    MB = 16; bytes = MB << 20
    bufs = [(device!(d); CUDA.zeros(UInt8, nd * bytes)) for d in dies]       # a slot per writer
    dst(j, i) = gptr(bufs[j]) + (i - 1) * bytes
    function run(pairs)               # (source index, destination pointer), each source's kernels in its stream's order
        evs = Dict{Int,Tuple{CuEvent,CuEvent}}()
        for (s, _) in pairs
            haskey(evs, s) && continue
            device!(dies[s]); e0 = CuEvent(); record(e0); evs[s] = (e0, CuEvent())
        end
        for (s, p) in pairs
            device!(dies[s]); @cuda threads = 256 blocks = 64 k_p2pbw!(p, Int32(bytes))
        end
        for (s, (e0, e1)) in evs
            device!(dies[s]); record(e1)
        end
        for (s, (e0, e1)) in evs
            device!(dies[s]); synchronize(e1)
        end
        maximum(elapsed(e0, e1) for (e0, e1) in values(evs))
    end
    best(pairs) = (run(pairs); minimum(run(pairs) for _ in 1:3))
    gb(n, t) = n * bytes / t / 1e9
    single = Dict((i, j) => gb(1, best([(i, dst(j, i))])) for i in 1:nd, j in 1:nd if i != j)
    @printf("dies %s, %d MB a transfer; single: one pair alone (min/median/max over the pairs); ring and all-to-all: all writers at once\n",
        join(dies, ","), MB)
    @printf("%-4s %-28s %-24s %-24s %s\n", "dies", "single GB/s", "ring GB/s (writer/total)", "all-to-all GB/s (total)", "all-gather 16 / 384 floats")
    for n in 2:nd
        s = sort([single[(i, j)] for i in 1:n, j in 1:n if i != j])
        tr = best([(k, dst(k % n + 1, k)) for k in 1:n])
        a2a = [(i, dst(j, i)) for i in 1:n for j in 1:n if i != j]
        ta = best(a2a)
        g16 = bench_comm_ll(dies[1:n], 16; modes = (:ll_p2p,), quiet = true)[:ll_p2p]
        g384 = bench_comm_ll(dies[1:n], 384; modes = (:ll_p2p,), quiet = true)[:ll_p2p]
        @printf("%-4d %5.2f / %5.2f / %5.2f        %5.2f / %6.2f          %6.2f                   %6.1f / %7.1f us\n", n, s[1],
            s[cld(length(s), 2)], s[end], gb(1, tr), gb(n, tr), gb(length(a2a), ta), g16, g384)
        metric("comm.sweep_single_GBs", "GB/s", s[1]; dies = n, median = s[cld(length(s), 2)], max = s[end])
        metric("comm.sweep_ring_GBs", "GB/s", gb(n, tr); dies = n, per_writer = gb(1, tr))
        metric("comm.sweep_a2a_GBs", "GB/s", gb(length(a2a), ta); dies = n)
    end
    println("single, every ordered pair (GB/s, row writes column):")
    for i in 1:nd
        println(@sprintf("  die %d:", dies[i]), join([i == j ? "     -" : @sprintf(" %5.2f", single[(i, j)]) for j in 1:nd]))
    end
end
# bulk bandwidth: CTAs stream 16-byte stores of n bytes into dst
function k_p2pbw!(dst::GPtr, n::Int32)
    i = gtid()
    q = reinterpret(Core.LLVMPtr{V4F,1}, dst)
    v = (VecElement(1.0f0), VecElement(2.0f0), VecElement(3.0f0), VecElement(Float32(i)))
    while i < n ÷ Int32(16)
        unsafe_store!(q, v, i + Int32(1), Val(16)); i += gridDim().x * blockDim().x
    end
    return
end
function bench_bw(dies)
    nd = length(dies)
    ctxs = [(device!(d); context()) for d in dies]
    for i in 1:nd, j in 1:nd
        i != j && (device!(dies[i]); try CUDA.enable_peer_access(ctxs[j]) catch end)
    end
    MB = 16
    bufs = [(device!(d); CUDA.zeros(UInt8, MB << 20)) for d in dies]
    hm = CUDA.alloc(CUDA.HostMemory, nd * (MB << 20), CUDA.MEMHOSTALLOC_DEVICEMAP | CUDA.MEMHOSTALLOC_PORTABLE)
    hp = UInt(convert(Ptr{UInt8}, hm))
    function run(pairs, bytes)        # pairs: (src index, destination pointer)
        evs = []
        for (s, dst) in pairs
            device!(dies[s]); e0 = CuEvent(); e1 = CuEvent(); record(e0)
            @cuda threads = 256 blocks = 64 k_p2pbw!(dst, Int32(bytes)); record(e1); push!(evs, (s, e0, e1))
        end
        for (s, e0, e1) in evs
            device!(dies[s]); synchronize(e1)
        end
        maximum(elapsed(e0, e1) for (s, e0, e1) in evs)
    end
    gb(bytes, t) = bytes / t / 1e9
    for (name, pairs, bytes) in [("die0 -> die1 (same board)", [(1, gptr(bufs[2]))], MB << 20),
                                 ("die0 -> die2 (other board)", [(1, gptr(bufs[3]))], MB << 20),
                                 ("die0 -> host", [(1, GPtr(hp))], MB << 20),
                                 ("all dies -> host at once", [(k, GPtr(hp + (k - 1) * (MB << 20))) for k in 1:nd], MB << 20),
                                 ("pairs 0-1, 2-3, 4-5 at once", [(k, gptr(bufs[k % 2 == 1 ? k + 1 : k - 1])) for k in 1:min(nd, 6)], MB << 20),
                                 ("ring k -> k+1 (all at once)", [(k, gptr(bufs[k % nd + 1])) for k in 1:nd], MB << 20)]
        run(pairs, bytes)
        t = minimum(run(pairs, bytes) for _ in 1:3)
        @printf("bw %-30s %6.2f GB/s per writer, %6.2f GB/s total\n", name, gb(bytes, t), gb(bytes * length(pairs), t))
        metric("comm.bw_GBs", "GB/s", gb(bytes * length(pairs), t); pattern = name)
    end
end
function bench_comm(dies)
    nd = length(dies)
    devs = [CuDevice(d) for d in dies]
    println("peer access (row can write column):")
    can = falses(nd, nd)
    for i in 1:nd
        line = ""
        for j in 1:nd
            r = Ref{Cint}(0)
            i != j && CUC.cuDeviceCanAccessPeer(r, devs[i], devs[j])
            can[i, j] = i == j || r[] == 1
            line *= can[i, j] ? " 1" : " 0"
        end
        println("  die ", dies[i], ":", line)
    end
    ctxs = [(device!(d); context()) for d in dies]
    n = Int32(384); iters = Int32(2000)
    pad(t, z) = ntuple(i -> i <= length(t) ? t[i] : z, 8)
    for mode in (all(can) ? (:p2p, :host) : (:host,))
        bufs = []; flags = []; outs = []
        if mode == :p2p
            for i in 1:nd, j in 1:nd
                i != j && (device!(dies[i]); try CUDA.enable_peer_access(ctxs[j]) catch end)
            end
            for d in dies
                device!(d)
                push!(bufs, CUDA.zeros(Float32, 8 * n)); push!(flags, CUDA.zeros(Int32, 8))
            end
            rb = pad(Tuple(fptr(b) for b in bufs), NOF); rf = pad(Tuple(reinterpret(IPtr, pointer(f)) for f in flags), NOI)
            mf = [reinterpret(IPtr, pointer(f)) for f in flags]
        else
            hm = CUDA.alloc(CUDA.HostMemory, nd * (8 * n * 4 + 64), CUDA.MEMHOSTALLOC_DEVICEMAP | CUDA.MEMHOSTALLOC_PORTABLE)
            hp = convert(Ptr{UInt8}, hm)
            ccall(:memset, Ptr{Cvoid}, (Ptr{UInt8}, Cint, Csize_t), hp, 0, nd * (8 * n * 4 + 64))
            base = UInt(hp)
            rb = pad(Tuple(FPtr(base + (k - 1) * (8 * n * 4 + 64)) for k in 1:nd), NOF)
            rf = pad(Tuple(IPtr(base + (k - 1) * (8 * n * 4 + 64) + 8 * n * 4) for k in 1:nd), NOI)
            mf = [rf[k] for k in 1:nd]
            push!(bufs, hm)
        end
        for (k, d) in enumerate(dies)
            device!(d); push!(outs, CUDA.zeros(UInt64, 1))
        end
        for (k, d) in enumerate(dies)
            device!(d)
            @cuda threads = 256 k_agbench!(Int32(k - 1), Int32(nd), rb, rf, mf[k], n, iters, reinterpret(Core.LLVMPtr{UInt64,1}, pointer(outs[k])))
        end
        for d in dies
            device!(d); synchronize()
        end
        mhz = CUDA.attribute(devs[1], CUDA.DEVICE_ATTRIBUTE_CLOCK_RATE) / 1e3
        us = [Array(outs[k])[1] / iters / mhz for k in 1:nd]
        @printf("all-gather over %-4s, %d dies x %d floats: %.2f us per round (max over dies; min %.2f)\n", mode, nd, n, maximum(us), minimum(us))
        metric("comm.allgather_us", "us", maximum(us); mode = mode, dies = nd, floats = Int(n))
    end
    for n in (16, 384, 1536)
        bench_comm_ll(dies, n)
    end
    length(dies) > 2 && bench_comm_ll(dies[1:2], 384)
end

# ---------------------------------------------------------------------------------------------
# 6. DFlash2: the draft. Five non-causal layers over a block of 4 positions (the anchor, then 3 mask tokens) with
# grouped-conv wrappers around attention and FFN, attending to a context cache made from the target's features (the
# mean of the 4 lanes of S after blocks 5, 15, 25, 35, 45: fc, its norm, each layer's k/v). Its hidden at positions
# 1..3 goes through the target head (top 16), and the selector walk picks each proposal. On DRAFT_DIE; the head's
# candidates on every die holding a head slice.
# ---------------------------------------------------------------------------------------------
const DRAFT_PATH = "/home/nate/models-cache/flash-sass/dflash2-q4km/Qwen3.8-Flash-Next-DFlash2-Q4_K_M.gguf"
const DWIDE = get(ENV, "FLASHNEXT_DWIDE", "0") == "1"     # slower: backbone 7.2 -> 24 ms a proposal
# the (format, lanes) k_gemv4! dispatches (the draft and the target head's slices)
const GEMV4_COMBOS = ((T_Q4_K, 10), (T_Q4_K, 6), (T_Q6_K, 10), (T_Q6_K, 6), (T_Q4_K, 2), (T_Q4_K, 8), (T_Q6_K, 2), (T_Q6_K, 8), (T_Q5_K, 2))
const H16P = Core.LLVMPtr{UInt16,1}
const GEMV4_TT = Tuple{MatI,Int32,FPtr,Int32,Int32,FPtr,Int32,Int32,Int32,Int32}
# y[i] = a x[i] + b x[i - 2560] over the block's positions (b only for p > 0), a, b = base + the projection's group
# coefficients (640 per position: side s at 320 s, a then b, 160 groups of 16 channels)
function k_dconv!(y::FPtr, x::FPtr, coeff::FPtr, base::FPtr, side::Int32, n::Int32)
    i = gtid()
    @inbounds if i < n * Int32(2560)
        p = i ÷ Int32(2560); c = i % Int32(2560); o = p * Int32(640) + side * Int32(320) + c ÷ Int32(16)
        a = ldf(base, side * Int32(5120) + c) + ldf(coeff, o)
        b = ldf(base, side * Int32(5120) + Int32(2560) + c) + ldf(coeff, o + Int32(160))
        stf(y, i, p > Int32(0) ? fma(a, ldf(x, i), b * ldf(x, i - Int32(2560))) : a * ldf(x, i))
    end
    return
end
# NeoX rotation of dims (j, j + 64) of every 128-dim head, position start + p
# the draft's position (its context length) is a device scalar sp, so captured graphs never change
function k_drope!(x::FPtr, freq::FPtr, heads::Int32, sp::IPtr, n::Int32)
    i = gtid(); start = unsafe_load(sp)
    @inbounds if i < n * heads * Int32(64)
        j = i & Int32(63); h = (i ÷ Int32(64)) % heads; p = i ÷ (heads * Int32(64))
        s, c = sincos(Float32(start + p) * ldf(freq, j))
        o = (p * heads + h) * Int32(128)
        a = ldf(x, o + j); b = ldf(x, o + j + Int32(64))
        stf(x, o + j, a * c - b * s); stf(x, o + j + Int32(64), a * s + b * c)
    end
    return
end
@inline h16(v::Float32) = reinterpret(UInt16, Float16(v))
@inline f16(u::UInt16) = Float32(reinterpret(Float16, u))
# context K/V (fp16) at positions start .. start + n - 1 of the 2048-position ring
function k_dcache!(kc::H16P, vc::H16P, k::FPtr, v::FPtr, sp::IPtr, n::Int32)
    i = gtid(); start = unsafe_load(sp)
    @inbounds if i < n * Int32(1024)
        p = i ÷ Int32(1024); e = i % Int32(1024); d = ((start + p) & Int32(2047)) * Int32(1024) + e
        unsafe_store!(kc, h16(ldf(k, i)), d + 1); unsafe_store!(vc, h16(ldf(v, i)), d + 1)
    end
    return
end
# attention of block position p, query head h (one warp): the context positions within the 2048 window, then all n
# block positions (non-causal; their K/V rounded to fp16 like the cache); 8 K/V heads for 32 query heads
function k_dattn!(out::FPtr, q::FPtr, ks::FPtr, vs::FPtr, kc::H16P, vc::H16P, sp::IPtr, n::Int32)
    context = unsafe_load(sp)
    b = blockIdx().x - Int32(1); p = b ÷ Int32(32); h = b % Int32(32); kv = h ÷ Int32(4)
    tid = threadIdx().x - Int32(1); lane = tid % Int32(32); warp = tid ÷ Int32(32); nw = blockDim().x ÷ Int32(32)
    @inbounds begin
        qv = Base.Cartesian.@ntuple 4 j -> ldf(q, (p * Int32(32) + h) * Int32(128) + lane + Int32(32(j - 1)))
        lo = context + p + Int32(1) > Int32(2048) ? context + p + Int32(1) - Int32(2048) : Int32(0)
        mx = -Inf32; den = 0.0f0; acc = Z4
        # the warps of the CTA take every nw-th key (flash-decoding split), merged below
        pos = lo + warp
        while pos < context + n
            if pos < context
                o = ((pos & Int32(2047)) * Int32(8) + kv) * Int32(128) + lane
                kk = Base.Cartesian.@ntuple 4 j -> f16(unsafe_load(kc, o + Int32(32(j - 1)) + 1))
                vv = Base.Cartesian.@ntuple 4 j -> f16(unsafe_load(vc, o + Int32(32(j - 1)) + 1))
            else
                o = ((pos - context) * Int32(8) + kv) * Int32(128) + lane
                kk = Base.Cartesian.@ntuple 4 j -> f16(h16(ldf(ks, o + Int32(32(j - 1)))))
                vv = Base.Cartesian.@ntuple 4 j -> f16(h16(ldf(vs, o + Int32(32(j - 1)))))
            end
            s = fma(qv[1], kk[1], fma(qv[2], kk[2], fma(qv[3], kk[3], qv[4] * kk[4])))
            s = wsum(s, Val(32)) * 0.08838834764831845f0
            nm = max(mx, s); os = exp(mx - nm); ns = exp(s - nm)
            den = fma(den, os, ns); mx = nm
            acc = Base.Cartesian.@ntuple 4 j -> fma(acc[j], os, vv[j] * ns)
            pos += nw
        end
        sh = CuStaticSharedArray(Float32, 8 * 130)       # per warp (up to 8): max, sum, 128 values
        lane == Int32(0) && (sh[warp*130+1] = mx; sh[warp*130+2] = den)
        Base.Cartesian.@nexprs 4 j -> (sh[warp*130+3+lane+32(j-1)] = acc[j])
        sync_threads()
        if warp == Int32(0)
            M = -Inf32
            for w in Int32(0):nw-Int32(1)
                M = max(M, sh[w*130+1])
            end
            D = 0.0f0; A = Z4
            for w in Int32(0):nw-Int32(1)
                f = sh[w*130+1] == -Inf32 ? 0.0f0 : exp(sh[w*130+1] - M)
                D = fma(sh[w*130+2], f, D)
                A = Base.Cartesian.@ntuple 4 j -> fma(sh[w*130+3+lane+32(j-1)], f, A[j])
            end
            Base.Cartesian.@nexprs 4 j -> stf(out, (p * Int32(32) + h) * Int32(128) + lane + Int32(32(j - 1)), A[j] / D)
        end
    end
    return
end
function k_dswiglu!(g::FPtr, u::FPtr, n::Int32)
    i = gtid()
    @inbounds i < n && stf(g, i, silu(ldf(g, i)) * ldf(u, i))
    return
end
# h[p F + j] = silu(gu[p 2F + j]) gu[p 2F + F + j]
function k_dswiglu2!(h::FPtr, gu::FPtr, F::Int32, n::Int32)
    i = gtid()
    @inbounds if i < n * F
        p = i ÷ F; j = i % F
        stf(h, i, silu(ldf(gu, Int32(2) * F * p + j)) * ldf(gu, Int32(2) * F * p + F + j))
    end
    return
end
function k_dadd!(x::FPtr, r::FPtr, n::Int32)
    i = gtid()
    @inbounds i < n && stf(x, i, ldf(x, i) + ldf(r, i))
    return
end
# the draft's features of positions start .. start + n - 1: per tap, the mean of S's 4 lanes, from the tap slots
function k_dfeat!(feat::FPtr, tap::FPtr, sp::IPtr, n::Int32)
    i = gtid(); start = unsafe_load(sp)
    @inbounds if i < n * Int32(12800)
        p = i ÷ Int32(12800); e = i % Int32(12800); k = e ÷ Int32(2560); c = e % Int32(2560)
        o = ((start + p) & Int32(7)) * Int32(TAPSLOT) + k * Int32(4 * 2560) + c
        stf(feat, i, 0.25f0 * (((ldf(tap, o) + ldf(tap, o + Int32(2560))) + ldf(tap, o + Int32(5120))) + ldf(tap, o + Int32(7680))))
    end
    return
end
# the 16 largest of logits[p ldl + 0 .. rows) (ties: the lower index first) as (value, r0 + index) pairs into
# out[p 32 + 2k, + 1]; one CTA per position, each round the next one in that order
# the top 16 of chunk ch (of nch) of position p's logits (CTA p nch + ch): out[(p nch + ch) 32 + 2k, +1] = value, row
function k_top16!(out::FPtr, logits::FPtr, rows::Int32, ldl::Int32, r0::Int32, nch::Int32)
    b = blockIdx().x - Int32(1); p = b ÷ nch; ch = b - p * nch
    tid = threadIdx().x - Int32(1); lane = tid % Int32(32); warp = tid ÷ Int32(32)
    lo = (ch * rows) ÷ nch; hi = ((ch + Int32(1)) * rows) ÷ nch
    sv = CuStaticSharedArray(Float32, 32); si = CuStaticSharedArray(Int32, 32); bv = CuStaticSharedArray(Float32, 1); bix = CuStaticSharedArray(Int32, 1)
    pv = Inf32; pi = Int32(-1)
    @inbounds for k in Int32(0):Int32(15)
        v = -Inf32; ix = Int32(typemax(Int32))
        i = lo + tid
        while i < hi
            x = ldf(logits, p * ldl + i)
            if (x < pv || (x == pv && i > pi)) && (x > v || (x == v && i < ix))
                v = x; ix = i
            end
            i += blockDim().x
        end
        for o in (16, 8, 4, 2, 1)
            ov = shfl_xor_sync(FULL, v, o); oi = shfl_xor_sync(FULL, ix, o)
            (ov > v || (ov == v && oi < ix)) && (v = ov; ix = oi)
        end
        lane == Int32(0) && (sv[warp+1] = v; si[warp+1] = ix)
        sync_threads()
        if warp == Int32(0)
            v = lane < blockDim().x ÷ Int32(32) ? sv[lane+1] : -Inf32; ix = lane < blockDim().x ÷ Int32(32) ? si[lane+1] : Int32(typemax(Int32))
            for o in (16, 8, 4, 2, 1)
                ov = shfl_xor_sync(FULL, v, o); oi = shfl_xor_sync(FULL, ix, o)
                (ov > v || (ov == v && oi < ix)) && (v = ov; ix = oi)
            end
            if lane == Int32(0)
                bv[1] = v; bix[1] = ix
                stf(out, b * Int32(32) + Int32(2) * k, v); stf(out, b * Int32(32) + Int32(2) * k + Int32(1), reinterpret(Float32, ix + r0))
            end
        end
        sync_threads()
        pv = bv[1]; pi = bix[1]
        sync_threads()
    end
    return
end

mutable struct DraftLayer
    attn_norm::FPtr; attn_cp::DevMat; attn_cb::FPtr; q::DevMat; k::DevMat; v::DevMat; qn::FPtr; kn::FPtr; o::Vector{DevMat}
    ffn_norm::FPtr; ffn_cp::DevMat; ffn_cb::FPtr; gate::DevMat; up::DevMat; down::Vector{DevMat}
end
mutable struct Draft
    g::GModel
    L::Vector{DraftLayer}
    fc::Vector{DevMat}; enc_norm::FPtr; out_norm::FPtr; sel::DevMat
    pred::GTensor; succ::GTensor
    predb::Vector{UInt8}; succb::Vector{UInt8}   # the selector codebooks in RAM (not page faults on the mapping)
    b::Dict{Symbol,FPtr}
    kc::Vector{H16P}; vc::Vector{H16P}
    freq::FPtr
    keep::Vector{Any}
    context::Int
    mask::Int32
    heads::Vector{Any}                # per die with a head slice: (ord, out, rows, r0, hx, logits, top)
    tap::FPtr
    ms::Dict{Symbol,Float64}          # host time of the draft's parts (s)
    pos::IPtr                         # the position the kernels read (device Int32): the context length or append start
    gprop::Union{Nothing,CUC.CuGraphExec}        # the backbone of a proposal, captured on its second use
    gapp::Dict{Int,CUC.CuGraphExec}              # the append of n positions, captured on its second use
    uses::Dict{Int,Int}                          # uses so far (0: proposal; n: append of n)
end
function load_draft(E::EngineB1, path = DRAFT_PATH)
    t0 = time()
    g = gguf_open([path])
    device!(DRAFT_DIE)
    keep = Any[]
    T(n) = g.tensors[n]
    f32(n) = (t = T(n); a = CuArray(copy(unsafe_wrap(Array, Ptr{Float32}(tptr(g, t)), prod(t.dims)))); push!(keep, a); fptr(a))
    # matrices: K slices of at most 2560 columns (the B4 x buffer)
    specs = Tuple{String,Int,Int}[]
    for l in 0:4, (nm, kc) in (("attn_conv_proj", 2560), ("attn_q", 2560), ("attn_k", 2560), ("attn_v", 2560), ("attn_output", 2560),
                               ("ffn_conv_proj", 2560), ("ffn_down", 2560))
        push!(specs, ("blk.$l.$nm.weight", kc, 0))
    end
    push!(specs, ("fc.weight", 2560, 0), ("selector_hidden.weight", 2560, 0))
    slices(t, kc) = [(c0, min(kc, t.dims[1] - c0)) for c0 in 0:kc:t.dims[1]-1]
    A = Arena()
    # gate and up as one matrix of 15360 rows (one phase; the SwiGLU takes its two halves)
    gup(l) = [(T("blk.$l.ffn_gate.weight"), 0, 7680), (T("blk.$l.ffn_up.weight"), 0, 7680)]
    # matrices of at most 4096 rows in wide layouts (6 or 10 lanes a row, 3-5 rows a group): with 2 lanes (16-row groups)
    # a 640-row matrix had 40 groups for 256 warps
    dwide(nm) = DWIDE && nrows(T(nm)) <= 4096
    for (nm, kc, _) in specs, (c0, n) in slices(T(nm), kc)
        put_mat!(A, g, [(T(nm), 0, nrows(T(nm)), c0, n)]; wide = dwide(nm))
    end
    for l in 0:4
        put_mat!(A, g, gup(l))
    end
    base = rawalloc(A.off); A = Arena(base, A.off)
    M = Dict{String,Vector{DevMat}}()
    for (nm, kc, _) in specs
        M[nm] = [put_mat!(A, g, [(T(nm), 0, nrows(T(nm)), c0, n)]; wide = dwide(nm)) for (c0, n) in slices(T(nm), kc)]
    end
    GUP = [put_mat!(A, g, gup(l)) for l in 0:4]
    L = [DraftLayer(f32("blk.$l.attn_norm.weight"), M["blk.$l.attn_conv_proj.weight"][1], f32("blk.$l.attn_conv_base"),
            M["blk.$l.attn_q.weight"][1], M["blk.$l.attn_k.weight"][1], M["blk.$l.attn_v.weight"][1],
            f32("blk.$l.attn_q_norm.weight"), f32("blk.$l.attn_k_norm.weight"), M["blk.$l.attn_output.weight"],
            f32("blk.$l.ffn_norm.weight"), M["blk.$l.ffn_conv_proj.weight"][1], f32("blk.$l.ffn_conv_base"),
            GUP[l+1], GUP[l+1], M["blk.$l.ffn_down.weight"]) for l in 0:4]
    mk(n) = (a = CUDA.zeros(Float32, n); push!(keep, a); fptr(a))
    b = Dict{Symbol,FPtr}()
    haskey(g.kv, "dflash.block_size") && Int(g.kv["dflash.block_size"]) != DBLK &&
        error("draft: block size $(g.kv["dflash.block_size"]), the engine's is $DBLK")
    for (k, n) in (:x => DBLK * 2560, :nrm => DBLK * 2560, :inp => DBLK * 2560, :coeff => DBLK * 640, :q => DBLK * 4096, :k => DBLK * 1024,
                   :v => DBLK * 1024, :att => DBLK * 4096, :prj => DBLK * 2560, :tmp => DBLK * 2560, :gate => DBLK * 7680, :up => DBLK * 7680,
                   :sel => DBLK * 256, :feat => DBLK * 12800, :ctxh => DBLK * 2560, :gu => DBLK * 15360)
        b[k] = mk(n)
    end
    kc = H16P[]; vc = H16P[]
    for _ in 1:5
        a = CUDA.zeros(UInt16, 2048 * 1024); c = CUDA.zeros(UInt16, 2048 * 1024); push!(keep, a, c)
        push!(kc, reinterpret(H16P, pointer(a))); push!(vc, reinterpret(H16P, pointer(c)))
    end
    th = Float32(g.kv["dflash.rope.freq_base"])
    fa = CuArray(Float32[th^(-Float32(j) / 64.0f0) for j in 0:63]); push!(keep, fa)
    heads = Any[]
    for DG in E.groups, D in DG
        H = D.rt.head; H === nothing && continue
        device!(D.rt.ord)
        rows = Int(H.out.rows)
        nd = DBLK - 1
        hx = CUDA.zeros(Float32, nd * 2560); lg = CUDA.zeros(Float32, nd * rows); top = CUDA.zeros(Float32, nd * 32 * TOPCH); push!(keep, hx, lg, top)
        push!(heads, (D.rt.ord, H.out, rows, Int(H.r0), fptr(hx), fptr(lg), fptr(top)))
    end
    device!(DRAFT_DIE)
    ram(t) = (v = Vector{UInt8}(undef, t.nbytes); GC.@preserve v unsafe_copyto!(pointer(v), Ptr{UInt8}(tptr(g, t)), t.nbytes); v)
    pa = CUDA.zeros(Int32, 1); push!(keep, pa)
    for ms in values(M), m in ms
        (Int(m.t), Int(m.L)) in GEMV4_COMBOS || error("draft: $(tname(m.t)) at $(m.L) lanes is not in k_gemv4!'s dispatch")
    end
    Dr = Draft(g, L, M["fc.weight"], f32("enc.output_norm.weight"), f32("output_norm.weight"), M["selector_hidden.weight"][1],
        T("selector_predecessor.weight"), T("selector_successor.weight"), ram(T("selector_predecessor.weight")), ram(T("selector_successor.weight")), b, kc, vc, fptr(fa), keep, 0, Int32(g.kv["tokenizer.ggml.mask_token_id"]),
        heads, fptr(E.tap), Dict{Symbol,Float64}(), reinterpret(IPtr, pointer(pa)), nothing, Dict{Int,CUC.CuGraphExec}(), Dict{Int,Int}())
    note(@sprintf("draft: %s, %.2f GB of matrices on die %d, %d head slices, in %.1f s", basename(path), A.off / 1e9, DRAFT_DIE, length(heads), time() - t0))
    Dr
end
function gemv4!(y::FPtr, m::DevMat, x::FPtr, P::Integer; ldx = m.K, xoff = 0, ldy = m.rows, accum = false)
    # (more than 4 positions: runs of 4, each its own read of the weights)
    if P > 4
        for p0 in 0:4:P-1
            gemv4!(y + 4 * p0 * ldy, m, x + 4 * p0 * ldx, min(4, P - p0); ldx, xoff, ldy, accum)
        end
        return
    end
    sh = 4 * 4 * (xu4(Int(m.K)) + Int(m.K) ÷ 16)
    k = getkernel(k_gemv4!, GEMV4_TT; maxregs = 128, maxthreads = 256)::CUC.HostKernel{typeof(k_gemv4!),GEMV4_TT}
    occ = Int(get!(() -> CUDA.active_blocks(k.fun, 256; shmem = sh), KCACHE, (:occ4, sh, CUDA.device()))::Integer)
    # fewer row groups than warps: each group's blocks over ks warps (K chunks), summed into y by atomic adds
    W = occ * NSM[] * 8; G = cld(Int(m.rows), 32 ÷ Int(m.L)); NI = (Int(m.K) ÷ devE(m.t)) ÷ Int(m.L)
    ks = GEMV4KS && G < W ? clamp(cld(W, G), 1, NI) : 1
    tm(() -> "gemv4 $(tname(m.t)) $(m.rows)x$(m.K) P$P") do
        if ks > 1 && !accum
            for p in 0:P-1
                CUC.cuMemsetD32Async(reinterpret(CUC.CUdeviceptr, y + 4 * p * ldy), UInt32(0), m.rows, stream())
            end
        end
        k(MatI(m), Int32(m.rows), x, Int32(ldx), Int32(xoff), y, Int32(ldy), Int32(accum), Int32(P), Int32(ks); threads = 256, blocks = occ * NSM[], shmem = sh)
    end
end
const GEMV4KS = get(ENV, "FLASHNEXT_GEMV4KS", "0") == "1"     # slower per op (2560x2560 P4: 1.08 -> 1.33 ms a round)
const DATTNW = parse(Int, get(ENV, "FLASHNEXT_DATTNW", "4"))     # warps per (position, head) of the draft's attention (<= 8)
const TOPCH = 8                       # chunks of a head slice's logits per position in the draft's top 16 (a CTA each)
const DBLK = P8MAX                    # the draft's block (dflash.block_size): the anchor and DBLK - 1 masked positions
dlaunch(f, args...; kw...) = tm(() -> string(nameof(f))) do
    launch(f, args...; kw...)
end
# --bench draft: an event recorded in the stream before and after each draft op (no host syncs in between; read after
# the rounds): the device time of each op and the idle gaps between them (the name is made only then)
const DTIME = Ref(false)
const DEVS = Tuple{Int,String,CuEvent,CuEvent}[]
function tm(f, name)
    DTIME[] || return f()
    e0 = CuEvent(); e1 = CuEvent()
    CUDA.record(e0, stream()); r = f(); CUDA.record(e1, stream())
    push!(DEVS, (CUDA.deviceid(CUDA.device()), name(), e0, e1))
    r
end
dnorm!(y, x, w, W, nv) = tm(() -> "norm W$W") do
    launch(k_rmsnorm!, y, x, w, Int32(W), Int32(0), EPS; threads = 256, blocks = nv)
end
function bench_draft(E::EngineB1)
    E.draft === nothing && (E.draft = load_draft(E))
    Dr = E.draft
    Dr.context = 64; draft_propose(Dr, E, 1000); draft_append!(Dr, 64, 1)   # warm: first launches load each kernel
    # (CUDA.@profile, CUPTI 11.8 under the 12.8 compat driver, hung the server here)
    empty!(DEVS); DTIME[] = true; t0 = time()
    for _ in 1:3
        Dr.context = 64; draft_propose(Dr, E, 1000); draft_append!(Dr, 64, 1)
    end
    DTIME[] = false
    for DG in E.groups, D in DG
        device!(D.rt.ord); synchronize()
    end
    wall = (time() - t0) / 3
    busy = Dict{String,Float64}(); gap = Dict{String,Float64}(); prev = Dict{Int,CuEvent}()
    for (dev, nm, e0, e1) in DEVS
        device!(dev)
        busy[nm] = get(busy, nm, 0.0) + CUDA.elapsed(e0, e1)
        haskey(prev, dev) && (gap[nm] = get(gap, nm, 0.0) + CUDA.elapsed(prev[dev], e0))   # idle before the op
        prev[dev] = e1
    end
    tb = sum(values(busy)); tg = sum(values(gap))
    for (k, v) in sort(collect(busy); by = last, rev = true)
        @printf("  %-34s %8.3f ms device, %7.3f ms idle before (per round, %4.1f%%)\n", k, v / 3 * 1e3, get(gap, k, 0.0) / 3 * 1e3, 100v / tb)
    end
    @printf("draft round (proposal + append, direct launches): %.2f ms on the devices, %.2f ms idle between ops, wall %.2f ms\n",
        tb / 3 * 1e3, tg / 3 * 1e3, wall * 1e3)
    empty!(DEVS)
end
el(n) = (threads = 256, blocks = cld(n, 256))
# the position the draft kernels read, set in stream order before them
setpos!(Dr::Draft, v::Integer) = CUC.cuMemsetD32Async(reinterpret(CUC.CUdeviceptr, Dr.pos), UInt32(v), 1, stream())
const DGRAPH = get(ENV, "FLASHNEXT_DGRAPH", "1") == "1"
# f's launches on the draft die: run directly on first use (that loads the kernels), captured into a CUDA graph on the
# second and relaunched from then on (one launch, not ~130; nothing in them changes: the position is Dr.pos)
function dgraph!(f, Dr::Draft, key::Int)
    u = get(Dr.uses, key, 0); Dr.uses[key] = u + 1
    (!DGRAPH || DTIME[] || u == 0) && return f()
    ex = key == 0 ? Dr.gprop : get(Dr.gapp, key, nothing)
    if ex === nothing
        ex = CUC.instantiate(CUC.capture(f))
        key == 0 ? (Dr.gprop = ex) : (Dr.gapp[key] = ex)
    end
    CUC.launch(ex)
end
# the context: the target's features of positions start .. start + n - 1 into each layer's K/V cache
function draft_append!(Dr::Draft, start::Integer, n::Integer)
    t0 = time()
    device!(DRAFT_DIE); b = Dr.b
    setpos!(Dr, start)
    dgraph!(Dr, n) do
        dlaunch(k_dfeat!, b[:feat], Dr.tap, Dr.pos, Int32(n); el(n * 12800)...)
        for (s, m) in enumerate(Dr.fc)
            gemv4!(b[:ctxh], m, b[:feat], n; ldx = 12800, xoff = 2560 * (s - 1), accum = s > 1)
        end
        dnorm!(b[:ctxh], b[:ctxh], Dr.enc_norm, 2560, n)
        for (l, W) in enumerate(Dr.L)
            gemv4!(b[:k], W.k, b[:ctxh], n; ldx = 2560); gemv4!(b[:v], W.v, b[:ctxh], n; ldx = 2560)
            dnorm!(b[:k], b[:k], W.kn, 128, 8n)
            dlaunch(k_drope!, b[:k], Dr.freq, Int32(8), Dr.pos, Int32(n); el(n * 8 * 64)...)
            dlaunch(k_dcache!, Dr.kc[l], Dr.vc[l], b[:k], b[:v], Dr.pos, Int32(n); el(n * 1024)...)
        end
    end
    Dr.context = start + n
    Dr.ms[:append] = get(Dr.ms, :append, 0.0) + time() - t0
end
# the proposals after anchor (at position Dr.context): [anchor, d1, d2, d3]
# The draft's head slices. FLASHNEXT_DHEAD=board (the default): on the draft's board only, each of its dies a copy of
# its share of output.weight (built on first use). The other boards run the early anchor (FLASHNEXT_SPECEARLY) during
# the draft, and their persistent kernels fill the SMs: a head slice queued there waited for it (draft head 7.8 ->
# 25.5 ms). =all: the verify's slices on every die.
# (die, slice of output.weight, rows, first row, hx, logits, top 16s): one head slice of the draft. The caches are
# Dict{Any,Any}: what the request path reads from them is asserted to its type, so its calls are static (the juliac
# executable compiled each one at its first use)
const DHeadSlice = Tuple{Int,DevMat,Int,Int,FPtr,FPtr,FPtr}
function draft_heads(Dr::Draft, E::EngineB1)::Vector{DHeadSlice}
    get(ENV, "FLASHNEXT_DHEAD", "board") == "all" && return DHeadSlice[h for h in Dr.heads]
    get!(KCACHE, :dheads_board) do
        # inhead: the target's own head slices, the last group's dies (they hold the whole vocabulary between them,
        # whichever group the draft is on: no second copy of output.weight); else a copy split over the draft's group
        DG = inhead(E) ? E.groups[end] : only([DG for DG in E.groups if any(D -> D.rt.ord == DRAFT_DIE, DG)])
        nd = length(DG); per = (248320 ÷ nd) ÷ 16 * 16
        t = E.g.tensors["output.weight"]
        hs = DHeadSlice[]
        for (j, D) in enumerate(DG)
            device!(D.rt.ord)
            if inhead(E)
                H = D.rt.head::DHead; m = H.out; r0 = H.r0; rows = Int(m.rows)
            else
                r0 = (j - 1) * per; rows = j == nd ? 248320 - r0 : per
                A = Arena(); put_mat!(A, E.g, [(t, r0, rows)])
                A = Arena(rawalloc(A.off), A.off); m = put_mat!(A, E.g, [(t, r0, rows)])
            end
            hx = CUDA.zeros(Float32, (DBLK - 1) * 2560); lg = CUDA.zeros(Float32, (DBLK - 1) * rows); top = CUDA.zeros(Float32, (DBLK - 1) * 32 * TOPCH)
            push!(Dr.keep, hx, lg, top)
            push!(hs, (D.rt.ord, m, rows, r0, fptr(hx), fptr(lg), fptr(top)))
        end
        s = sort([(h[4], h[3]) for h in hs])          # (first row, rows): contiguous over all 248320 rows
        (s[1][1] == 0 && all(s[j][1] + s[j][2] == s[j+1][1] for j in 1:length(s)-1) && s[end][1] + s[end][2] == 248320) ||
            error("draft: head slices do not cover the vocabulary: $s")
        note(@sprintf("draft: head slices on dies %s", join([h[1] for h in hs], "+")))
        hs
    end
end
function draft_propose(Dr::Draft, E::EngineB1, anchor::Integer)
    t0 = time()
    hs = draft_heads(Dr, E)
    device!(DRAFT_DIE); b = Dr.b; n = DBLK; nd = DBLK - 1
    ul!(b[:x], vcat(embed(E.g, anchor), repeat(embed(E.g, Dr.mask), nd)))
    setpos!(Dr, Dr.context)
    dgraph!(Dr, 0) do
        for (l, W) in enumerate(Dr.L)
            dnorm!(b[:nrm], b[:x], W.attn_norm, 2560, n)
            gemv4!(b[:coeff], W.attn_cp, b[:nrm], n)
            dlaunch(k_dconv!, b[:inp], b[:nrm], b[:coeff], W.attn_cb, Int32(0), Int32(n); el(n * 2560)...)
            gemv4!(b[:q], W.q, b[:inp], n); gemv4!(b[:k], W.k, b[:inp], n); gemv4!(b[:v], W.v, b[:inp], n)
            dnorm!(b[:q], b[:q], W.qn, 128, 32n); dnorm!(b[:k], b[:k], W.kn, 128, 8n)
            dlaunch(k_drope!, b[:q], Dr.freq, Int32(32), Dr.pos, Int32(n); el(n * 32 * 64)...)
            dlaunch(k_drope!, b[:k], Dr.freq, Int32(8), Dr.pos, Int32(n); el(n * 8 * 64)...)
            dlaunch(k_dattn!, b[:att], b[:q], b[:k], b[:v], Dr.kc[l], Dr.vc[l], Dr.pos, Int32(n); threads = 32 * DATTNW, blocks = 32n)
            ko = 0
            for (s, m) in enumerate(W.o)
                gemv4!(b[:prj], m, b[:att], n; ldx = 4096, xoff = ko, accum = s > 1); ko += Int(m.K)
            end
            dlaunch(k_dconv!, b[:tmp], b[:prj], b[:coeff], W.attn_cb, Int32(1), Int32(n); el(n * 2560)...)
            dlaunch(k_dadd!, b[:x], b[:tmp], Int32(n * 2560); el(n * 2560)...)
            dnorm!(b[:nrm], b[:x], W.ffn_norm, 2560, n)
            gemv4!(b[:coeff], W.ffn_cp, b[:nrm], n)
            dlaunch(k_dconv!, b[:inp], b[:nrm], b[:coeff], W.ffn_cb, Int32(0), Int32(n); el(n * 2560)...)
            gemv4!(b[:gu], W.gate, b[:inp], n)                  # [gate | up] per position
            dlaunch(k_dswiglu2!, b[:gate], b[:gu], Int32(7680), Int32(n); el(n * 7680)...)
            for (s, m) in enumerate(W.down)
                gemv4!(b[:prj], m, b[:gate], n; ldx = 7680, xoff = 2560 * (s - 1), accum = s > 1)
            end
            dlaunch(k_dconv!, b[:tmp], b[:prj], b[:coeff], W.ffn_cb, Int32(1), Int32(n); el(n * 2560)...)
            dlaunch(k_dadd!, b[:x], b[:tmp], Int32(n * 2560); el(n * 2560)...)
        end
        dnorm!(b[:x], b[:x], Dr.out_norm, 2560, n)
        gemv4!(b[:sel], Dr.sel, b[:x], n)
    end
    # the hidden rows of positions 1 .. nd to each head slice's die: peer copies after an event on the draft's stream
    # (not through the host)
    ev = get!(() -> CuEvent(CUDA.EVENT_DISABLE_TIMING), KCACHE, (:draftev, DRAFT_DIE))::CuEvent
    CUDA.record(ev, stream())
    t1 = time()
    # the target head's top 16 at positions 1 .. nd, from every head slice
    for (ord, out, rows, r0, hx, lg, top) in hs
        device!(ord)
        CUC.cuStreamWaitEvent(stream(), ev, 0)
        CUC.cuMemcpyDtoDAsync_v2(reinterpret(CUC.CUdeviceptr, hx), reinterpret(CUC.CUdeviceptr, b[:x] + 4 * 2560), 4 * nd * 2560, stream())
        gemv4!(lg, out, hx, nd; ldx = 2560, ldy = rows)
        dlaunch(k_top16!, top, lg, Int32(rows), Int32(rows), Int32(r0), Int32(TOPCH); threads = 256, blocks = nd * TOPCH)
    end
    device!(DRAFT_DIE); sel = dl(b[:sel], n * 256)        # (meanwhile the heads run)
    # every head die's top 16s into pinned memory asynchronously, then one wait per die
    nt = nd * 32 * TOPCH
    pinp = get!(KCACHE, :draftpin) do
        hb = CUDA.alloc(CUDA.HostMemory, 4 * nt * 8, CUDA.MEMHOSTALLOC_PORTABLE)
        (hb, convert(Ptr{Float32}, hb))
    end[2]::Ptr{Float32}
    for (j, (ord, out, rows, r0, hx, lg, top)) in enumerate(hs)
        device!(ord); CUC.cuMemcpyDtoHAsync_v2(pinp + 4 * nt * (j - 1), reinterpret(CUC.CUdeviceptr, top), 4 * nt, stream())
    end
    cand = [Tuple{Float32,Int32}[] for _ in 1:nd]
    for (j, (ord, out, rows, r0, hx, lg, top)) in enumerate(hs)
        device!(ord); synchronize()
        t = unsafe_wrap(Array, pinp + 4 * nt * (j - 1), nt)
        for p in 1:nd, ch in 0:TOPCH-1, k in 0:15
            o = ((p - 1) * TOPCH + ch) * 32
            push!(cand[p], (t[o+2k+1], reinterpret(Int32, t[o+2k+2])))
        end
    end
    t2 = time()
    # the selector walk: score = logit + sum_r pred[previous][r] sel[p][r] succ[candidate][r]
    toks = Int32[anchor]
    pr = zeros(Float32, 256); su = zeros(Float32, 256)
    for p in 1:nd
        cs = sort(cand[p]; by = c -> (-c[1], c[2]))[1:16]
        GC.@preserve Dr dequant_row!(pr, Dr.pred.typ, pointer(Dr.predb) + Int(toks[end]) * rowbytes(Dr.pred), 256)
        best = -Inf; bi = Int32(0)
        for (v, id) in cs
            GC.@preserve Dr dequant_row!(su, Dr.succ.typ, pointer(Dr.succb) + Int(id) * rowbytes(Dr.succ), 256)
            s = Float64(v) + sum(Float64(pr[r]) * sel[p*256+r] * su[r] for r in 1:256)
            s > best && (best = s; bi = id)
        end
        push!(toks, bi)
    end
    Dr.ms[:backbone] = get(Dr.ms, :backbone, 0.0) + t1 - t0; Dr.ms[:head] = get(Dr.ms, :head, 0.0) + t2 - t1
    Dr.ms[:select] = get(Dr.ms, :select, 0.0) + time() - t2; Dr.ms[:n] = get(Dr.ms, :n, 0.0) + 1
    toks
end

# ---- pipelining: the tokens of a known sequence (a prompt, a round of drafts) launched back to back. Each die runs
# them in its stream's order; the boards hand S over per position slot (t & 3), so board k + 1 works on position t
# while board k works on t + 1. A slot is reused by position t + 4 only after t is done on the last board (and the
# draft has taken t's taps: slot t & 7 of the tap buffer is reused by t + 8). ----
const PIPEEV = Dict{Tuple{Int,Symbol,Int},Any}()
pipe_ev(ord, kind, s) = get!(PIPEEV, (ord, kind, s)) do
    device!(ord); CuEvent(CUDA.EVENT_DISABLE_TIMING)
end::CuEvent
const pipe_used = Set{Tuple{Int,Symbol,Int}}()
# the dies holding the placement's last block
lastdies(E::EngineB1) = [D for DG in E.groups for D in DG if !isempty(D.rt.blks) && maximum(B.l for B in D.rt.blks) == lastblock(E)]
function launch_pipelined!(E::EngineB1, id::Integer; Dr = nothing)
    t = length(E.ids); s = t & 3
    for D in lastdies(E)
        (D.rt.ord, :tok, s) in pipe_used && CUDA.synchronize(pipe_ev(D.rt.ord, :tok, s))
    end
    Dr !== nothing && (DRAFT_DIE, :app, t & 7) in pipe_used && CUDA.synchronize(pipe_ev(DRAFT_DIE, :app, t & 7))
    seq0, t = launch_token!(E, id)
    for D in lastdies(E)
        device!(D.rt.ord); CUDA.record(pipe_ev(D.rt.ord, :tok, s), stream()); push!(pipe_used, (D.rt.ord, :tok, s))
    end
    if Dr !== nothing                  # the draft's context: position t's taps once every die is done with t
        draft_wait_dies!(E)
        draft_append!(Dr, t, 1)
        device!(DRAFT_DIE); CUDA.record(pipe_ev(DRAFT_DIE, :app, t & 7), stream()); push!(pipe_used, (DRAFT_DIE, :app, t & 7))
    end
    seq0, t
end
# A round's first position (the anchor) needs nothing from the draft: launched before the draft runs, on every board
# but the draft's (there a waiting persistent kernel would hold the SMs the draft needs), so the boards before
# work on it during the draft. The draft board's kernels follow the draft (finish_anchor!).
function launch_anchor_early!(E::EngineB1, id::Integer)
    t = length(E.ids); s = t & 3
    for D in lastdies(E)
        (D.rt.ord, :tok, s) in pipe_used && CUDA.synchronize(pipe_ev(D.rt.ord, :tok, s))
    end
    defer = [D.rt.ord for DG in E.groups if any(D -> D.rt.ord == DRAFT_DIE, DG) for D in DG]
    seq0, t = launch_token!(E, id; defer)
    (seq0, t, defer)
end
function finish_anchor!(E::EngineB1, early)
    seq0, t, defer = early; s = t & 3
    for DG in E.groups, D in DG
        D.rt.ord in defer && D.nblk > 0 && launch_die!(E, D, seq0, t)
    end
    for D in lastdies(E)
        device!(D.rt.ord); CUDA.record(pipe_ev(D.rt.ord, :tok, s), stream()); push!(pipe_used, (D.rt.ord, :tok, s))
    end
    seq0
end
# ---- early stop (FLASHNEXT_EARLYSTOP=1): the round's positions are decided one at a time as each leaves the last
# board, and the positions after the first rejected draft are stopped instead of verified ----
# Stop the positions of seq numbers [lo, hi) on every die: the window into each die's ABW by a copy on a stream of its
# own (the kernels in flight read it); then, behind them in each die's stream, the phase and reduction counters zeroed
# (a stopped position leaves them partly counted)
llreg(ord) = (KCACHE[:llreg]::Dict{Int,Any})[ord]::Tuple{U64P,Int}     # a die's LL region (words, count)
function abort_positions!(E::EngineB1, lo::UInt32, hi::UInt32)
    nmax = 0
    for DG in E.groups, D in DG
        nmax = max(nmax, llreg(D.rt.ord)[2])
    end
    pinp = get!(KCACHE, :abwpin) do
        hb = CUDA.alloc(CUDA.HostMemory, 8 + 8 * nmax, CUDA.MEMHOSTALLOC_PORTABLE); p = convert(Ptr{UInt32}, hb)
        mark = UInt64(LLMARK) << 32
        for i in 1:nmax
            unsafe_store!(Ptr{UInt64}(p + 8), mark, i)
        end
        (hb, p)
    end[2]::Ptr{UInt32}
    unsafe_store!(pinp, lo, 1); unsafe_store!(pinp, hi, 2)
    # on each die's own stream (the copy engine, beside the kernels): the window, then the marked LL words from the
    # host (a same-die copy from a marked buffer needs the SMs the waiting kernels hold: it hung)
    pend = get!(() -> Set{Int}(), KCACHE, :abwpend)::Set{Int}
    for DG in E.groups, D in DG
        device!(D.rt.ord)
        s = get!(() -> CuStream(; flags = CUDA.STREAM_NON_BLOCKING), KCACHE, (:abwstream, D.rt.ord))::CuStream
        CUC.cuMemcpyHtoDAsync_v2(reinterpret(CUC.CUdeviceptr, D.ll[:ABW]), pinp, 8, s)
        r, n = llreg(D.rt.ord)
        CUC.cuMemcpyHtoDAsync_v2(reinterpret(CUC.CUdeviceptr, r), pinp + 8, 8 * n, s)
        ev = get!(() -> CuEvent(CUDA.EVENT_DISABLE_TIMING), KCACHE, (:abwev, D.rt.ord))::CuEvent; CUDA.record(ev, s)
        push!(pend, D.rt.ord)
    end
end
# before a die's next k_token! after a stop (launch_die!): behind the stopped kernels and the marking, the LL words
# and the phase and reduction counters zeroed (a stopped position leaves them partly counted). Here, not at the stop:
# the draft on die 6 comes first in the next round and need not wait for the marking, 1 MB over board 3's PCH Gen2 x2
# link (draft backbone 0.8 -> 2.1 ms when it did)
function abort_settle!(D::B1Die)
    pend = get(KCACHE, :abwpend, nothing)::Union{Nothing,Set{Int}}
    (pend === nothing || !(D.rt.ord in pend)) && return
    delete!(pend, D.rt.ord)
    CUC.cuStreamWaitEvent(stream(), KCACHE[(:abwev, D.rt.ord)]::CuEvent, 0)
    r, n = llreg(D.rt.ord)
    CUC.cuMemsetD32Async(reinterpret(CUC.CUdeviceptr, r), UInt32(0), 2 * n, stream())
    D.nblk == 0 && return
    CUC.cuMemsetD32Async(reinterpret(CUC.CUdeviceptr, D.ctr), UInt32(0), 32 * D.nblk, stream())
    CUC.cuMemsetD32Async(reinterpret(CUC.CUdeviceptr, D.cnt), UInt32(0), 80, stream())
end
# head slices for the early decisions: output.weight split over the collapse die's board (done with its share of the
# round when a position leaves the last board), each die a copy of its share (built on first use)
function early_heads(E::EngineB1)
    get!(KCACHE, :earlyhead) do
        c, _ = collapse_target(E)
        DG = only([DG for DG in E.groups if any(D -> D === c, DG)])
        nd = length(DG); per = (248320 ÷ nd) ÷ 16 * 16
        t = E.g.tensors["output.weight"]
        hs = EHeadSlice[]
        for (j, D) in enumerate(DG)
            device!(D.rt.ord)
            r0 = (j - 1) * per; rows = j == nd ? 248320 - r0 : per
            A = Arena(); put_mat!(A, E.g, [(t, r0, rows)])
            A = Arena(rawalloc(A.off), A.off); m = put_mat!(A, E.g, [(t, r0, rows)])
            a = (CUDA.zeros(Float32, 2560), CUDA.zeros(Float32, rows), CUDA.zeros(Float32, 8), CUDA.zeros(UInt64, 2560))
            push!(get!(() -> Any[], KCACHE, :keep)::Vector{Any}, a)
            push!(hs, (D, m, rows, r0, fptr(pointer(a[1])), fptr(pointer(a[2])), fptr(pointer(a[3])), reinterpret(U64P, pointer(a[4]))))
        end
        note(@sprintf("early stop: head slices on dies %s", join([h[1].rt.ord for h in hs], "+")))
        hs
    end::Vector{EHeadSlice}
end
# (die, slice of output.weight, rows, first row, hx, logits, argmax, the row's LL words): one early-stop head slice
const EHeadSlice = Tuple{B1Die,DevMat,Int,Int,FPtr,FPtr,FPtr,U64P}
# wait for the given dies' streams, at most 10 s (as hangguard: then the busy ones are named and the server exits,
# a hung round is not recoverable in the process; a plain synchronize there waited forever)
function waitdone(ords; what = "", limit = 10.0)
    t0 = time()
    while true
        busy = [o for o in ords if (device!(o); !CUDA.isdone(stream()))]
        isempty(busy) && return
        if time() - t0 > limit
            println(stderr, "flashnext: ", what, ": dies ", join(busy, ", "), " still busy after 10 s (hang)", SERVING[] ? "; the server exits" : "")
            flush(stderr)
            SERVING[] && ccall(:_exit, Cvoid, (Cint,), 3)
            error("B1 kernel hang")
        end
    end
end
# the target's argmax after position k of the round (its collapsed row in hz4 slot k on the collapse die c, queued):
# the row to the other early-head dies over LL, each slice's logits and argmax, read back; waits for them
function early_argmax!(E::EngineB1, c::B1Die, k::Int, sq::UInt32, hz4p::FPtr)
    hs = early_heads(E)
    pinp = get!(KCACHE, :earlypin) do
        hb = CUDA.alloc(CUDA.HostMemory, 4 * 8 * 8, CUDA.MEMHOSTALLOC_PORTABLE); (hb, convert(Ptr{Float32}, hb))
    end[2]::Ptr{Float32}
    src = hz4p + 4 * 2560 * (k - 1); sqe = sq + UInt32(5)
    others = [h for h in hs if h[1] !== c]
    if !isempty(others)
        device!(c.rt.ord)
        dst = ntuple(i -> i <= length(others) ? others[i][8] : U64P(0), 8)
        launch(k_put_ll!, dst, Int32(length(others)), src, Int32(2560), sqe; threads = 256, blocks = 10)
    end
    for (j, (D, m, rows, r0, hx, lg, am, hzll)) in enumerate(hs)
        device!(D.rt.ord)
        if D === c
            hx = src
        else
            launch(k_ll2f!, hx, hzll, Int32(2560), sqe; threads = 256, blocks = 10)
        end
        gemv4!(lg, m, hx, 1; ldx = 2560, ldy = rows)
        launch(k_argmax!, am, lg, Int32(rows), Int32(r0); threads = 1024, blocks = 1)
        CUC.cuMemcpyDtoHAsync_v2(pinp + 4 * 8 * (j - 1), reinterpret(CUC.CUdeviceptr, am), 4 * 2, stream())
    end
    best = -Inf32; out = Int32(0)
    waitdone([h[1].rt.ord for h in hs]; what = "early-stop head of position $k")
    for (j, h) in enumerate(hs)
        v = unsafe_load(pinp, 8 * (j - 1) + 1); i = reinterpret(Int32, unsafe_load(pinp, 8 * (j - 1) + 2))
        (v > best || (v == best && i < out)) && (best = v; out = i)
    end
    out
end
# P <= 4 known tokens at positions t .. t + P - 1, pipelined; the target's argmax after each (the collapse of each
# position on the last die as its S arrives, then every head slice's logits for the P positions at once)
function verify_round!(E::EngineB1, toks::Vector{Int32}; early = nothing)
    inhead(E) && return verify_round_inhead!(E, toks; early)
    VEARLY = get(ENV, "FLASHNEXT_VEARLY", "1") == "1"         # read per call: a request's FLASHNEXT_VEARLY=0 sets it
    P = length(toks); t0 = early === nothing ? length(E.ids) : early[2]
    c, H = collapse_target(E); b = c.rt.b
    # a collapse die off the last board runs every position's collapse after its own share of the round (queued
    # before its next block kernels it would wait there for the last board and stall its own board)
    VEARLY &= c in E.groups[end]
    # FLASHNEXT_EARLYSTOP=1 (read per call): decide position by position and stop after a rejected draft (needs the
    # collapse off the last board, FLASHNEXT_COLLAPSE_DIE at the server's start)
    EARLY = get(ENV, "FLASHNEXT_EARLYSTOP", "1") == "1" && !(c in E.groups[end])
    lastl = lastblock(E)
    vh = get!(() -> Dict{Int,Any}(), KCACHE, :verifyhead8)::Dict{Int,Any}           # (buffers for P8MAX positions)
    hz4 = get!(() -> (device!(c.rt.ord); CUDA.zeros(Float32, P8MAX * 2560)), vh, -1)
    hz4p = get!(() -> (device!(c.rt.ord); fptr(pointer(hz4))), vh, -2)::FPtr
    # position k's collapse on the head's die, into hz4 slot k (VEARLY: launched right after position k's kernels,
    # so it runs between them on that die, not all at the end)
    function collapse!(k, sq)
        device!(c.rt.ord)
        t = t0 + k - 1
        launch(k_ll2f!, b[:Sfull], fullS(c, t), Int32(10240), sqof(sq, lastl, 16); threads = 256, blocks = 40)
        launch(k_rmsnorm!, b[:xh], b[:Sfull], H.norm, Int32(2560), Int32(2560), EPS; threads = 256, blocks = 4)
        gemv!(b[:down], H.down, b[:xh])
        launch(k_hc_silu!, b[:ht], b[:down], Int32(320); threads = 320, blocks = 1)
        gemv!(b[:hup], H.up, b[:ht])
        launch(k_hc_mix!, b[:hz], b[:hup], b[:xh]; threads = 256, blocks = 10)
        CUC.cuMemcpyDtoDAsync_v2(reinterpret(CUC.CUdeviceptr, hz4p) + 4 * 2560 * (k - 1), reinterpret(CUC.CUdeviceptr, b[:hz]), 4 * 2560, stream())
    end
    seqs = UInt32[]
    # FLASHNEXT_VPROF=1: a timing event after each position's kernel on every die (printed after the round)
    vprof = get(ENV, "FLASHNEXT_VPROF", "0") == "1"
    evs = Dict{Tuple{Int,Int},CuEvent}(); tv0 = time()
    if vprof
        for DG in E.groups, D in DG
            device!(D.rt.ord); e = CuEvent(); CUDA.record(e, stream()); evs[(D.rt.ord, 0)] = e
        end
    end
    for (k, id) in enumerate(toks)
        push!(seqs, k == 1 && early !== nothing ? finish_anchor!(E, early) : first(launch_pipelined!(E, id)))
        if vprof
            for DG in E.groups, D in DG
                device!(D.rt.ord); e = CuEvent(); CUDA.record(e, stream()); evs[(D.rt.ord, k)] = e
            end
        end
        VEARLY && !EARLY && collapse!(k, seqs[k])
    end
    if EARLY
        # one position at a time as it leaves the last board: its collapse and argmax; the positions after the first
        # rejected draft are stopped (their kernels leave at the next block start), not verified
        outs = Int32[]
        for k in 1:P
            collapse!(k, seqs[k])
            a = early_argmax!(E, c, k, seqs[k], hz4p)
            push!(outs, a)
            if k < P && toks[k+1] != a
                abort_positions!(E, seqs[k+1], seqs[P] + UInt32(2048))
                break
            end
        end
        return outs
    end
    VEARLY || foreach(k -> collapse!(k, seqs[k]), 1:P)
    # (peer copies to the head dies after an event, not through the host, measured slower: verify 60.3 -> 66.0 ms)
    # The collapsed rows through pinned host memory, the copies to and from every head die asynchronous: queued on
    # all of them at once (synchronous copies, one die after another, started the last head 0.5 ms after the first)
    pin = get!(KCACHE, :verifypin) do
        hb = CUDA.alloc(CUDA.HostMemory, 4 * (P8MAX * 2560 + 2P8MAX * 8), CUDA.MEMHOSTALLOC_PORTABLE)
        (hb, convert(Ptr{Float32}, hb))
    end
    hzp = pin[2]; amp = hzp + 4 * P8MAX * 2560
    tq0 = time()
    # the head stage queued now, behind each die's kernels of the round: the collapse's die writes the collapsed
    # rows into an LL buffer on every other head die (k_put_ll!), which turns them back into floats as they arrive
    # (k_ll2f! waits on their seq), then its slice's logits and argmax, read back into pinned memory. No host wait
    # and no device-to-device copy calls in between (a peer cuMemcpy serializes with all work on both dies).
    sqv = seqs[P] + UInt32(4)
    hds = [D for DG in E.groups for D in DG if D.rt.head !== nothing]
    # (raw pointers, taken once on each array's own die: pointer() of a CuArray last used on another die's stream
    # synchronizes that stream, which here waited out the whole round)
    for D in hds
        rows = Int(D.rt.head.out.rows)
        get!(vh, D.rt.ord) do
            device!(D.rt.ord)
            a = (CUDA.zeros(Float32, P8MAX * 2560), CUDA.zeros(Float32, P8MAX * rows), CUDA.zeros(Float32, 2P8MAX), CUDA.zeros(UInt64, P8MAX * 2560))
            (a, fptr(pointer(a[1])), fptr(pointer(a[2])), fptr(pointer(a[3])), reinterpret(U64P, pointer(a[4])))
        end
    end
    others =[D for D in hds if D !== c]
    device!(c.rt.ord)
    dst = ntuple(k -> k <= length(others) ? vh[others[k].rt.ord][5] : U64P(0), 8)
    launch(k_put_ll!, dst, Int32(length(others)), hz4p, Int32(P * 2560), sqv; threads = 256, blocks = 40)
    tq1 = tq0; tq2 = tq0
    outs = fill(Int32(0), P); best = fill(-Inf32, P)
    heads = Int[]
    for D in hds
        rt = D.rt
        device!(rt.ord); rows = Int(rt.head.out.rows)
        _, hx, lg, am, hzll = vh[rt.ord]
        if D === c
            hx = hz4p
        else
            launch(k_ll2f!, hx, hzll, Int32(P * 2560), sqv; threads = 256, blocks = 40)
        end
        gemv4!(lg, rt.head.out, hx, P; ldx = 2560, ldy = rows)
        for p in 0:P-1
            launch(k_argmax!, am + 8p, lg + 4 * p * rows, Int32(rows), Int32(rt.head.r0); threads = 1024, blocks = 1)
        end
        CUC.cuMemcpyDtoHAsync_v2(amp + 4 * 2P8MAX * length(heads), reinterpret(CUC.CUdeviceptr, am), 4 * 2P8MAX, stream())
        push!(heads, rt.ord)
    end
    tq3 = time()
    SERVING[] && hangguard(E)
    for (j, o) in enumerate(heads)
        device!(o); synchronize()
        for p in 1:P
            v = unsafe_load(amp, 2P8MAX * (j - 1) + 2p - 1); i = reinterpret(Int32, unsafe_load(amp, 2P8MAX * (j - 1) + 2p))
            (v > best[p] || (v == best[p] && i < outs[p])) && (best[p] = v; outs[p] = i)
        end
    end
    tq4 = time()
    if vprof
        for DG in E.groups, D in DG          # (each die from its own start event: those were recorded on idle streams together)
            o = D.rt.ord; device!(o)
            ends = [CUDA.elapsed(evs[(o, 0)], evs[(o, k)]) * 1e3 for k in 1:P]
            @printf("vprof die %d: position ends %s ms (periods %s)\n", o, join([@sprintf("%.2f", x) for x in ends], " "),
                join([@sprintf("%.2f", ends[k] - ends[k-1]) for k in 2:P], " "))
        end
        @printf("vprof host: launches %.2f ms, then collapse wait %.2f, rows to host %.2f, head launches %.2f, heads %.2f\n",
            (tq0 - tv0) * 1e3, (tq1 - tq0) * 1e3, (tq2 - tq1) * 1e3, (tq3 - tq2) * 1e3, (tq4 - tq3) * 1e3)
    end
    outs
end
# speculative decoding: DFlash2 drafts 3 tokens after the anchor; the target verifies the 4 positions pipelined;
# the matched drafts and the target's next token are kept (the target's output stays its own greedy output)
function generate_spec(E::EngineB1, ids::Vector{Int32}, n::Int; echo = true)
    E.draft === nothing && (E.draft = load_draft(E))
    Dr = E.draft::Draft; Dr.context = 0; empty!(Dr.ms)        # (the field is Any: asserted, the draft's calls are static)
    empty!(pipe_used)
    tp = time()
    for id in ids[1:end-1]
        launch_pipelined!(E, id; Dr)
    end
    SERVING[] && hangguard(E)
    for D in lastdies(E); device!(D.rt.ord); synchronize(); end
    device!(DRAFT_DIE); synchronize()
    tpp = time() - tp
    out = Int32[]; hist = zeros(Int, P8MAX); anchor = ids[end]
    td = time(); tv = 0.0
    while length(out) < n
        t = length(E.ids)
        # the anchor's target pass starts now, beside the draft (FLASHNEXT_SPECEARLY=0: after it)
        early = get(ENV, "FLASHNEXT_SPECEARLY", "1") == "1" && !inhead(E) ? launch_anchor_early!(E, anchor) : nothing
        props = draft_propose(Dr, E, anchor)
        # FLASHNEXT_SPECP (read per call, default the draft's block): the round's positions, the anchor and SPECP - 1
        # drafts
        SP = clamp(parse(Int, get(ENV, "FLASHNEXT_SPECP", string(DBLK))), 1, length(props)); props = props[1:SP]
        t1 = time()
        outs = verify_round!(E, props; early)
        tv += time() - t1
        m = 0
        while m < SP - 1 && props[m+2] == outs[m+1]
            m += 1
        end
        hist[m+1] += 1
        resize!(E.ids, t + m + 1)                  # the accepted inputs: the anchor, then the matched drafts
        draft_wait_dies!(E)
        draft_append!(Dr, t, m + 1)
        stop = false
        for tok in vcat(props[2:m+1], outs[m+1])
            push!(out, tok)
            echo && !hidden(E.tk, tok) && (write(stdout, piece(E.tk, tok)); flush(stdout))
            (tok == E.tk.eos || length(out) >= n) && (stop = true; break)
        end
        stop && break
        anchor = outs[m+1]
    end
    tdd = time() - td
    echo && println()
    r = sum(hist); acc = sum(k * hist[k] for k in 1:P8MAX) / max(1, r); k = 1e3 / max(1, Dr.ms[:n])
    pp = (length(ids) - 1) / tpp; tg = length(out) / tdd
    @printf("prompt %d tokens in %.2f s (%.2f tok/s, pipelined); generated %d tokens in %.2f s (%.2f tok/s, speculative)\n", length(ids), tpp, pp, length(out), tdd, tg)
    @printf("spec: %d rounds, accepted %s: %s, mean %.3f; per round %.1f ms (verify %.1f, draft %.2f: backbone %.2f, head %.2f, select %.2f; append %.2f)\n",
        r, join(1:P8MAX, "/"), join(hist, "/"), acc, tdd / max(1, r) * 1e3, tv / max(1, r) * 1e3, (Dr.ms[:backbone] + Dr.ms[:head] + Dr.ms[:select]) * k,
        Dr.ms[:backbone] * k, Dr.ms[:head] * k, Dr.ms[:select] * k, get(Dr.ms, :append, 0.0) * k)
    metric("pp.tokens_s", "tokens/s", pp; tokens = length(ids), mode = "pipelined")
    metric("tg.tokens_s", "tokens/s", tg; tokens = length(out), mode = "spec", reference_flash_sass = 40.3)
    metric("spec.accepted_mean", "tokens", acc; rounds = r, h1 = hist[1], h2 = hist[2], h3 = hist[3], h4 = hist[4],
        h5 = hist[5], h6 = hist[6], h7 = hist[7], h8 = hist[8])
    get(ENV, "FLASHNEXT_B8PROF", "0") == "1" && b8_report()
    out
end
# ---- the B4 engine (host): the B1 engine's weights and heads, its own LL buffers (4 position slots), recurrent
# state copies, conv/PLE rings and KV caches; one k_token4! per die per round of P <= 4 positions ----
mutable struct B4Die
    D::B1Die
    ll::Dict{Symbol,U64P}
    keep::Vector{Any}
    dd::Any; bd::Any
    nblk::Int
    head::Any                         # (hz4, logits4, am4) on a die with a head slice
    tim::Ptr{UInt64}                  # CTA 0's op-end stamps (host-mapped): [16 block + op]
end
mutable struct EngineB4
    E::EngineB1
    groups::Vector{Vector{B4Die}}
    host::Any; emb::Ptr{UInt64}; pleg::Ptr{UInt64}
    kern::Dict{Int,Any}
    xsm::Int
    rounds::Int                       # rounds since the last report
    seqn::Int                         # rounds ever launched: seq0 = 2048 seqn (LL words of earlier rounds never match)
    bank::Int32
    ids::Vector{Int32}                # the accepted tokens (positions 0 .. length - 1)
end
llsizes4(n) = (nc = 2560 ÷ n; nkh = 16 ÷ n; nq = 24 ÷ n; nkv = 2 ÷ n; F = 640 ÷ n;
    Dict(:S => 4 * 4nc, :DP => 4 * PDP4, :Z => 4 * PZ4, :INJ => 4 * PINJ4, :RP => 4 * PRP4, :QKV => 4 * 1280nkh, :GATE => 4 * 768nkh,
        :AB => 4 * 12nkh, :ON => 4 * 768nkh, :YP => 4 * 8nc, :QG => 4 * 512nq, :KK => 4 * 256nkv, :VV => 4 * 256nkv, :IX => 4 * 640,
        :QO => 4 * 256nq, :AO => 4 * 256nq, :KVR => 8, :GU => 4 * 84F, :IDXW => PIDXW4, :SHO => 8, :OUTP => 4 * 4nc, :PK => 4 * 4nc,
        :PV => 4 * nc, :PPT => 4 * 32, :LS => 4 * PLS4))
function load_b4!(E::EngineB1)
    error("B4 (the speculative verify) is not ported to the tensor-split layout (v$LAYOUT_VERSION): pairs and partner pointers")
    t0 = time()
    host = CUDA.alloc(CUDA.HostMemory, 2 * 4 * 2560 * 8, CUDA.MEMHOSTALLOC_DEVICEMAP | CUDA.MEMHOSTALLOC_PORTABLE)
    hp = convert(Ptr{UInt64}, host)
    ccall(:memset, Ptr{Cvoid}, (Ptr{UInt64}, Cint, Csize_t), hp, 0, 2 * 4 * 2560 * 8)
    groups = Vector{B4Die}[]
    mc = MAXCTX[]
    for DG in E.groups
        G = B4Die[]
        for D in DG
            rt = D.rt; device!(rt.ord)
            keep = Any[]; ll = Dict{Symbol,U64P}()
            if !isempty(rt.blks) || rt.n == 1
                for (k, sz) in (isempty(rt.blks) ? Dict(:S => 4 * 4 * 2560) : llsizes4(rt.n))
                    a = CUDA.zeros(UInt64, sz); push!(keep, a); ll[k] = reinterpret(U64P, pointer(a))
                end
            end
            hd = nothing
            if rt.head !== nothing
                rows = Int(rt.head.out.rows)
                hz = CUDA.zeros(Float32, 4 * 2560); lg = CUDA.zeros(Float32, 4 * rows); am = CUDA.zeros(Float32, 8); push!(keep, hz, lg, am)
                hd = (fptr(hz), fptr(lg), fptr(am))
            end
            tm = CUDA.alloc(CUDA.HostMemory, 8 * 16 * 24, CUDA.MEMHOSTALLOC_DEVICEMAP | CUDA.MEMHOSTALLOC_PORTABLE); push!(keep, tm)
            push!(G, B4Die(D, ll, keep, nothing, nothing, length(rt.blks), hd, convert(Ptr{UInt64}, tm)))
        end
        push!(groups, G)
    end
    tapof(l) = (k = findfirst(==(l), DRAFT_TAPS); k === nothing ? NOF : fptr(E.tap) + 4 * (k - 1) * 4 * 2560)
    for (gi, G) in enumerate(groups)
        for D4 in G
            D4.nblk == 0 && continue
            rt = D4.D.rt; device!(rt.ord); n = rt.n; nc = 2560 ÷ n; nkh = 16 ÷ n; nkv = 2 ÷ n
            n == 2 || error("B4: boards of 2 dies only (die $(rt.ord) has n = $n)")
            peer = G[3-(rt.d+1)]
            pp(k) = peer.ll[k]
            mk(T, k) = (a = CUDA.zeros(T, k); push!(D4.keep, a); a)
            red = mk(Float32, 41 * 4 * 2560); cnt = mk(Int32, 80); ctr = mk(Int32, 32 * D4.nblk)
            ll = D4.ll
            dd = DD4(Int32(n), Int32(rt.d), Int32(nc), Int32(16 ÷ n), Int32(24 ÷ n), Int32(2 ÷ n), Int32(640 ÷ n), Int32(0),
                ll[:S], ll[:DP], pp(:DP), ll[:Z], pp(:Z), ll[:INJ], ll[:RP], pp(:RP),
                ll[:QKV], ll[:GATE], ll[:AB], ll[:ON], ll[:YP], pp(:YP),
                ll[:QG], ll[:KK], ll[:VV], ll[:IX], ll[:QO], ll[:AO], ll[:KVR],
                ll[:GU], ll[:IDXW], ll[:SHO], ll[:OUTP], pp(:OUTP),
                ll[:PK], ll[:PV], ll[:PPT], pp(:PPT), ll[:LS], pp(:LS),
                reinterpret(U64P, hp), reinterpret(U64P, hp + 4 * 2560 * 8), rt.rope,
                fptr(red), reinterpret(IPtr, pointer(cnt)), Core.LLVMPtr{UInt64,1}(0), reinterpret(IPtr, pointer(ctr)), U64P(0), U64P(0),
                Core.LLVMPtr{UInt32,1}(0))
            nxt = gi < length(groups) ? groups[gi+1] : nothing
            bds = BD[]
            for (i, B) in enumerate(rt.blks)
                hand = U64P(0); hls = 0; hoff = 0
                if i == length(rt.blks) && nxt !== nothing
                    if length(nxt) == n
                        hand = nxt[rt.d+1].ll[:S]; hls = nc
                    else
                        hand = nxt[1].ll[:S]; hls = 2560; hoff = nc * rt.d
                    end
                end
                if B.attn
                    st = LState(NOF, NOF, gptr(mk(UInt8, mc * 256nkv)), gptr(mk(UInt8, mc * 16nkv)), gptr(mk(UInt8, mc * 256nkv)), gptr(mk(UInt8, mc * 16nkv)),
                        gptr(mk(UInt8, (mc ÷ 4) * 256)), fptr(mk(Float32, 512)), fptr(mk(Float32, mc ÷ 4)), reinterpret(IPtr, pointer(mk(Int32, 512))))
                else
                    st = LState(fptr(mk(Float32, 8 * 640nkh)), fptr(mk(Float32, 4 * 3nkh * 16384)), GPtr(0), GPtr(0), GPtr(0), GPtr(0), GPtr(0), NOF, NOF, NOI)
                end
                pler = B.l == 1 ? fptr(mk(Float32, 16 * 4nc)) : NOF
                push!(bds, make_bd(B, st, pler, hand, hls, hoff, 1 + (i - 1) * NSTB, tapof(B.l)))
            end
            D4.dd = CuArray([dd]); D4.bd = CuArray(bds)
        end
    end
    xsm = 4 * (xu4(2560) + 160)
    E4 = EngineB4(E, groups, host, hp, hp + 4 * 2560 * 8, Dict{Int,Any}(), xsm, 0, 0, Int32(0), Int32[])
    for G in groups, D4 in G
        D4.nblk == 0 && continue
        device!(D4.D.rt.ord)
        k = getkernel(k_token4!, KTOKEN4_TT; maxregs = 128, maxthreads = B4THREADS)
        E4.kern[D4.D.rt.ord] = k
        occ = CUDA.active_blocks(k.fun, B4THREADS; shmem = 4 * xsm)
        mem = CUDA.memory(k)
        occ >= 1 || error("k_token4!: no CTA of $B4THREADS threads fits on die $(D4.D.rt.ord): $(CUDA.registers(k)) registers, $(mem.shared) B static + $(4 * xsm) B dynamic shared, $(mem.local) B local")
    end
    note(@sprintf("B4: engine on %d dies in %.1f s", sum(length, groups), time() - t0))
    E4
end
# a new sequence: no tokens; DeltaNet states (all copies), conv/PLE rings zero
function reset_b4!(E4::EngineB4)
    empty!(E4.ids); E4.bank = Int32(0)
    z!(p, n) = CUC.cuMemsetD32_v2(reinterpret(CUC.CUdeviceptr, p), UInt32(0), n)
    for G in E4.groups, D4 in G
        D4.nblk == 0 && continue
        device!(D4.D.rt.ord); nkh = 16 ÷ D4.D.rt.n; nc = 2560 ÷ D4.D.rt.n
        for b in Array(D4.bd)
            if b.attn == 0
                z!(b.conv, 8 * 640nkh); z!(b.dn, 4 * 3nkh * 16384)
            else
                z!(b.ring, 512)
            end
            b.pler != NOF && z!(b.pler, 16 * 4nc)
        end
        synchronize()
    end
end
# one round: tokens toks (P <= 4) at positions t .. t + P - 1 (t = length(E4.ids)); the target's argmax after each
function round4!(E4::EngineB4, toks::Vector{Int32})
    E = E4.E; P = length(toks); t = length(E4.ids)
    1 <= P <= 4 || error("round4!: 1 to 4 positions")
    E4.rounds += 1; E4.seqn += 1
    seq0 = UInt32(2048 * E4.seqn)
    for G in E4.groups, D4 in G
        D4.nblk == 0 && continue
        device!(D4.D.rt.ord)
        E4.kern[D4.D.rt.ord](reinterpret(Core.LLVMPtr{DD4,1}, pointer(D4.dd::CuVector{DD4})), reinterpret(Core.LLVMPtr{BD,1}, pointer(D4.bd::CuVector{BD})), Int32(D4.nblk), seq0,
            Int32(t), Int32(P), E4.bank, Int32(E4.xsm), PROF[] ? reinterpret(Core.LLVMPtr{UInt64,1}, D4.tim) : Core.LLVMPtr{UInt64,1}(0);
            threads = B4THREADS, blocks = NSM[], shmem = 4 * E4.xsm)
    end
    ctx = vcat(E4.ids, toks)
    for p in 1:P
        emb = embed(E.g, toks[p])
        for i in 1:2560
            unsafe_store!(E4.emb, llword(seq0 + UInt32(1), emb[i]), (p - 1) * 2560 + i)
        end
        gi = ple_input(E.ple, ctx, t + p - 1)
        for i in 1:2560
            unsafe_store!(E4.pleg, llword(seq0 + UInt32(2), gi[i]), (p - 1) * 2560 + i)
        end
    end
    # the head: the collapse of each position on the last board's die, its hz to every head slice, logits, argmax
    lastl = lastblock(E)
    c4 = E4.groups[end][end]; c = c4.D; H = c.rt.head; b = c.rt.b
    device!(c.rt.ord)
    hzs = Vector{Float32}(undef, 4 * 2560)
    for p in 0:P-1
        launch(k_ll2f!, b[:Sfull], c4.ll[:S] + 8 * p * 4 * 2560, Int32(10240), sqof(seq0, lastl, 16); threads = 256, blocks = 40)
        launch(k_rmsnorm!, b[:xh], b[:Sfull], H.norm, Int32(2560), Int32(2560), EPS; threads = 256, blocks = 4)
        gemv!(b[:down], H.down, b[:xh])
        launch(k_hc_silu!, b[:ht], b[:down], Int32(320); threads = 320, blocks = 1)
        gemv!(b[:hup], H.up, b[:ht])
        launch(k_hc_mix!, b[:hz], b[:hup], b[:xh]; threads = 256, blocks = 10)
        hzs[p*2560+1:p*2560+2560] = dl(b[:hz], 2560)
    end
    for G in E4.groups, D4 in G
        D4.head === nothing && continue
        rt = D4.D.rt; device!(rt.ord)
        hz, lg, am = D4.head; rows = Int(rt.head.out.rows)
        ul!(hz, hzs)
        gemv4!(lg, rt.head.out, hz, P; ldx = 2560, ldy = rows)
        for p in 0:P-1
            launch(k_argmax!, am + 8p, lg + 4 * p * rows, Int32(rows), Int32(rt.head.r0); threads = 1024, blocks = 1)
        end
    end
    PROF[] && prof4_acc!(E4)
    best = fill(-Inf32, P); bi = zeros(Int32, P)
    for G in E4.groups, D4 in G
        D4.head === nothing && continue
        device!(D4.D.rt.ord); r = dl(D4.head[3], 8)
        for p in 1:P
            v = r[2p-1]; i = reinterpret(Int32, r[2p])
            (v > best[p] || (v == best[p] && i < bi[p])) && (best[p] = v; bi[p] = i)
        end
    end
    bi
end
# --prof: per op, CTA 0's time from the previous op's end (summed over blocks and rounds), and each die's kernel time
const PROF4 = Dict{Tuple{Int,Int},Float64}()
const OP4NAMES = ["hc_down(attn)", "hc_up+mix(attn)", "proj", "deltanet/attn_prep", "attn", "out_proj", "hc_write(attn)",
    "hc_down(ffn)", "hc_up+mix(ffn)", "router+shared", "experts_gate_up", "experts_down", "hc_write(ffn)"]
function prof4_acc!(E4::EngineB4)
    for G in E4.groups, D4 in G
        D4.nblk == 0 && continue
        device!(D4.D.rt.ord); synchronize()
        s = unsafe_wrap(Array, D4.tim, 16 * 24)
        for bi in 1:D4.nblk, k in 1:13
            PROF4[(D4.D.rt.ord, k)] = get(PROF4, (D4.D.rt.ord, k), 0.0) + (s[16bi+k+1] - s[16bi+k]) / 1e3
        end
        PROF4[(D4.D.rt.ord, 0)] = get(PROF4, (D4.D.rt.ord, 0), 0.0) + (s[16D4.nblk+14] - s[1]) / 1e3
    end
end
function prof4_report(E4::EngineB4, rounds)
    println("B4 profile (CTA 0, us per round, summed over blocks):")
    dies = sort(unique(first.(collect(keys(PROF4)))))
    @printf("  %-18s %s\n", "op", join([@sprintf("%9s", "die $d") for d in dies], ""))
    for k in 1:13
        @printf("  %-18s %s\n", OP4NAMES[k], join([@sprintf("%9.1f", get(PROF4, (d, k), 0.0) / rounds) for d in dies], ""))
    end
    @printf("  %-18s %s\n", "kernel", join([@sprintf("%9.1f", get(PROF4, (d, 0), 0.0) / rounds) for d in dies], ""))
    empty!(PROF4)
end
# --b4 --teacher --ids PROMPT --expect ORACLE: B4 rounds of 4 over prompt + oracle; its argmax against the oracle
function b4_teacher_check(E::EngineB1, ids::Vector{Int32}, ex::Vector{Int32})
    E.b4 === nothing && (E.b4 = load_b4!(E))
    t0 = time()
    out = teacher4(E.b4, vcat(ids, ex))
    dt = time() - t0
    pr = out[length(ids):length(ids)+length(ex)-1]
    diff = findall(k -> pr[k] != ex[k], eachindex(ex))
    @printf("B4 teacher forcing: %d of %d argmax match the oracle%s; %d rounds in %.2f s (%.1f ms a round, %.1f positions/s)\n",
        length(ex) - length(diff), length(ex), isempty(diff) ? "" : " (differ at " * join(diff[1:min(end, 8)], ",") * ")",
        E.b4.rounds, dt, dt / max(1, E.b4.rounds) * 1e3, (length(ids) + length(ex)) / dt)
    metric("b4.teacher_match", "positions", length(ex) - length(diff); compared = length(ex))
    metric("b4.round_ms", "ms", dt / max(1, E.b4.rounds) * 1e3)
    PROF[] && prof4_report(E.b4, E.b4.rounds)
    E.b4.rounds = 0
    isempty(diff) || length(diff) <= 2 ? 0 : 1
end
# --b4 --debug: the first board alone, blocks 0 .. k - 1 for k = 1 .. its blocks, on position 0 of ids (states reset each
# time): S after them from B1 and from B4 (P = 1), compared per lane
function b4_debug(E::EngineB1, ids::Vector{Int32})
    E.b4 === nothing && (E.b4 = load_b4!(E))
    E4 = E.b4; G1 = E.groups[1]; G4 = E4.groups[1]
    nb = G1[1].nblk
    tok = ids[1]
    for k in 1:nb
        # B1
        reset_state!(E); E.ntok += 1; seq0 = UInt32(2048 * E.ntok)
        for D in G1
            device!(D.rt.ord)
            E.kern[D.rt.ord](reinterpret(Core.LLVMPtr{DD,1}, pointer(D.dd)), reinterpret(Core.LLVMPtr{BD,1}, pointer(D.bd)), Int32(k), seq0, Int32(0), Int32(D.xsm),
                Core.LLVMPtr{UInt64,1}(0); threads = B1THREADS[], blocks = D.grid, shmem = b1shmem(D))
        end
        emb = embed(E.g, tok); gi = ple_input(E.ple, Int32[tok], 0)
        for i in 1:2560
            unsafe_store!(E.emb, llword(seq0 + UInt32(1), emb[i]), i); unsafe_store!(E.pleg, llword(seq0 + UInt32(2), gi[i]), i)
        end
        for D in G1; device!(D.rt.ord); synchronize(); end
        l = G1[1].rt.blks[k].l
        s1 = [lldl(D.ll[:S], 0, 4 * (2560 ÷ D.rt.n)) for D in G1]
        # B4
        reset_b4!(E4); E4.seqn += 1; seq4 = UInt32(2048 * E4.seqn)
        for D4 in G4
            device!(D4.D.rt.ord)
            E4.kern[D4.D.rt.ord](reinterpret(Core.LLVMPtr{DD4,1}, pointer(D4.dd)), reinterpret(Core.LLVMPtr{BD,1}, pointer(D4.bd)), Int32(k), seq4,
                Int32(0), Int32(1), Int32(0), Int32(E4.xsm), Core.LLVMPtr{UInt64,1}(0); threads = B4THREADS, blocks = NSM[], shmem = 4 * E4.xsm)
        end
        for i in 1:2560
            unsafe_store!(E4.emb, llword(seq4 + UInt32(1), emb[i]), i); unsafe_store!(E4.pleg, llword(seq4 + UInt32(2), gi[i]), i)
        end
        for D4 in G4; device!(D4.D.rt.ord); synchronize(); end
        s4 = [lldl(D4.ll[:S], 0, 4 * (2560 ÷ D4.D.rt.n)) for D4 in G4]
        errs = Float64[]
        for (a, b) in zip(s1, s4), ln in 0:3
            nc = length(a[1]) ÷ 4; r = ln*nc+1:ln*nc+nc
            push!(errs, maximum(abs.(a[1][r] .- b[1][r])) / max(maximum(abs.(a[1][r])), 1f-30))
        end
        ok1 = all(==(sqof(seq0, l, 16)), s1[1][2]); ok4 = all(==(sqof(seq4, l, 16)), s4[1][2])
        @printf("after block %2d: S relative error per lane (die 0 then die 1): %s%s\n", l, join([@sprintf("%.1e", e) for e in errs], " "),
            ok1 && ok4 ? "" : "  (seq: B1 $(ok1) B4 $(ok4))")
    end
    0
end
# teacher forcing through B4 rounds of 4: the target's argmax at every position of ids
function teacher4(E4::EngineB4, ids::Vector{Int32})
    reset_b4!(E4)
    out = Int32[]
    i = 1
    while i <= length(ids)
        toks = ids[i:min(i + parse(Int, get(ENV, "FLASHNEXT_B4P", "4")) - 1, end)]
        append!(out, round4!(E4, toks))
        append!(E4.ids, toks)
        E4.bank = Int32((E4.bank + length(toks)) & 3)         # all accepted: the state after the last position
        i += length(toks)
    end
    out
end

# the entry point of the juliac executable (and of jlrun, which calls main)
function (@main)(args)
    # FLASHNEXT_COMPILETIME=1: the time spent compiling during the run, at its end (what the juliac executable still
    # compiles at runtime)
    ct = get(ENV, "FLASHNEXT_COMPILETIME", "0") == "1"
    ct && Base.cumulative_compile_timing(true)
    t0 = time()
    try
        st = main_(args)
        ct && note(@sprintf("compile time %.1f s of %.1f s", Base.cumulative_compile_time_ns()[1] / 1e9, time() - t0))
        return st
    catch e
        # never print the exception object itself: its fields can hold the 37 GB model mapping
        msg = e isa MethodError ? "MethodError: $(e.f) with argument types $(map(typeof, e.args))" : first(sprint(showerror, e), 2000)
        println(stderr, "flashnext: error: ", msg)
        PROF[] && LASTB1[] !== nothing && fault_report(LASTB1[])
        for (i, fr) in enumerate(stacktrace(catch_backtrace()))
            i > 25 && break
            println(stderr, "  ", fr)
        end
        return 1
    end
end
function main_(args)
    a = String.(args)
    if !isempty(a) && a[1] == "--build-kernels"          # flashnext.ptx and flashnext.sass into DIR (no GPU needed)
        return build_kernels(length(a) > 1 ? a[2] : ".")
    end
    !isempty(a) && a[1] == "--kernel-bc" && return kernel_bc(a[2], parse.(Int, split(a[3], ",")))
    k = findfirst(==("--model"), a)                       # --model FILE: one GGUF (merged) or a split's first shard
    if k !== nothing
        MODEL[] = model_paths(a[k+1]); deleteat!(a, k:k+1)
    end
    k = findfirst(==("--merge"), a)                       # --merge OUT: the model's files as one GGUF
    k !== nothing && return merge_gguf(MODEL[], a[k+1])
    k = findfirst(==("--kernels"), a)                     # --kernels jit | ptx FILE | sass FILE
    if k !== nothing
        if a[k+1] == "jit"
            deleteat!(a, k:k+1)
        else
            a[k+1] in ("ptx", "sass") || error("--kernels jit | ptx FILE | sass FILE")
            KMODE[] = kread(a[k+1], a[k+2]); KPATH[] = a[k+2]; KSTAT[] = stat(a[k+2]); note("kernels from ", a[k+2], " (", a[k+1], ", ", length(KMODE[]), " bytes)")
            deleteat!(a, k:k+2)
        end
    end
    if !isempty(a) && a[1] == "--bench" && a[2] == "comm"
        dl_ = length(a) > 2 ? parse.(Int, split(a[3], ",")) : collect(0:6)
        get(ENV, "FLASHNEXT_BW_ONLY", "") == "1" || bench_comm(dl_)
        bench_bw(dl_)
        return 0
    end
    if !isempty(a) && a[1] == "--meta"
        g = gguf_open(length(a) > 1 ? a[2:end] : MODEL[])
        show_meta(g)
        return 0
    end
    if !isempty(a) && a[1] == "--bench" && a[2] == "phase"
        B1THREADS[] = parse(Int, get(ENV, "FLASHNEXT_B1_THREADS", "256")); B1MAXREGS[] = parse(Int, get(ENV, "FLASHNEXT_B1_MAXREGS", "128"))
        bench_phase(gguf_open(MODEL[]))
        return 0
    end
    if !isempty(a) && a[1] == "--bench" && a[2] == "gemv4"
        B1MAXREGS[] = parse(Int, get(ENV, "FLASHNEXT_B1_MAXREGS", "128"))
        bench_gemv4(gguf_open(MODEL[]))
        return 0
    end
    if !isempty(a) && a[1] == "--bench" && a[2] == "gemv"
        NSM[] = CUDA.attribute(device(), CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
        g = gguf_open(MODEL[])
        bench_gemv(g)
        return 0
    end
    if !isempty(a) && a[1] == "--ptx"
        return ptx_all(length(a) > 1 ? a[2] : ".") == 0 ? 0 : 1
    end
    o = parse_args(a)
    MAXCTX[] = o["ctx"]
    PROF[] = get(o, "prof", false)
    B1THREADS[] = parse(Int, get(ENV, "FLASHNEXT_B1_THREADS", "256")); B1MAXREGS[] = parse(Int, get(ENV, "FLASHNEXT_B1_MAXREGS", "128"))
    g = gguf_open(MODEL[])
    note("model: ", join(basename.(MODEL[]), ", "))
    tk = Tok(g)
    if haskey(o, "tokenize")
        ids = encode(tk, o["tokenize"])
        println(join(ids, " "))
        return 0
    end
    prompt = get(o, "p", "The capital of France is")
    ids = haskey(o, "ids") ? o["ids"] : encode(tk, o["raw"] ? prompt : chat_prompt(prompt))
    note("prompt ids: ", join(ids, " "))
    if get(o, "stage", "a") in ("b", "b1")
        # every block over all 8 dies (no layer split: one group); --groups 0+1,2+3,... --bblocks 12,12,... the pipeline
        groups = [parse.(Int, split(s, "+")) for s in split(get(o, "groups", "0+1+2+3+4+5+6+7"), ",")]
        counts = haskey(o, "bblocks") ? o["bblocks"] : (length(groups) == 1 ? [48] : fill(48 ÷ length(groups), length(groups)))
        pl = placement_b(groups, counts)
        get(o, "hold", false) && hold(g, pl)
        if o["stage"] == "b1"
            E = load_engine_b1(g, pl)
            get(o, "check", false) && return run_check_b1(E, ids) ? 0 : 1
            get(o, "debug", false) && return (debug_b1(E, ids[1]); 0)
        else
            E = load_engine_b(g, pl)
            get(o, "check", false) && return run_check_b(E, ids) ? 0 : 1
        end
    else
        pl = placement_a(o["dies"], o["blocks"], o["layers"])
        get(o, "hold", false) && hold(g, pl)
        E = load_engine(g, pl)
        get(o, "check", false) && return run_check(E, ids) ? 0 : 1
    end
    get(o, "serve", false) && E isa EngineB1 && return serve(E)
    run_request(E, tk, o, ids)
end
function run_request(E, tk, o, ids)
    if get(o, "teacher", false) && haskey(o, "expect")
        oracle_margins(E, ids, o["expect"])
        return 0
    end
    if get(o, "teacher", false)
        preds = teacher(E, ids, o["n"])
        PROF[] && E isa EngineB1 && profile_b1(E)
        println("teacher-forced argmax: ", join(preds, " "))
        println("text: ", String(vcat([piece(tk, i) for i in preds]...)))
        return 0
    end
    out = get(o, "spec", "off") == "on" && E isa EngineB1 ? generate_spec(E, ids, o["n"]) :
        generate(E, ids, o["n"]; greedy = o["greedy"], seed = o["seed"], spec = get(o, "spec", "off"))
    PROF[] && E isa EngineB1 && get(o, "spec", "off") != "on" && profile_b1(E)
    println("ids: ", join(out, " "))
    if haskey(o, "expect")         # the output against an oracle: the length of the common prefix
        ex = o["expect"]; k = findfirst(i -> i > length(out) || i > length(ex) || out[i] != ex[i], 1:max(length(out), length(ex)) + 1) - 1
        println("oracle: $k of $(min(length(out), length(ex))) leading ids match", k < min(length(out), length(ex)) ? " (first difference at output $(k + 1): $(out[k+1]) vs $(ex[k+1]))" : "")
        metric("check.oracle_prefix", "tokens", k; compared = min(length(out), length(ex)))
        return k == min(length(out), length(ex)) ? 0 : 1
    end
    return 0
end

# ---- the juliac executable (built with FLASHNEXT_EXE=1): the request path compiled into its image. main_ holds the
# engine in a variable of several types (stages A, B, B1), so inference stops at run_request, and everything below
# it was compiled at the executable's first (and only) request: 18.7 tok/s speculative. These give it the B1 types;
# the image keeps what the build compiles. (Not in jlrun or the server: there it would only move compile time to
# the start.) ----
if get(ENV, "FLASHNEXT_EXE", "0") == "1"
    for (f, tt) in ((run_request, (EngineB1, Tok, Dict{String,Any}, Vector{Int32})),
                    (load_engine_b1, (GModel, PlacementB)), (load_draft, (EngineB1,)),
                    (generate_spec, (EngineB1, Vector{Int32}, Int)), (generate, (EngineB1, Vector{Int32}, Int)),
                    (verify_round!, (EngineB1, Vector{Int32})), (draft_propose, (Draft, EngineB1, Int32)),
                    (draft_append!, (Draft, Int, Int)), (early_argmax!, (EngineB1, B1Die, Int, UInt32, FPtr)),
                    (abort_positions!, (EngineB1, UInt32, UInt32)), (launch_pipelined!, (EngineB1, Int32)),
                    (launch_anchor_early!, (EngineB1, Int32)), (step_greedy!, (EngineB1, Int32)))
        precompile(f, tt) || println(stderr, "flashnext: precompile of ", f, tt, " failed")
    end
end
