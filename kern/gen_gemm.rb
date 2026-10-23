#!/usr/bin/env ruby
# GEMM_FMT=q4k|bf16 gen_gemm.rb > gemm_<fmt>.sass: the prefill GEMM of a weight tile against fp32 activations in k-major
# layout, C[row][pos] = sum_k W[row][k] * Xt[k][pos], as a MaxAs kernel for sm_52 (one source for every weight format;
# the format decides how a thread turns its bytes of a row into the 4 fp32 weights it puts into the shared tile).
#
# Shape: one CTA of 256 threads computes a 128 (rows) x 128 (positions) tile of C over the whole K; grid
# (ceil(rows / 128), ldx / 128). The main loop is sgemm128's (chain/maxas/sgemm): every thread owns an 8 x 8
# fragment, the 8-k tiles of both operands live in shared memory double buffered (16 KB), each k line is 64 FFMAs
# per thread, 8 lines per pass. What differs from sgemm: the "A" tile (positions) is one LDG.128 of Xt per thread
# and the "B" tile (rows) is dequantised on the way in by thread (row = tid & 127, half = tid >> 7), which stores
# 4 fp32 weights to Bs[k][row] per pass. Rows past `rows` are clamped (duplicate work, no store); positions are
# padded by the caller (ldx a multiple of 128, Xt zero beyond npos); C is written with ldc = ldx.
#
#   q4k   the repacked Q4_K tile (ref/quant.c repack_q4_k: 16 B headers per 256-k block, then 128 B of nibbles per
#         block, 16 B per 32-k sub-block, low nibble = even element). Thread half h takes elements 4h..4h+3 of each
#         8-k group: two bytes of the sub-block's word (I2F of masked bytes, then q * ws - nm with ws = d * scale and
#         nm = dmin * min). Scales are decoded once per sub-block (every 4 passes) in a branch at the top of the
#         pass; the next sub-block's 16 B and the next block's header are prefetched into registers. K % 256 == 0.
#   bf16  rows of bf16 (K * 2 bytes per row). Thread half h loads 16 B = 8 consecutive weights every second pass
#         (k = 16 g + 8 h .. + 7 for passes 2 g, 2 g + 1), so a pass's 8 k lines are k = 16 g + 8 (l >> 2) + 4 (pass & 1)
#         + (l & 3) and the activation loader follows that order (alternating steps of 4 and 12 k rows). A bf16 is
#         a shift into fp32. K % 16 == 0.
#
# Params: 0x140 W (tile), 0x148 Xt, 0x150 C, 0x158 rows, 0x15c npos, 0x160 K, 0x164 ldx, 0x168 debug.

#   q2    Q2_0: 64-k blocks of 16 B of 2-bit codes (weight = d (q - 1), fp16 d per block) addressed with strides, so the
#         dense row tile (ref/quant.c repack_q2_0: W = the tile's code area, crow = row stride, cstride 16; S = the
#         tile, srow = row stride, sstride 2) and the expert units of kern/hold2.c (codes 16 B per row, units 46080 B
#         apart, the scales 40960 B into a unit) use one kernel. Thread half h holds bytes 8 h .. 8 h + 7 of a block
#         (k = 64 b + 32 h + 4 i .. + 3 at pass i of the block), so a pass's 8 k lines are k = 64 b + 32 (l >> 2)
#         + 4 (pass & 7) + (l & 3) (activation steps of 4 rows, 36 at a block end). K % 64 == 0.
#         Extra params: 0x170 S, 0x178 crow, 0x17c cstride, 0x180 srow, 0x184 sstride.

#   q3    the repacked Q3_K tile (ref/quant.c repack_q3_k): a 16 Q-byte area of fp16 d per block (d of block b at
#         16 (b mod Q) + 2 (b / Q), Q = lanes / 8), then 112 B records per 256-k block: 16 words of 2-bit codes (group g
#         of 16 k, element l at bits 16 (l >> 3) + 2 (l & 7)), 32 B of inverted high bits (bit l of bytes 64 + 2 g,
#         65 + 2 g), 16 int8 scales (sc - 32) at 96 + g; weight = d (sc - 32) (code - 4 hinv). A thread works through
#         a quad of 4 groups (8 passes) rotating its code words 16 bits, its high bytes 8 bits and its scale word 8
#         bits every second pass; code - 4 hinv is I2F.S8 of (code + 4 hinv) << 5 against d (sc - 32) / 32.
#         W = the records (tile + 16 Q), crow = row stride; S = the tile (the d area), srow = row stride, sstride = log2 Q.

#   iq4   the repacked IQ4_XS tile (ref/quant.c repack_iq4_xs): the d area as q3's, 128 B of nibbles per 256-k block (16 B per
#         32-k sub-block, low nibble = even element, as q4k), then 8 int8 scales (ls - 32) per block; weight = d sc kv[code]
#         with the IQ4_NL codebook kv. A thread's 4 codes of a pass are 4 nibbles of a half word h; two PRMTs look the 4
#         values up in the codebook held in 4 registers (entries 0..7 and 8..15, selector = the nibbles & 7), a third PRMT
#         spreads each code's bit 3 into a byte mask (sign replication) and LOP3 picks the bytes; I2F.S8 per byte, times
#         d sc (decoded once per sub-block at the top of the pass, as q4k). W = the nibbles (tile + 16 Q), crow = row stride;
#         S = the tile, srow = row stride, sstride = log2 Q; the scales follow the row's nibbles (W + rr crow + 128 nb).

#   q5k   the repacked Q5_K tile (ref/quant.c repack_q5_k): q4k's headers and nibbles (176 B per block per row), then 32 B
#         of high bits per block, one word per sub-block (bit e = element e). Per pass the thread's 4 high bits are a nibble
#         b of that word (the word rotates 8 bits a pass); b * 0x410 and b * 0x208 put elements 0, 2 and 1, 3 at bit 4 of
#         the bytes of m0 and m1 (mask 0x1010), so the q4k byte conversions read code = nibble + 16 bit.

#   q6k   the repacked Q6_K tile (ref/quant.c repack_q6_k): the d area as q3's, then 208 B records per 256-k block: 128 B of
#         nibbles (16 B per 32-k sub-block, as q4k), 8 B of high 2-bit fields per sub-block (one word per 16-k group, element
#         e at bits 16 (e >> 3) + 2 (e & 7)), 16 int8 scales (one per 16-k group). The nibbles go as q4k's; the thread's 4
#         fields are byte h of the group word's current half (the pair rotates 16 bits a pass), split even / odd (& 0x33,
#         & 0xcc) and moved to bits 4-5 and 12-13 by one XMAD each (0x110, 0x44; the overlap carries stay below bit 11), so the
#         byte conversions read code = nibble + 16 field and the weight is code ws - nm with ws = d sc, nm = 32 ws (the scale
#         of the second group takes over at the third pass). The sub-block loads (nibbles, fields, the two scales) and, a
#         block early, the next block's d are issued at the sub-block boundary from the record pointer.
#         W = the records (tile + 16 Q), crow = row stride; S = the tile, srow = row stride, sstride = log2 Q.

#   q40, q50, iq4nl   the repacked 32-k block tiles (ref/quant.c repack_b32): per row 48 B of fp16 d (block b at 2 b), then
#         16 B of nibbles per block (as q4k's sub-blocks), q50 then one high-bit word per block (as q5k's). A block is
#         q4k's sub-block with d for its scale: q40 / q50 weight = d code - 8 d / 16 d (q4k's and q5k's conversions),
#         iq4nl = d kv[code] (iq4's). W = the tile, crow = row stride. K % 32 == 0.
#   q80   the same tile with 32 int8 per block: a 16 B quad serves two passes (thread half h takes bytes 4 h .. 4 h + 3 of
#         the pass's 8, one PRMT of the quad's first two words), so the quads are prefetched every second pass and d
#         every fourth; weight = d code.

# GEMM_OP=1: the body of a die-kernel op (kern/die.pl) instead of a standalone kernel: the CTA walks the tiles
# tile = vid, vid + units, ... (by = tile / ntx, bx = tile mod ntx), the parameters come from the descriptor.
OP = ENV['GEMM_OP'] == '1'

FMT = ENV.fetch('GEMM_FMT', 'q4k')
abort "GEMM_FMT must be q4k, q5k, q6k, bf16, q2, q3, iq4, q40, q50, iq4nl or q80" unless %w[q4k q5k q6k bf16 q2 q3 iq4 q40 q50 iq4nl q80].include?(FMT)
Q80 = FMT == 'q80'
Q5K = FMT == 'q5k'
Q6K = FMT == 'q6k'
Q40 = FMT == 'q40'
Q50 = FMT == 'q50'
IQNL = FMT == 'iq4nl'
B32N = Q40 || Q50 || IQNL   # 32-k blocks of nibbles with one fp16 d (ref/quant.c repack_b32)
B32OFF = Q40 ? '8.0' : '16.0'   # weight = d (code - offset)
Q4K = FMT == 'q4k' || Q5K   # q5k is q4k plus the high bits
BF16 = FMT == 'bf16'
Q2 = FMT == 'q2'
Q3 = FMT == 'q3'
IQ4 = FMT == 'iq4'
# the IQ4_NL codebook as four little-endian words of int8: entries 0..3, 4..7, 8..11, 12..15
KV = [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113].freeze
KVW = KV.each_slice(4).map { |q| q.each_with_index.sum { |v, i| (v & 0xff) << (8 * i) } }.freeze

out = +''
emit = ->(s) { out << s << "\n" }

# the register-name templates: MaxAs understands x<00-03|64-67>; the die kernel's op loader (die.pl expand_names) only
# x<00-03>, so an op body lists such names in full
XY = [0, 1, 2, 3, 64, 65, 66, 67].freeze
tmpl = ->(prefix, nums) { OP ? nums.map { |n| format('%s%02d', prefix, n) }.join(', ') : "#{prefix}<00-03|64-67>" }

emit.call <<~HDR
  # Kernel: gemm_#{FMT}  (generated by GEMM_FMT=#{FMT} gen_gemm.rb; do not edit)
  # SharedSize: 16384
  <CONSTANT_MAPPING>
      param_W0   : c[0x0][0x140]
      param_W1   : c[0x0][0x144]
      param_X0   : c[0x0][0x148]
      param_X1   : c[0x0][0x14c]
      param_C0   : c[0x0][0x150]
      param_C1   : c[0x0][0x154]
      param_rows : c[0x0][0x158]
      param_npos : c[0x0][0x15c]
      param_K    : c[0x0][0x160]
      param_ldx  : c[0x0][0x164]
      param_D0   : c[0x0][0x168]
      param_D1   : c[0x0][0x16c]
      param_S0   : c[0x0][0x170]
      param_S1   : c[0x0][0x174]
      param_crow : c[0x0][0x178]
      param_cstr : c[0x0][0x17c]
      param_srow : c[0x0][0x180]
      param_sstr : c[0x0][0x184]
      #{OP ? 'gridDimX   : c[0x0][0x14]' : ''}
  </CONSTANT_MAPPING>

  <REGISTER_MAPPING>
      // the 64 C registers, placed against the fragment registers as sgemm128 does (no bank conflicts)
       3, 2,11,10,19,18,27,26 : #{tmpl.call('cx00y', XY)}
       7, 6,15,14,23,22,31,30 : #{tmpl.call('cx01y', XY)}
       1, 0, 9, 8,17,16,25,24 : #{tmpl.call('cx02y', XY)}
       5, 4,13,12,21,20,29,28 : #{tmpl.call('cx03y', XY)}
      35,34,43,42,51,50,59,58 : #{tmpl.call('cx64y', XY)}
      39,38,47,46,55,54,63,62 : #{tmpl.call('cx65y', XY)}
      33,32,41,40,49,48,57,56 : #{tmpl.call('cx66y', XY)}
      37,36,45,44,53,52,61,60 : #{tmpl.call('cx67y', XY)}
      // setup temporaries (the C registers are zeroed after the setup)
      0-13    : t0, t1, t2, ka, ldx, r, h, rr, k, nb, stride, tid7, tid1, tid128
      // double buffered fragments
      64-79   : #{tmpl.call('j0Ax', XY)}, #{tmpl.call('j0By', XY)}
      80-95   : #{tmpl.call('j1Ax', XY)}, #{tmpl.call('j1By', XY)}
      // epilogue (the fragments are dead)
      64-71   : cs<0-7>
      72-79   : pa<0-1>, pb<0-1>, pq<0-1>, pr<0-1>
      80-83   : cy00, cy04, cy08, cy12
      84-95   : ldc, ldc1, ldc4, ldc8, ldc60, writeCs, readCs, cx, ci, tmpc, tid96, tid128e
      // loaders and state
      96-99   : loadX<0-3>
      100-103 : wq<0-3>
      104-107 : sq<0-3>
      108-111 : hdr<0-3>
      112-113 : track<0-1>
      114-115 : wtrack<0-1>
      116-117 : hptr<0-1>
      #{BF16 ? '116-117 : xstepB, xs' : '142-143 : xstepB, xs'}
      144-145 : cst, sst
      118-121 : readAs, readBs, writeS, writeW
      122-130 : ki, pc, jn, selh, ws, nm, dd, dm, xstep
      131-133 : hw, m0, m1
      134-137 : f<0-3>
      138-141 : wv<0-3>
      142-150 : u1, u2, u3, sc, mn, shj, tt, scf, mnf
      #{Q3 ? '108-111 : wh<0-1>, sh<0-1>' : ''}
      #{Q3 ? '124-125 : sh8h, sh4h' : ''}
      #{Q3 ? '142-143 : qn, ss' : ''}
      #{Q3 ? '144-145 : dbase<0-1>' : ''}
      #{Q3 ? '146-147 : dh, cb' : ''}
      #{Q3 ? '149-150 : hb, bn' : ''}
      151-154 : tid, bx, by, tid31
      #{OP ? '155-158 : tile, ntx, nty, ntiles' : ''}
      #{IQ4 || IQNL ? '159-165 : bn, dbase<0-1>, tb<0-3>' : ''}
      #{Q5K || Q50 ? '159-164 : hq, hbtrack<0-1>, sqh, bfp, cmask' : ''}
      #{B32N || Q80 ? '166-168 : dtrack<0-1>, dsq' : ''}
      #{Q6K ? '160-185 : hq<0-1>, sqh<0-1>, rec<0-1>, dbase<0-1>, s16, sqs, ws1, nm1, jp, jc, bnp, dhn, sh8h, cmask, ta<0-1>, tc<0-1>, te<0-1>, td<0-1>' : ''}
  </REGISTER_MAPPING>
HDR

# 32 x 32 -> low 32 multiply (MaxAs's XMAD.LO expansion), d = a * b + c; x is a temporary, d must differ from a
xmad_lo = lambda do |d, a, b, c, x|
  "--:-:-:-:6      XMAD.MRG #{x}, #{a}, #{b}.H1, RZ;\n" \
  "--:-:-:-:6      XMAD #{d}, #{a}, #{b}, #{c};\n" \
  "--:-:-:-:6      XMAD.PSL.CBCC #{d}, #{a}.H1, #{x}.H1, #{d};\n"
end

# ---------------------------------------------------------------- setup (hand scheduled, stall 6 unless independent)
if OP
  # the tiles this CTA owns: tile = vid + n units; its (bx, by) by repeated subtraction (a few steps per tile)
  emit.call <<~TILES
    --:-:1:-:1      S2R tid, SR_TID.X;
    --:-:2:-:6      S2R tile, SR_CTAID.X;
    03:-:-:-:6      MOV ntx, param_rows;
    --:-:-:-:6      IADD ntx, ntx, 0x7f;
    --:-:-:-:6      SHR.U32 ntx, ntx, 0x7;
    --:-:-:-:6      MOV nty, param_ldx;
    --:-:-:-:6      SHR.U32 nty, nty, 0x7;
    #{xmad_lo.call('ntiles', 'ntx', 'nty', 'RZ', 'bx')}
    --:-:-:-:6      ISETP.GE.U32.AND P0, PT, tile, ntiles, PT;
    --:-:-:-:6      NOP;
    --:-:-:-:6      NOP;
    --:-:-:-:6      NOP;
    --:-:-:-:6  @P0 EXIT;
    TILE:
    --:-:-:-:5      BAR.SYNC 0x0;
    --:-:-:-:6      MOV bx, tile;
    --:-:-:-:6      MOV by, RZ;
    TDIV:
    --:-:-:-:6      ISETP.GE.U32.AND P0, PT, bx, ntx, PT;
    --:-:-:-:6      NOP;
    --:-:-:-:6      NOP;
    --:-:-:-:6      NOP;
    --:-:-:-:6 @!P0 BRA TDIVD;
    --:-:-:-:6      IADD bx, bx, -ntx;
    --:-:-:-:6      IADD by, by, 0x1;
    --:-:-:-:6      BRA TDIV;
    TDIVD:
  TILES
else
  emit.call <<~IDS
    --:-:1:-:1      S2R tid, SR_TID.X;
    --:-:2:-:1      S2R bx, SR_CTAID.X;
    --:-:3:-:6      S2R by, SR_CTAID.Y;
  IDS
end
emit.call <<~SETUP
  07:-:-:-:6      LOP32I.AND tid31, tid, 0x1f;
  --:-:-:-:6      SHR.U32 ka, tid, 0x5;
  --:-:-:-:6      MOV ldx, param_ldx;
SETUP
if Q4K || Q3 || IQ4 || Q6K || B32N || Q80
  emit.call xmad_lo.call('t0', 'ka', 'ldx', 'RZ', 't1')   # activation row of k line ka: k = ka
else
  emit.call <<~KOFF
    --:-:-:-:6      SHR.U32 t2, ka, 0x2;
    --:-:-:-:6      LOP32I.AND t1, ka, 0x3;
    --:-:-:-:6      ISCADD t2, t2, t1, #{BF16 ? '0x3' : '0x5'};
  KOFF
  emit.call xmad_lo.call('t0', 't2', 'ldx', 'RZ', 't1')   # k = (bf16: 8, q2: 32) (ka >> 2) + (ka & 3)
end
emit.call <<~SETUP2
  --:-:-:-:6      ISCADD t0, by, t0, 0x7;
  --:-:-:-:6      ISCADD t0, tid31, t0, 0x2;
  --:-:-:-:6      SHL t0, t0, 0x2;
  --:-:-:-:6      IADD track0.CC, t0, param_X0;
  --:-:-:-:6      IADD.X track1, RZ, param_X1;
  --:-:-:-:6      SHL writeS, tid, 0x4;
  --:-:-:-:6      LOP32I.AND r, tid, 0x7f;
  --:-:-:-:6      SHR.U32 h, tid, 0x7;
  --:-:-:-:6      ISCADD rr, bx, r, 0x7;
  --:-:-:-:6      MOV t1, param_rows;
  --:-:-:-:6      IADD t1, t1, -0x1;
  --:-:-:-:6      IMNMX.U32 rr, rr, t1, PT;
  --:-:-:-:6      MOV k, param_K;
SETUP2
if Q4K
  emit.call <<~Q4SET
    --:-:-:-:6      SHL xstep, ldx, 0x5;
    --:-:-:-:6      SHR.U32 nb, k, 0x8;
    --:-:-:-:6      SHL t2, nb, 0x7;
    --:-:-:-:6      ISCADD stride, nb, t2, 0x4;
    #{Q5K ? '--:-:-:-:6      ISCADD stride, nb, stride, 0x5;' : ''}
    #{xmad_lo.call('t0', 'rr', 'stride', 'RZ', 't1')}
    --:-:-:-:6      IADD hptr0.CC, t0, param_W0;
    --:-:-:-:6      IADD.X hptr1, RZ, param_W1;
    --:-:-:-:6      SHL t2, nb, 0x4;
    --:-:-:-:6      IADD wtrack0.CC, hptr0, t2;
    --:-:-:-:6      IADD.X wtrack1, hptr1, RZ;
  Q4SET
  # q5k: the high-bit words 144 nb into the row, the thread's nibble of a word (BFE position 4 h, width 4), the byte mask
  emit.call <<~Q5SET if Q5K
    --:-:-:-:6      SHL t2, nb, 0x7;
    --:-:-:-:6      ISCADD t2, nb, t2, 0x4;
    --:-:-:-:6      IADD hbtrack0.CC, hptr0, t2;
    --:-:-:-:6      IADD.X hbtrack1, hptr1, RZ;
    --:-:-:-:6      ISCADD bfp, h, 0x400, 0x2;
    --:-:-:-:6      MOV32I cmask, 0x1010;
  Q5SET
  emit.call <<~Q4SET
    --:-:-:-:6      MOV32I t0, 0x4410;
    --:-:-:-:6      MOV32I t1, 0x4432;
    --:-:-:-:6      ISETP.NE.U32.AND P1, PT, h, RZ, PT;
  Q4SET
elsif BF16
  emit.call <<~BFSET
    --:-:-:-:6      SHL xstep, ldx, 0x4;
    --:-:-:-:6      SHL xstepB, ldx, 0x5;
    --:-:-:-:6      IADD xstepB, xstepB, xstep;
    --:-:-:-:6      MOV stride, param_crow;
    #{xmad_lo.call('t0', 'rr', 'stride', 'RZ', 't1')}
    --:-:-:-:6      SHL t2, h, 0x4;
    --:-:-:-:6      IADD t0, t0, t2;
    --:-:-:-:6      IADD wtrack0.CC, t0, param_W0;
    --:-:-:-:6      IADD.X wtrack1, RZ, param_W1;
  BFSET
elsif IQ4
  # nibbles at W + rr crow (wtrack walks the sub-blocks), the scales 128 nb on (hptr walks the blocks), the d area at S + rr srow
  emit.call <<~IQSET
    --:-:-:-:6      SHL xstep, ldx, 0x5;
    --:-:-:-:6      SHR.U32 nb, k, 0x8;
    --:-:-:-:6      MOV stride, param_crow;
    #{xmad_lo.call('t0', 'rr', 'stride', 'RZ', 't1')}
    --:-:-:-:6      IADD wtrack0.CC, t0, param_W0;
    --:-:-:-:6      IADD.X wtrack1, RZ, param_W1;
    --:-:-:-:6      SHL t2, nb, 0x7;
    --:-:-:-:6      IADD hptr0.CC, wtrack0, t2;
    --:-:-:-:6      IADD.X hptr1, wtrack1, RZ;
    --:-:-:-:6      MOV stride, param_srow;
    #{xmad_lo.call('t0', 'rr', 'stride', 'RZ', 't1')}
    --:-:-:-:6      IADD dbase0.CC, t0, param_S0;
    --:-:-:-:6      IADD.X dbase1, RZ, param_S1;
    --:-:-:-:1      MOV32I tb0, #{format('0x%08x', KVW[0])};
    --:-:-:-:1      MOV32I tb1, #{format('0x%08x', KVW[1])};
    --:-:-:-:1      MOV32I tb2, #{format('0x%08x', KVW[2])};
    --:-:-:-:1      MOV32I tb3, #{format('0x%08x', KVW[3])};
    --:-:-:-:6      MOV bn, RZ;
    --:-:-:-:6      MOV32I t0, 0x4410;
    --:-:-:-:6      MOV32I t1, 0x4432;
    --:-:-:-:6      ISETP.NE.U32.AND P1, PT, h, RZ, PT;
  IQSET
elsif Q6K
  # the row's records at W + rr crow (rec: the block of the next sub-block load), its d area at S + rr srow; sub-blocks 0
  # and 1 are loaded in the setup, so the first boundary loads sub-block 2 (jp) and starts sub-block 1 (jc)
  emit.call <<~Q6SET
    --:-:-:-:6      SHL xstep, ldx, 0x5;
    --:-:-:-:6      MOV stride, param_crow;
    #{xmad_lo.call('t0', 'rr', 'stride', 'RZ', 't1')}
    --:-:-:-:6      IADD rec0.CC, t0, param_W0;
    --:-:-:-:6      IADD.X rec1, RZ, param_W1;
    --:-:-:-:6      MOV stride, param_srow;
    #{xmad_lo.call('t0', 'rr', 'stride', 'RZ', 't1')}
    --:-:-:-:6      IADD dbase0.CC, t0, param_S0;
    --:-:-:-:6      IADD.X dbase1, RZ, param_S1;
    --:-:-:-:6      SHL sh8h, h, 0x3;
    --:-:-:-:6      MOV32I cmask, 0x3030;
    --:-:-:-:6      MOV32I jp, 0x2;
    --:-:-:-:6      MOV32I jc, 0x1;
    --:-:-:-:6      MOV bnp, RZ;
    --:-:-:-:6      MOV32I t0, 0x4410;
    --:-:-:-:6      MOV32I t1, 0x4432;
    --:-:-:-:6      ISETP.NE.U32.AND P1, PT, h, RZ, PT;
  Q6SET
elsif B32N
  # the row's d at W + rr crow (dtrack walks the blocks), its nibbles 48 B on (wtrack), q50's high-bit words after the
  # nibbles (hbtrack, the thread's nibble of a word as q5k)
  emit.call <<~B32SET
    --:-:-:-:6      SHL xstep, ldx, 0x5;
    --:-:-:-:6      MOV stride, param_crow;
    #{xmad_lo.call('t0', 'rr', 'stride', 'RZ', 't1')}
    --:-:-:-:6      IADD dtrack0.CC, t0, param_W0;
    --:-:-:-:6      IADD.X dtrack1, RZ, param_W1;
    --:-:-:-:6      IADD wtrack0.CC, dtrack0, 0x30;
    --:-:-:-:6      IADD.X wtrack1, dtrack1, RZ;
    --:-:-:-:6      MOV32I t0, 0x4410;
    --:-:-:-:6      MOV32I t1, 0x4432;
    --:-:-:-:6      ISETP.NE.U32.AND P1, PT, h, RZ, PT;
  B32SET
  emit.call <<~Q50SET if Q50
    --:-:-:-:6      SHR.U32 nb, k, 0x5;
    --:-:-:-:6      SHL t2, nb, 0x4;
    --:-:-:-:6      IADD hbtrack0.CC, wtrack0, t2;
    --:-:-:-:6      IADD.X hbtrack1, wtrack1, RZ;
    --:-:-:-:6      ISCADD bfp, h, 0x400, 0x2;
    --:-:-:-:6      MOV32I cmask, 0x1010;
  Q50SET
  emit.call <<~NLSET if IQNL
    --:-:-:-:1      MOV32I tb0, #{format('0x%08x', KVW[0])};
    --:-:-:-:1      MOV32I tb1, #{format('0x%08x', KVW[1])};
    --:-:-:-:1      MOV32I tb2, #{format('0x%08x', KVW[2])};
    --:-:-:-:6      MOV32I tb3, #{format('0x%08x', KVW[3])};
  NLSET
elsif Q80
  # d at W + rr crow (dtrack), the int8 codes 48 B on (wtrack); selh picks words 0, 1 of a quad for half h
  emit.call <<~Q80SET
    --:-:-:-:6      SHL xstep, ldx, 0x5;
    --:-:-:-:6      MOV stride, param_crow;
    #{xmad_lo.call('t0', 'rr', 'stride', 'RZ', 't1')}
    --:-:-:-:6      IADD dtrack0.CC, t0, param_W0;
    --:-:-:-:6      IADD.X dtrack1, RZ, param_W1;
    --:-:-:-:6      IADD wtrack0.CC, dtrack0, 0x30;
    --:-:-:-:6      IADD.X wtrack1, dtrack1, RZ;
    --:-:-:-:6      MOV32I t0, 0x3210;
    --:-:-:-:6      MOV32I t1, 0x7654;
    --:-:-:-:6      ISETP.NE.U32.AND P1, PT, h, RZ, PT;
  Q80SET
else
  # codes at W + rr crow + 8 h (+ b cstride), the block scale at S + rr srow (+ b sstride)
  emit.call <<~Q2SET
    --:-:-:-:6      SHL xstep, ldx, 0x4;
    --:-:-:-:6      SHL xstepB, ldx, 0x7;
    --:-:-:-:6      IADD xstepB, xstepB, xstep;
    --:-:-:-:6      MOV stride, param_crow;
    #{xmad_lo.call('t0', 'rr', 'stride', 'RZ', 't1')}
    --:-:-:-:6      SHL t2, h, 0x3;
    --:-:-:-:6      IADD t0, t0, t2;
    --:-:-:-:6      IADD wtrack0.CC, t0, param_W0;
    --:-:-:-:6      IADD.X wtrack1, RZ, param_W1;
    --:-:-:-:6      MOV stride, param_srow;
    #{xmad_lo.call('t0', 'rr', 'stride', 'RZ', 't1')}
    --:-:-:-:6      IADD hptr0.CC, t0, param_S0;
    --:-:-:-:6      IADD.X hptr1, RZ, param_S1;
    --:-:-:-:6      MOV cst, param_cstr;
    --:-:-:-:6      MOV sst, param_sstr;
  Q2SET
end
if Q3
  # the row's records at W + rr crow (hptr walks the blocks), its d area at S + rr srow; the thread's byte and nibble shifts
  emit.call <<~Q3SET
    --:-:-:-:6      SHL xstep, ldx, 0x5;
    --:-:-:-:6      MOV stride, param_crow;
    #{xmad_lo.call('t0', 'rr', 'stride', 'RZ', 't1')}
    --:-:-:-:6      IADD hptr0.CC, t0, param_W0;
    --:-:-:-:6      IADD.X hptr1, RZ, param_W1;
    --:-:-:-:6      MOV stride, param_srow;
    #{xmad_lo.call('t0', 'rr', 'stride', 'RZ', 't1')}
    --:-:-:-:6      IADD dbase0.CC, t0, param_S0;
    --:-:-:-:6      IADD.X dbase1, RZ, param_S1;
    --:-:-:-:6      SHL sh8h, h, 0x3;
    --:-:-:-:6      SHL sh4h, h, 0x2;
    --:-:-:-:6      MOV32I qn, 0x1;
    --:-:-:-:6      MOV bn, RZ;
  Q3SET
end
emit.call <<~SETUP3
  --:-:-:-:6      SHL writeW, h, 0xb;
  --:-:-:-:6      ISCADD writeW, r, writeW, 0x2;
  --:-:-:-:6      IADD writeW, writeW, 0x1000;
  --:-:-:-:6      LOP32I.AND tid128, tid, 0x80;
  --:-:-:-:6      BFE.U32 tid7, tid, 0x301;
  --:-:-:-:6      SHR.U32 readAs, tid128, 0x4;
  --:-:-:-:6      NOP;
SETUP3
emit.call '  --:-:-:-:6      SEL selh, t1, t0, P1;' if Q4K || IQ4 || Q6K || B32N || Q80
emit.call <<~SETUP4
  --:-:-:-:6      LOP.OR readAs, readAs, tid7;
  --:-:-:-:6      SHL readAs, readAs, 0x4;
  --:-:-:-:6      LOP32I.AND tid1, tid, 0x1;
  --:-:-:-:6      LOP32I.AND readBs, tid, 0x70;
  --:-:-:-:6      SHR.U32 readBs, readBs, 0x3;
  --:-:-:-:6      LOP.OR readBs, readBs, tid1;
  --:-:-:-:6      ISCADD readBs, readBs, 4x<8*128>, 0x4;
  --:-:-:-:6      SHR.U32 ki, k, 0x3;
  --:-:-:-:6      MOV pc, RZ;
SETUP4
emit.call '  --:-:-:-:6      MOV32I jn, 0x1;' if Q4K || IQ4   # (q3 keeps its byte shift in that register)
post_zero = nil   # pointer bumps that wait until after the C registers are zeroed
if Q4K
  emit.call <<~Q4LD
    --:-:5:-:1      LDG.E.128 hdr0, [hptr];
    --:-:3:-:1      LDG.E.128 wq0, [wtrack];
    #{Q5K ? '--:-:3:-:1      LDG.E hq, [hbtrack];' : ''}
    --:-:2:-:6      LDG.E.128 loadX0, [track];
    --:-:-:-:6      ISETP.GT.U32.AND P0, PT, ki, 0x1, PT;
    --:-:-:-:6      ISETP.GT.U32.AND P1, PT, ki, 0x4, PT;
    --:-:-:-:6      ISETP.NE.U32.AND P3, PT, RZ, RZ, PT;
    --:-:-:-:6      IADD wtrack0.CC, wtrack0, 0x10;
    --:-:-:-:6      IADD.X wtrack1, wtrack1, RZ;
    --:-:-:-:6      IADD hptr0.CC, hptr0, 0x10;
    --:-:-:-:6      IADD.X hptr1, hptr1, RZ;
    --:-:-:-:6      IADD track0.CC, track0, xstep;
    --:-:-:-:6      IADD.X track1, track1, RZ;
  Q4LD
  post_zero = <<~Q5BUMP if Q5K
    --:-:-:-:6      IADD hbtrack0.CC, hbtrack0, 0x4;
    --:-:-:-:6      IADD.X hbtrack1, hbtrack1, RZ;
  Q5BUMP
elsif IQ4
  # block 0's scales and d, sub-block 0's nibbles, the activations of pass 0 (P1: the sub-block prefetch, as q4k)
  # (the pointers move on after the C registers are zeroed, well after the loads read them; see q3)
  emit.call <<~IQLD
    --:-:2:-:1      LDG.E.128 loadX0, [track];
    --:-:3:-:1      LDG.E.128 wq0, [wtrack];
    --:-:5:-:1      LDG.E.64 hdr0, [hptr];
    --:-:5:-:6      LDG.E.U16 hdr2, [dbase];
    --:-:-:-:6      ISETP.GT.U32.AND P0, PT, ki, 0x1, PT;
    --:-:-:-:6      ISETP.GT.U32.AND P1, PT, ki, 0x4, PT;
    --:-:-:-:6      ISETP.NE.U32.AND P3, PT, RZ, RZ, PT;
  IQLD
  post_zero = <<~IQBUMP
    --:-:-:-:6      IADD wtrack0.CC, wtrack0, 0x10;
    --:-:-:-:6      IADD.X wtrack1, wtrack1, RZ;
    --:-:-:-:6      IADD hptr0.CC, hptr0, 0x8;
    --:-:-:-:6      IADD.X hptr1, hptr1, RZ;
    --:-:-:-:6      IADD track0.CC, track0, xstep;
    --:-:-:-:6      IADD.X track1, track1, RZ;
  IQBUMP
elsif Q6K
  # sub-block 0 (nibbles, fields, scales) and block 0's d, sub-block 1 into the prefetch registers, the activations of pass 0;
  # the loop's P3 marks the last pass of a sub-block, P6 the second (the next pass starts the second 16-k group)
  emit.call <<~Q6LD
    --:-:2:-:1      LDG.E.128 loadX0, [track];
    --:-:3:-:1      LDG.E.128 wq0, [rec];
    --:-:3:-:1      LDG.E.64 hq0, [rec+0x80];
    --:-:3:-:1      LDG.E.U16 s16, [rec+0xc0];
    --:-:5:-:1      LDG.E.U16 dhn, [dbase];
    --:-:3:-:1      LDG.E.128 sq0, [rec+0x10];
    --:-:3:-:1      LDG.E.64 sqh0, [rec+0x88];
    --:-:3:-:6      LDG.E.U16 sqs, [rec+0xc2];
    --:-:-:-:6      ISETP.GT.U32.AND P0, PT, ki, 0x1, PT;
    --:-:-:-:6      ISETP.NE.U32.AND P3, PT, RZ, RZ, PT;
    --:-:-:-:6      ISETP.NE.U32.AND P6, PT, RZ, RZ, PT;
  Q6LD
  post_zero = <<~Q6BUMP
    --:-:-:-:6      IADD track0.CC, track0, xstep;
    --:-:-:-:6      IADD.X track1, track1, RZ;
  Q6BUMP
elsif B32N
  # block 0's nibbles, (q50) high bits and d, the activations of pass 0 (P1: the block prefetch, as q4k's sub-blocks)
  emit.call <<~B32LD
    --:-:2:-:1      LDG.E.128 loadX0, [track];
    --:-:3:-:1      LDG.E.128 wq0, [wtrack];
    #{Q50 ? '--:-:3:-:1      LDG.E hq, [hbtrack];' : ''}
    --:-:5:-:6      LDG.E.U16 dsq, [dtrack];
    --:-:-:-:6      ISETP.GT.U32.AND P0, PT, ki, 0x1, PT;
    --:-:-:-:6      ISETP.GT.U32.AND P1, PT, ki, 0x4, PT;
    --:-:-:-:6      ISETP.NE.U32.AND P3, PT, RZ, RZ, PT;
  B32LD
  post_zero = <<~B32BUMP
    --:-:-:-:6      IADD wtrack0.CC, wtrack0, 0x10;
    --:-:-:-:6      IADD.X wtrack1, wtrack1, RZ;
    --:-:-:-:6      IADD dtrack0.CC, dtrack0, 0x2;
    --:-:-:-:6      IADD.X dtrack1, dtrack1, RZ;
    #{Q50 ? "--:-:-:-:6      IADD hbtrack0.CC, hbtrack0, 0x4;\n--:-:-:-:6      IADD.X hbtrack1, hbtrack1, RZ;" : ''}
    --:-:-:-:6      IADD track0.CC, track0, xstep;
    --:-:-:-:6      IADD.X track1, track1, RZ;
  B32BUMP
elsif Q80
  # quad 0 of block 0 and its d, the activations of pass 0; the loop's P3 marks an odd pass (the next one starts a quad),
  # P6 the last pass of a block (the next quad takes the new d), P1 the quad prefetch (even passes), P4 the d prefetch
  emit.call <<~Q80LD
    --:-:2:-:1      LDG.E.128 loadX0, [track];
    --:-:3:-:1      LDG.E.128 wq0, [wtrack];
    --:-:5:-:6      LDG.E.U16 dsq, [dtrack];
    --:-:-:-:6      ISETP.GT.U32.AND P0, PT, ki, 0x1, PT;
    --:-:-:-:6      ISETP.GT.U32.AND P1, PT, ki, 0x2, PT;
    --:-:-:-:6      ISETP.NE.U32.AND P3, PT, RZ, RZ, PT;
    --:-:-:-:6      ISETP.NE.U32.AND P4, PT, RZ, RZ, PT;
    --:-:-:-:6      ISETP.NE.U32.AND P6, PT, RZ, RZ, PT;
  Q80LD
  post_zero = <<~Q80BUMP
    --:-:-:-:6      IADD wtrack0.CC, wtrack0, 0x10;
    --:-:-:-:6      IADD.X wtrack1, wtrack1, RZ;
    --:-:-:-:6      IADD dtrack0.CC, dtrack0, 0x2;
    --:-:-:-:6      IADD.X dtrack1, dtrack1, RZ;
    --:-:-:-:6      IADD track0.CC, track0, xstep;
    --:-:-:-:6      IADD.X track1, track1, RZ;
  Q80BUMP
elsif Q3
  # quad 0 of block 0 (codes, high bytes, scales), quad 1 prefetched, d of block 0, the activations of pass 0; the loop's P3
  # marks the last pass of a quad, P2 an even pass (the scale word moves on after the odd pass that follows it)
  # (the activation load goes first and its pointer moves on only after the C registers are zeroed: behind seven
  # other loads it read the bumped pointer, which fed pass 0 with pass 1's activations)
  emit.call <<~Q3LD
    --:-:2:-:1      LDG.E.128 loadX0, [track];
    --:-:3:-:1      LDG.E.128 wq0, [hptr];
    --:-:3:-:1      LDG.E.64 wh0, [hptr+0x40];
    --:-:3:-:1      LDG.E ws, [hptr+0x60];
    --:-:6:-:1      LDG.E.128 sq0, [hptr+0x10];
    --:-:6:-:1      LDG.E.64 sh0, [hptr+0x48];
    --:-:6:-:1      LDG.E ss, [hptr+0x64];
    --:-:5:-:6      LDG.E.U16 dh, [dbase];
    --:-:-:-:6      ISETP.GT.U32.AND P0, PT, ki, 0x1, PT;
    --:-:-:-:6      ISETP.GT.U32.AND P1, PT, ki, 0x9, PT;
    --:-:-:-:6      ISETP.NE.U32.AND P3, PT, RZ, RZ, PT;
    --:-:-:-:6      ISETP.EQ.U32.AND P2, PT, RZ, RZ, PT;
  Q3LD
  post_zero = <<~Q3BUMP
    --:-:-:-:6      IADD track0.CC, track0, xstep;
    --:-:-:-:6      IADD.X track1, track1, RZ;
  Q3BUMP
elsif Q2
  # block 0's codes and scale, the activations of pass 0; the loop's P3 marks the last pass of a block (the next pass
  # starts a block), P6 the pass before it (the activation step to the next block), P1 the block prefetch
  emit.call <<~Q2LD
    --:-:3:-:1      LDG.E.64 wq0, [wtrack];
    --:-:5:-:1      LDG.E.U16 hdr0, [hptr];
    --:-:2:-:6      LDG.E.128 loadX0, [track];
    --:-:-:-:6      ISETP.GT.U32.AND P0, PT, ki, 0x1, PT;
    --:-:-:-:6      ISETP.GT.U32.AND P1, PT, ki, 0x8, PT;
    --:-:-:-:6      ISETP.NE.U32.AND P3, PT, RZ, RZ, PT;
    --:-:-:-:6      ISETP.NE.U32.AND P6, PT, RZ, RZ, PT;
    --:-:-:-:6      IADD wtrack0.CC, wtrack0, cst;
    --:-:-:-:6      IADD.X wtrack1, wtrack1, RZ;
    --:-:-:-:6      IADD hptr0.CC, hptr0, sst;
    --:-:-:-:6      IADD.X hptr1, hptr1, RZ;
    --:-:-:-:6      IADD track0.CC, track0, xstep;
    --:-:-:-:6      IADD.X track1, track1, RZ;
  Q2LD
else
  # groups 0 and 1 of the row, the activations of pass 0; the loop's P3 is "pass even" (the next pass is odd), P1 the group prefetch
  # (a load reads its address registers late: every pointer moves on at least 24 cycles after the load that used it)
  emit.call <<~BFLD
    --:-:3:-:1      LDG.E.128 wq0, [wtrack];
    --:-:2:-:6      LDG.E.128 loadX0, [track];
    --:-:-:-:6      ISETP.GT.U32.AND P0, PT, ki, 0x1, PT;
    --:-:-:-:6      ISETP.GT.U32.AND P1, PT, ki, 0x4, PT;
    --:-:-:-:6      ISETP.EQ.U32.AND P3, PT, RZ, RZ, PT;
    --:-:-:-:6      NOP;
    --:-:-:-:6      IADD wtrack0.CC, wtrack0, 0x20;
    --:-:-:-:6      IADD.X wtrack1, wtrack1, RZ;
    --:-:-:-:6      IADD track0.CC, track0, xstep;
    --:-:-:-:6      IADD.X track1, track1, RZ;
    --:-:3:-:6      LDG.E.128 sq0, [wtrack];
    --:-:-:-:6      NOP;
    --:-:-:-:6      NOP;
    --:-:-:-:6      NOP;
    --:-:-:-:6      IADD wtrack0.CC, wtrack0, 0x20;
    --:-:-:-:6      IADD.X wtrack1, wtrack1, RZ;
  BFLD
end

# zero C
64.times do |i|
  x = [0, 1, 2, 3, 64, 65, 66, 67][i / 8]
  y = [0, 1, 2, 3, 64, 65, 66, 67][i % 8]
  emit.call format('--:-:-:-:%d      MOV cx%02dy%02d, RZ;', i == 63 ? 6 : 1, x, y)
end
emit.call post_zero if post_zero

# the first pass dequantised into buffer 0
# q5k: the thread's 4 high bits b (then the word moves on a pass), b * 0x410 -> bits 4, 12 = elements 0, 2 (m0),
# b * 0x208 -> bits 4, 12 = elements 1, 3 (m1)
q5pre = !(Q5K || Q50) ? '' : [
  '--:-:-:-:6      BFE.U32 u1, hq, bfp;',
  '--:-:-:-:6      SHR.U32 hq, hq, 0x8;',
  '--:-:-:-:6      XMAD u3, u1, 0x410, RZ;',
  '--:-:-:-:6      XMAD u2, u1, 0x208, RZ;',
  '--:-:-:-:6      LOP3.LUT m0, m0, u3, cmask, 0xf8;',
  '--:-:-:-:6      LOP3.LUT m1, m1, u2, cmask, 0xf8;',
].join("\n    ")
if Q4K
  # scale of sub-block 0 (j = 0: scale = scales[0] & 63, min = scales[4] & 63)
  emit.call <<~PRE
    10:-:4:-:1      F2F.F32.F16 dd, hdr0;
    --:-:4:-:6      F2F.F32.F16 dm, hdr0.H1;
    --:-:-:-:6      LOP32I.AND sc, hdr1, 0x3f;
    --:-:-:-:6      LOP32I.AND mn, hdr2, 0x3f;
    --:-:4:-:1      I2F.F32.U32 scf, sc;
    --:-:4:-:6      I2F.F32.U32 mnf, mn;
    08:-:-:-:6      FMUL ws, dd, scf;
    --:-:-:-:6      FMUL nm, dm, mnf;
    04:-:-:-:6      PRMT hw, wq0, selh, RZ;
    --:-:-:-:6      LOP32I.AND m0, hw, 0x0f0f;
    --:-:-:-:6      SHR.U32 m1, hw, 0x4;
    --:-:-:-:6      LOP32I.AND m1, m1, 0x0f0f;
    #{q5pre}
    --:-:4:-:1      I2F.F32.U8 f0, m0.B0;
    --:-:4:-:1      I2F.F32.U8 f2, m0.B1;
    --:-:4:-:1      I2F.F32.U8 f1, m1.B0;
    --:-:4:-:6      I2F.F32.U8 f3, m1.B1;
    08:-:-:-:1      FFMA wv0, f0, ws, -nm;
    --:-:-:-:1      FFMA wv1, f1, ws, -nm;
    --:-:-:-:1      FFMA wv2, f2, ws, -nm;
    --:-:-:-:6      FFMA wv3, f3, ws, -nm;
    --:-:-:-:1      MOV wq0, wq1;
    --:-:-:-:1      MOV wq1, wq2;
    --:-:-:-:1      MOV wq2, wq3;
  PRE
elsif BF16
  emit.call <<~PRE
    04:-:-:-:1      SHL wv0, wq0, 0x10;
    --:-:-:-:1      LOP32I.AND wv1, wq0, 0xffff0000;
    --:-:-:-:1      SHL wv2, wq1, 0x10;
    --:-:-:-:6      LOP32I.AND wv3, wq1, 0xffff0000;
  PRE
elsif IQ4
  # ws = d sc of sub-block 0 (scale byte 0); the codebook lookup of the thread's 4 codes (bytes 2 h, 2 h + 1 of word 0)
  emit.call <<~PRE
    10:-:4:-:1      F2F.F32.F16 dd, hdr2;
    --:-:4:-:6      I2F.F32.S8 scf, hdr0;
    04:-:-:-:6      PRMT hw, wq0, selh, RZ;
    --:-:-:-:6      LOP32I.AND m0, hw, 0x7777;
    --:-:-:-:6      SHL m1, hw, 0x4;
    --:-:-:-:6      PRMT u1, tb0, m0, tb1;
    --:-:-:-:6      PRMT u2, tb2, m0, tb3;
    --:-:-:-:6      PRMT u3, m1, 0xd9c8, hw;
    --:-:-:-:6      LOP3.LUT tt, u1, u2, u3, 0xd8;
    08:-:-:-:6      FMUL ws, dd, scf;
    --:-:4:-:1      I2F.F32.S8 f0, tt.B0;
    --:-:4:-:1      I2F.F32.S8 f1, tt.B1;
    --:-:4:-:1      I2F.F32.S8 f2, tt.B2;
    --:-:4:-:6      I2F.F32.S8 f3, tt.B3;
    08:-:-:-:1      FMUL wv0, f0, ws;
    --:-:-:-:1      FMUL wv1, f1, ws;
    --:-:-:-:1      FMUL wv2, f2, ws;
    --:-:-:-:6      FMUL wv3, f3, ws;
    --:-:-:-:1      MOV wq0, wq1;
    --:-:-:-:1      MOV wq1, wq2;
    --:-:-:-:1      MOV wq2, wq3;
  PRE
elsif B32N && !IQNL
  # ws = d, nm = offset d of block 0; the first pass's nibbles (and q50's high bits) as q4k's (q5k's)
  emit.call <<~PRE
    10:-:4:-:6      F2F.F32.F16 dd, dsq;
    08:-:-:-:6      MOV ws, dd;
    --:-:-:-:6      FMUL nm, dd, #{B32OFF};
    04:-:-:-:6      PRMT hw, wq0, selh, RZ;
    --:-:-:-:6      LOP32I.AND m0, hw, 0x0f0f;
    --:-:-:-:6      SHR.U32 m1, hw, 0x4;
    --:-:-:-:6      LOP32I.AND m1, m1, 0x0f0f;
    #{q5pre}
    --:-:4:-:1      I2F.F32.U8 f0, m0.B0;
    --:-:4:-:1      I2F.F32.U8 f2, m0.B1;
    --:-:4:-:1      I2F.F32.U8 f1, m1.B0;
    --:-:4:-:6      I2F.F32.U8 f3, m1.B1;
    08:-:-:-:1      FFMA wv0, f0, ws, -nm;
    --:-:-:-:1      FFMA wv1, f1, ws, -nm;
    --:-:-:-:1      FFMA wv2, f2, ws, -nm;
    --:-:-:-:6      FFMA wv3, f3, ws, -nm;
    --:-:-:-:1      MOV wq0, wq1;
    --:-:-:-:1      MOV wq1, wq2;
    --:-:-:-:1      MOV wq2, wq3;
  PRE
elsif IQNL
  # ws = d of block 0; the first pass's codes through the codebook as iq4's
  emit.call <<~PRE
    10:-:4:-:6      F2F.F32.F16 dd, dsq;
    04:-:-:-:6      PRMT hw, wq0, selh, RZ;
    --:-:-:-:6      LOP32I.AND m0, hw, 0x7777;
    --:-:-:-:6      SHL m1, hw, 0x4;
    --:-:-:-:6      PRMT u1, tb0, m0, tb1;
    --:-:-:-:6      PRMT u2, tb2, m0, tb3;
    --:-:-:-:6      PRMT u3, m1, 0xd9c8, hw;
    --:-:-:-:6      LOP3.LUT tt, u1, u2, u3, 0xd8;
    08:-:-:-:6      MOV ws, dd;
    --:-:4:-:1      I2F.F32.S8 f0, tt.B0;
    --:-:4:-:1      I2F.F32.S8 f1, tt.B1;
    --:-:4:-:1      I2F.F32.S8 f2, tt.B2;
    --:-:4:-:6      I2F.F32.S8 f3, tt.B3;
    08:-:-:-:1      FMUL wv0, f0, ws;
    --:-:-:-:1      FMUL wv1, f1, ws;
    --:-:-:-:1      FMUL wv2, f2, ws;
    --:-:-:-:6      FMUL wv3, f3, ws;
    --:-:-:-:1      MOV wq0, wq1;
    --:-:-:-:1      MOV wq1, wq2;
    --:-:-:-:1      MOV wq2, wq3;
  PRE
elsif Q80
  # ws = d of block 0; the first pass's 4 int8 (word h of the quad), then the quad moves on two words
  emit.call <<~PRE
    10:-:4:-:6      F2F.F32.F16 dd, dsq;
    04:-:-:-:6      PRMT hw, wq0, selh, wq1;
    --:-:-:-:1      MOV wq0, wq2;
    --:-:-:-:6      MOV wq1, wq3;
    08:-:-:-:6      MOV ws, dd;
    --:-:4:-:1      I2F.F32.S8 f0, hw.B0;
    --:-:4:-:1      I2F.F32.S8 f1, hw.B1;
    --:-:4:-:1      I2F.F32.S8 f2, hw.B2;
    --:-:4:-:6      I2F.F32.S8 f3, hw.B3;
    08:-:-:-:1      FMUL wv0, f0, ws;
    --:-:-:-:1      FMUL wv1, f1, ws;
    --:-:-:-:1      FMUL wv2, f2, ws;
    --:-:-:-:6      FMUL wv3, f3, ws;
  PRE
elsif Q6K
  # ws, nm of both 16-k groups of sub-block 0; the first pass: nibbles as q4k, the fields of byte h of the group word
  emit.call <<~PRE
    10:-:4:-:1      F2F.F32.F16 dd, dhn;
    04:-:4:-:1      I2F.F32.S8 scf, s16;
    --:-:4:-:6      I2F.F32.S8 mnf, s16.B1;
    --:-:-:-:6      PRMT hw, wq0, selh, RZ;
    --:-:-:-:6      SHR.U32 u1, hq0, sh8h;
    --:-:-:-:6      LOP32I.AND m0, hw, 0x0f0f;
    --:-:-:-:6      SHR.U32 m1, hw, 0x4;
    --:-:-:-:6      LOP32I.AND u2, u1, 0xcc;
    --:-:-:-:6      LOP32I.AND u1, u1, 0x33;
    --:-:-:-:6      LOP32I.AND m1, m1, 0x0f0f;
    --:-:-:-:6      XMAD u3, u1, 0x110, RZ;
    --:-:-:-:6      XMAD u2, u2, 0x44, RZ;
    --:-:-:-:6      PRMT hq0, hq0, 0x5432, hq1;
    --:-:-:-:6      SHR.U32 hq1, hq1, 0x10;
    --:-:-:-:6      LOP3.LUT m0, m0, u3, cmask, 0xf8;
    --:-:-:-:6      LOP3.LUT m1, m1, u2, cmask, 0xf8;
    08:-:-:-:6      FMUL ws, dd, scf;
    --:-:-:-:6      FMUL ws1, dd, mnf;
    --:-:-:-:6      FMUL nm, ws, 32.0;
    --:-:-:-:6      FMUL nm1, ws1, 32.0;
    --:-:4:-:1      I2F.F32.U8 f0, m0.B0;
    --:-:4:-:1      I2F.F32.U8 f2, m0.B1;
    --:-:4:-:1      I2F.F32.U8 f1, m1.B0;
    --:-:4:-:6      I2F.F32.U8 f3, m1.B1;
    08:-:-:-:1      FFMA wv0, f0, ws, -nm;
    --:-:-:-:1      FFMA wv1, f1, ws, -nm;
    --:-:-:-:1      FFMA wv2, f2, ws, -nm;
    --:-:-:-:6      FFMA wv3, f3, ws, -nm;
    --:-:-:-:1      MOV wq0, wq1;
    --:-:-:-:1      MOV wq1, wq2;
    --:-:-:-:1      MOV wq2, wq3;
  PRE
elsif Q3
  # d / 32 of block 0; the first pass: codes = bits 8 h .. of word 0, high bits = bits 4 h .. of byte 0, scale byte 0
  emit.call <<~PRE
    10:-:4:-:6      F2F.F32.F16 dd, dh;
    08:-:-:-:6      FMUL dm, dd, 0.03125;
    04:-:-:-:6      SHR.U32 cb, wq0, sh8h;
    --:-:-:-:6      SHR.U32 hb, wh0, sh4h;
    --:-:4:-:6      I2F.F32.S8 nm, ws;
    --:-:-:-:1      BFE.U32 hw, cb, 0x200;
    --:-:-:-:1      BFE.U32 m0, cb, 0x202;
    --:-:-:-:1      BFE.U32 m1, cb, 0x204;
    --:-:-:-:1      BFE.U32 tt, cb, 0x206;
    --:-:-:-:1      BFE.U32 f0, hb, 0x100;
    --:-:-:-:1      BFE.U32 f1, hb, 0x101;
    --:-:-:-:1      BFE.U32 f2, hb, 0x102;
    --:-:-:-:6      BFE.U32 f3, hb, 0x103;
    --:-:-:-:1      PRMT wq0, wq0, 0x5432, wq1;
    --:-:-:-:1      PRMT wq1, wq1, 0x5432, wq2;
    --:-:-:-:1      PRMT wq2, wq2, 0x5432, wq3;
    --:-:-:-:1      SHR.U32 wq3, wq3, 0x10;
    --:-:-:-:1      PRMT wh0, wh0, 0x4321, wh1;
    --:-:-:-:6      SHR.U32 wh1, wh1, 0x8;
    --:-:-:-:1      ISCADD hw, f0, hw, 0x2;
    --:-:-:-:1      ISCADD m0, f1, m0, 0x2;
    --:-:-:-:1      ISCADD m1, f2, m1, 0x2;
    --:-:-:-:6      ISCADD tt, f3, tt, 0x2;
    08:-:-:-:6      FMUL nm, nm, dm;
    --:-:-:-:1      SHL hw, hw, 0x5;
    --:-:-:-:1      SHL m0, m0, 0x5;
    --:-:-:-:1      SHL m1, m1, 0x5;
    --:-:-:-:6      SHL tt, tt, 0x5;
    --:-:4:-:1      I2F.F32.S8 f0, hw;
    --:-:4:-:1      I2F.F32.S8 f1, m0;
    --:-:4:-:1      I2F.F32.S8 f2, m1;
    --:-:4:-:6      I2F.F32.S8 f3, tt;
    08:-:-:-:1      FMUL wv0, f0, nm;
    --:-:-:-:1      FMUL wv1, f1, nm;
    --:-:-:-:1      FMUL wv2, f2, nm;
    --:-:-:-:6      FMUL wv3, f3, nm;
  PRE
else
  # d and -d of block 0, byte 0 of the thread's codes (then the pair rotates one byte down)
  emit.call <<~PRE
    10:-:4:-:6      F2F.F32.F16 dd, hdr0;
    08:-:-:-:6      FADD nm, RZ, -dd;
    04:-:-:-:1      LOP32I.AND hw, wq0, 0x3;
    --:-:-:-:1      BFE.U32 m0, wq0, 0x202;
    --:-:-:-:1      BFE.U32 m1, wq0, 0x204;
    --:-:-:-:6      BFE.U32 tt, wq0, 0x206;
    --:-:-:-:6      PRMT wq0, wq0, 0x4321, wq1;
    --:-:-:-:6      SHR.U32 wq1, wq1, 0x8;
    --:-:4:-:1      I2F.F32.U32 f0, hw;
    --:-:4:-:1      I2F.F32.U32 f1, m0;
    --:-:4:-:1      I2F.F32.U32 f2, m1;
    --:-:4:-:6      I2F.F32.U32 f3, tt;
    08:-:-:-:1      FFMA wv0, f0, dd, nm;
    --:-:-:-:1      FFMA wv1, f1, dd, nm;
    --:-:-:-:1      FFMA wv2, f2, dd, nm;
    --:-:-:-:6      FFMA wv3, f3, dd, nm;
  PRE
end
# GEMM_DEBUG_PRE=1: CTA (0,0) stores, per thread, the first pass's dequantisation inputs and outputs and leaves
if ENV['GEMM_DEBUG_PRE']
  regs = Q3 ? %w[cb hb nm wv0 wv1 wv2 wv3 ws dd dm wq0 wh0] : %w[hw m0 nm wv0 wv1 wv2 wv3 ws dd dm wq0 wq1]
  emit.call <<~DBGP
    --:-:-:-:6      ISETP.EQ.U32.AND P4, PT, bx, RZ, PT;
    --:-:-:-:6      SHL tmpc, tid, 0x2;
    --:-:-:-:6      IADD pa0.CC, tmpc, param_D0;
    --:-:-:-:6      IADD.X pa1, RZ, param_D1;
    --:-:-:-:6      ISETP.EQ.U32.AND P4, PT, by, RZ, P4;
    --:-:-:-:6      NOP;
    --:-:-:-:6      NOP;
    --:-:-:-:6      NOP;
  DBGP
  regs.each_with_index { |r, i| emit.call format('--:-:-:-:1  @P4 STG.E [pa + %#x], %s;', 0x400 * i, r) }
  emit.call '--:-:-:-:6      EXIT;'
end

emit.call <<~PRE2
  --:-:-:-:1      STS [writeW + 4x<0*128>], wv0;
  --:-:-:-:1      STS [writeW + 4x<1*128>], wv1;
  --:-:-:-:1      STS [writeW + 4x<2*128>], wv2;
  --:-:-:-:1      STS [writeW + 4x<3*128>], wv3;
  02:-:-:-:6      STS.128 [writeS], loadX0;
  --:-:-:-:5      BAR.SYNC 0x0;
  --:-:-:-:1      LOP.XOR writeS, writeS, 4x<16*128>;
  --:-:-:-:6      LOP.XOR writeW, writeW, 4x<16*128>;
  --:-:-:-:1      LDS.U.128 j0Ax00, [readAs + 4x<0*128 + 00>];
  --:-:-:-:1      LDS.U.128 j0By00, [readBs + 4x<0*128 + 00>];
  --:-:-:-:1      LDS.U.128 j0Ax64, [readAs + 4x<0*128 + 64>];
  --:-:1:-:6      LDS.U.128 j0By64, [readBs + 4x<0*128 + 64>];
PRE2

# ---------------------------------------------------------------- the main loop: one 8-k pass per iteration
emit.call 'LOOP:'
if Q4K
  # sub-block boundary (pc == 3): the next sub-block's nibble quad and scale, the next block's header
  emit.call <<~DEC
    --:-:-:-:6 @!P3 BRA NODEC;
    04:-:-:-:1      MOV wq0, sq0;
    --:-:-:-:1      MOV wq1, sq1;
    --:-:-:-:1      MOV wq2, sq2;
    --:-:-:-:6      MOV wq3, sq3;
    #{Q5K ? '--:-:-:-:6      MOV hq, sqh;' : ''}
    10:-:4:-:1      F2F.F32.F16 dd, hdr0;
    --:-:4:-:6      F2F.F32.F16 dm, hdr0.H1;
    --:-:-:-:6      LOP32I.AND shj, jn, 0x3;
    --:-:-:-:6      ISETP.GE.U32.AND P2, PT, jn, 0x4, PT;
    --:-:-:-:6      ISETP.EQ.U32.AND P4, PT, jn, 0x7, PT;
    --:-:-:-:6      SHL shj, shj, 0x3;
    --:-:-:-:1      SHR.U32 u1, hdr1, shj;
    --:-:-:-:1      SHR.U32 u2, hdr2, shj;
    --:-:-:-:6      SHR.U32 u3, hdr3, shj;
    --:-:-:-:6      NOP;
    --:-:-:-:1 @!P2 LOP32I.AND sc, u1, 0x3f;
    --:-:-:-:1 @!P2 LOP32I.AND mn, u2, 0x3f;
    --:-:-:-:1  @P2 LOP32I.AND sc, u3, 0xf;
    --:-:-:-:1  @P2 SHR.U32 mn, u3, 0x4;
    --:-:-:-:1  @P2 SHR.U32 u1, u1, 0x6;
    --:-:-:-:6  @P2 SHR.U32 u2, u2, 0x6;
    --:-:-:-:1  @P2 LOP32I.AND u1, u1, 0x3;
    --:-:-:-:1  @P2 LOP32I.AND u2, u2, 0x3;
    --:-:-:-:6  @P2 LOP32I.AND mn, mn, 0xf;
    --:-:-:-:1  @P2 ISCADD sc, u1, sc, 0x4;
    --:-:-:-:6  @P2 ISCADD mn, u2, mn, 0x4;
    --:-:4:-:1      I2F.F32.U32 scf, sc;
    --:-:4:-:6      I2F.F32.U32 mnf, mn;
    --:-:-:-:6      ISETP.GT.U32.AND P4, PT, ki, 0x5, P4;
    --:-:-:-:6      IADD jn, jn, 0x1;
    --:-:-:-:6      LOP32I.AND jn, jn, 0x7;
    08:-:-:-:6      FMUL ws, dd, scf;
    --:-:-:-:6      FMUL nm, dm, mnf;
    --:-:-:-:6      NOP;
    --:-:5:-:6  @P4 LDG.E.128 hdr0, [hptr];
    --:-:-:-:6      NOP;
    --:-:-:-:6  @P4 IADD hptr0.CC, hptr0, 0x10;
    --:-:-:-:6  @P4 IADD.X hptr1, hptr1, RZ;
    NODEC:
  DEC
elsif IQ4
  # sub-block boundary (pc == 3): the next sub-block's nibble quad, ws = d sc of sub-block jn (scale byte jn); after the
  # block's last sub-block the next block's scales and d (at 16 (b mod Q) + 2 (b / Q) of the d area, b = bn + 1)
  emit.call <<~DEC
    --:-:-:-:6 @!P3 BRA NODEC;
    04:-:-:-:1      MOV wq0, sq0;
    --:-:-:-:1      MOV wq1, sq1;
    --:-:-:-:1      MOV wq2, sq2;
    --:-:-:-:6      MOV wq3, sq3;
    10:-:4:-:6      F2F.F32.F16 dd, hdr2;
    --:-:-:-:6      LOP32I.AND shj, jn, 0x3;
    --:-:-:-:6      ISETP.GE.U32.AND P2, PT, jn, 0x4, PT;
    --:-:-:-:6      ISETP.EQ.U32.AND P4, PT, jn, 0x7, PT;
    --:-:-:-:6      SHL shj, shj, 0x3;
    --:-:-:-:6      IADD u2, bn, 0x1;
    --:-:-:-:6      MOV32I u3, 0x1;
    --:-:-:-:6      SHL u3, u3, param_sstr;
    --:-:-:-:6      SEL u1, hdr1, hdr0, P2;
    --:-:-:-:6      ISETP.GT.U32.AND P4, PT, ki, 0x5, P4;
    --:-:-:-:6      SHR.U32 u1, u1, shj;
    --:-:-:-:6      IADD u3, u3, -0x1;
    --:-:4:-:6      I2F.F32.S8 scf, u1;
    --:-:-:-:6      LOP.AND u3, u2, u3;
    --:-:-:-:6      SHL u3, u3, 0x4;
    --:-:-:-:6      SHR.U32 tt, u2, param_sstr;
    --:-:-:-:6      ISCADD u3, tt, u3, 0x1;
    --:-:-:-:6      IADD f0.CC, dbase0, u3;
    --:-:-:-:6      IADD.X f1, dbase1, RZ;
    --:-:-:-:6      IADD jn, jn, 0x1;
    08:-:-:-:6      FMUL ws, dd, scf;
    --:-:-:-:6      LOP32I.AND jn, jn, 0x7;
    --:-:5:-:1  @P4 LDG.E.64 hdr0, [hptr];
    --:-:5:-:6  @P4 LDG.E.U16 hdr2, [f0];
    --:-:-:-:6      NOP;
    --:-:-:-:6      NOP;
    --:-:-:-:6      NOP;
    --:-:-:-:6  @P4 IADD hptr0.CC, hptr0, 0x8;
    --:-:-:-:6  @P4 IADD.X hptr1, hptr1, RZ;
    --:-:-:-:6  @P4 MOV bn, u2;
    NODEC:
  DEC
elsif Q6K
  # sub-block boundary (pc == 3): the sub-block's nibbles, fields and scales (loaded a sub-block ago), ws, nm of its two
  # 16-k groups (a new block's d when jc == 0); then the loads of sub-block jp of rec's block (ki > 5: it exists), after
  # the last one of a block the next record and its d (P2; never with P4, so dhn is not read and loaded at once)
  emit.call <<~DEC
    --:-:-:-:6 @!P3 BRA NODEC;
    04:-:-:-:1      MOV wq0, sq0;
    --:-:-:-:1      MOV wq1, sq1;
    --:-:-:-:1      MOV wq2, sq2;
    --:-:-:-:1      MOV wq3, sq3;
    --:-:-:-:1      MOV hq0, sqh0;
    --:-:-:-:1      MOV hq1, sqh1;
    --:-:-:-:6      MOV s16, sqs;
    --:-:-:-:6      ISETP.GT.U32.AND P1, PT, ki, 0x5, PT;
    --:-:-:-:6      ISETP.EQ.U32.AND P4, PT, jc, RZ, PT;
    --:-:4:-:1      I2F.F32.S8 scf, s16;
    --:-:4:-:6      I2F.F32.S8 mnf, s16.B1;
    --:-:-:-:6      ISETP.EQ.U32.AND P2, PT, jp, 0x8, P1;
    10:-:4:-:6  @P4 F2F.F32.F16 dd, dhn;
    --:-:-:-:6      IADD jc, jc, 0x1;
    --:-:-:-:6      LOP32I.AND jc, jc, 0x7;
    --:-:-:-:6      NOP;
    --:-:-:-:6  @P2 IADD rec0.CC, rec0, 0xd0;
    --:-:-:-:6  @P2 IADD.X rec1, rec1, RZ;
    --:-:-:-:6  @P2 MOV jp, RZ;
    --:-:-:-:6  @P2 IADD bnp, bnp, 0x1;
    --:-:-:-:6      MOV32I u3, 0x1;
    --:-:-:-:6      SHL u3, u3, param_sstr;
    --:-:-:-:6      IADD u3, u3, -0x1;
    --:-:-:-:6      LOP.AND u3, bnp, u3;
    --:-:-:-:6      SHL u3, u3, 0x4;
    --:-:-:-:6      SHR.U32 tt, bnp, param_sstr;
    --:-:-:-:6      ISCADD u3, tt, u3, 0x1;
    --:-:-:-:6      IADD td0.CC, dbase0, u3;
    --:-:-:-:6      IADD.X td1, dbase1, RZ;
    --:-:-:-:6      SHL u1, jp, 0x4;
    --:-:-:-:6      IADD ta0.CC, rec0, u1;
    --:-:-:-:6      IADD.X ta1, rec1, RZ;
    --:-:-:-:6      ISCADD u1, jp, 0x80, 0x3;
    --:-:-:-:6      IADD tc0.CC, rec0, u1;
    --:-:-:-:6      IADD.X tc1, rec1, RZ;
    --:-:-:-:6      ISCADD u1, jp, 0xc0, 0x1;
    --:-:-:-:6      IADD te0.CC, rec0, u1;
    --:-:-:-:6      IADD.X te1, rec1, RZ;
    08:-:-:-:6      FMUL ws, dd, scf;
    --:-:-:-:6      FMUL ws1, dd, mnf;
    --:-:-:-:6      FMUL nm, ws, 32.0;
    --:-:-:-:6      FMUL nm1, ws1, 32.0;
    --:-:5:-:1  @P2 LDG.E.U16 dhn, [td];
    --:-:3:-:1  @P1 LDG.E.128 sq0, [ta];
    --:-:3:-:1  @P1 LDG.E.64 sqh0, [tc];
    --:-:3:-:6  @P1 LDG.E.U16 sqs, [te];
    --:-:-:-:6      IADD jp, jp, 0x1;
    NODEC:
  DEC
elsif B32N
  # block boundary (pc == 3): the next block's nibbles, (q50) high bits and d (prefetched three passes ago)
  emit.call <<~DEC
    --:-:-:-:6 @!P3 BRA NODEC;
    04:-:-:-:1      MOV wq0, sq0;
    --:-:-:-:1      MOV wq1, sq1;
    --:-:-:-:1      MOV wq2, sq2;
    --:-:-:-:6      MOV wq3, sq3;
    #{Q50 ? '--:-:-:-:6      MOV hq, sqh;' : ''}
    10:-:4:-:6      F2F.F32.F16 dd, dsq;
    08:-:-:-:6      MOV ws, dd;
    #{IQNL ? '' : "--:-:-:-:6      FMUL nm, dd, #{B32OFF};"}
    NODEC:
  DEC
elsif Q80
  # quad boundary (pc odd): the next quad (prefetched a pass ago); at a block boundary (P6) its d
  emit.call <<~DEC
    --:-:-:-:6 @!P3 BRA NODEC;
    04:-:-:-:1      MOV wq0, sq0;
    --:-:-:-:1      MOV wq1, sq1;
    --:-:-:-:1      MOV wq2, sq2;
    --:-:-:-:6      MOV wq3, sq3;
    10:-:4:-:6  @P6 F2F.F32.F16 dd, dsq;
    08:-:-:-:6  @P6 MOV ws, dd;
    NODEC:
  DEC
elsif Q3
  # quad boundary (pc == 7): the next quad's words (prefetched seven passes ago), then the prefetch of the one after:
  # quad qn of block bn, a new block's record 112 B on and its d a quad early (ki > 9: that quad exists)
  emit.call <<~QUAD
    --:-:-:-:6 @!P3 BRA NOQD;
    20:-:-:-:1      MOV wq0, sq0;
    --:-:-:-:1      MOV wq1, sq1;
    --:-:-:-:1      MOV wq2, sq2;
    --:-:-:-:1      MOV wq3, sq3;
    --:-:-:-:1      MOV wh0, sh0;
    --:-:-:-:1      MOV wh1, sh1;
    --:-:-:-:6      MOV ws, ss;
    --:-:-:-:6      ISETP.EQ.U32.AND P4, PT, qn, RZ, PT;
    --:-:-:-:6      IADD qn, qn, 0x1;
    --:-:-:-:6      LOP32I.AND qn, qn, 0x3;
    --:-:-:-:6      ISETP.GT.U32.AND P1, PT, ki, 0x9, PT;
    --:-:-:-:6      ISETP.EQ.U32.AND P6, PT, qn, RZ, PT;
    --:-:-:-:6      NOP;
    10:-:4:-:6  @P4 F2F.F32.F16 dd, dh;
    --:-:-:-:6      NOP;
    --:-:-:-:6      NOP;
    08:-:-:-:6  @P4 FMUL dm, dd, 0.03125;
    --:-:-:-:6  @P6 IADD bn, bn, 0x1;
    --:-:-:-:6  @P6 IADD hptr0.CC, hptr0, 0x70;
    --:-:-:-:6  @P6 IADD.X hptr1, hptr1, RZ;
    --:-:-:-:6      SHL cb, qn, 0x4;
    --:-:-:-:6      IADD f0.CC, hptr0, cb;
    --:-:-:-:6      IADD.X f1, hptr1, RZ;
    --:-:-:-:6      SHL cb, qn, 0x3;
    --:-:-:-:6      IADD cb, cb, 0x40;
    --:-:-:-:6      IADD f2.CC, hptr0, cb;
    --:-:-:-:6      IADD.X f3, hptr1, RZ;
    --:-:-:-:6      SHL cb, qn, 0x2;
    --:-:-:-:6      IADD cb, cb, 0x60;
    --:-:-:-:6      IADD wv0.CC, hptr0, cb;
    --:-:-:-:6      IADD.X wv1, hptr1, RZ;
    --:-:6:-:1  @P1 LDG.E.128 sq0, [f0];
    --:-:6:-:1  @P1 LDG.E.64 sh0, [f2];
    --:-:6:-:6  @P1 LDG.E ss, [wv0];
    --:-:-:-:6  @P6 MOV32I cb, 0x1;
    --:-:-:-:6  @P6 SHL cb, cb, param_sstr;
    --:-:-:-:6  @P6 IADD cb, cb, -0x1;
    --:-:-:-:6  @P6 LOP.AND cb, bn, cb;
    --:-:-:-:6  @P6 SHL cb, cb, 0x4;
    --:-:-:-:6  @P6 SHR.U32 hb, bn, param_sstr;
    --:-:-:-:6  @P6 ISCADD cb, hb, cb, 0x1;
    --:-:-:-:6  @P6 IADD wv2.CC, dbase0, cb;
    --:-:-:-:6  @P6 IADD.X wv3, dbase1, RZ;
    --:-:5:-:6  @P6 LDG.E.U16 dh, [wv2];
    NOQD:
  QUAD
elsif Q2
  # block boundary (pc == 7): the next block's codes (prefetched seven passes ago) and its d, -d
  emit.call <<~BLK
    --:-:-:-:6 @!P3 BRA NOBLK;
    04:-:-:-:1      MOV wq0, sq0;
    --:-:-:-:6      MOV wq1, sq1;
    10:-:4:-:6      F2F.F32.F16 dd, hdr0;
    08:-:-:-:6      FADD nm, RZ, -dd;
    NOBLK:
  BLK
end

swirl = [[2, 0], [2, 1], [0, 1], [0, 0]]
xvals = [0, 1, 64, 65]
corder = []
[0, 2, 64, 66].each do |y|
  xvals.each { |x| swirl.each { |dx, dy| corder << [x + dx, y + dy] } }
  xvals = xvals.reverse
end

insert = {}
if Q4K || Q40 || Q50
  insert.merge!(
    # the dequantisation of the next pass's 4 weights (the word of the sub-block quad, two bytes of it per half)
    'j0c1'  => "--:-:-:-:1      PRMT hw, wq0, selh, RZ;\n",
    'j0c8'  => "--:-:-:-:1      LOP32I.AND m0, hw, 0x0f0f;\n",
    'j0c9'  => "--:-:-:-:1      SHR.U32 m1, hw, 0x4;\n",
    'j0c10' => "--:-:-:-:1      MOV wq0, wq1;\n",
    'j0c11' => "--:-:-:-:1      MOV wq1, wq2;\n",
    'j0c12' => "--:-:-:-:1      MOV wq2, wq3;\n",
    'j0c16' => "--:-:-:-:1      LOP32I.AND m1, m1, 0x0f0f;\n",
    'j0c20' => "--:-:4:-:1      I2F.F32.U8 f0, m0.B0;\n",
    'j0c22' => "--:-:4:-:1      I2F.F32.U8 f2, m0.B1;\n",
    'j0c24' => "--:-:4:-:1      I2F.F32.U8 f1, m1.B0;\n",
    'j0c26' => "--:-:4:-:1      I2F.F32.U8 f3, m1.B1;\n",
    # the next pass's activations and (every fourth pass) the next sub-block's nibble quad
    'j0c31' => "--:-:2:-:1  @P0 LDG.E.128 loadX0, [track];\n",
    'j0c33' => "--:-:3:-:1  @P1 LDG.E.128 sq0, [wtrack];\n",
    'j0c40' => "08:-:-:-:1      FFMA wv0, f0, ws, -nm;\n",
    'j0c42' => "--:-:-:-:1      FFMA wv1, f1, ws, -nm;\n",
    'j0c44' => "--:-:-:-:1      FFMA wv2, f2, ws, -nm;\n",
    'j0c46' => "--:-:-:-:1      FFMA wv3, f3, ws, -nm;\n",
    'j1c24' => "--:-:-:-:1  @P1 IADD wtrack0.CC, wtrack0, 0x10;\n",
    'j1c32' => "--:-:-:-:1  @P1 IADD.X wtrack1, wtrack1, RZ;\n",
    'j7c48' => "--:-:-:-:1      IADD pc, pc, 0x1;\n",
    'j7c50' => "--:-:-:-:1      IADD ki, ki, -0x1;\n",
    'j7c56' => "--:-:-:-:1      LOP32I.AND pc, pc, 0x3;\n",
    'j7c58' => "--:-:-:-:1      IADD track0.CC, track0, xstep;\n",
    # ki counts the passes left including the next one: the loop goes on while one is left (P5), and that pass
    # prefetches (P0) only when another follows it
    'j7c63' => "--:-:-:-:6      ISETP.NE.U32.AND P5, PT, ki, RZ, PT;\n" \
               "--:-:-:-:6      ISETP.GT.U32.AND P0, PT, ki, 0x1, PT;\n" \
               "--:-:-:-:6      ISETP.GT.U32.AND P1, PT, ki, 0x4, PT;\n" \
               "--:-:-:-:6      ISETP.EQ.U32.AND P3, PT, pc, 0x3, PT;\n" \
               "--:-:-:-:6      IADD.X track1, track1, RZ;\n" \
               "--:-:-:-:6      ISETP.EQ.U32.AND P1, PT, pc, RZ, P1;\n" \
               "--:-:-:Y:5  @P5 BRA LOOP;\n",
  )
  if Q5K || Q50
    # the high bits go into m0, m1 before the byte conversions, which move later (as iq4's) and so do the weights
    %w[j0c20 j0c22 j0c24 j0c26 j0c40 j0c42 j0c44 j0c46].each { |s| insert.delete(s) }
    insert.merge!(
      'j0c13' => "--:-:-:-:1      BFE.U32 u1, hq, bfp;\n",
      'j0c14' => "--:-:-:-:1      SHR.U32 hq, hq, 0x8;\n",
      'j0c21' => "--:-:-:-:1      XMAD u3, u1, 0x410, RZ;\n",
      'j0c23' => "--:-:-:-:1      XMAD u2, u1, 0x208, RZ;\n",
      'j0c28' => "--:-:-:-:1      LOP3.LUT m0, m0, u3, cmask, 0xf8;\n",
      'j0c30' => "--:-:-:-:1      LOP3.LUT m1, m1, u2, cmask, 0xf8;\n",
      'j0c34' => "--:-:3:-:1  @P1 LDG.E sqh, [hbtrack];\n",
      'j0c37' => "--:-:4:-:1      I2F.F32.U8 f0, m0.B0;\n",
      'j0c39' => "--:-:4:-:1      I2F.F32.U8 f2, m0.B1;\n",
      'j0c41' => "--:-:4:-:1      I2F.F32.U8 f1, m1.B0;\n",
      'j0c43' => "--:-:4:-:1      I2F.F32.U8 f3, m1.B1;\n",
      'j0c50' => "08:-:-:-:1      FFMA wv0, f0, ws, -nm;\n",
      'j0c52' => "--:-:-:-:1      FFMA wv1, f1, ws, -nm;\n",
      'j0c54' => "--:-:-:-:1      FFMA wv2, f2, ws, -nm;\n",
      'j0c56' => "--:-:-:-:1      FFMA wv3, f3, ws, -nm;\n",
      'j1c40' => "--:-:-:-:1  @P1 IADD hbtrack0.CC, hbtrack0, 0x4;\n",
      'j1c48' => "--:-:-:-:1  @P1 IADD.X hbtrack1, hbtrack1, RZ;\n",
    )
  end
elsif Q6K
  insert.merge!(
    # the next pass's 4 nibbles (as q4k) and fields (byte h of the group word's half, then the pair moves 16 bits on);
    # at the third pass of a sub-block (P6) the second group's scale takes over
    'j0c1'  => "--:-:-:-:1      PRMT hw, wq0, selh, RZ;\n",
    'j0c3'  => "--:-:-:-:1  @P6 MOV ws, ws1;\n",
    'j0c5'  => "--:-:-:-:1  @P6 MOV nm, nm1;\n",
    'j0c7'  => "--:-:-:-:1      SHR.U32 u1, hq0, sh8h;\n",
    'j0c8'  => "--:-:-:-:1      LOP32I.AND m0, hw, 0x0f0f;\n",
    'j0c9'  => "--:-:-:-:1      SHR.U32 m1, hw, 0x4;\n",
    'j0c10' => "--:-:-:-:1      MOV wq0, wq1;\n",
    'j0c11' => "--:-:-:-:1      MOV wq1, wq2;\n",
    'j0c12' => "--:-:-:-:1      MOV wq2, wq3;\n",
    'j0c14' => "--:-:-:-:1      LOP32I.AND u2, u1, 0xcc;\n",
    'j0c15' => "--:-:-:-:1      LOP32I.AND u1, u1, 0x33;\n",
    'j0c16' => "--:-:-:-:1      LOP32I.AND m1, m1, 0x0f0f;\n",
    'j0c17' => "--:-:-:-:1      PRMT hq0, hq0, 0x5432, hq1;\n",
    'j0c18' => "--:-:-:-:1      SHR.U32 hq1, hq1, 0x10;\n",
    'j0c22' => "--:-:-:-:1      XMAD u3, u1, 0x110, RZ;\n",
    'j0c23' => "--:-:-:-:1      XMAD u2, u2, 0x44, RZ;\n",
    'j0c29' => "--:-:-:-:1      LOP3.LUT m0, m0, u3, cmask, 0xf8;\n",
    'j0c30' => "--:-:-:-:1      LOP3.LUT m1, m1, u2, cmask, 0xf8;\n",
    'j0c31' => "--:-:2:-:1  @P0 LDG.E.128 loadX0, [track];\n",
    'j0c37' => "--:-:4:-:1      I2F.F32.U8 f0, m0.B0;\n",
    'j0c39' => "--:-:4:-:1      I2F.F32.U8 f2, m0.B1;\n",
    'j0c41' => "--:-:4:-:1      I2F.F32.U8 f1, m1.B0;\n",
    'j0c43' => "--:-:4:-:1      I2F.F32.U8 f3, m1.B1;\n",
    'j0c50' => "08:-:-:-:1      FFMA wv0, f0, ws, -nm;\n",
    'j0c52' => "--:-:-:-:1      FFMA wv1, f1, ws, -nm;\n",
    'j0c54' => "--:-:-:-:1      FFMA wv2, f2, ws, -nm;\n",
    'j0c56' => "--:-:-:-:1      FFMA wv3, f3, ws, -nm;\n",
    'j7c48' => "--:-:-:-:1      IADD pc, pc, 0x1;\n",
    'j7c50' => "--:-:-:-:1      IADD ki, ki, -0x1;\n",
    'j7c56' => "--:-:-:-:1      LOP32I.AND pc, pc, 0x3;\n",
    'j7c58' => "--:-:-:-:1      IADD track0.CC, track0, xstep;\n",
    'j7c63' => "--:-:-:-:6      ISETP.NE.U32.AND P5, PT, ki, RZ, PT;\n" \
               "--:-:-:-:6      ISETP.EQ.U32.AND P6, PT, pc, 0x1, PT;\n" \
               "--:-:-:-:6      ISETP.GT.U32.AND P0, PT, ki, 0x1, PT;\n" \
               "--:-:-:-:6      ISETP.EQ.U32.AND P3, PT, pc, 0x3, PT;\n" \
               "--:-:-:-:6      IADD.X track1, track1, RZ;\n" \
               "--:-:-:-:6      NOP;\n" \
               "--:-:-:Y:5  @P5 BRA LOOP;\n",
  )
elsif Q80
  insert.merge!(
    # the next pass's 4 int8 (word h of the quad's first two), then the quad moves on two words
    'j0c1'  => "--:-:-:-:1      PRMT hw, wq0, selh, wq1;\n",
    'j0c8'  => "--:-:-:-:1      MOV wq0, wq2;\n",
    'j0c9'  => "--:-:-:-:1      MOV wq1, wq3;\n",
    'j0c15' => "--:-:4:-:1      I2F.F32.S8 f0, hw.B0;\n",
    'j0c17' => "--:-:4:-:1      I2F.F32.S8 f1, hw.B1;\n",
    'j0c19' => "--:-:4:-:1      I2F.F32.S8 f2, hw.B2;\n",
    'j0c21' => "--:-:4:-:1      I2F.F32.S8 f3, hw.B3;\n",
    # the next pass's activations, at an even pass the next quad and at the third pass of a block the next d
    'j0c31' => "--:-:2:-:1  @P0 LDG.E.128 loadX0, [track];\n",
    'j0c33' => "--:-:3:-:1  @P1 LDG.E.128 sq0, [wtrack];\n",
    'j0c35' => "--:-:5:-:1  @P4 LDG.E.U16 dsq, [dtrack];\n",
    'j0c40' => "08:-:-:-:1      FMUL wv0, f0, ws;\n",
    'j0c42' => "--:-:-:-:1      FMUL wv1, f1, ws;\n",
    'j0c44' => "--:-:-:-:1      FMUL wv2, f2, ws;\n",
    'j0c46' => "--:-:-:-:1      FMUL wv3, f3, ws;\n",
    'j1c24' => "--:-:-:-:1  @P1 IADD wtrack0.CC, wtrack0, 0x10;\n",
    'j1c32' => "--:-:-:-:1  @P1 IADD.X wtrack1, wtrack1, RZ;\n",
    'j1c40' => "--:-:-:-:1  @P4 IADD dtrack0.CC, dtrack0, 0x2;\n",
    'j1c48' => "--:-:-:-:1  @P4 IADD.X dtrack1, dtrack1, RZ;\n",
    'j7c48' => "--:-:-:-:1      IADD pc, pc, 0x1;\n",
    'j7c50' => "--:-:-:-:1      IADD ki, ki, -0x1;\n",
    'j7c56' => "--:-:-:-:1      LOP32I.AND pc, pc, 0x3;\n",
    'j7c58' => "--:-:-:-:1      IADD track0.CC, track0, xstep;\n",
    'j7c62' => "--:-:-:-:1      LOP32I.AND tt, pc, 0x1;\n",
    # P1: a quad follows the next one and the next pass is even; P4: that quad starts a block (the next pass is the
    # third of its block); both read P1 = ki > 2 well after it is set
    'j7c63' => "--:-:-:-:6      ISETP.NE.U32.AND P5, PT, ki, RZ, PT;\n" \
               "--:-:-:-:6      ISETP.GT.U32.AND P1, PT, ki, 0x2, PT;\n" \
               "--:-:-:-:6      ISETP.GT.U32.AND P0, PT, ki, 0x1, PT;\n" \
               "--:-:-:-:6      ISETP.NE.U32.AND P3, PT, tt, RZ, PT;\n" \
               "--:-:-:-:6      ISETP.EQ.U32.AND P6, PT, pc, 0x3, PT;\n" \
               "--:-:-:-:6      IADD.X track1, track1, RZ;\n" \
               "--:-:-:-:6      ISETP.EQ.U32.AND P4, PT, pc, 0x2, P1;\n" \
               "--:-:-:-:6      ISETP.EQ.U32.AND P1, PT, tt, RZ, P1;\n" \
               "--:-:-:Y:5  @P5 BRA LOOP;\n",
  )
elsif IQ4 || IQNL
  insert.merge!(
    # the next pass's 4 codes (two bytes of the quad's word per half): codebook lookup, byte select, I2F, times d sc
    'j0c1'  => "--:-:-:-:1      PRMT hw, wq0, selh, RZ;\n",
    'j0c8'  => "--:-:-:-:1      LOP32I.AND m0, hw, 0x7777;\n",
    'j0c9'  => "--:-:-:-:1      SHL m1, hw, 0x4;\n",
    'j0c10' => "--:-:-:-:1      MOV wq0, wq1;\n",
    'j0c11' => "--:-:-:-:1      MOV wq1, wq2;\n",
    'j0c12' => "--:-:-:-:1      MOV wq2, wq3;\n",
    'j0c16' => "--:-:-:-:1      PRMT u1, tb0, m0, tb1;\n",
    'j0c17' => "--:-:-:-:1      PRMT u2, tb2, m0, tb3;\n",
    'j0c18' => "--:-:-:-:1      PRMT u3, m1, 0xd9c8, hw;\n",
    'j0c25' => "--:-:-:-:1      LOP3.LUT tt, u1, u2, u3, 0xd8;\n",
    'j0c31' => "--:-:2:-:1  @P0 LDG.E.128 loadX0, [track];\n",
    'j0c33' => "--:-:3:-:1  @P1 LDG.E.128 sq0, [wtrack];\n",
    'j0c35' => "--:-:4:-:1      I2F.F32.S8 f0, tt.B0;\n",
    'j0c37' => "--:-:4:-:1      I2F.F32.S8 f1, tt.B1;\n",
    'j0c39' => "--:-:4:-:1      I2F.F32.S8 f2, tt.B2;\n",
    'j0c41' => "--:-:4:-:1      I2F.F32.S8 f3, tt.B3;\n",
    'j0c50' => "08:-:-:-:1      FMUL wv0, f0, ws;\n",
    'j0c52' => "--:-:-:-:1      FMUL wv1, f1, ws;\n",
    'j0c54' => "--:-:-:-:1      FMUL wv2, f2, ws;\n",
    'j0c56' => "--:-:-:-:1      FMUL wv3, f3, ws;\n",
    'j1c24' => "--:-:-:-:1  @P1 IADD wtrack0.CC, wtrack0, 0x10;\n",
    'j1c32' => "--:-:-:-:1  @P1 IADD.X wtrack1, wtrack1, RZ;\n",
    'j7c48' => "--:-:-:-:1      IADD pc, pc, 0x1;\n",
    'j7c50' => "--:-:-:-:1      IADD ki, ki, -0x1;\n",
    'j7c56' => "--:-:-:-:1      LOP32I.AND pc, pc, 0x3;\n",
    'j7c58' => "--:-:-:-:1      IADD track0.CC, track0, xstep;\n",
    'j7c63' => "--:-:-:-:6      ISETP.NE.U32.AND P5, PT, ki, RZ, PT;\n" \
               "--:-:-:-:6      ISETP.GT.U32.AND P0, PT, ki, 0x1, PT;\n" \
               "--:-:-:-:6      ISETP.GT.U32.AND P1, PT, ki, 0x4, PT;\n" \
               "--:-:-:-:6      ISETP.EQ.U32.AND P3, PT, pc, 0x3, PT;\n" \
               "--:-:-:-:6      IADD.X track1, track1, RZ;\n" \
               "--:-:-:-:6      ISETP.EQ.U32.AND P1, PT, pc, RZ, P1;\n" \
               "--:-:-:Y:5  @P5 BRA LOOP;\n",
  )
elsif Q3
  insert.merge!(
    # the next pass's codes (byte h of word 0), high bits (nibble h of byte 0) and scale (byte 0); the words then rotate
    'j0c1'  => "--:-:-:-:1      SHR.U32 cb, wq0, sh8h;\n",
    'j0c3'  => "--:-:-:-:1      SHR.U32 hb, wh0, sh4h;\n",
    'j0c5'  => "--:-:4:-:1      I2F.F32.S8 nm, ws;\n",
    'j0c7'  => "--:-:-:-:1      BFE.U32 hw, cb, 0x200;\n",
    'j0c8'  => "--:-:-:-:1      BFE.U32 m0, cb, 0x202;\n",
    'j0c9'  => "--:-:-:-:1      BFE.U32 m1, cb, 0x204;\n",
    'j0c10' => "--:-:-:-:1      BFE.U32 tt, cb, 0x206;\n",
    'j0c11' => "--:-:-:-:1      BFE.U32 f0, hb, 0x100;\n",
    'j0c12' => "--:-:-:-:1      BFE.U32 f1, hb, 0x101;\n",
    'j0c13' => "--:-:-:-:1      BFE.U32 f2, hb, 0x102;\n",
    'j0c14' => "--:-:-:-:1      BFE.U32 f3, hb, 0x103;\n",
    'j0c15' => "--:-:-:-:1      PRMT wq0, wq0, 0x5432, wq1;\n",
    'j0c16' => "--:-:-:-:1      PRMT wq1, wq1, 0x5432, wq2;\n",
    'j0c17' => "--:-:-:-:1      PRMT wq2, wq2, 0x5432, wq3;\n",
    'j0c18' => "--:-:-:-:1      SHR.U32 wq3, wq3, 0x10;\n",
    'j0c19' => "--:-:-:-:1      PRMT wh0, wh0, 0x4321, wh1;\n",
    'j0c20' => "--:-:-:-:1      SHR.U32 wh1, wh1, 0x8;\n",
    'j0c21' => "--:-:-:-:1  @P2 SHR.U32 ws, ws, 0x8;\n",
    'j0c23' => "--:-:-:-:1      ISCADD hw, f0, hw, 0x2;\n",
    'j0c24' => "--:-:-:-:1      ISCADD m0, f1, m0, 0x2;\n",
    'j0c25' => "--:-:-:-:1      ISCADD m1, f2, m1, 0x2;\n",
    'j0c26' => "--:-:-:-:1      ISCADD tt, f3, tt, 0x2;\n",
    'j0c27' => "08:-:-:-:1      FMUL nm, nm, dm;\n",
    'j0c29' => "--:-:-:-:1      SHL hw, hw, 0x5;\n",
    'j0c30' => "--:-:-:-:1      SHL m0, m0, 0x5;\n",
    'j0c31' => "--:-:2:-:1  @P0 LDG.E.128 loadX0, [track];\n",
    'j0c33' => "--:-:-:-:1      SHL m1, m1, 0x5;\n",
    'j0c34' => "--:-:-:-:1      SHL tt, tt, 0x5;\n",
    'j0c37' => "--:-:4:-:1      I2F.F32.S8 f0, hw;\n",
    'j0c39' => "--:-:4:-:1      I2F.F32.S8 f1, m0;\n",
    'j0c41' => "--:-:4:-:1      I2F.F32.S8 f2, m1;\n",
    'j0c43' => "--:-:4:-:1      I2F.F32.S8 f3, tt;\n",
    'j0c50' => "08:-:-:-:1      FMUL wv0, f0, nm;\n",
    'j0c52' => "--:-:-:-:1      FMUL wv1, f1, nm;\n",
    'j0c54' => "--:-:-:-:1      FMUL wv2, f2, nm;\n",
    'j0c56' => "--:-:-:-:1      FMUL wv3, f3, nm;\n",
    'j7c48' => "--:-:-:-:1      IADD pc, pc, 0x1;\n",
    'j7c50' => "--:-:-:-:1      IADD ki, ki, -0x1;\n",
    'j7c56' => "--:-:-:-:1      LOP32I.AND pc, pc, 0x7;\n",
    'j7c58' => "--:-:-:-:1      IADD track0.CC, track0, xstep;\n",
    'j7c62' => "--:-:-:-:1      LOP32I.AND cb, pc, 0x1;\n",
    'j7c63' => "--:-:-:-:6      ISETP.NE.U32.AND P5, PT, ki, RZ, PT;\n" \
               "--:-:-:-:6      ISETP.GT.U32.AND P0, PT, ki, 0x1, PT;\n" \
               "--:-:-:-:6      ISETP.EQ.U32.AND P3, PT, pc, 0x7, PT;\n" \
               "--:-:-:-:6      ISETP.EQ.U32.AND P2, PT, cb, RZ, PT;\n" \
               "--:-:-:-:6      IADD.X track1, track1, RZ;\n" \
               "--:-:-:-:6      NOP;\n" \
               "--:-:-:Y:5  @P5 BRA LOOP;\n",
  )
elsif Q2
  insert.merge!(
    # the next pass's 4 codes are the low byte of the pair, which then rotates one byte down; weight = d q - d
    'j0c1'  => "--:-:-:-:1      LOP32I.AND hw, wq0, 0x3;\n",
    'j0c3'  => "--:-:-:-:1      BFE.U32 m0, wq0, 0x202;\n",
    'j0c5'  => "--:-:-:-:1      BFE.U32 m1, wq0, 0x204;\n",
    'j0c7'  => "--:-:-:-:1      BFE.U32 tt, wq0, 0x206;\n",
    'j0c13' => "--:-:-:-:1      PRMT wq0, wq0, 0x4321, wq1;\n",
    'j0c14' => "--:-:-:-:1      SHR.U32 wq1, wq1, 0x8;\n",
    'j0c15' => "--:-:4:-:1      I2F.F32.U32 f0, hw;\n",
    'j0c17' => "--:-:4:-:1      I2F.F32.U32 f1, m0;\n",
    'j0c19' => "--:-:4:-:1      I2F.F32.U32 f2, m1;\n",
    'j0c21' => "--:-:4:-:1      I2F.F32.U32 f3, tt;\n",
    # the next pass's activations and (every eighth pass) the next block's codes and scale
    'j0c31' => "--:-:2:-:1  @P0 LDG.E.128 loadX0, [track];\n",
    'j0c33' => "--:-:3:-:1  @P1 LDG.E.64 sq0, [wtrack];\n",
    'j0c35' => "--:-:5:-:1  @P1 LDG.E.U16 hdr0, [hptr];\n",
    'j0c40' => "08:-:-:-:1      FFMA wv0, f0, dd, nm;\n",
    'j0c42' => "--:-:-:-:1      FFMA wv1, f1, dd, nm;\n",
    'j0c44' => "--:-:-:-:1      FFMA wv2, f2, dd, nm;\n",
    'j0c46' => "--:-:-:-:1      FFMA wv3, f3, dd, nm;\n",
    'j1c24' => "--:-:-:-:1  @P1 IADD wtrack0.CC, wtrack0, cst;\n",
    'j1c32' => "--:-:-:-:1  @P1 IADD.X wtrack1, wtrack1, RZ;\n",
    'j1c40' => "--:-:-:-:1  @P1 IADD hptr0.CC, hptr0, sst;\n",
    'j1c48' => "--:-:-:-:1  @P1 IADD.X hptr1, hptr1, RZ;\n",
    'j7c40' => "--:-:-:-:1      SEL xs, xstepB, xstep, P6;\n",
    'j7c48' => "--:-:-:-:1      IADD pc, pc, 0x1;\n",
    'j7c50' => "--:-:-:-:1      IADD ki, ki, -0x1;\n",
    'j7c56' => "--:-:-:-:1      LOP32I.AND pc, pc, 0x7;\n",
    'j7c58' => "--:-:-:-:1      IADD track0.CC, track0, xs;\n",
    'j7c63' => "--:-:-:-:6      ISETP.NE.U32.AND P5, PT, ki, RZ, PT;\n" \
               "--:-:-:-:6      ISETP.GT.U32.AND P0, PT, ki, 0x1, PT;\n" \
               "--:-:-:-:6      ISETP.GT.U32.AND P1, PT, ki, 0x8, PT;\n" \
               "--:-:-:-:6      ISETP.EQ.U32.AND P3, PT, pc, 0x7, PT;\n" \
               "--:-:-:-:6      ISETP.EQ.U32.AND P6, PT, pc, 0x6, PT;\n" \
               "--:-:-:-:6      IADD.X track1, track1, RZ;\n" \
               "--:-:-:-:6      ISETP.EQ.U32.AND P1, PT, pc, RZ, P1;\n" \
               "--:-:-:Y:5  @P5 BRA LOOP;\n",
  )
else
  insert.merge!(
    # the next pass's 4 weights: words 2,3 of the group quad after an even pass (P3), words 0,1 after an odd one
    'j0c1'  => "--:-:-:-:1      SEL hw, wq2, wq0, P3;\n",
    'j0c3'  => "--:-:-:-:1      SEL m0, wq3, wq1, P3;\n",
    'j0c8'  => "--:-:-:-:1      SHL wv0, hw, 0x10;\n",
    'j0c9'  => "--:-:-:-:1      LOP32I.AND wv1, hw, 0xffff0000;\n",
    'j0c10' => "--:-:-:-:1      SHL wv2, m0, 0x10;\n",
    'j0c11' => "--:-:-:-:1      LOP32I.AND wv3, m0, 0xffff0000;\n",
    # after an even pass the quad moves on to the next group (prefetched two passes ago) and the one after is fetched
    'j0c16' => "04:-:-:-:1  @P3 MOV wq0, sq0;\n",
    'j0c17' => "--:-:-:-:1  @P3 MOV wq1, sq1;\n",
    'j0c18' => "--:-:-:-:1  @P3 MOV wq2, sq2;\n",
    'j0c19' => "--:-:-:-:1  @P3 MOV wq3, sq3;\n",
    'j0c31' => "--:-:2:-:1  @P0 LDG.E.128 loadX0, [track];\n",
    'j0c33' => "--:-:3:-:1  @P1 LDG.E.128 sq0, [wtrack];\n",
    'j1c24' => "--:-:-:-:1  @P1 IADD wtrack0.CC, wtrack0, 0x20;\n",
    'j1c32' => "--:-:-:-:1  @P1 IADD.X wtrack1, wtrack1, RZ;\n",
    'j7c40' => "--:-:-:-:1      SEL xs, xstepB, xstep, P3;\n",
    'j7c48' => "--:-:-:-:1      IADD pc, pc, 0x1;\n",
    'j7c50' => "--:-:-:-:1      IADD ki, ki, -0x1;\n",
    'j7c56' => "--:-:-:-:1      LOP32I.AND pc, pc, 0x1;\n",
    'j7c58' => "--:-:-:-:1      IADD track0.CC, track0, xs;\n",
    'j7c63' => "--:-:-:-:6      ISETP.NE.U32.AND P5, PT, ki, RZ, PT;\n" \
               "--:-:-:-:6      ISETP.GT.U32.AND P0, PT, ki, 0x1, PT;\n" \
               "--:-:-:-:6      ISETP.GT.U32.AND P1, PT, ki, 0x4, PT;\n" \
               "--:-:-:-:6      ISETP.EQ.U32.AND P3, PT, pc, RZ, PT;\n" \
               "--:-:-:-:6      IADD.X track1, track1, RZ;\n" \
               "--:-:-:-:6      ISETP.EQ.U32.AND P1, PT, pc, RZ, P1;\n" \
               "--:-:-:Y:5  @P5 BRA LOOP;\n",
  )
end
if B32N
  # the next block's d with its nibbles (P1, the first pass of a block); the d pointer moves on 2 B
  insert.merge!(
    'j0c36' => "--:-:5:-:1  @P1 LDG.E.U16 dsq, [dtrack];\n",
    'j1c52' => "--:-:-:-:1  @P1 IADD dtrack0.CC, dtrack0, 0x2;\n",
    'j1c60' => "--:-:-:-:1  @P1 IADD.X dtrack1, dtrack1, RZ;\n",
  )
end
insert.merge!(
  'j1c10' => "--:-:-:-:1      STS [writeW + 4x<0*128>], wv0;\n",
  'j1c12' => "--:-:-:-:1      STS [writeW + 4x<1*128>], wv1;\n",
  'j1c14' => "--:-:-:-:1      STS [writeW + 4x<2*128>], wv2;\n",
  'j1c16' => "--:-:-:-:1      STS [writeW + 4x<3*128>], wv3;\n",
  'j6c30' => "02:-:-:-:1  @P0 STS.128 [writeS], loadX0;\n",
  'j6c62' => "01:-:-:-:5      BAR.SYNC 0x0;\n" \
             "--:-:-:-:1  @P0 LOP.XOR readAs, readAs, 4x<16*128>;\n" \
             "--:-:-:-:1  @P0 LOP.XOR readBs, readBs, 4x<16*128>;\n" \
             "--:-:-:-:1  @P0 LOP.XOR writeS, writeS, 4x<16*128>;\n" \
             "--:-:-:-:1  @P0 LOP.XOR writeW, writeW, 4x<16*128>;\n",
)

8.times do |j|
  odd = j & 1
  nodd = 1 - odd
  rs = (j + 1) % 8
  pred = j == 7 ? '@P0' : '   '
  %w[0 2 4 6].each { |c| abort "slot j#{j}c#{c} is the fragment loads'" if insert.key?("j#{j}c#{c}") }
  insert["j#{j}c0"] = format("--:-:-:-:1  %s LDS.U.128 j%dAx00, [readAs + 4x<%d*128 + 00>];\n", pred, nodd, rs)
  insert["j#{j}c2"] = format("--:-:-:-:1  %s LDS.U.128 j%dBy00, [readBs + 4x<%d*128 + 00>];\n", pred, nodd, rs)
  insert["j#{j}c4"] = format("--:-:-:-:1  %s LDS.U.128 j%dAx64, [readAs + 4x<%d*128 + 64>];\n", pred, nodd, rs)
  insert["j#{j}c6"] = format("--:-:1:-:1  %s LDS.U.128 j%dBy64, [readBs + 4x<%d*128 + 64>];\n", pred, nodd, rs)
  64.times do |c|
    x, y = corder[c]
    ins = insert["j#{j}c#{c}"] || ''
    yld = c == 32 ? 'Y' : '-'
    wait = c == 0 && j < 7 ? '01' : '--'
    stall = ins =~ /LDS|LDG|STS|BAR/ ? 0 : 1
    out << format("%s:-:-:%s:%d      FFMA cx%02dy%02d, j%dAx%02d, j%dBy%02d, cx%02dy%02d;\n%s", wait, yld, stall, x, y, odd, x, odd, y, x, y, ins)
  end
end

# GEMM_DEBUG=1: CTA (0,0) stores per thread its first C value and a few state registers to the debug buffer
if ENV['GEMM_DEBUG']
  emit.call <<~DBG
    --:-:-:-:6      ISETP.EQ.U32.AND P4, PT, bx, RZ, PT;
    --:-:-:-:6      SHL tmpc, tid, 0x2;
    --:-:-:-:6      IADD pa0.CC, tmpc, param_D0;
    --:-:-:-:6      IADD.X pa1, RZ, param_D1;
    --:-:-:-:6      ISETP.EQ.U32.AND P4, PT, by, RZ, P4;
    --:-:-:-:6      NOP;
    --:-:-:-:6      NOP;
    --:-:-:-:6      NOP;
    --:-:-:-:1  @P4 STG.E [pa], cx00y00;
    --:-:-:-:1  @P4 STG.E [pa + 0x400], ws;
    --:-:-:-:1  @P4 STG.E [pa + 0x800], nm;
    --:-:-:-:1  @P4 STG.E [pa + 0xc00], wv0;
    --:-:-:-:1  @P4 STG.E [pa + 0x1000], j0Ax00;
    --:-:-:-:1  @P4 STG.E [pa + 0x1400], wq0;
    --:-:-:-:1  @P4 STG.E [pa + 0x1800], hdr0;
    --:2:-:-:6  @P4 STG.E [pa + 0x1c00], readBs;
    02:-:-:-:6      NOP;
  DBG
end

# ---------------------------------------------------------------- the C tile: shuffled through shared memory per warp, coalesced rows
emit.call <<~EPI
  --:-:-:-:6      LOP32I.AND readAs, readAs, 0xfff;
  --:-:-:-:6      LOP32I.AND readBs, readBs, 0xfff;
  --:-:-:-:6      ISCADD writeCs, readBs, readAs, 0x5;
  --:-:-:-:6      LOP32I.AND tid128e, tid, 0x80;
  --:-:-:-:6      LOP32I.AND tid96, tid, 0x60;
  --:-:-:-:6      SHR.U32 cx, tid128e, 0x2;
  --:-:-:-:6      LOP.OR cx, tid31, cx;
  --:-:-:-:6      SHL readCs, tid96, 0x4;
  --:-:-:-:6      LOP.OR readCs, readCs, cx;
  --:-:-:-:6      SHL readCs, readCs, 0x2;
  --:-:-:-:6      ISCADD cx, by, cx, 0x7;
  --:-:-:-:6      SHR.U32 cy00, tid96, 0x1;
  --:-:-:-:6      ISCADD cy00, bx, cy00, 0x7;
  --:-:-:-:6      MOV ldc, param_ldx;
  #{xmad_lo.call('ci', 'cy00', 'ldc', 'cx', 'tmpc')}
  --:-:-:-:6      SHL tmpc, ci, 0x2;
  --:-:-:-:6      SHR.U32 ci, ci, 0x1e;
  --:-:-:-:6      IADD pa0.CC, tmpc, param_C0;
  --:-:-:-:6      IADD.X pa1, ci, param_C1;
  --:-:-:-:6      ISETP.LT.U32.AND P5, PT, cx, param_npos, PT;
  --:-:-:-:6      IADD tmpc, cx, 0x40;
  --:-:-:-:6      ISETP.LT.U32.AND P6, PT, tmpc, param_npos, PT;
  --:-:-:-:6      IADD cy00, cy00, -0x1;
  --:-:-:-:6      IADD cy04, cy00, 0x4;
  --:-:-:-:6      IADD cy08, cy00, 0x8;
  --:-:-:-:6      IADD cy12, cy00, 0xc;
  --:-:-:-:6      SHL ldc1, ldc, 0x2;
  --:-:-:-:6      SHL ldc4, ldc, 0x4;
  --:-:-:-:6      SHL ldc8, ldc, 0x5;
  --:-:-:-:6      ISCADD ldc60, ldc, -ldc4, 0x8;
  --:-:-:-:6      IADD pa0.CC, pa0, -ldc1;
  --:-:-:-:6      IADD.X pa1, pa1, -0x1;
  --:-:-:-:6      IADD pb0.CC, pa0, ldc4;
  --:-:-:-:6      IADD.X pb1, pa1, RZ;
  --:-:-:-:6      IADD pq0.CC, pa0, ldc8;
  --:-:-:-:6      IADD.X pq1, pa1, RZ;
  --:-:-:-:6      IADD pr0.CC, pb0, ldc8;
  --:-:-:-:6      IADD.X pr1, pb1, RZ;
EPI

[0, 1, 2, 3, 64, 65, 66, 67].each do |y|
  if y == 64
    emit.call <<~JUMP
      --:-:-:-:1      IADD cy00, cy00, 0x3c;
      --:-:-:-:1      IADD cy04, cy04, 0x3c;
      --:-:-:-:1      IADD cy08, cy08, 0x3c;
      --:-:-:-:6      IADD cy12, cy12, 0x3c;
      02:-:-:-:6      IADD pa0.CC, pa0, ldc60;
      --:-:-:-:6      IADD.X pa1, pa1, RZ;
      --:-:-:-:6      IADD pb0.CC, pb0, ldc60;
      --:-:-:-:6      IADD.X pb1, pb1, RZ;
      --:-:-:-:6      IADD pq0.CC, pq0, ldc60;
      --:-:-:-:6      IADD.X pq1, pq1, RZ;
      --:-:-:-:6      IADD pr0.CC, pr0, ldc60;
      --:-:-:-:6      IADD.X pr1, pr1, RZ;
    JUMP
  end
  wait = y == 0 || y == 64 ? '--' : '02'
  [0, 1, 2, 3, 64, 65, 66, 67].each_with_index do |x, i|
    emit.call format('%s:-:-:-:%d      MOV cs%d, cx%02dy%02d;', i.zero? ? wait : '--', i == 7 ? 6 : 1, i, x, y)
  end
  emit.call '--:-:-:-:5      CAL STORE_C;'
end
if OP
  emit.call <<~NEXT
    02:-:-:-:6      IADD tile, tile, gridDimX;
    --:-:-:-:6      ISETP.LT.U32.AND P0, PT, tile, ntiles, PT;
    --:-:-:-:6      NOP;
    --:-:-:-:6      NOP;
    --:-:-:-:6      NOP;
    --:-:-:-:6  @P0 BRA TILE;
  NEXT
end
emit.call '--:-:-:-:5      EXIT;'

emit.call <<~STORE
  STORE_C:
  --:-:-:-:1      STS.128 [writeCs + 4x<00>], cs0;
  --:-:-:-:6      STS.128 [writeCs + 4x<64>], cs4;
  --:-:-:-:1      LDS cs0, [readCs + 4x<0*128 + 00>];
  --:-:-:-:1      LDS cs1, [readCs + 4x<0*128 + 64>];
  --:-:-:-:1      LDS cs2, [readCs + 4x<1*128 + 00>];
  --:-:-:-:1      LDS cs3, [readCs + 4x<1*128 + 64>];
  --:-:-:-:1      LDS cs4, [readCs + 4x<2*128 + 00>];
  --:-:-:-:1      LDS cs5, [readCs + 4x<2*128 + 64>];
  --:-:-:-:1      LDS cs6, [readCs + 4x<3*128 + 00>];
  --:-:1:-:1      LDS cs7, [readCs + 4x<3*128 + 64>];
  --:-:-:-:1      IADD cy00, cy00, 0x1;
  --:-:-:-:1      IADD cy04, cy04, 0x1;
  --:-:-:-:1      IADD cy08, cy08, 0x1;
  --:-:-:-:6      IADD cy12, cy12, 0x1;
  --:-:-:-:6      IADD pa0.CC, pa0, ldc1;
  --:-:-:-:1      ISETP.LT.U32.AND P0, PT, cy00, param_rows, P5;
  --:-:-:-:1      ISETP.LT.U32.AND P1, PT, cy00, param_rows, P6;
  --:-:-:-:1      ISETP.LT.U32.AND P2, PT, cy04, param_rows, P5;
  --:-:-:-:6      ISETP.LT.U32.AND P3, PT, cy04, param_rows, P6;
  --:-:-:-:6      IADD.X pa1, pa1, RZ;
  --:-:-:-:6      IADD pb0.CC, pb0, ldc1;
  --:-:-:-:6      NOP;
  --:-:-:-:6      IADD.X pb1, pb1, RZ;
  01:-:-:-:1  @P0 STG.E [pa + 4x<00>], cs0;
  --:-:-:-:1  @P1 STG.E [pa + 4x<64>], cs1;
  --:-:-:-:1  @P2 STG.E [pb + 4x<00>], cs2;
  --:-:-:-:6  @P3 STG.E [pb + 4x<64>], cs3;
  --:-:-:-:1      ISETP.LT.U32.AND P0, PT, cy08, param_rows, P5;
  --:-:-:-:1      ISETP.LT.U32.AND P1, PT, cy08, param_rows, P6;
  --:-:-:-:1      ISETP.LT.U32.AND P2, PT, cy12, param_rows, P5;
  --:-:-:-:6      ISETP.LT.U32.AND P3, PT, cy12, param_rows, P6;
  --:-:-:-:6      IADD pq0.CC, pq0, ldc1;
  --:-:-:-:6      IADD.X pq1, pq1, RZ;
  --:-:-:-:6      IADD pr0.CC, pr0, ldc1;
  --:-:-:-:6      IADD.X pr1, pr1, RZ;
  --:-:-:-:1  @P0 STG.E [pq + 4x<00>], cs4;
  --:-:-:-:1  @P1 STG.E [pq + 4x<64>], cs5;
  --:-:-:-:1  @P2 STG.E [pr + 4x<00>], cs6;
  --:2:-:-:6  @P3 STG.E [pr + 4x<64>], cs7;
  --:-:-:-:5      RET;
STORE

print out
