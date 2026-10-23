#!/usr/bin/env ruby
# stalltune.rb: shorten the stall counts of the persistent kernel's dispatcher / barrier code to what its register dependences need.
#   perl die.pl | ruby stalltune.rb > die.sass
#
# The framework code of die.pl (descriptor install, exchange wait, dispatch, POSITION_END, PHASE_END, barrier) was written with a stall of 6
# on almost every instruction, a dependent chain being safe that way. The warm-cache floor of a phase (phase_floor) is the sum of those stalls,
# 1.7 us without and 3.4 us with the grid barrier, and all eight warps of a CTA run the same chain.
# Only the regions between the lines "# STALLTUNE ON" and "# STALLTUNE OFF" are touched, nothing is reordered and no instruction is added or
# removed; only the stall field of ALU instructions and plain NOPs that set no barrier changes.
#
# Model (Maxwell): an ALU result (IADD SHL SHR LOP MOV SEL ISCADD IMNMX XMAD FADD FMUL FFMA ...) is readable 6 cycles after the issue of its
# instruction; a predicate written by ISETP is usable 18 cycles after it (the margin the sources already use); the carry written by ".CC" is read
# by ".X" 6 cycles later. Every other instruction (loads, stores, S2R, SHFL, VOTE, BAR, MEMBAR, branches, anything unknown) keeps its own stall and
# keeps the original issue-time distance to every instruction within 24 cycles after it (so a late operand read or a predicate written
# without latency model stays covered). A label or a branch ends a block, and the last 24 cycles of a block keep the original issue times, so
# the producers at the end of a block are as old for the next block as they were before.
#
# STALLTUNE_REPORT=1 prints the number of shortened stalls and the cycles removed to stderr.

LINE = /\A(?<w>[0-9a-f-]{2}):(?<r>[0-9-]):(?<wr>[0-9-]):(?<y>[0-9a-zA-Z-]):(?<s>[0-9a-f])(?<ws>\s+)(?<body>.*?)(?<tail>\s*)\z/
ALU = %w[IADD IADD3 IADD32I ISCADD SHL SHR SHF LOP LOP32I LOP3 MOV MOV32I SEL IMNMX FMNMX XMAD FADD FMUL FFMA FSEL PRMT BFE BFI FLO POPC IMAD ICMP NOP].freeze
SETP = %w[ISETP FSETP PSETP].freeze
LAT = 6
PLAT = 18
WINDOW = 24
TAIL = 20   # the last cycles of a block keep their original stalls: its producers are as old for the next block as they were

text = $stdin.read
lines = text.lines

# ---- register map ----
regmap = {}
in_map = false
lines.each do |l|
  if l =~ /<REGISTER_MAPPING>/ then in_map = true; next end
  if l =~ %r{</REGISTER_MAPPING>} then in_map = false; next end
  next unless in_map
  next unless (m = l.match(%r{\A\s*([\d,\-]+)\s*:\s*(.+?)\s*\z}))
  regs = m[1].split(',').flat_map { |p| p.include?('-') ? (a, b = p.split('-').map(&:to_i); (a..b).to_a) : [p.to_i] }
  names = m[2].split(/\s*,\s*/)
  names.each do |nm|
    if (r = nm.match(/\A(\w+?)<(\d+)-(\d+)>\z/))
      w = r[2].start_with?('0') ? r[2].size : 0
      (r[2].to_i..r[3].to_i).each_with_index { |n, k| regmap[format('%s%0*d', r[1], w, n)] = regs[k] if regs[k] }
    elsif nm =~ /\A\w+\z/ && names.size == 1 && regs.size == 1
      regmap[nm] = regs[0]
    end
  end
end

def parse(l)
  m = LINE.match(l.chomp) or return nil
  body = m[:body]
  pred = nil
  if body =~ /\A@(!?)(P\d|PT)\s+/ then pred = Regexp.last_match(2); body = Regexp.last_match.post_match end
  return nil unless body =~ /\A([A-Z0-9_]+)((?:\.[A-Za-z0-9_]+)*)\s*(.*?);\s*\z/
  { m: m, pred: pred, op: Regexp.last_match(1), mods: Regexp.last_match(2), ops: Regexp.last_match(3).split(/,(?![^\[]*\])/).map(&:strip) }
end

NOT_REG = %w[RZ PT CC X H0 H1 reuse U32 S32 U16 S16 E CI CV CG WT S8 U8 HI LO].freeze
def regs_of(tok, regmap, width = 1)
  out = []
  tok.scan(/[A-Za-z_][A-Za-z_0-9]*/) do |name|
    next if NOT_REG.include?(name) || name =~ /\AP\d\z/
    base = regmap[name] || regmap[name + '0']
    out.concat((base...base + width).to_a) if base
  end
  out
end

def preds_of(tok) = tok.scan(/(?<![A-Za-z_0-9])(P\d)(?![A-Za-z_0-9])/).flatten

def width_of(mods) = mods.include?('.128') ? 4 : mods.include?('.64') ? 2 : 1

# ---- one straight-line block ----
def tune_block(block, report)
  n = block.size
  return 0 if n < 2
  orig_t = []
  t = 0
  block.each { |b| orig_t << t; t += b[:orig_stall] }
  total = t
  tail_start = n
  (n - 1).downto(0) { |i| tail_start = i; break if total - orig_t[i] >= TAIL }
  pin = lambda do |new_t, i|   # earliest time of i from the pinned (non-ALU) predecessors
    r = 0
    (i - 1).downto(0) do |x|
      break if orig_t[i] - orig_t[x] > WINDOW
      r = [r, new_t[x] + (orig_t[i] - orig_t[x])].max unless block[x][:reducible]
    end
    r
  end
  new_t = Array.new(n, 0)
  lat = Hash.new(-1_000_000)
  plat = Hash.new(-1_000_000)
  cc = -1_000_000
  n.times do |i|
    b = block[i]
    ready = 0
    if i.positive?
      pb = block[i - 1]
      ready = new_t[i - 1] + (pb[:reducible] && i - 1 < tail_start ? 1 : pb[:orig_stall])
    end
    b[:reads].each { |r| ready = [ready, lat[r]].max }
    b[:preads].each { |p| ready = [ready, plat[p]].max }
    ready = [ready, cc].max if b[:reads_cc]
    ready = [ready, pin.call(new_t, i)].max
    ready = [ready, new_t[i - 1] + block[i - 1][:orig_stall]].max if i.positive? && i >= tail_start
    new_t[i] = ready
    if b[:alu]
      b[:writes].each { |r| lat[r] = ready + LAT }
      cc = ready + LAT if b[:writes_cc]
      b[:pwrites].each { |p| plat[p] = ready + PLAT }
    else
      b[:pwrites].each { |p| plat[p] = [plat[p], ready + PLAT].max }
    end
  end
  stalls = block.map { |b| b[:orig_stall] }
  saved = 0
  n.times do |i|
    b = block[i]
    next unless b[:reducible] && i + 1 < n && i < tail_start
    want = [[new_t[i + 1] - new_t[i], 1].max, 15].min
    next if want >= b[:orig_stall]
    saved += b[:orig_stall] - want
    stalls[i] = want
  end
  # verify against the dependence model with the stalls actually chosen
  tt = []
  t = 0
  stalls.each { |s| tt << t; t += s }
  lat2 = Hash.new(-1_000_000); plat2 = Hash.new(-1_000_000); cc2 = -1_000_000
  ok = true
  n.times do |i|
    b = block[i]
    b[:reads].each { |r| ok &&= tt[i] >= lat2[r] }
    b[:preads].each { |p| ok &&= tt[i] >= plat2[p] }
    ok &&= tt[i] >= cc2 if b[:reads_cc]
    ok &&= tt[i] >= pin.call(tt, i)
    if b[:alu]
      b[:writes].each { |r| lat2[r] = tt[i] + LAT }
      cc2 = tt[i] + LAT if b[:writes_cc]
    end
    b[:pwrites].each { |p| plat2[p] = [plat2[p], tt[i] + PLAT].max }
  end
  return 0 unless ok
  changed = 0
  n.times { |i| if stalls[i] != block[i][:orig_stall] then block[i][:new_stall] = stalls[i]; changed += 1 end }
  report[:cycles] += saved
  changed
end

# ---- driver ----
out = lines.dup
in_region = false
block = []
idx = []
changed_total = 0
report = Hash.new(0)

flush = lambda do
  changed_total += tune_block(block, report)
  block.each_with_index do |b, k|
    next unless b[:new_stall]
    m = b[:m]
    out[idx[k]] = format("%s:%s:%s:%s:%x%s%s%s\n", m[:w], m[:r], m[:wr], m[:y], b[:new_stall], m[:ws], m[:body], m[:tail])
  end
  block = []
  idx = []
end

lines.each_with_index do |l, i|
  if l =~ /\A\s*#\s*STALLTUNE ON/ then in_region = true; next end
  if l =~ /\A\s*#\s*STALLTUNE OFF/ then flush.call; in_region = false; next end
  next unless in_region
  if l =~ /\A[A-Za-z_][A-Za-z_0-9]*:\s*\z/ then flush.call; next end
  p = parse(l)
  next unless p
  m = p[:m]
  op = p[:op]
  ops = p[:ops]
  is_alu = ALU.include?(op) || SETP.include?(op)
  reducible = is_alu && m[:r] == '-' && m[:wr] == '-'
  reads = []
  writes = []
  preads = []
  pwrites = []
  if SETP.include?(op)
    pwrites = ops[0, 2].flat_map { |t| preds_of(t) }
    ops[2..].to_a.each { |t| preads.concat(preds_of(t)); reads.concat(regs_of(t, regmap)) }
  elsif op == 'NOP'
    # no operands
  elsif is_alu
    writes = regs_of(ops[0].to_s, regmap)
    ops[1..].to_a.each { |t| preads.concat(preds_of(t)); reads.concat(regs_of(t, regmap)) }
  else
    # variable-latency or control instruction: its operand reads and the predicates it names are pinned by the window rule
    ops.each { |t| preads.concat(preds_of(t)); reads.concat(regs_of(t, regmap, t.include?('[') ? 2 : width_of(p[:mods]))) }
  end
  preads << p[:pred] if p[:pred] && p[:pred] != 'PT'
  block << { m: m, orig_stall: m[:s].to_i(16), reducible: reducible, alu: is_alu && reducible, reads: reads, writes: writes, preads: preads.uniq,
             pwrites: pwrites, reads_cc: p[:mods].include?('.X'), writes_cc: ops[0].to_s.include?('.CC') }
  idx << i
  flush.call if %w[BRA BRX EXIT RET CAL JCAL JMP JMX SYNC SSY BRK CONT PBK PCNT PRET].include?(op)
end
flush.call
$stdout.write(out.join)
warn "stalltune: #{changed_total} stall fields shortened, #{report[:cycles]} issue cycles removed" if ENV['STALLTUNE_REPORT']
