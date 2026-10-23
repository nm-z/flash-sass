#!/usr/bin/env ruby
# progdecode.rb BASE [DIE]: decode the FLASH_DUMP_PROG descriptor dumps  BASE.w<positions>.d<die>  (128 B per phase) into one line per phase:
# index, op name (from kern/dieops.h), units, word 30 flags (S send, W wait, N nobar, G group, F fused, C chain, B inner barrier), word 31 (tok mask).
# With two BASEs ("a b DIE") the op / flags sequences are compared.
OPS = {}
File.readlines(File.join(__dir__, 'source/kern/dieops.h')).each { |l| OPS[$2.to_i] = $1 if l =~ /#define OP_(\w+) (\d+)/ }
OPS.merge!(File.readlines(File.join(__dir__, 'chain/kern/dieops.h')).each_with_object({}) { |l, h| h[$2.to_i] = $1 if l =~ /#define OP_(\w+) (\d+)/ }) if File.exist?(File.join(__dir__, 'chain/kern/dieops.h'))

def load(base, die)
  f = Dir.glob("#{base}.w*.d#{die}").first or abort "no dump #{base}.w*.d#{die}"
  File.binread(f).unpack('V*').each_slice(32).to_a
end

def flags(w)
  s = +''
  s << 'S' if w & 1 != 0
  s << 'W' if w & 2 != 0
  s << 'N' if w & 4 != 0
  s << 'G' if w & 8 != 0
  s << 'F' if w & 16 != 0
  s << 'C' if w & 32 != 0
  s << 'B' if w & 64 != 0
  s
end

def row(i, d)
  format('%4d %-12s units=%-4d %-6s mask=%08x', i, OPS[d[0]] || "op#{d[0]}", d[1], flags(d[30]), d[31])
end

if ARGV.size >= 3
  a, b, die = ARGV[0], ARGV[1], ARGV[2].to_i
  pa, pb = load(a, die), load(b, die)
  puts "phases: #{pa.size} vs #{pb.size}"
  n = [pa.size, pb.size].max
  shown = 0
  n.times do |i|
    ra = pa[i] && row(i, pa[i]); rb = pb[i] && row(i, pb[i])
    next if ra && rb && ra.sub(/\A\s*\d+ /, '') == rb.sub(/\A\s*\d+ /, '')
    puts "#{(ra || '-').ljust(56)} | #{rb || '-'}"
    break if (shown += 1) > 60
  end
else
  base, die = ARGV[0], (ARGV[1] || 0).to_i
  load(base, die).each_with_index { |d, i| puts row(i, d) }
end
