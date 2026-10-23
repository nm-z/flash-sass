#!/usr/bin/env ruby
# nopfold.rb: fold the NOPs that only pad a predicate / register latency into the stall count of the instruction before them.
#   perl die.pl | ruby nopfold.rb > die.sass
#
# The instruction cache of a Maxwell SM fetches at 1.3 ns per instruction from the 8 KB level, 5 ns from the 32 KB level and 10 ns from
# L2 (icache_probe), and the dispatcher / barrier / op prologues are re-fetched cold in every phase, so every instruction that only
# burns cycles costs fetch time. A group "I (stall s); NOP (stall a); NOP (stall b) ..." issues the next real instruction s + a + b + ...
# cycles after I. This rewrites the group as I with stall min(15, s + a + b + ...) followed by one NOP carrying the remainder (when the
# sum is above 15; more NOPs only above 30), so the issue times of everything after the group are unchanged.
#
# Only plain NOPs ("--:-:-:-:S      NOP;", no barrier fields, no predicate) on the lines directly after an instruction line are folded (a
# label, comment or blank line ends the group), and never into a branch, exit, barrier or the like (those keep their delay slots; the jump
# table of die.pl, entries of one 32 B bundle each, stays aligned) nor into another NOP.
MAX = 15
INSN = /\A(?<w>[0-9a-f-]{2}):(?<r>[0-9-]):(?<wr>[0-9-]):(?<y>[0-9a-zA-Z-]):(?<s>[0-9a-f])(?<rest>\s+.*)\z/
NOT_FOLDED_INTO = /\b(BRA|BRX|EXIT|BAR|SYNC|SSY|PBK|PCNT|PRET|RET|CAL|JCAL|JMP|JMX|BRK|CONT|NOP|MEMBAR|DEPBAR|VOTE)\b/

def parse(line) = INSN.match(line.chomp)
def pure_nop?(m) = m && m[:w] == '--' && m[:r] == '-' && m[:wr] == '-' && m[:rest] =~ /\A\s+NOP;\s*\z/
def emit(m, stall) = format("%s:%s:%s:%s:%x%s\n", m[:w], m[:r], m[:wr], m[:y], stall, m[:rest])

lines = $stdin.readlines
out = []
folded = 0
i = 0
while i < lines.size
  m = parse(lines[i])
  if m && !pure_nop?(m) && m[:rest] !~ NOT_FOLDED_INTO && i + 1 < lines.size && pure_nop?(parse(lines[i + 1]))
    total = m[:s].to_i(16)
    j = i + 1
    nops = 0
    while j < lines.size && pure_nop?(parse(lines[j]))
      total += parse(lines[j])[:s].to_i(16)
      nops += 1
      j += 1
    end
    first = [total, MAX].min
    rest = total - first
    out << emit(m, first)
    kept = 0
    while rest > 0
      take = [rest, MAX].min
      out << format("--:-:-:-:%x      NOP;\n", take)
      rest -= take
      kept += 1
    end
    folded += nops - kept
    i = j
  else
    out << lines[i]
    i += 1
  end
end
$stdout.write(out.join)
warn "nopfold: #{lines.size - out.size} fewer lines (#{folded} NOPs folded)"
