#!/usr/bin/env ruby
# floorexp.rb NAME...: build variants of the generated die image (chain/kern/die_all_c.sass) with parts of the per-phase path removed (NOT correct
# kernels: only the empty-phase timing of phase_floor is meaningful) and assemble them to fl_<NAME>.cubin next to it.
#   notimestamp : the per-phase %globaltimer store
#   nomembar    : the MEMBAR.GL.IVALLD after the barrier spin
#   nospintimer : the 300 s check inside the barrier spin loop
#   nobarsync   : the BAR.SYNC 0x1 after the barrier
#   all         : all of the above
require 'fileutils'
KERN = '/home/nate/codex/flash-goal-140-1200/chain/kern'
Dir.chdir(KERN)
src = File.readlines('die_all_c.sass')

def cut_between(lines, from_re, to_re, keep_from: true)
  i = lines.index { |l| l =~ from_re } or raise "no #{from_re}"
  j = (i + 1...lines.size).find { |k| lines[k] =~ to_re } or raise "no #{to_re} after #{from_re}"
  lines[(keep_from ? i + 1 : i)...j] = []
  lines
end

def edit(name, lines)
  case name
  when 'notimestamp'
    cut_between(lines, /^BDONE:/, /^# ---- next phase ----|^\s*--:-:1:-:2\s+LDS bar_t, \[RZ\+0xbf50\]/, keep_from: true)
    # keep the BAR.SYNC that follows BDONE
    lines.insert(lines.index { |l| l =~ /^BDONE:/ } + 1, "02:-:-:-:6      BAR.SYNC 0x1;\n") unless lines[lines.index { |l| l =~ /^BDONE:/ } + 1] =~ /BAR.SYNC/
  when 'nomembar'
    i = lines.index { |l| l =~ /\@!P4 BRA BSPIN;/ } or raise 'no BSPIN branch'
    j = (i + 1..i + 3).find { |k| lines[k] =~ /MEMBAR.GL.IVALLD/ } or raise 'no membar'
    lines.delete_at(j)
  when 'nospintimer'
    i = lines.index { |l| l =~ /^BSPIN:/ } or raise 'no BSPIN'
    a = (i...i + 40).find { |k| lines[k] =~ /S2R bar_t, SR_GLOBALTIMERHI/ } or raise 'no timer'
    b = (a...a + 25).find { |k| lines[k] =~ /\@P5 BRA BEXP;/ } or raise 'no bexp'
    lines[a..b] = []
  when 'nobarsync'
    i = lines.index { |l| l =~ /^BDONE:/ } or raise 'no BDONE'
    j = (i + 1..i + 3).find { |k| lines[k] =~ /BAR.SYNC 0x1/ } or raise 'no bar.sync'
    lines.delete_at(j)
  else raise "unknown #{name}"
  end
  lines
end

ARGV.each do |name|
  lines = src.dup
  names = name == 'all' ? %w[notimestamp nomembar nospintimer nobarsync] : [name]
  names.each { |n| lines = edit(n, lines) }
  File.write("fl_#{name}.sass", lines.join)
  FileUtils.rm_f("fl_#{name}.cubin")
  FileUtils.cp('skel_die.cubin', "fl_#{name}.cubin")
  ok = system({ 'PATH' => "/opt/cuda/bin:#{ENV['PATH']}" }, 'perl', '-I../maxas/lib', '../maxas/bin/maxas.pl', '-i', "fl_#{name}.sass", "fl_#{name}.cubin", out: File::NULL, err: File::NULL)
  puts "#{name}: #{ok ? 'built' : 'MAXAS FAILED'} (#{src.size - lines.size} lines removed)"
end
