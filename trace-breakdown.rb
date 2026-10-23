#!/usr/bin/env ruby
# Aggregate flash-sass phase-trace CSVs (trace <die>) into per-phase interval totals.
# Usage: ruby trace-breakdown.rb die0.csv [die1.csv ...]
# Columns: position,phase,layer,opcode,units,flags,selected_keys,start_ns,wait_start_ns,wait_end_ns,body_end_ns,end_ns,ple_wait_start_ns,ple_wait_end_ns,publication_ns
# Intervals per phase: dispatch = wait_start-start; xwait = wait_end-wait_start; body = body_end-wait_end; barrier = end-body_end.
require 'csv'
def fmt(us) format('%8.1f', us) end
ARGV.each do |path|
  text = File.read(path)
  header = text.lines.find { |l| l.start_with?('trace_begin') }
  rows = text.lines.select { |l| l =~ /\A\d+,\d+,-?\d+,/ }.map { |l| l.chomp.split(',') }
  positions = rows.map { |r| r[0].to_i }.uniq.sort
  puts "== #{File.basename(path)} #{header&.strip}"
  puts "positions=#{positions.inspect} phases/position=#{rows.length / [positions.length,1].max}"
  # Use generated positions only (skip the first, which is the last prompt position with the full head)
  gen = positions.drop(1)
  gen = positions if gen.empty?
  sel = rows.select { |r| gen.include?(r[0].to_i) }
  by_op = Hash.new { |h,k| h[k] = {n:0,dispatch:0.0,xwait:0.0,body:0.0,barrier:0.0,total:0.0} }
  tot = {n:0,dispatch:0.0,xwait:0.0,body:0.0,barrier:0.0,total:0.0}
  gaps = 0.0; prev_end = nil; prev_pos = nil
  sel.each do |r|
    pos = r[0].to_i; op = r[3]
    s,ws,we,be,e = r[7..11].map(&:to_i)
    d = (ws-s)/1000.0; x = (we-ws)/1000.0; b = (be-we)/1000.0; bar = (e-be)/1000.0; t = (e-s)/1000.0
    h = by_op[op]; h[:n]+=1; h[:dispatch]+=d; h[:xwait]+=x; h[:body]+=b; h[:barrier]+=bar; h[:total]+=t
    tot[:n]+=1; tot[:dispatch]+=d; tot[:xwait]+=x; tot[:body]+=b; tot[:barrier]+=bar; tot[:total]+=t
    gaps += (s-prev_end)/1000.0 if prev_end && prev_pos == pos
    prev_end = e; prev_pos = pos
  end
  npos = gen.length.to_f
  puts format('per token (avg over %d generated positions): phases=%d total=%.2f ms  dispatch=%.2f  xwait=%.2f  body=%.2f  barrier=%.2f  inter-phase gaps=%.2f ms',
              npos, tot[:n]/npos, tot[:total]/npos/1000, tot[:dispatch]/npos/1000, tot[:xwait]/npos/1000, tot[:body]/npos/1000, tot[:barrier]/npos/1000, gaps/npos/1000)
  puts format('%-12s %6s %9s %9s %9s %9s %9s %7s', 'opcode','n/tok','total_us','dispatch','xwait','body','barrier','us/ph')
  by_op.sort_by { |_,h| -h[:total] }.first(24).each do |op,h|
    puts format('%-12s %6.0f %s %s %s %s %s %7.1f', op, h[:n]/npos, fmt(h[:total]/npos), fmt(h[:dispatch]/npos), fmt(h[:xwait]/npos), fmt(h[:body]/npos), fmt(h[:barrier]/npos), h[:total]/h[:n])
  end
  # distribution of per-phase fixed cost (dispatch+barrier) for trivial ops
  fixed = sel.map { |r| s,ws,we,be,e = r[7..11].map(&:to_i); ((ws-s)+(e-be))/1000.0 }.sort
  puts format('dispatch+barrier per phase: p10=%.1f p50=%.1f p90=%.1f us', fixed[fixed.length/10], fixed[fixed.length/2], fixed[fixed.length*9/10]) unless fixed.empty?
  puts
end
