#!/usr/bin/env ruby
# Synthetic CPU regressions for the analyzer; these are not GPU measurements.
require 'tmpdir'
require 'json'
require 'zlib'
require 'open3'
require 'rbconfig'
require_relative 'trace_lifecycle_check'

GPU_COLUMNS = %w[position phase layer opcode units flags selected_keys start_ns wait_start_ns wait_end_ns body_end_ns end_ns ple_wait_start_ns ple_wait_end_ns publication_ns cta body_start_ns send_before_ns send_after_ns exchange_generation].freeze
HOST_COLUMNS = %w[position observed_ns callback_start_ns callback_end_ns ple_staging_start_ns ple_staging_end_ns ple_ready_ns ple_bytes ple_short_reads ple_read_start_ns ple_read_end_ns ple_ready_start_ns poll_start_ns poll_end_ns readback_start_ns readback_end_ns].freeze

def fixture(dir, shifts, bad: nil)
  paths = []
  2.times do |die|
    path = File.join(dir,"die-#{die}.gz"); paths << path
    Zlib::GzipWriter.open(path) do |f|
      f.puts("trace_begin request=1 die=#{die} first=0 count=2 nph=3 ctas=16 nblk=1 schema=2 status=completed aborted=0 ended=0 completed=2 requested=2")
      f.puts(GPU_COLUMNS.join(','))
      2.times do |t|
        3.times do |phase|
          16.times do |cta|
            next if bad == :missing && die == 1 && t == 1 && phase == 2 && cta == 15
            origin = 10**18 + shifts[die] + t*100000 + die*1000 + cta*10
            a = origin + phase*30000
            ws = a+100; we = phase == 2 ? a+10000 : a+200
            bs = phase == 2 ? a+11000 : a+300
            be = phase == 2 ? a+20000 : a+10000
            z = phase == 2 ? a+30000 : a+20000
            s0 = phase == 1 ? a+15000 : 0; s1 = phase == 1 ? a+16000 : 0
            pub = phase == 2 ? a+31000 : 0
            flags = [0,1,2][phase]; gen = phase == 2 ? t+1 : t
            gen += 5 if bad == :generation && die == 1 && phase == 1
            row = [t,phase,-1,%w[embrow xcopy idxsel][phase],0,flags,1,a,ws,we,be,z,0,0,pub,cta,bs,s0,s1,gen]
            f.puts(row.join(','))
          end
        end
      end
      f.puts('trace_end')
    end
  end
  path = File.join(dir,'host.gz'); paths << path
  Zlib::GzipWriter.open(path) do |f|
    f.puts('hosttrace_begin request=1 first=0 count=2 clock=CLOCK_MONOTONIC status=completed aborted=0 ended=0 completed=2 requested=2 schema=2 abort_start_ns=0 abort_end_ns=0 abort_position=-1 abort_reason=none request_poll_calls=2 request_poll_ns=2000 request_readback_calls=2 request_readback_ns=2000')
    f.puts(HOST_COLUMNS.join(','))
    2.times do |t|
      a = 4000000000+t*100000
      f.puts([t,a,a+10,a+20,0,0,0,0,0,0,0,0,a-2000,a-1000,a-1000,a].join(','))
    end
    f.puts('hosttrace_end')
  end
  paths
end

def analyze(dir,paths,*args)
  out = File.join(dir,'report.json')
  stdout,stderr,status = Open3.capture3(RbConfig.ruby,File.join(__dir__,'trace_reconcile.rb'),
    '--dies','2','--quantum-ns','0','--out',out,*args,*paths)
  [status.success?,File.exist?(out) ? JSON.parse(File.read(out)) : nil,stdout+stderr]
end

Dir.mktmpdir('trace-reconcile-cpu-') do |root|
  a = File.join(root,'a'); b = File.join(root,'b'); c = File.join(root,'c'); d = File.join(root,'d')
  [a,b,c,d].each { |p| Dir.mkdir(p) }
  paths = fixture(a,[0,900000000000000])
  paths.each do |p|
    kind = p.include?('host') ? 'hosttrace' : 'trace'
    header = Zlib::GzipReader.open(p) { |io| io.readline.strip }
    TraceChecks.validate_file(p,kind,TraceChecks.fields(header),false)
  end
  ok,first,log = analyze(a,paths); raise log unless ok
  raise 'wrong CTA/token reconciliation count' unless first['local_cta_reconciliation'].length == 64
  raise 'nonzero residual' unless first['local_cta_reconciliation'].all? { |r| r['residual_ns'] == 0 }
  raise 'missing grid joins' unless first['causal_analysis']['grid_barrier_joins'] == 12
  true_span = 192150
  ca = first['causal_analysis']
  raise 'bounds exclude true synthetic span' unless ca['makespan_lower_bound_ns'] <= true_span && true_span <= ca['enclosing_span_upper_bound_ns']
  puts 'PASS exact per-CTA/token and same-die reconciliation; grid joins; known makespan bounds'
  ok,shifted,log = analyze(b,fixture(b,[77777770000000,-99999990000000])); raise log unless ok
  keys = %w[chain_sampled_ns_after_crossing_allowance makespan_lower_bound_ns enclosing_span_upper_bound_ns]
  raise 'epoch shift changed elapsed/causal result' unless keys.all? { |k| first['causal_analysis'][k] == shifted['causal_analysis'][k] }
  raise 'epoch shift changed local partitions' unless first['local_cta_reconciliation'] == shifted['local_cta_reconciliation']
  puts 'PASS independent epoch shifts above 2^53 preserve durations and causal bounds'
  ok,selected,log = analyze(a,paths,'--position','1'); raise log unless ok
  raise 'single position selection failed' unless selected['positions'] == [1] && selected['local_cta_reconciliation'].all? { |r| r['interpublication_ns'].nil? }
  puts 'PASS single-position scope keeps earlier-token time outside its first bucket'
  ok,_r,log = analyze(c,fixture(c,[0,0],bad: :missing)); raise 'missing CTA row accepted' if ok
  puts 'PASS incomplete CTA coverage rejected'
  ok,_r,log = analyze(d,fixture(d,[0,0],bad: :generation)); raise 'bad generation accepted' if ok
  puts 'PASS inconsistent generation coverage rejected'
end
