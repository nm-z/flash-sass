#!/usr/bin/env ruby
require 'digest'
require 'fileutils'
require 'json'
require 'open3'
require 'rbconfig'
require 'zlib'

evidence, lifecycle_name, output = ARGV
raise 'usage: reconcile_lifecycle.rb EVIDENCE_DIRECTORY LIFECYCLE_BASENAME OUTPUT_DIRECTORY' unless evidence && lifecycle_name && output
evidence = File.realpath(evidence)
lifecycle = File.join(evidence, lifecycle_name)
output = File.expand_path(output)
results = JSON.parse(File.read(File.join(lifecycle, 'results.json')))
FileUtils.mkdir_p(output)

reports = []; skipped = []
results.fetch('cases').each do |entry|
  name = entry.fetch('case')
  traces = (0...6).map { |die| File.join(lifecycle, "#{name}-trace-#{die}.txt.gz") }
  host = File.join(lifecycle, "#{name}-hosttrace.txt.gz")
  unless traces.all? { |path| File.file?(path) && File.size(path).positive? } && File.file?(host)
    skipped << {case: name, reason: 'missing trace files'}
    next
  end
  first = Zlib::GzipReader.open(traces.first, &:readline)
  if first.start_with?('error:')
    skipped << {case: name, reason: 'request has no trace records'}
    next
  end
  compressed = File.join(output, "#{name}-capture.json.gz")
  temporary = compressed.delete_suffix('.gz')
  command = [RbConfig.ruby, 'kern/trace_reconcile.rb', '--dies', '6', '--allow-incomplete',
             '--out', temporary, *traces, host]
  _stdout, stderr, status = Open3.capture3(*command)
  raise "#{name}: reconciliation failed: #{stderr}" unless status.success?
  Zlib::GzipWriter.open(compressed) do |gzip|
    File.open(temporary, 'rb') { |file| IO.copy_stream(file, gzip) }
  end
  File.delete(temporary)
  report = JSON.parse(Zlib::GzipReader.open(compressed, &:read))
  local = report.fetch('local_cta_reconciliation')
  envelopes = report.fetch('same_die_envelopes')
  raise "#{name}: nonzero local residual" unless local.all? { |row| row.fetch('residual_ns').zero? }
  raise "#{name}: nonzero die residual" unless envelopes.all? { |row| row.fetch('residual_ns').zero? }
  reports << {case: name, report: File.basename(compressed), local_partitions: local.length,
              die_envelopes: envelopes.length, sha256: Digest::SHA256.file(compressed).hexdigest}
end

index = {schema: 1, reconciled: reports, skipped: skipped}
File.write(File.join(output, 'index.json'), JSON.pretty_generate(index) + "\n")
puts JSON.pretty_generate(index)
