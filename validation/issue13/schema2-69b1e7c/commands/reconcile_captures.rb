#!/usr/bin/env ruby
require 'digest'
require 'fileutils'
require 'json'
require 'open3'
require 'rbconfig'
require 'zlib'

evidence, output = ARGV
raise 'usage: reconcile_captures.rb EVIDENCE_DIRECTORY OUTPUT_DIRECTORY' unless evidence && output
FileUtils.mkdir_p(output)

samples = Dir.glob(File.join(evidence, 'overhead-cta*', 'slot-*-traced', 'measured-01')).sort
complete, skipped = samples.partition do |sample|
  File.file?(File.join(sample, 'hosttrace.raw.gz')) &&
    (0...6).all? { |die| File.file?(File.join(sample, "trace-#{die}.raw.gz")) }
end
raise 'no complete six-die traced samples found' if complete.empty?

reports = complete.map do |sample|
  slot = File.basename(File.dirname(sample))
  block = File.basename(File.dirname(File.dirname(sample)))
  name = "#{block.delete_prefix('overhead-')}-#{slot}-capture.json.gz"
  compressed = File.join(output, name)
  temporary = compressed.delete_suffix('.gz')
  inputs = (0...6).map { |die| File.join(sample, "trace-#{die}.raw.gz") }
  inputs << File.join(sample, 'hosttrace.raw.gz')
  command = [RbConfig.ruby, 'kern/trace_reconcile.rb', '--dies', '6', '--out', temporary, *inputs]
  _stdout, stderr, status = Open3.capture3(*command)
  raise "#{name}: reconciliation failed: #{stderr}" unless status.success?
  Zlib::GzipWriter.open(compressed) { |gzip| File.open(temporary, 'rb') { |file| IO.copy_stream(file, gzip) } }
  File.delete(temporary)
  {sample: sample, report: name, bytes: File.size(compressed), sha256: Digest::SHA256.file(compressed).hexdigest}
end

index = {schema: 1, reconciled: reports, skipped_incomplete_samples: skipped}
File.write(File.join(output, 'index.json'), JSON.pretty_generate(index) + "\n")
puts JSON.pretty_generate(index)
