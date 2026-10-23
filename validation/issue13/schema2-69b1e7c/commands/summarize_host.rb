#!/usr/bin/env ruby
require 'json'
require 'zlib'

layout, reports_dir, output = ARGV
raise 'usage: summarize_host.rb LAYOUT REPORTS_DIRECTORY OUTPUT.json' unless layout && reports_dir && output

paths = Dir.glob(File.join(reports_dir, "#{layout.downcase}-*-capture.json.gz")).sort
raise "no #{layout} reports under #{reports_dir}" if paths.empty?
rows = paths.flat_map do |path|
  report = JSON.parse(Zlib::GzipReader.open(path, &:read))
  report.fetch('host').fetch('bounded_token_rows')
end

stats = lambda do |values|
  numbers = values.map { |value| Float(value) }
  {n: numbers.length, mean: numbers.sum / numbers.length, min: numbers.min, max: numbers.max}
end

raw = rows.map { |row| row.fetch('raw_fields') }
result = {
  layout: layout,
  capture_count: paths.length,
  host_token_rows: rows.length,
  ple_bytes: stats.call(raw.map { |row| row.fetch('ple_bytes') }),
  ple_short_reads_total: raw.sum { |row| Integer(row.fetch('ple_short_reads')) },
  ple_read_ns: stats.call(rows.map { |row| row.fetch('ple_read_ns') }),
  ple_staging_ns: stats.call(rows.map { |row| row.fetch('ple_staging_ns') }),
  ple_ready_bracket_ns: stats.call(rows.map { |row| row.fetch('ple_ready_bracket_ns') }),
  callback_ns: stats.call(rows.map { |row| row.fetch('callback_ns') })
}
File.write(output, JSON.pretty_generate(result) + "\n")
puts JSON.pretty_generate(result)
