#!/usr/bin/env ruby
require 'json'

METRICS = %w[wall_ns first_token_ns steady_decode_ns_per_token run_ms_rounded].freeze

def need(ok, message)
  raise message unless ok
end

layout, output, *inputs = ARGV
need(layout && output && inputs.length == 3,
     'usage: summarize_abba.rb LAYOUT OUTPUT BLOCK1.json BLOCK2.json BLOCK3.json')

blocks = inputs.each_with_index.map do |path, i|
  document = JSON.parse(File.read(path))
  metrics = METRICS.to_h do |name|
    block = document.fetch(name).fetch('abba_blocks')
    need(block.length == 1, "#{path}: expected one ABBA block for #{name}")
    [name, block.first]
  end
  {block: i + 1, metrics: metrics}
end

overhead = METRICS.to_h do |name|
  values = blocks.map { |block| Float(block.fetch(:metrics).fetch(name).fetch('overhead_percent')) }
  [name, {n: values.length, mean: values.sum / values.length,
          min: values.min, max: values.max}]
end

result = {layout: layout, blocks: blocks, overhead_percent: overhead}
File.write(output, JSON.pretty_generate(result) + "\n")
puts JSON.pretty_generate(result)
