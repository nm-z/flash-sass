#!/usr/bin/env ruby
require 'rbconfig'

root, config, output = ARGV
raise 'usage: run_trace_validation.rb REPOSITORY_ROOT CONFIG.json OUTPUT_BASENAME' unless root && config && output
root = File.realpath(root)
evidence = File.join(root, 'evidence/schema2')
config = File.realpath(config)
raise 'output must be a basename' unless File.basename(output) == output && output.match?(/\A[A-Za-z0-9._-]+\z/)

Dir.chdir(evidence)
exec RbConfig.ruby, File.join(root, 'kern/trace_validation.rb'),
     '--config', config, '--output', output
