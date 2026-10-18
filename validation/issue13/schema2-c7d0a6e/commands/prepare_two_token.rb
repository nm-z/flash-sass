#!/usr/bin/env ruby
# Preserve the failed four-token setup and create a matched two-output matrix.
require 'json'
root = File.realpath(ARGV.fetch(0))
control_file = File.join(root,'control.json')
control = JSON.parse(File.read(control_file))
config = JSON.parse(File.read(File.join(root,'config-cta16.json')))
config['new'] = 1
%w[ordinary traced].each do |mode|
  old_label = mode+'16'; new_label = old_label+'two'
  selected = Marshal.load(Marshal.dump(control.fetch('modes').fetch(old_label)))
  count = mode == 'ordinary' ? 0 : 2
  selected['environment']['FLASH_TRACE'] = count.to_s
  selected['expected_config']['trace_tokens'] = count.to_s
  control['modes'][new_label] = selected
  config['modes'][mode]['trace_tokens'] = count
  config['modes'][mode]['prepare_argv'][-1] = new_label
end
File.write(control_file,JSON.pretty_generate(control)+"\n")
File.write(File.join(root,'config-cta16-two.json'),JSON.pretty_generate(config)+"\n")
puts 'Prepared ordinary/traced two-output matrix with 16-CTA capture and unchanged shared runtime fields'
