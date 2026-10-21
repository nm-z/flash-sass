#!/usr/bin/env ruby
require 'fileutils'
require 'open3'

root, source_sha, output, control_output = ARGV
raise 'usage: run_lifecycle.rb ROOT SOURCE_SHA OUTPUT_DIRECTORY CONTROL_OUTPUT_DIRECTORY' unless root && source_sha && output && control_output
root = File.expand_path(root)
output = File.expand_path(output)
control_output = File.expand_path(control_output)
raise 'output paths must be new' if File.exist?(output) || File.exist?(control_output)
control = File.join(root, 'evidence/schema2/control.json')
checker = File.join(root, 'kern/trace_lifecycle_check.rb')
controller = File.join(root, 'kern/trace_control.rb')
preflight = File.join(root, 'kern/trace_preflight')
config = File.join(root, 'evidence/schema2/config-cta1.json')
lock_path = '/tmp/flash-sass-engine-validation.lock'

FileUtils.mkdir_p(control_output)
lock = File.open(lock_path, File::RDWR | File::CREAT, 0o600)
raise 'validation lock is already held' unless lock.flock(File::LOCK_EX | File::LOCK_NB)
failure = nil

run_and_record = lambda do |label, env, *command|
  stdout, stderr, status = Open3.capture3(env, *command)
  File.write(File.join(control_output, "#{label}.stdout"), stdout)
  File.write(File.join(control_output, "#{label}.stderr"), stderr)
  File.write(File.join(control_output, "#{label}.status"), "#{status.exitstatus}\n")
  raise "#{label} failed with status #{status.exitstatus}: #{stderr}" unless status.success?
  stdout
end

begin
  run_and_record.call('prepare', {'FLASH_TRACE_VALIDATION_LOCK_HELD' => '1'},
                      'ruby', controller, control, 'lifecycle')
  env = {
    'FLASH_TRACE_VALIDATION_RUNNER' => File.join(root, 'kern/trace_validation.rb'),
    'FLASH_TRACE_VALIDATION_CONFIG' => config
  }
  run_and_record.call('lifecycle', env, 'ruby', checker,
                      '--socket', '/tmp/flash.sock', '--out', output,
                      '--source-sha', source_sha, '--pair-check', '--preflight', preflight)
rescue StandardError => error
  failure = error
ensure
  begin
    run_and_record.call('restore', {'FLASH_TRACE_VALIDATION_LOCK_HELD' => '1'},
                        'ruby', controller, control, 'restore')
  rescue StandardError => error
    failure ||= error
  end
  lock.flock(File::LOCK_UN)
  lock.close
end

raise failure if failure
puts "PASS lifecycle evidence saved to #{output}"
