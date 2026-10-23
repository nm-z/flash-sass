#!/usr/bin/env ruby
require 'fileutils'
require 'open3'

root, source_sha, output_name, control_output_name = ARGV
raise 'usage: run_lifecycle.rb ROOT SOURCE_SHA OUTPUT_BASENAME CONTROL_OUTPUT_BASENAME' unless root && source_sha && output_name && control_output_name
root = File.realpath(root)
valid_name = lambda { |name| File.basename(name) == name && name.match?(/\A[A-Za-z0-9._-]+\z/) }
raise 'output arguments must be basenames' unless valid_name.call(output_name) && valid_name.call(control_output_name)
output_root = File.join(root, 'evidence/schema2')
output = File.join(output_root, output_name)
control_output = File.join(output_root, control_output_name)
raise 'output paths must be new' if File.exist?(output) || File.exist?(control_output)
control = File.join(root, 'evidence/schema2/control.json')
checker = File.join(root, 'kern/trace_lifecycle_check.rb')
controller = File.join(root, 'kern/trace_control.rb')
preflight = File.join(root, 'kern/trace_preflight')
config = File.join(root, 'evidence/schema2/config-cta1.json')
lock_path = '/tmp/fs13-validation.lock'

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
  control_env = {'FLASH_TRACE_VALIDATION_LOCK_HELD' => '1'}
  run_and_record.call('prepare-ordinary', control_env,
                      'ruby', controller, control, 'lifecycle-ordinary')
  run_and_record.call('prepare-traced', control_env,
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
