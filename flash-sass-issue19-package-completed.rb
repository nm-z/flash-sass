#!/usr/bin/env ruby
# Archive completed, hash-verified validation records without binaries or weights.
require 'json'
require 'digest'
require 'fileutils'
require 'open3'
require 'time'

old_root, new_root = ARGV
abort 'Usage: package-completed.rb OLD_EVIDENCE_DIR MERGED_EVIDENCE_DIR' unless old_root && new_root
[old_root, new_root].each do |root|
  doc = JSON.parse(File.read(File.join(root, 'model.json')))
  raise "Incomplete model evidence: #{root}" unless doc['status'] == 'PASS' && doc['original_engine_restored'] == true
end
raise 'Merged driver did not finish successfully' unless File.read(File.join(new_root, 'integration-driver.exit')).strip == '0'
corpus = File.join(old_root, 'model-corpus.txt')
FileUtils.cp(corpus, File.join(new_root, 'model-corpus.txt'))
audit = File.expand_path('flash-sass-issue19-audit-completed.rb', __dir__)
raise 'Completed evidence audit failed' unless system('ruby', audit, old_root, new_root)

baseline = File.binread(File.join(old_root, 'resident-stderr-during-65536.log'))
final_log = File.binread('/tmp/flashcore.err')
raise 'Resident error log changed or rotated unexpectedly' unless final_log.start_with?(baseline)
File.binwrite(File.join(new_root, 'resident-stderr-through-restoration.log'), final_log)
errors = final_log.lines.grep(/error|failed|short.read|watchdog|fault|timeout/i)
log_note = {captured_utc: Time.now.utc.iso8601, core_pid: 3575268, source: '/tmp/flashcore.err',
            file: 'resident-stderr-through-restoration.log', bytes: final_log.bytesize,
            sha256: Digest::SHA256.hexdigest(final_log), earlier_snapshot_preserved_as_prefix: true,
            error_pattern: 'error|failed|short.read|watchdog|fault|timeout (case insensitive)',
            matching_lines: errors,
            scope: 'The full resident log includes activity before issue #19 validation. The baseline snapshot was captured during the 65536-position run; subsequent reload/restoration messages are retained.'}
File.write(File.join(new_root, 'resident-log-audit.json'), JSON.pretty_generate(log_note) + "\n")
raise 'Resident engine log needs error review' unless errors.empty?

def package(root, modes, extras, name)
  files = []
  record_hashes = 0
  modes.each do |mode|
    file = "#{mode}.json"
    doc = JSON.parse(File.read(File.join(root, file)))
    raise "Incomplete #{mode}" unless doc.fetch('status') == 'PASS'
    files << file
    doc.fetch('results').each do |row|
      raise "Failed record #{row['name']}" if row.key?('passes') && row.fetch('passes') != true
      if row.key?('expected_exit')
        raise "Unexpected exit #{row['name']}" unless row.fetch('exit_code') == row.fetch('expected_exit')
      end
      {'stdout' => 'stdout_sha256', 'stderr' => 'stderr_sha256',
       'commands_file' => 'commands_sha256'}.each do |key, hash_key|
        next unless row.key?(key)
        path = row.fetch(key)
        raise "Unexpected path #{path}" unless path == File.basename(path)
        raise "Hash mismatch #{path}" unless Digest::SHA256.file(File.join(root, path)).hexdigest == row.fetch(hash_key)
        record_hashes += 1
        files << path
      end
    end
    if mode == 'model'
      files += doc.fetch('model_contexts').map { |c| "model-#{c.fetch('positions')}.tokens.json" }
    end
  end
  files = (files + extras).uniq.sort
  entries = files.to_h do |file|
    raise "Unexpected archive path #{file}" unless file == File.basename(file)
    path = File.join(root, file)
    raise "Missing archive member #{path}" unless File.file?(path)
    [file, {sha256: Digest::SHA256.file(path).hexdigest, bytes: File.size(path)}]
  end
  list = File.join(root, name + '.members')
  archive = File.join(root, name + '.tar.gz')
  File.binwrite(list, files.join("\0") + "\0")
  command = ['tar', '-czf', archive, '-C', root, '--null', '-T', list]
  stdout, stderr, status = Open3.capture3(*command)
  raise "Archive failed: #{stderr}" unless status.success?
  entries.each do |file, data|
    raise "Evidence changed while packaging #{file}" unless Digest::SHA256.file(File.join(root, file)).hexdigest == data.fetch(:sha256)
  end
  note = {status: 'PASS', created_utc: Time.now.utc.iso8601, archive: File.basename(archive),
          archive_sha256: Digest::SHA256.file(archive).hexdigest, archive_bytes: File.size(archive),
          files: files.size, recorded_file_hash_checks: record_hashes, command: command,
          command_exit: status.exitstatus, command_stdout: stdout, command_stderr: stderr,
          entries: entries, scope: 'Recorded commands, inputs and results only; no engine binaries, cubins or checkpoint weights.'}
  File.write(File.join(root, name + '-package.json'), JSON.pretty_generate(note) + "\n")
  puts JSON.generate(note.reject { |key, _| %i[entries command].include?(key) })
end

package(old_root, ['model'],
        %w[model-corpus.txt model-audit.json model-layer-tables.md model-driver.stdout model-driver.stderr
           model-driver.pid model-process.json resident-stderr-during-65536.log resident-stderr-baseline.json],
        'model-raw')
package(new_root, %w[cpu build gpu model merge-targets],
        %w[model-corpus.txt model-audit.json parity-128.json artifact-equivalence.json source-equivalence.json
           integration-driver.sh integration-driver.stdout integration-driver.stderr integration-driver.pid
           integration-driver.exit resident-stderr-through-restoration.log resident-log-audit.json],
        'integration-raw')
