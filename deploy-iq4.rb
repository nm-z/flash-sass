require 'socket'
require 'timeout'
require 'fileutils'
root = '/home/nate/codex/flash-astra-perf-20261018'
candidate = '/home/nate/codex/flash-astra-build-20261018/iq4-bias/kern/die.cubin'
query = lambda do |cmd|
  Timeout.timeout(180) do
    UNIXSocket.open('/tmp/flash.sock') do |s|
      s.puts(cmd)
      line = s.gets or raise 'control EOF'
      raise line if line.start_with?('error')
      puts line
      line
    end
  end
end
parse = ->(line) { line.split.drop(1).map { |v| v.split('=',2) }.to_h }
original = parse.call(query.call('config'))
raise 'unexpected runtime' unless original.values_at('layers','dies','maxt','img_rows') == %w[48/48 6 230000 4096]
previous = original.fetch('cubin')
changed = false
complete = false
begin
  changed = true
  query.call("cubin #{candidate}")
  actual = query.call('config')
  fields = parse.call(actual)
  original.each { |k,v| raise "config changed: #{k}" unless %w[cubin cubin_fnv].include?(k) || fields[k] == v }
  File.write(root+'/iq4-deployed-config.txt',actual)
  STDOUT.flush
  raise 'HTTP failed' unless system({'FRESH_SET'=>'6'},RbConfig.ruby,root+'/http-real.rb',root+'/http-iq4-live')
  complete = true
ensure
  query.call("cubin #{previous}") if changed && !complete
end
