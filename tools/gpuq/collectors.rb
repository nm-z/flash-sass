require 'open3'
require 'timeout'
require 'etc'
require 'csv'
require 'socket'
require_relative 'importer'

module GPUQ
  class Collectors
    GPU_FIELDS={
      'index'=>nil,'uuid'=>nil,'utilization.gpu'=>['gpu.utilization_percent','percent'],
      'utilization.memory'=>['gpu.memory_utilization_percent','percent'],
      'memory.used'=>['gpu.memory_used_mib','MiB'],'memory.total'=>['gpu.memory_total_mib','MiB'],
      'temperature.gpu'=>['gpu.temperature_c','C'],'clocks.sm'=>['gpu.sm_clock_mhz','MHz'],
      'clocks.mem'=>['gpu.memory_clock_mhz','MHz'],'power.draw'=>['gpu.power_w','W'],
      'power.limit'=>['gpu.power_limit_w','W'],'pcie.link.gen.current'=>['gpu.pcie_generation','generation'],
      'pcie.link.width.current'=>['gpu.pcie_width','lanes'],
      'clocks_throttle_reasons.active'=>['gpu.throttle_mask','bitmask']
    }.freeze
    attr_reader :errors
    def initialize(db,job,excluded,engine_socket,importer)
      @db,@job,@excluded,@socket,@importer=db,job,excluded,engine_socket,importer
      @seq=Hash.new(0);@errors={};@available={}
    end
    def enabled?(name) = !@excluded.include?(name)
    def doc(kind,source,bytes)
      ordinal=@seq[[kind,source]];@seq[[kind,source]]+=1
      @db.document(@job,kind,source,bytes,ordinal:ordinal)
    end
    def command(name,argv,timeout:15)
      started=GPUQ.mono_ns
      out=err='';status=nil
      Open3.popen3(*argv,pgroup:true) do |input,output,error,thread|
        input.close
        readers=[Thread.new { output.read },Thread.new { error.read }]
        begin
          Timeout.timeout(timeout) { status=thread.value;out=readers[0].value;err=readers[1].value }
        rescue Timeout::Error
          Process.kill('KILL',-thread.pid) rescue Errno::ESRCH
          thread.join;out=readers[0].value;err=readers[1].value
          raise "collector command timed out: #{argv.first}"
        ensure
          readers.each(&:join)
        end
      end
      id=doc('collector',GPUQ.json(argv),GPUQ.json(stdout:out,stderr:err,exit_code:status.exitstatus))
      raise "#{argv.first}: exit #{status.exitstatus}: #{err.strip}" unless status.success?
      @db.metric(@job,'collector.command_ns',GPUQ.mono_ns-started,unit:'ns',entity:name,document:id)
      [out,id]
    end
    def attempt(name)
      return unless enabled?(name)
      started=GPUQ.mono_ns
      begin
        yield
        @available[name]=true
        @db.collector(@job,name,'collecting',nil,elapsed_ns:GPUQ.mono_ns-started)
      rescue SQLite3::Exception
        raise
      rescue StandardError => e
        @errors[name]=e.message
        doc('collector_error',name,"#{e.class}: #{e.message}")
        @db.collector(@job,name,@available[name] ? 'partial' : 'unavailable',e.message,elapsed_ns:GPUQ.mono_ns-started)
      end
    end
    def gpu
      attempt('gpu') do
        out,source=command('gpu',['nvidia-smi',"--query-gpu=#{GPU_FIELDS.keys.join(',')}",'--format=csv,noheader,nounits'])
        @db.batch do
        CSV.parse(out).each do |row|
          raise 'unexpected GPU field count' unless row.length==GPU_FIELDS.length
          values=GPU_FIELDS.keys.zip(row.map(&:strip)).to_h
          GPU_FIELDS.each do |key,definition|
            next unless definition
            value=values[key]
            if value.match?(/\A0x[0-9a-f]+\z/i)
              value=value.to_i(16)
            elsif value.match?(/\A-?\d+(?:\.\d+)?\z/)
              value=value.include?('.') ? value.to_f : value.to_i
            else
              next # N/A is retained in the raw query, never interpreted as zero.
            end
            @db.metric(@job,definition[0],value,unit:definition[1],entity:values['uuid'],document:source,
              dimensions:{gpu_index:values['index'],scope:'whole device, including concurrent workloads'})
          end
        end
        end
      end
    end
    def driver_snapshot
      attempt('gpu') do
        out,_=command('gpu',['nvidia-smi','-q','-x'])
        doc('gpu_xml','nvidia-smi -q -x',out)
        command('gpu',['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv'])
      end
    end
    def process_tree(root)
      attempt('process') do
        pending=[root];seen={};rss=cpu=reads=writes=0;members=[]
        until pending.empty?
          pid=pending.shift;next if seen[pid];seen[pid]=true
          begin
            stat=File.read("/proc/#{pid}/stat").split(') ',2).last.split
            rss+=stat[21].to_i*Etc.sysconf(Etc::SC_PAGESIZE)
            cpu+=stat[11].to_i+stat[12].to_i
            io=File.read("/proc/#{pid}/io").scan(/^(\w+):\s+(\d+)/).to_h
            reads+=io.fetch('read_bytes','0').to_i;writes+=io.fetch('write_bytes','0').to_i
            members << pid
            pending.concat(File.read("/proc/#{pid}/task/#{pid}/children").split.map(&:to_i))
          rescue Errno::ENOENT,Errno::ESRCH,Errno::EACCES
            next
          end
        end
        raise 'process exited before a tree sample; waited resource usage is recorded separately' if members.empty?
        @db.batch do
        dims={pids:members,scope:'currently observed process tree; exited children may be absent'}
        {'process.tree_rss_bytes'=>[rss,'bytes'],'process.tree_cpu_ticks'=>[cpu,'ticks'],
         'process.tree_read_bytes'=>[reads,'bytes'],'process.tree_write_bytes'=>[writes,'bytes']}.each do |name,(value,unit)|
          @db.metric(@job,name,value,unit:unit,dimensions:dims)
        end
        mem=File.read('/proc/meminfo').scan(/^(\w+):\s+(\d+)/).to_h
        %w[MemAvailable MemFree Cached Dirty SwapFree].each { |k| @db.metric(@job,"host.memory.#{k.downcase}_bytes",mem[k].to_i*1024,unit:'bytes',entity:'host') if mem[k] }
        %w[cpu io memory].each do |kind|
          text=File.read("/proc/pressure/#{kind}") rescue next
          text.each_line do |line|
            scope=line.split.first
            line.scan(/(avg10|avg60|avg300|total)=([\d.]+)/).each do |field,value|
              @db.metric(@job,"host.pressure.#{kind}.#{scope}.#{field}",field=='total' ? value.to_i : value.to_f,unit:field=='total' ? 'us' : 'percent',entity:'host')
            end
          end
        end
        end
      end
    end
    def source_files(argv,cwd)
      candidates=argv.filter_map do |arg|
        path=File.expand_path(arg,cwd)
        path if File.file?(path) && path.match?(/\.(rb|pl|sh|bash|lua|c|h|zig|rs|go|sass|ptx|sql)\z/)
      end
      candidates.uniq.each { |path| @db.document(@job,'source',path,File.binread(path)) }
    end
    def provenance(argv,cwd)
      attempt('provenance') do
        command('provenance',['uname','-a'])
        executable=argv.first.include?('/') ? File.expand_path(argv.first,cwd) : ENV.fetch('PATH','').split(':').map { |p| File.join(p,argv.first) }.find { |p| File.executable?(p) && File.file?(p) }
        paths=([executable]+argv.filter_map { |a| p=File.expand_path(a,cwd);p if File.file?(p) }).compact.uniq
        identities=paths.map do |path|
          before=File.stat(path);stamp={path:path,device:before.dev,inode:before.ino,bytes:before.size,mtime_ns:(before.mtime.to_r*1_000_000_000).to_i,ctime_ns:(before.ctime.to_r*1_000_000_000).to_i}
          if before.size<=64*1024*1024
            hash=Digest::SHA256.file(path).hexdigest
            after=File.stat(path)
            raise "changed while hashing #{path}" unless [before.size,before.mtime,before.ctime,before.ino]==[after.size,after.mtime,after.ctime,after.ino]
            stamp[:sha256]=hash
          else
            stamp[:sha256]=nil;stamp[:hash_status]='not hashed automatically: large model/input; supply its existing hash manifest as an artifact'
          end
          stamp
        end
        doc('identity','command files',GPUQ.json(identities))
        roots=([cwd]+paths.map { |p| File.dirname(p) }).uniq
        found={}
        roots.each do |dir|
          out,err,status=Open3.capture3('git','-C',dir,'rev-parse','--show-toplevel')
          next unless status.success?
          root=out.strip;next if found[root];found[root]=true
          %w[HEAD].each { |ref| command('provenance',['git','-C',root,'rev-parse',ref]) }
          command('provenance',['git','-C',root,'status','--porcelain=v1'])
          command('provenance',['git','-C',root,'diff','--binary','HEAD'])
        end
        @db.collector(@job,'provenance','partial','large input hashes require the supplied manifest') if identities.any? { |x| !x[:sha256] }
      end
    end
    def engine_query(cmd)
      Timeout.timeout(5) do
        UNIXSocket.open(@socket) do |s|
          pid=s.getsockopt(Socket::SOL_SOCKET,Socket::SO_PEERCRED).unpack('iii').first
          s.puts(cmd);data=''
          while (line=s.gets)
            data << line
            break if cmd=='config' || line.start_with?('error:','trace_end','hosttrace_end')
          end
          [pid,data]
        end
      end
    end
    def engine(stage)
      return unless enabled?('engine') || enabled?('trace')
      attempt('engine') do
        pid,text=engine_query('config')
        doc('engine_context',"#{stage}:config",text)
        doc('engine_context',"#{stage}:process",File.binread("/proc/#{pid}/cmdline"))
        @config=text.split.drop(1).filter_map { |x| x.split('=',2) if x.include?('=') }.to_h
      end
      return unless stage=='after' && enabled?('trace')
      attempt('trace') do
        if !@config || @config.fetch('trace_tokens','0').to_i==0
          raise 'resident engine has no enabled trace producer; raw job-provided traces are still imported'
        end
        commands=['hosttrace']+(0...@config.fetch('dies','0').to_i).map { |d| "trace #{d}" }
        commands.each do |cmd|
          pid,text=engine_query(cmd)
          id=doc('engine_trace_context',cmd,text)
          # Shared engines may serve requests outside the queue. Keep these as
          # explicitly unattributed context, never assert they belong to this job.
          @importer.parse(id,text,"engine_snapshot_unattributed:#{pid}:#{cmd}")
        end
      end
    end
    def finish
      GROUPS.each do |name|
        next unless enabled?(name)
        row=@db.one('SELECT status,reason FROM collectors WHERE job_id=? AND name=?',[@job,name])
        next unless row && %w[pending collecting].include?(row['status'])
        reason=@errors[name]
        @db.collector(@job,name,reason ? 'partial' : 'complete',reason)
      end
    end
  end
end
