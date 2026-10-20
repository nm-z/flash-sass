require 'shellwords'
require 'find'
require 'set'
require_relative 'collectors'

module GPUQ
  class Runner
    def initialize(db,options)
      @db,@o=db,options;@signal=nil;@fatal=nil;@pid=nil
      @lock_path=ENV.fetch('GPUQ_LOCK','/tmp/flash-sass-engine-validation.lock')
      @queue=ENV.fetch('GPUQ_QUEUE','/tmp/gpuq')
    end
    def run(argv,replay_of:nil)
      raise 'missing command' if argv.empty?
      excluded=@o.fetch(:exclude,[])
      unknown=excluded-GROUPS
      raise "unknown optional collectors: #{unknown.join(',')}" unless unknown.empty?
      # Install handlers before the queued row becomes visible to other clients.
      traps=%w[INT TERM HUP].to_h { |s| [s,Signal.trap(s) { @signal=Signal.list.fetch(s) }] }
      cwd=Dir.pwd
      parent=ENV['GPUQ_JOB_ID']
      parent=nil unless parent && @db.one("SELECT id FROM jobs WHERE id=? AND status='running'",[parent])
      @id=@db.create_job(argv:argv,cwd:cwd,exclusions:excluded,label:@o[:name]||File.basename(argv.first),parent:parent,replay_of:replay_of)
      @importer=Importer.new(@db,@id,excluded:excluded)
      @collectors=Collectors.new(@db,@id,excluded,@o.fetch(:socket,ENV.fetch('FLASH_SOCK','/tmp/flash.sock')),@importer)
      staging=File.join(File.dirname(@db.path),'staging',@id)
      FileUtils.mkdir_p(staging,mode:0700)
      roots=@o.fetch(:capture,[]).map { |p| File.expand_path(p,cwd) }
      # Nonexistent path arguments are common --out/positional run directories.
      argv.each { |a| p=File.expand_path(a,cwd);roots << p if a.include?('/') && !a.start_with?('-') && !File.exist?(p) }
      roots=roots.uniq
      @existing_paths=Set.new
      roots.each do |root|
        next unless File.exist?(root)
        Find.find(root) { |p| File.directory?(p) && File.symlink?(p) ? Find.prune : @existing_paths.add(p) }
      end
      status=nil;start_mono=nil;ended=nil;ended_mono=nil;code=70
      begin
        files=Dir[File.join(__dir__,'*.{rb,sql,json}')].sort.to_h { |p| [File.basename(p),Digest::SHA256.file(p).hexdigest] }
        @db.document(@id,'recorder','implementation',GPUQ.json(files:files,ruby:RUBY_VERSION,sqlite_gem:SQLite3::VERSION,sqlite:SQLite3::SQLITE_VERSION,profile:1))
        @collectors.source_files(argv,cwd) # recipe source and output capture are mandatory
        @db.collector(@id,'identity','complete','argv, cwd, profile and supplied scripts retained')
        acquire(parent)
        if @signal
          code=128+@signal
          @db.update_job(@id,status:'cancelled',ended_ns:GPUQ.now_ns,exit_code:code,termination_signal:@signal)
          return code
        end
        prepare=GPUQ.mono_ns
        @collectors.provenance(argv,cwd)
        @collectors.driver_snapshot
        @collectors.engine('before')
        @db.metric(@id,'collection.prepare_ns',GPUQ.mono_ns-prepare,unit:'ns')
        if @signal
          code=128+@signal
          @db.update_job(@id,status:'cancelled',ended_ns:GPUQ.now_ns,exit_code:code,termination_signal:@signal)
          return code
        end
        env={'GPUQ_JOB_ID'=>@id,'GPUQ_DB'=>@db.path,'GPUQ_RUN_DIR'=>staging,'GPUQ_ARTIFACT_DIR'=>staging,
             'GPUQ_EXCLUDE'=>excluded.join(','),'GPUQ_LOCK_FD'=>'9','FLASH_VALIDATION_LOCK_FD'=>'9','TMPDIR'=>staging}
        command=argv
        usage_r=usage_w=nil
        unless excluded.include?('process')
          raise '/usr/bin/time is required for exact process resource accounting' unless File.executable?('/usr/bin/time')
          usage_r,usage_w=IO.pipe
          format='{"user_s":%U,"system_s":%S,"max_rss_kib":%M,"major_faults":%F,"minor_faults":%R,"fs_inputs":%I,"fs_outputs":%O,"voluntary_switches":%w,"involuntary_switches":%c}'
          command=['/usr/bin/time','--quiet',"--format=#{format}",'--output=/proc/self/fd/4','--',*argv]
        end
        out_r,out_w=IO.pipe;err_r,err_w=IO.pipe
        input_source=@o.fetch(:input,STDIN)
        input_r,input_w=IO.pipe unless input_source.tty?
        spawn_options={in:input_r||File::NULL,out:out_w,err:err_w,pgroup:true,9=>@lock,close_others:true}
        spawn_options[4]=usage_w if usage_w
        start_mono=GPUQ.mono_ns
        @db.update_job(@id,status:'running',started_ns:GPUQ.now_ns)
        @pid=Process.spawn(env,*command,**spawn_options)
        out_w.close;err_w.close;usage_w&.close
        input_r&.close
        if input_w
          input_thread=Thread.new do
            ordinal=0
            begin
              loop do
                bytes=input_source.readpartial(65536)
                @db.document(@id,'stdin','stdin',bytes,ordinal:ordinal);ordinal+=1
                input_w.write(bytes)
              end
            rescue EOFError,Errno::EPIPE
            rescue StandardError => e
              @fatal=e
            ensure
              input_w.close unless input_w.closed?
            end
          end
        end
        readers=[capture(out_r,'stdout'),capture(err_r,'stderr')]
        sampler=Thread.new do
          until @sampling_stop
            @collectors.gpu
            @collectors.process_tree(@pid)
            @sleep_mutex.synchronize { @sleep_cv.wait(@sleep_mutex,@o.fetch(:interval,1.0)) unless @sampling_stop }
          end
        rescue StandardError => e
          @fatal=e
        end
        waiter=Thread.new do
          waited=Process.waitpid2(@pid)
          [waited[1],GPUQ.now_ns,GPUQ.mono_ns]
        end
        deadline=@o[:timeout] && start_mono+(@o[:timeout]*1_000_000_000).to_i
        sent=nil
        loop do
          if waiter.join(0)
            status,ended,ended_mono=waiter.value
            break
          end
          if (@signal || @fatal || (deadline && GPUQ.mono_ns>=deadline)) && !sent
            sig=@signal || Signal.list['TERM'];Process.kill(sig,-@pid) rescue Errno::ESRCH
            sent=GPUQ.mono_ns
          elsif sent && GPUQ.mono_ns-sent>5_000_000_000
            Process.kill('KILL',-@pid) rescue Errno::ESRCH
          end
          waiter.join(0.02)
        end
        elapsed=ended_mono-start_mono
        @sampling_stop=true;@sleep_mutex.synchronize { @sleep_cv.broadcast };sampler.join
        input_thread&.kill&.join
        readers.each do |thread|
          unless thread.join(5)
            @db.document(@id,'capture_error','stdout/stderr','A background descendant retained the output pipe after command exit; capture is incomplete.')
            thread.kill.join
            @capture_incomplete=true
          end
        end
        code=status.signaled? ? 128+status.termsig : status.exitstatus
        code=128+@signal if @signal
        code=70 if @fatal
        @db.update_job(@id,status:'collecting',ended_ns:ended,exit_code:code,termination_signal:status.termsig||@signal)
        @db.metric(@id,'job.command_wall_ns',elapsed,unit:'ns') unless excluded.include?('timing')
        if usage_r
          raw=usage_r.read;usage_r.close
          doc=@db.document(@id,'resource_usage','waited command and children',raw)
          begin
            JSON.parse(raw).each { |k,v| @db.metric(@id,"process.#{k}",v,unit:k.end_with?('_s') ? 's' : k.end_with?('_kib') ? 'KiB' : 'count',document:doc) }
          rescue JSON::ParserError
            @db.collector(@id,'process','partial','resource usage unavailable after termination')
          end
        end
        finish=GPUQ.mono_ns
        @collectors.engine('after')
        @collectors.driver_snapshot
        collect_files(staging,roots)
        @db.metric(@id,'collection.finish_ns',GPUQ.mono_ns-finish,unit:'ns')
        @db.collector(@id,'artifacts','complete',nil,config:{files:@importer.count['files'],new_text_files_moved:true})
        @db.collector(@id,'output',@capture_incomplete || @fatal ? 'partial' : 'complete','stdout/stderr/stdin retained')
        trace_rows=@importer.count['phase']+@importer.count['host_trace']
        @db.collector(@id,'trace',trace_rows>0 ? 'complete' : 'unavailable',trace_rows>0 ? nil : 'no valid instrumented producer supplied a trace',config:{rows:trace_rows}) unless excluded.include?('trace')
        @db.collector(@id,'profiler',@importer.count['profiler']>0 ? 'complete' : 'unavailable',@importer.count['profiler']>0 ? nil : 'no profiler counter producer supplied records') unless excluded.include?('profiler')
        @collectors.finish
        @db.document(@id,'runner_error','storage/capture',"#{@fatal.class}: #{@fatal.message}") if @fatal
        final=@signal ? 'cancelled' : deadline && sent ? 'timed_out' : @fatal || @capture_incomplete ? 'capture_failed' : code==0 ? 'succeeded' : 'failed'
        @db.update_job(@id,status:final,exit_code:code)
        code
      rescue StandardError => e
        Process.kill('TERM',-@pid) rescue Errno::ESRCH if @pid && !status
        failure=e.is_a?(Errno::ENOENT) ? 127 : e.is_a?(Errno::EACCES) ? 126 : 70
        @db.document(@id,'runner_error','exception',"#{e.class}: #{e.message}\n#{e.backtrace.join("\n")}")
        @db.update_job(@id,status:'failed',ended_ns:GPUQ.now_ns,exit_code:failure)
        failure
      ensure
        @sampling_stop=true;@sleep_mutex&.synchronize { @sleep_cv&.broadcast }
        input_thread&.kill&.join if defined?(input_thread)
        sampler&.join if defined?(sampler)
        if defined?(readers) && readers
          readers.each { |t| t.kill.join if t.alive? }
        end
        [out_r,out_w,err_r,err_w,usage_r,usage_w].each { |io| io.close if io && !io.closed? } if defined?(out_r)
        [input_r,input_w].each { |io| io.close if io && !io.closed? } if defined?(input_r)
        File.unlink(@ticket_file) if @ticket_file && File.exist?(@ticket_file)
        @lock&.close
        traps&.each { |s,handler| Signal.trap(s,handler) }
      end
    end
    def acquire(parent)
      @sleep_mutex=Mutex.new;@sleep_cv=ConditionVariable.new
      FileUtils.mkdir_p(@queue)
      if parent && ENV['GPUQ_LOCK_FD']=='9'
        inherited=IO.for_fd(9,autoclose:false)
        expected=File.stat(@lock_path)
        raise 'invalid inherited queue lock' unless inherited.stat.ino==expected.ino && inherited.stat.dev==expected.dev
        # dup preserves the open-file description and its flock ownership.
        @lock=inherited.dup
        return
      end
      @lock=File.open(@lock_path,'w')
      ticket=GPUQ.now_ns
      loop do
        @ticket_file=File.join(@queue,"#{ticket}.job")
        begin
          File.open(@ticket_file,File::WRONLY|File::CREAT|File::EXCL,0600) { |f| f.puts("#{Process.pid} #{@id}") };break
        rescue Errno::EEXIST
          ticket+=1
        end
      end
      File.chmod(0644,@ticket_file) # other queue users need PID/order, never argv
      @db.update_job(@id,ticket:ticket.to_s)
      start=GPUQ.mono_ns
      loop do
        break if @signal
        older=Dir.glob(File.join(@queue,'*.job')).any? do |path|
          first=File.basename(path,'.job').to_i
          pid=File.read(path).split.first.to_i rescue 0
          alive=pid>0 && (Process.kill(0,pid) rescue false)
          File.unlink(path) rescue nil unless alive
          alive && first<ticket
        end
        break if !older && @lock.flock(File::LOCK_EX|File::LOCK_NB)
        sleep 0.2
      end
      @db.metric(@id,'job.queue_wait_ns',GPUQ.mono_ns-start,unit:'ns')
      File.unlink(@ticket_file) if File.exist?(@ticket_file)
      @ticket_file=nil
    end
    def capture(io,stream)
      Thread.new do
        ordinal=0;buffer=''.b;doc=nil
        loop do
          bytes=io.readpartial(65536)
          doc=@db.document(@id,stream,stream,bytes,ordinal:ordinal);ordinal+=1
          buffer << bytes
          while (n=buffer.index("\n"))
            @importer.line(doc,buffer.slice!(0,n+1),{'stream'=>stream,'chunk'=>ordinal-1})
          end
          if buffer.bytesize>8*1024*1024
            @db.document(@id,'parse_notice',stream,'Oversized line retained verbatim; structured line parsing skipped.',ordinal:ordinal)
            buffer.clear
          end
        end
      rescue EOFError
        begin
          @importer.line(doc,buffer,{'stream'=>stream,'chunk'=>ordinal-1}) unless buffer.empty?
        rescue StandardError => e
          @fatal=e
        end
      rescue StandardError => e
        @fatal=e
      ensure
        io.close unless io.closed?
      end
    end
    def collect_files(staging,roots)
      [staging,*roots].uniq.each do |root|
        next unless File.exist?(root)
        if File.file?(root)
          @importer.file(root,remove:!@existing_paths.include?(root))
        elsif File.directory?(root) && !File.symlink?(root)
          repositories=[]
          Find.find(root) do |path|
            if File.directory?(path)
              Find.prune if %w[.git node_modules].include?(File.basename(path)) || File.symlink?(path)
              repositories << path if File.exist?(File.join(path,'.git'))
              next
            end
            next unless File.file?(path) && !File.symlink?(path)
            managed=path.start_with?(staging+'/')
            repository_file=repositories.any? { |repo| path.start_with?(repo+'/') }
            @importer.file(path,remove:!repository_file && (managed || !@existing_paths.include?(path)))
          end
        end
      end
    end
  end
end
