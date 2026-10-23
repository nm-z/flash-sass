require 'rubygems'
vendor=ENV.fetch('GPUQ_GEM_PATH','/usr/local/lib/gpuq/gems')
Gem.use_paths(Gem.dir,[vendor,*Gem.path]) if File.directory?(vendor)
require 'sqlite3'
require 'json'
require 'zlib'
require 'digest'
require 'fileutils'
require 'socket'
require 'securerandom'
require 'monitor'

module GPUQ
  GROUPS = %w[timing process gpu engine trace profiler provenance].freeze
  MANDATORY = %w[identity output artifacts].freeze
  ENV_KEYS = /\A(?:FLASH_|GEMV_|DIE_|CUDA_VISIBLE_DEVICES\z|CUDA_DEVICE_ORDER\z|SPLIT_SHARES\z|ATTN_|OMP_NUM_THREADS\z)/
  def self.now_ns = Process.clock_gettime(Process::CLOCK_REALTIME, :nanosecond)
  def self.mono_ns = Process.clock_gettime(Process::CLOCK_MONOTONIC, :nanosecond)
  def self.clean(value)
    case value
    when String then value.encode('UTF-8',invalid: :replace,undef: :replace)
    when Hash then value.to_h { |k,v| [clean(k),clean(v)] }
    when Array then value.map { |v| clean(v) }
    else value
    end
  end
  def self.json(value) = JSON.generate(clean(value))
  def self.process_start(pid)
    File.read("/proc/#{pid}/stat").split(') ',2).last.split[19]
  rescue Errno::ENOENT,Errno::EACCES
    nil
  end
  def self.default_db
    File.join(ENV.fetch('XDG_STATE_HOME',File.expand_path('~/.local/state')),'gpuq','history.sqlite3')
  end
  class Store
    attr_reader :path
    def initialize(path)
      @path=File.expand_path(path)
      FileUtils.mkdir_p(File.dirname(@path),mode:0700)
      @db=SQLite3::Database.new(@path)
      File.chmod(0600,@path)
      @db.busy_timeout=15_000
      @db.results_as_hash=true
      @db.execute('PRAGMA journal_mode=WAL')
      @db.execute('PRAGMA synchronous=FULL')
      @db.execute('PRAGMA foreign_keys=ON')
      @db.execute_batch(File.read(File.join(__dir__,'schema.sql')))
      raise 'unsupported database schema' if @db.get_first_value('SELECT max(version) FROM schema_version')>2
      columns=@db.execute('PRAGMA table_info(metric_definitions)').map { |r| r['name'] }
      @db.execute("ALTER TABLE metric_definitions ADD COLUMN collector TEXT NOT NULL DEFAULT 'timing'") unless columns.include?('collector')
      @db.execute('INSERT OR IGNORE INTO schema_version VALUES(2)')
      definitions=JSON.parse(File.read(File.join(__dir__,'metrics.json')))
      @db.transaction do
        definitions.each do |row|
          @db.execute('INSERT OR IGNORE INTO metric_definitions(name,unit,description,collector) VALUES(?,?,?,?)',row.values_at('name','unit','description','collector'))
        end
      end
      @mutex=Monitor.new
    end
    def close = @mutex.synchronize { @db.close }
    def execute(sql,args=[]) = @mutex.synchronize { @db.execute(sql,args) }
    def one(sql,args=[]) = execute(sql,args).first
    def batch
      @mutex.synchronize do
        if @db.transaction_active?
          yield
        else
          # Reserve the writer before document identity reads. A deferred WAL
          # snapshot cannot be upgraded after another gpuq process commits.
          @db.transaction(:immediate) { yield }
        end
      end
    end
    def create_job(argv:,cwd:,exclusions:[],label:nil,parent:nil,replay_of:nil,origin:'live',source_key:nil,status:'queued',submitted_ns:GPUQ.now_ns)
      id=SecureRandom.uuid
      boot=File.read('/proc/sys/kernel/random/boot_id').strip rescue nil
      env=ENV.select { |k,_| k.match?(ENV_KEYS) }
      execute('INSERT INTO jobs(id,parent_id,replay_of,label,host,boot_id,pid,process_start,cwd,argv_json,environment_json,submitted_ns,status,exclusions_json,origin,source_key) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
        [id,parent,replay_of,label,origin=='live' ? Socket.gethostname : 'unknown',boot,Process.pid,GPUQ.process_start(Process.pid),cwd,GPUQ.json(argv),GPUQ.json(env),submitted_ns,status,GPUQ.json(exclusions),origin,source_key])
      GROUPS.each do |name|
        excluded=exclusions.include?(name)
        collector(id,name,excluded ? 'excluded' : (origin=='live' ? 'pending' : 'not_applicable'),excluded ? 'explicit --exclude' : nil)
      end
      MANDATORY.each { |name| collector(id,name,origin=='live' ? 'pending' : 'not_applicable','mandatory capture') }
      id
    end
    def update_job(id,fields)
      allowed=%w[started_ns ended_ns status exit_code termination_signal ticket]
      raise 'invalid job fields' unless (fields.keys.map(&:to_s)-allowed).empty?
      execute("UPDATE jobs SET #{fields.keys.map { |k| "#{k}=?" }.join(',')} WHERE id=?",fields.values+[id])
    end
    def collector(id,name,status,reason=nil,config:{},elapsed_ns:0)
      execute('INSERT INTO collectors(job_id,name,status,reason,config_json,elapsed_ns) VALUES(?,?,?,?,?,?) ON CONFLICT(job_id,name) DO UPDATE SET status=excluded.status,reason=excluded.reason,config_json=excluded.config_json,elapsed_ns=collectors.elapsed_ns+excluded.elapsed_ns',
        [id,name,status,reason,GPUQ.json(config),elapsed_ns])
    end
    def document(id,kind,source,bytes,ordinal:0,at:GPUQ.now_ns,index_text:nil)
      raw=bytes.b;hash=Digest::SHA256.hexdigest(raw)
      packed=Zlib::Deflate.deflate(raw)
      codec=packed.bytesize<raw.bytesize ? 'zlib' : 'raw'
      data=codec=='zlib' ? packed : raw
      @mutex.synchronize do
        batch do
          prior=@db.get_first_row('SELECT id,sha256 FROM documents WHERE job_id=? AND kind=? AND source=? AND ordinal=?',[id,kind,source,ordinal])
          if prior
            raise 'document identity collision' unless prior['sha256']==hash
            prior['id']
          else
            @db.execute('INSERT INTO documents(job_id,kind,source,ordinal,observed_ns,bytes,sha256,codec,data) VALUES(?,?,?,?,?,?,?,?,?)',
            [id,kind,source,ordinal,at,raw.bytesize,hash,codec,SQLite3::Blob.new(data)])
          doc=@db.last_insert_row_id
            @db.execute('INSERT INTO document_search(rowid,text) VALUES(?,?)',[doc,(index_text||raw).dup.force_encoding('UTF-8').scrub])
            doc
          end
        end
      end
    end
    def contents(doc)
      row=one('SELECT * FROM documents WHERE id=?',[doc]) or raise 'unknown document'
      bytes=row['codec']=='zlib' ? Zlib::Inflate.inflate(row['data']) : row['data']
      raise 'corrupt document' unless Digest::SHA256.hexdigest(bytes)==row['sha256']
      bytes
    end
    def metric(id,name,value,unit:,entity:'job',dimensions:{},document:nil,at:GPUQ.now_ns,description:nil)
      return if value.nil? || (value.is_a?(Float) && !value.finite?)
      @mutex.synchronize do
        group=name.start_with?('gpu.') ? 'gpu' : name.start_with?('process.','host.') ? 'process' : name.start_with?('phase.') ? 'trace' : name.start_with?('counter.') ? 'profiler' : 'timing'
        @db.execute('INSERT OR IGNORE INTO metric_definitions(name,unit,description,collector) VALUES(?,?,?,?)',[name,unit,description||name,group])
        expected=@db.get_first_value('SELECT unit FROM metric_definitions WHERE name=?',[name])
        raise "unit conflict for #{name}: #{expected} vs #{unit}" unless expected==unit
        int=value.is_a?(Integer) && value.between?(-(2**63),2**63-1) ? value : nil
        num=value.is_a?(Numeric) ? value.to_f : nil
        @db.execute('INSERT INTO samples(job_id,document_id,metric,observed_ns,entity,dimensions_json,value_json,value_number,value_integer) VALUES(?,?,?,?,?,?,?,?,?)',
          [id,document,name,at,entity,GPUQ.json(dimensions),GPUQ.json(value),num,int])
      end
    end
    def records(id,doc,rows)
      @mutex.synchronize do
        batch do
          rows.each { |n,kind,dims,data,valid| @db.execute('INSERT OR IGNORE INTO records(job_id,document_id,row_number,kind,dimensions_json,data_json,valid) VALUES(?,?,?,?,?,?,?)',[id,doc,n,kind,GPUQ.json(dims),GPUQ.json(data),valid ? 1 : 0]) }
        end
      end
    end
    def readonly(sql)
      raise 'query accepts SELECT/WITH/EXPLAIN only' unless sql.lstrip.match?(/\A(?:SELECT|WITH|EXPLAIN)\b/i)
      reader=SQLite3::Database.new(@path,flags:SQLite3::Constants::Open::READONLY)
      reader.results_as_hash=true;reader.busy_timeout=15_000
      reader.execute('PRAGMA query_only=ON')
      reader.execute(sql)
    ensure
      reader&.close
    end
    def reconcile
      host=Socket.gethostname
      boot=File.read('/proc/sys/kernel/random/boot_id').strip rescue nil
      execute("SELECT * FROM jobs WHERE host=? AND origin='live' AND status IN ('queued','running','collecting')",[host]).each do |job|
        next if job['boot_id']==boot && GPUQ.process_start(job['pid'])==job['process_start']
        update_job(job['id'],status:'lost',ended_ns:GPUQ.now_ns)
        document(job['id'],'lifecycle','reconcile','Owner process disappeared; completion and final resource usage are unknown.')
      end
    end
  end
end
