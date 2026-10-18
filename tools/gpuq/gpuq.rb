#!/usr/bin/env ruby
require 'rubygems'
vendor=ENV.fetch('GPUQ_GEM_PATH','/usr/local/lib/gpuq/gems')
Gem.use_paths(Gem.dir,[vendor,*Gem.path]) if File.directory?(vendor)
require 'optparse'
require 'base64'
require_relative 'runner'

module GPUQ
  class CLI
    HELP={
      run:'gpuq run [--name NAME] [--exclude gpu,trace,...] [--capture DIR] [--interval SECONDS] -- COMMAND ARGS...',
      jobs:'gpuq jobs [LIMIT] / gpuq latest / gpuq ls',
      logs:'gpuq logs JOB_ID [LIMIT] (JSON, with exact bytes retained in documents)',
      document:'gpuq document DOCUMENT_ID [--raw] (explicit database read)',
      metrics:'gpuq metrics JOB_ID / gpuq compare METRIC [JOB_ID...]',
      search:'gpuq search WORDS (FTS5 query)',
      query:'gpuq query SELECT_SQL (read-only SQLite connection)',
      rerun:'gpuq rerun JOB_ID [--exclude GROUPS] (reuse saved argv; current source identity is recorded)',
      import:'gpuq import PATH... (idempotent historical artifact import; never invents missing run timestamps)',
      emit:'gpuq emit METRIC UNIT JSON_VALUE (inside a job; writes directly to its database)',
      put:'gpuq put SOURCE_LABEL (inside a job; stdin goes directly to its database)',
      schema:'gpuq schema / gpuq doctor',
      storage:'--db PATH or GPUQ_DB; default ~/.local/state/gpuq/history.sqlite3',
      exclusions:GROUPS,
      mandatory:'Job identity, exclusions, stdin/stdout/stderr, recipe scripts, text artifacts and collection errors cannot be excluded. run never streams command output.'
    }.freeze
    def initialize(argv)
      @args=argv.dup;@original=argv.dup
      @path=ENV.fetch('GPUQ_DB',GPUQ.default_db)
      boundary=@args.index('--')||@args.length
      if (i=@args.take(boundary).index('--db'))
        @path=@args.fetch(i+1);@args.slice!(i,2)
      end
      @command=@args.shift||'help'
    end
    def emit(value)
      convert=lambda do |v|
        case v
        when Integer then v.abs>9_007_199_254_740_991 ? v.to_s : v
        when Hash then v.to_h { |k,x| [k,convert.call(x)] }
        when Array then v.map { |x| convert.call(x) }
        else v
        end
      end
      puts JSON.pretty_generate(convert.call(GPUQ.clean(value)))
    end
    def run_options(args,initial={})
      opts={exclude:[],capture:[],interval:1.0}.merge(initial)
      parser=OptionParser.new do |p|
        p.on('--exclude GROUPS') { |s| opts[:exclude]=s=='all' ? GROUPS.dup : s.split(',').uniq }
        p.on('--name NAME') { |s| opts[:name]=s }
        p.on('--capture PATH') { |s| opts[:capture] << s }
        p.on('--interval SECONDS',Float) { |s| raise 'interval must be at least 0.05 seconds' unless s.finite? && s>=0.05;opts[:interval]=s }
        p.on('--timeout SECONDS',Float) { |s| raise 'timeout must be positive' unless s.finite? && s>0;opts[:timeout]=s }
        p.on('--engine-socket PATH') { |s| opts[:socket]=s }
      end
      parser.order!(args)
      opts
    end
    def job_id(value)
      return value unless value=='latest'
      row=@db.one("SELECT id FROM jobs WHERE origin='live' ORDER BY submitted_ns DESC LIMIT 1")
      raise 'no live jobs recorded' unless row
      row['id']
    end
    def main
      @db=Store.new(@path)
      case @command
      when 'run'
        options=run_options(@args)
        Runner.new(@db,options).run(@args)
      when 'rerun'
        old=@db.one('SELECT * FROM jobs WHERE id=?',[job_id(@args.shift)]) or raise 'unknown job'
        options=run_options(@args,exclude:JSON.parse(old['exclusions_json']),name:old['label'])
        input=@db.execute("SELECT id FROM documents WHERE job_id=? AND kind='stdin' ORDER BY ordinal",[old['id']]).map { |r| @db.contents(r['id']) }.join
        options[:input]=StringIO.new(input) unless input.empty?
        raise 'unexpected rerun arguments' unless @args.empty?
        Dir.chdir(old['cwd']) { Runner.new(@db,options).run(JSON.parse(old['argv_json']),replay_of:old['id']) }
      when 'jobs','latest','ls'
        @db.reconcile
        limit=@command=='latest' ? 1 : Integer(@args.shift||100)
        raise 'limit must be positive' unless limit>0
        where=@command=='ls' ? "WHERE status IN ('queued','running','collecting')" : @command=='latest' ? "WHERE origin='live'" : ''
        emit(@db.execute("SELECT * FROM jobs #{where} ORDER BY submitted_ns DESC LIMIT ?",[limit]));0
      when 'logs'
        job=job_id(@args.fetch(0));limit=Integer(@args.fetch(1,'100'))
        rows=@db.execute("SELECT id,job_id,kind,source,ordinal,observed_ns,bytes,sha256 FROM documents WHERE job_id=? AND kind IN ('stdout','stderr','runner_error','capture_error','collector_error') ORDER BY id DESC LIMIT ?",[job,limit]).reverse
        emit(rows.map { |r| r.merge('text'=>@db.contents(r['id']).dup.force_encoding('UTF-8').scrub) });0
      when 'document'
        id=Integer(@args.fetch(0));data=@db.contents(id)
        if @args[1]=='--raw'
          STDOUT.binmode;STDOUT.write(data)
        else
          emit(id:id,text:data.dup.force_encoding('UTF-8').scrub,bytes:data.bytesize,base64:Base64.strict_encode64(data))
        end
        0
      when 'metrics'
        emit(@db.execute('SELECT * FROM metric_history WHERE job_id=? ORDER BY metric,entity,sample_id',[job_id(@args.fetch(0))]));0
      when 'compare'
        metric=@args.shift or raise 'metric required'
        if metric.start_with?('phase.')
          column=metric.delete_prefix('phase.')
          raise 'unknown phase interval' unless %w[dispatch_ns exchange_wait_ns body_ns barrier_ns ple_wait_ns].include?(column)
          where=@args.empty? ? '' : "AND p.job_id IN (#{(['?']*@args.length).join(',')})"
          emit(@db.execute("SELECT p.job_id,j.label,j.status,p.dimensions_json,p.data_json,p.#{column} AS value_ns FROM phase_intervals p JOIN jobs j ON j.id=p.job_id WHERE 1=1 #{where} ORDER BY j.submitted_ns,p.row_number",@args))
        else
          where=@args.empty? ? '' : "AND job_id IN (#{(['?']*@args.length).join(',')})"
          emit(@db.execute("SELECT * FROM metric_history WHERE metric=? #{where} ORDER BY submitted_ns,sample_id",[metric,*@args]))
        end
        0
      when 'search'
        advanced=@args.first=='--fts';@args.shift if advanced
        term=@args.join(' ');raise 'search text required' if term.empty?
        term=term.split.map { |s| '"'+s.gsub('"','""')+'"' }.join(' AND ') unless advanced
        emit(@db.execute('SELECT d.id,d.job_id,d.kind,d.source,d.sha256,d.bytes,j.label FROM document_search f JOIN documents d ON d.id=f.rowid JOIN jobs j ON j.id=d.job_id WHERE document_search MATCH ? ORDER BY d.observed_ns DESC LIMIT 100',[term]));0
      when 'query'
        emit(@db.readonly(@args.join(' ')));0
      when 'schema'
        emit(tables:@db.execute("SELECT name,sql FROM sqlite_master WHERE type IN ('table','view') AND name NOT LIKE 'document_search_%' ORDER BY name"),metrics:@db.execute('SELECT * FROM metric_definitions ORDER BY name'),collectors:GROUPS);0
      when 'doctor'
        emit(database:@db.path,sqlite:SQLite3::SQLITE_VERSION,integrity:@db.execute('PRAGMA quick_check'),queue:ENV.fetch('GPUQ_QUEUE','/tmp/gpuq'),lock:ENV.fetch('GPUQ_LOCK','/tmp/flash-sass-engine-validation.lock'));0
      when 'import'
        raise 'import path required' if @args.empty?
        groups={}
        @args.each do |input|
          root=File.expand_path(input)
          raise "missing input #{root}" unless File.exist?(root)
          Find.find(root) do |path|
            if File.directory?(path)
              Find.prune if %w[.git node_modules].include?(File.basename(path)) || File.symlink?(path)
            elsif File.file?(path) && !File.symlink?(path) && path.match?(Importer::TEXT_EXT)
              (groups[File.dirname(path)]||=[]) << path
            end
          end
        end
        groups.sort.each do |dir,paths|
          key="import:#{Socket.gethostname}:#{dir}"
          row=@db.one('SELECT id FROM jobs WHERE source_key=?',[key])
          id=row ? row['id'] : @db.create_job(argv:[],cwd:dir,label:File.basename(dir),origin:'imported',source_key:key,status:'imported')
          importer=Importer.new(@db,id)
          paths.sort.each { |path| importer.file(path,kind:'historical_artifact') }
          @db.collector(id,'artifacts',importer.count['errors']>0 ? 'partial' : 'complete',nil,config:importer.count)
        end
        0 # imports are writes; inspect them through jobs/query, not printed logs
      when 'emit'
        job=ENV.fetch('GPUQ_JOB_ID');name,unit,value=@args
        raise 'metric, unit and JSON value required' unless value
        @db.metric(job,name,JSON.parse(value),unit:unit);0
      when 'put'
        job=ENV.fetch('GPUQ_JOB_ID');source=@args.fetch(0)
        id=@db.document(job,'supplied',source,STDIN.binmode.read,ordinal:GPUQ.now_ns)
        Importer.new(@db,job,excluded:ENV.fetch('GPUQ_EXCLUDE','').split(',')).parse(id,@db.contents(id),source);0
      when 'help','--help','-h'
        emit(HELP.merge(database:@db.path));0
      else
        raise "unknown command #{@command}"
      end
    rescue StandardError => e
      begin
        if @db
          id=@db.create_job(argv:@original,cwd:Dir.pwd,label:'gpuq command rejected',origin:'control',status:'rejected')
          @db.document(id,'runner_error',@command,"#{e.class}: #{e.message}")
          @db.update_job(id,ended_ns:GPUQ.now_ns,exit_code:2)
        end
      rescue StandardError
        # Storage failure must never fall back to executing an unrecorded job.
      end
      emit(error:e.message,database:@path) unless %w[run rerun import emit put].include?(@command)
      2
    ensure
      @db&.close
    end
  end
end

exit GPUQ::CLI.new(ARGV).main if $PROGRAM_NAME==__FILE__
