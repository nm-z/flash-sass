require 'csv'
require 'stringio'
require 'time'
require_relative 'store'

module GPUQ
  class Importer
    ALIASES={
      'engine_decode_tps'=>['decode.engine.tokens_s','tokens/s'],
      'decode_tps'=>['decode.client.tokens_s','tokens/s'],
      'predicted_per_second'=>['decode.reported.tokens_s','tokens/s'],
      'prompt_per_second'=>['prefill.reported.tokens_s','tokens/s'],
      'engine_prefill_ms'=>['prefill.engine.first_ms','ms'],
      'engine_run_ms'=>['request.engine.run_ms','ms'],
      'first_observed_ms'=>['request.client.first_ms','ms'],
      'wall_ms'=>['request.client.wall_ms','ms'],
      'prompt_ms'=>['prefill.reported.ms','ms'],
      'decode_ms'=>['decode.reported.ms','ms'],
      'decode_intervals'=>['decode.intervals','count'],
      'prompt_n'=>['request.prompt_tokens','count'],
      'predicted_n'=>['request.generated_tokens','count'],
      'cache_n'=>['request.cached_tokens','count']
    }.freeze
    TEXT_EXT=/\.(?:jsonl?|csv|tsv|log|txt|text|xml|protocol|tap|md|sass|ptx|rb|pl|sh|c|h|sql|yaml|yml|toml|diff|patch)(?:\.gz)?\z/i
    SOURCE_EXT=/\.(?:c|h|rb|pl|sh|bash|lua|zig|rs|go|sass|ptx|sql)\z/i
    attr_reader :count
    def initialize(store,job,excluded:[])
      @db,@job,@excluded=store,job,excluded
      @count=Hash.new(0)
      @seen={}
      @row_sequences=Hash.new(0)
      info=@db.one('SELECT origin,started_ns FROM jobs WHERE id=?',[@job])
      @historical=info['origin']!='live';@at=info['started_ns']
    end
    def enabled?(group) = !@excluded.include?(group)
    def file(path,kind:'artifact',remove:false)
      return false unless File.file?(path) && !File.symlink?(path)
      stamp=File.stat(path)
      key=[File.expand_path(path),stamp.size,stamp.mtime.to_r,stamp.ctime.to_r]
      return false if @seen[key]
      return false if path.match?(/\.(?:gguf|cubin|so|o|a|sqlite3?|db|bin|png|jpe?g|zip|tar)\z/i)
      probe=File.binread(path,8192)
      return false unless path.match?(TEXT_EXT) || (!probe.include?("\0") && probe.dup.force_encoding('UTF-8').valid_encoding?)
      bytes=File.binread(path)
      after=File.stat(path)
      raise "artifact changed while reading: #{path}" unless [stamp.ino,stamp.size,stamp.mtime,stamp.ctime]==[after.ino,after.size,after.mtime,after.ctime]
      decoded=path.end_with?('.gz') ? Zlib::GzipReader.new(StringIO.new(bytes)).read : bytes
      prior=@db.one('SELECT ordinal,sha256 FROM documents WHERE job_id=? AND kind=? AND source=? ORDER BY ordinal DESC LIMIT 1',[@job,kind,File.expand_path(path)])
      return false if prior && prior['sha256']==Digest::SHA256.hexdigest(bytes)
      doc=@db.document(@job,kind,File.expand_path(path),bytes,ordinal:prior ? prior['ordinal']+1 : 0,index_text:decoded)
      @count['files']+=1;@seen[key]=true
      parse(doc,decoded,path)
      # Delete only files in the run-owned staging directory, after the durable insert.
      if remove && !path.match?(SOURCE_EXT) && !%w[Makefile Gemfile AGENTS.md].include?(File.basename(path))
        final=File.stat(path)
        File.unlink(path) if [after.ino,after.size,after.mtime,after.ctime]==[final.ino,final.size,final.mtime,final.ctime]
      end
      true
    rescue JSON::ParserError,CSV::MalformedCSVError,Zlib::Error => e
      @db.document(@job,'parse_error',path,e.message,ordinal:@count['errors']);@count['errors']+=1
      false
    end
    def parse(doc,bytes,source)
      text=bytes.dup.force_encoding('UTF-8')
      return unless text.valid_encoding?
      stripped=text.lstrip
      if stripped.start_with?('{','[')
        begin
          object=JSON.parse(text)
          @db.batch { json(doc,object,{'source'=>source}) }
          return
        rescue JSON::ParserError
          # Protocols and JSONL have multiple records, not one JSON value.
        end
      end
      if source.sub(/\.gz\z/,'').end_with?('.csv') || text.include?('position,phase,') || text.include?('position,observed_ns,')
        csv(doc,text,source)
      else
        text.each_line.with_index(1) { |line,n| line(doc,line,{'source'=>source,'line'=>n}) }
      end
    end
    def line(doc,text,dims)
      s=text.strip
      if s.start_with?('{','[')
        @db.batch { json(doc,JSON.parse(s),dims) }
      elsif s.start_with?('done ') && enabled?('timing')
        v=s.split
        if v.size>=8 && v[1..3].all? { |x| x.match?(/\A\d+\z/) }
          generated=v[1].to_i;run=Float(v[4]);first=Float(v[5]);intervals=[generated-1,0].max
          {'request.generated_tokens'=>[generated,'count'],'request.prompt_tokens'=>[v[3].to_i,'count'],
           'request.engine.run_ms'=>[run,'ms'],'prefill.engine.first_ms'=>[first,'ms'],
           'decode.intervals'=>[intervals,'count']}.each { |name,(value,unit)| metric(doc,name,value,unit,dims) }
          metric(doc,'decode.engine.tokens_s',intervals*1000.0/(run-first),'tokens/s',dims) if intervals>0 && run>first
        end
      end
    rescue JSON::ParserError,ArgumentError
      # Raw bytes remain in documents, including ordinary non-metric text.
    end
    def metric(doc,name,value,unit,dims)
      @db.metric(@job,name,value,unit:unit,dimensions:dims,document:doc,at:@historical ? @at : GPUQ.now_ns)
      @count['metrics']+=1
    end
    def json(doc,obj,dims,path='',depth=0)
      return if depth>32
      case obj
      when Hash
        if @historical && obj['started_utc'].is_a?(String)
          @at=(Time.iso8601(obj['started_utc']).to_r*1_000_000_000).to_i rescue nil
          @db.update_job(@job,started_ns:@at) if @at
        end
        if obj.key?('gpuq_metric') && obj.key?('value')
          return unless enabled?(obj.fetch('collector','timing'))
          metric(doc,obj.fetch('gpuq_metric'),obj['value'],obj.fetch('unit'),dims.merge(obj.fetch('dimensions',{})))
          return
        end
        row_dims=dims.merge(obj.slice('case','label','request','request_id','repetition','warmup','die','position','phase','status','placement'))
        obj.each do |key,value|
          field=path.empty? ? key : "#{path}.#{key}"
          if value.is_a?(Numeric) && enabled?('timing') && ALIASES[key]
            name,unit=ALIASES[key];metric(doc,name,value,unit,row_dims.merge('field'=>field))
          elsif key=='text' && value.is_a?(String)
            if value.include?('position,phase,') || value.include?('position,observed_ns,')
              csv(doc,value,"#{row_dims['source']}:#{field}")
            else
              line(doc,value,row_dims)
            end
          elsif value.is_a?(Numeric) && enabled?('profiler') && field.match?(/profiler|counter/)
            metric(doc,"counter.#{field}",value,'reported',row_dims)
            @count['profiler']+=1
          elsif value.is_a?(Hash) || value.is_a?(Array)
            json(doc,value,row_dims,field,depth+1)
          end
        end
        @db.records(@job,doc,[[next_record(doc),'json',row_dims,obj,true]]) if path.empty?
      when Array
        if path.match?(/(?:token_arrival_ms|client_observed_ns|intervals_ns)\z/) && enabled?('timing') && obj.all? { |v| v.is_a?(Numeric) }
          intervals=path.end_with?('intervals_ns') ? obj : obj.each_cons(2).map { |a,b| (b-a)*(path.end_with?('_ms') ? 1_000_000 : 1) }
          intervals.each_with_index { |v,i| metric(doc,'decode.client.interval_ns',v,'ns',dims.merge('interval'=>i,'field'=>path)) }
        else
          obj.each_with_index { |v,i| json(doc,v,dims.merge('record_index'=>i),path,depth+1) if v.is_a?(Hash) || v.is_a?(Array) }
        end
      end
    end
    def next_record(doc)
      @count['records']+=1
      @row_sequences[doc]+=1
    end
    def csv(doc,text,source)
      if text.include?('trace_begin') || text.include?('hosttrace_begin')
        meta={};headers=nil;rows=[]
        text.each_line.with_index(1) do |line,n|
          if line.start_with?('trace_begin','hosttrace_begin')
            meta=line.scan(/(\w+)=([^\s]+)/).to_h;headers=nil
          elsif line.start_with?('trace_end','hosttrace_end')
            headers=nil
          elsif line.start_with?('position,')
            headers=CSV.parse_line(line)
          elsif headers && !line.strip.empty?
            fields=CSV.parse_line(line)
            append_csv_row(doc,headers.zip(fields).to_h,meta,source,n,rows)
            if rows.length>=500
              @db.records(@job,doc,rows);rows=[]
            end
          end
        end
        @db.records(@job,doc,rows) unless rows.empty?
        return
      end
      io=StringIO.new(text);meta={};header=nil
      until io.eof?
        offset=io.pos;l=io.gets
        if l.include?(',') && !l.start_with?('#','trace_begin','hosttrace_begin')
          header=offset;break
        end
        meta.merge!(l.scan(/(\w+)=([^\s]+)/).to_h)
      end
      return unless header
      io.pos=header
      rows=[]
      CSV.new(io,headers:true).each_with_index do |row,i|
        break if row.fields.first.to_s.start_with?('trace_end','hosttrace_end')
        append_csv_row(doc,row.to_h,meta,source,i+1,rows)
        if rows.length>=500
          @db.records(@job,doc,rows);rows=[]
        end
      end
      @db.records(@job,doc,rows) unless rows.empty?
    end
    def append_csv_row(doc,data,meta,source,n,rows)
      kind=data.key?('phase') && data.key?('position') ? 'phase' : data.key?('observed_ns') ? 'host_trace' : 'csv'
      return if %w[phase host_trace].include?(kind) && !enabled?('trace')
      dims=meta.merge('source'=>source,'source_row'=>n).merge(data.slice('position','phase','layer','opcode','die','cta','uuid'))
      valid=true
      if kind=='phase'
        keys=%w[start_ns wait_start_ns wait_end_ns body_end_ns end_ns]
        values=keys.map { |k| Integer(data.fetch(k,''),10) rescue nil }
        valid=values.all? { |x| x && x>0 && x<2**63 } && values.each_cons(2).all? { |a,b| a<=b }
      end
      rows << [next_record(doc),kind,dims,data,valid]
      @count[kind]+=1
    end
  end
end
