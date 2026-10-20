#!/usr/bin/env ruby
require 'tmpdir'
require 'open3'
require 'timeout'
require_relative 'store'

checks=[]
check=lambda { |name,condition| raise "FAILED: #{name}" unless condition;checks << name }
cli=File.join(__dir__,'gpuq.rb')
Dir.mktmpdir('gpuq-integration-') do |dir|
  path=File.join(dir,'history.sqlite3')
  env={'GPUQ_DB'=>path,'GPUQ_LOCK'=>File.join(dir,'lock'),'GPUQ_QUEUE'=>File.join(dir,'queue'),'GPUQ_JOB_ID'=>nil,'GPUQ_LOCK_FD'=>nil}
  call=lambda { |*args| Open3.capture3(env,RbConfig.ruby,cli,*args) }
  db=GPUQ::Store.new(path)
  latest=lambda { db.one("SELECT * FROM jobs WHERE origin='live' ORDER BY submitted_ns DESC LIMIT 1") }
  out,err,status=call.call('run','--exclude','all','--',RbConfig.ruby,'-e','STDOUT.binmode.write("alpha\x00\xff\n"); STDERR.write("omega"); exit 7')
  job=latest.call
  check.call('quiet run and exact child exit',out.empty? && err.empty? && status.exitstatus==7 && job['exit_code']==7)
  stdout=db.one("SELECT id FROM documents WHERE job_id=? AND kind='stdout'",[job['id']])
  stderr=db.one("SELECT id FROM documents WHERE job_id=? AND kind='stderr'",[job['id']])
  check.call('lossless binary stdout',db.contents(stdout['id'])=="alpha\x00\xff\n".b)
  check.call('unterminated stderr',db.contents(stderr['id'])=='omega')
  check.call('all exclusions persisted',db.execute("SELECT * FROM collectors WHERE job_id=? AND status='excluded'",[job['id']]).length==GPUQ::GROUPS.length)
  out,err,status=Open3.capture3(env,RbConfig.ruby,cli,'run','--exclude','all','--',RbConfig.ruby,'-e','STDOUT.write(STDIN.read.reverse)',stdin_data:'fixture input')
  input_job=latest.call
  input_doc=db.one("SELECT id FROM documents WHERE job_id=? AND kind='stdin'",[input_job['id']])
  check.call('stdin captured without terminal passthrough',status.success? && out.empty? && err.empty? && input_doc && db.contents(input_doc['id'])=='fixture input')
  out,err,status=call.call('rerun',input_job['id'])
  replay_input=latest.call
  output_doc=db.one("SELECT id FROM documents WHERE job_id=? AND kind='stdout'",[replay_input['id']])
  check.call('rerun restores recorded stdin',status.success? && output_doc && db.contents(output_doc['id'])=='tupni erutxif')

  payload='require "json"; puts({gpuq_metric:"test.exact_integer",unit:"count",value:9007199254740993}.to_json); File.write(File.join(ENV.fetch("GPUQ_RUN_DIR"),"results.json"),JSON.generate(engine_decode_tps:40.5,decode_tps:39.5)); sleep 0.12'
  out,err,status=call.call('run','--exclude','gpu,engine,trace,profiler,provenance','--',RbConfig.ruby,'-e',payload)
  job=latest.call
  check.call('measured job succeeded quietly',status.success? && out.empty? && err.empty?)
  sample=db.one("SELECT * FROM samples WHERE job_id=? AND metric='test.exact_integer'",[job['id']])
  check.call('integer precision retained',sample && sample['value_integer']==9007199254740993 && sample['value_json']=='9007199254740993')
  check.call('engine and client rates distinguished',db.execute("SELECT metric FROM samples WHERE job_id=? AND metric LIKE 'decode.%tokens_s'",[job['id']]).map { |r| r['metric'] }.sort==%w[decode.client.tokens_s decode.engine.tokens_s])
  check.call('managed text moved into database',!File.exist?(File.join(dir,'staging',job['id'],'results.json')) && db.one("SELECT id FROM documents WHERE job_id=? AND kind='artifact'",[job['id']]))
  check.call('waited resource usage collected',db.one("SELECT value_json FROM samples WHERE job_id=? AND metric='process.max_rss_kib'",[job['id']]))

  a=Process.spawn(env,RbConfig.ruby,cli,'run','--name','fifo-a','--exclude','all','--',RbConfig.ruby,'-e','sleep 0.5',out:File::NULL,err:File::NULL)
  Timeout.timeout(5) { sleep 0.01 until db.one("SELECT id FROM jobs WHERE label='fifo-a' AND status='running'") }
  b=Process.spawn(env,RbConfig.ruby,cli,'run','--name','fifo-b','--exclude','all','--',RbConfig.ruby,'-e','raise "cancelled job executed"',out:File::NULL,err:File::NULL)
  Timeout.timeout(5) { sleep 0.01 until db.one("SELECT id FROM jobs WHERE label='fifo-b' AND status='queued'") }
  Process.kill('TERM',b);_,bs=Process.waitpid2(b)
  c=Process.spawn(env,RbConfig.ruby,cli,'run','--name','fifo-c','--exclude','all','--',RbConfig.ruby,'-e','print "third"',out:File::NULL,err:File::NULL)
  Process.wait(a);Process.wait(c)
  aa=db.one("SELECT * FROM jobs WHERE label='fifo-a'");bb=db.one("SELECT * FROM jobs WHERE label='fifo-b'");cc=db.one("SELECT * FROM jobs WHERE label='fifo-c'")
  check.call('queued cancellation does not execute',bs.exitstatus==143 && bb['status']=='cancelled' && bb['started_ns'].nil?)
  check.call('FIFO and mutual exclusion',aa['ended_ns']<=cc['started_ns'] && cc['status']=='succeeded')

  out,err,status=call.call('run','--name','nested-parent','--exclude','all','--',RbConfig.ruby,cli,'run','--name','nested-child','--exclude','all','--',RbConfig.ruby,'-e','print "nested"')
  parent=db.one("SELECT * FROM jobs WHERE label='nested-parent'");child=db.one("SELECT * FROM jobs WHERE label='nested-child'")
  check.call('nested queue inherits lease without deadlock',status.success? && out.empty? && err.empty? && child && child['parent_id']==parent['id'] && child['status']=='succeeded')

  ns=9007199254741000
  trace="trace_begin request=3 die=1\nposition,phase,layer,opcode,start_ns,wait_start_ns,wait_end_ns,body_end_ns,end_ns,ple_wait_start_ns,ple_wait_end_ns\n2,0,-1,embrow,#{ns},#{ns+10},#{ns+20},#{ns+30},#{ns+40},0,0\ntrace_end\n"
  program='File.write(File.join(ENV.fetch("GPUQ_RUN_DIR"),"trace.csv"),ARGV.fetch(0))'
  out,err,status=call.call('run','--exclude','gpu,engine,profiler,provenance','--',RbConfig.ruby,'-e',program,trace)
  traced=latest.call
  check.call('new-job trace capture quiet',status.success? && out.empty? && err.empty?)
  phase=db.one('SELECT * FROM phase_intervals WHERE job_id=?',[traced['id']])
  check.call('phase arithmetic beyond binary64 exact integer',phase && phase['dispatch_ns']==10 && phase['body_ns']==10 && phase['barrier_ns']==10)

  out,err,status=call.call('search','omega')
  check.call('database full-text lookup',status.success? && JSON.parse(out).any? { |r| r['kind']=='stderr' })
  out,err,status=call.call('query','WITH x AS (SELECT 1) DELETE FROM jobs')
  check.call('query cannot mutate database',!status.success? && db.one('SELECT count(*) AS n FROM jobs')['n']>0)
  out,err,status=call.call('run','--exclude','all','--','/definitely-missing-gpuq-test-command')
  check.call('missing executable returns 127 quietly',status.exitstatus==127 && out.empty? && err.empty?)
  old=aa['id'];out,err,status=call.call('rerun',old)
  check.call('saved command reusable',status.success? && latest.call['replay_of']==old)

  lost=db.create_job(argv:['fixture'],cwd:dir,status:'running')
  db.execute('UPDATE jobs SET process_start=? WHERE id=?',['not-the-current-process',lost]);db.reconcile
  check.call('lost process is not called success',db.one('SELECT status FROM jobs WHERE id=?',[lost])['status']=='lost')
  db.close
end
puts GPUQ.json(status:'passed',checks:checks,count:checks.length)
