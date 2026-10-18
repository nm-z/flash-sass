#!/usr/bin/env ruby
# Reconcile schema-2 sampled intervals without assuming shared GPU epochs.
require 'csv'
require 'json'
require 'optparse'
require 'zlib'
require_relative 'trace_lifecycle_check'

options = {quantum: 1024}
OptionParser.new do |p|
  p.banner = 'Usage: trace_reconcile.rb --dies N [--position N] [--out FILE] TRACE... HOSTTRACE'
  p.on('--dies N',Integer) { |v| options[:dies] = v }
  p.on('--position N',Integer) { |v| options[:position] = v }
  p.on('--quantum-ns N',Integer) { |v| options[:quantum] = v }
  p.on('--allow-incomplete') { options[:incomplete] = true }
  p.on('--out FILE') { |v| options[:out] = v }
end.parse!

def need(ok, message)
  raise ArgumentError, message unless ok
end
def integer(s)
  need(s && s.match?(/\A-?\d+\z/), "invalid integer #{s.inspect}")
  Integer(s,10)
end
def numeric(h)
  h.transform_values { |s| s.match?(/\A-?\d+\z/) ? Integer(s,10) : s }
end
def union_ns(intervals)
  finish = nil; total = 0
  intervals.sort.each do |a,b|
    if finish.nil? || a > finish
      total += b-a; finish = b
    elsif b > finish
      total += b-finish; finish = b
    end
  end
  total
end
def node_info(n)
  %i[die cta position phase event timestamp_ns generation opcode].zip(n).to_h
end

begin
  need(options[:dies] && options[:dies].positive? && !ARGV.empty?, '--dies and input files required')
  need(options[:quantum] >= 0, 'negative quantum')
  need(!options[:position] || options[:position] >= 0, 'negative position')
  quantum = options[:quantum]; gpu = {}; hosts = []
  ARGV.each do |path|
    text = path.end_with?('.gz') ? Zlib::GzipReader.open(path, &:read) : File.read(path)
    lines = text.lines
    first = lines.shift&.strip
    need(first && first.start_with?('trace_begin ','hosttrace_begin '), "#{path}: full schema-2 response required")
    kind = first.start_with?('hosttrace_') ? 'hosttrace' : 'trace'
    meta = TraceChecks.fields(first)
    need(meta['schema'] == '2', "#{path}: schema 2 required")
    need(lines.pop&.strip == "#{kind}_end", "#{path}: missing terminator")
    table = CSV.parse(lines.join,headers: true)
    need(table.headers.uniq == table.headers, "#{path}: duplicate columns")
    rows = table.map do |r|
      r.to_h.transform_values.with_index { |v,i| table.headers[i] == 'opcode' ? v : integer(v) }
    end
    need(!rows.empty?, "#{path}: empty trace")
    data = {path: path, meta: meta, rows: rows, original_rows: rows.length}
    if kind == 'trace'
      die = integer(meta.fetch('die')); need(!gpu.key?(die), 'duplicate die'); gpu[die] = data
    else
      hosts << data
    end
  end
  dies = gpu.keys.sort
  need(dies == (0...options[:dies]).to_a, 'missing or unexpected participating dies')
  need(hosts.length == 1, 'exactly one matching host trace required')
  reference = gpu[dies.first][:meta]
  (gpu.values + hosts).each do |data|
    meta = data[:meta]
    %w[request first count status aborted ended completed requested].each do |k|
      need(meta.fetch(k) == reference.fetch(k), "inconsistent #{k}")
    end
    unless options[:incomplete]
      need(meta['status'] == 'completed' && meta['aborted'] == '0' && meta['completed'] == meta['requested'],
           'stopped/aborted/failed request is not a complete benchmark')
    end
  end

  # Validate complete original geometry before selecting a token.
  original = []
  positions = nil; ctas_by_die = {}
  gpu.each do |die,data|
    m = data[:meta]; ctas = integer(m.fetch('ctas')); nph = integer(m.fetch('nph'))
    need([1,16].include?(ctas) && nph.positive?, 'invalid geometry')
    ctas_by_die[die] = ctas
    ps = (integer(m.fetch('first'))...(integer(m.fetch('first')) + integer(m.fetch('count')))).to_a
    positions ||= ps; need(positions == ps, 'different position windows')
    need(data[:rows].length == ps.length * nph * ctas, 'wrong original record count')
    data[:rows].each_with_index do |r,i|
      need(r['position'] == ps.first + i/(nph*ctas) && r['phase'] == i/ctas%nph && r['cta'] == i%ctas,
           "die #{die}: original row geometry mismatch")
    end
    original << {die: die,path: data[:path],metadata: numeric(m),rows: data[:original_rows]}
    if options[:position]
      need(ps.include?(options[:position]), 'selected position unavailable')
      data[:rows] = data[:rows].select { |r| r['position'] == options[:position] }
    end
  end
  need(ctas_by_die.values.uniq.length == 1, 'mixed CTA layouts')
  original << {kind: 'host',path: hosts[0][:path],metadata: numeric(hosts[0][:meta]),rows: hosts[0][:original_rows]}
  need(hosts[0][:rows].map { |r| r['position'] } == positions, 'host position coverage differs')
  if options[:position]
    positions = [options[:position]]
    hosts[0][:rows] = hosts[0][:rows].select { |r| r['position'] == options[:position] }
  end

  nodes = []; edges = []; indegree = []
  add_node = lambda do |info|
    id = nodes.length; nodes << info; edges << []; indegree << 0; id
  end
  add_edge = lambda do |a,b,w,kind,buckets=nil|
    edges[a] << [b,w,kind,buckets]; indegree[b] += 1
  end
  starts = []; finishes = []
  sends = Hash.new { |h,g| h[g] = {} }
  waits = Hash.new { |h,g| h[g] = Hash.new { |x,p| x[p] = [] } }
  grids = Hash.new { |h,k| h[k] = {body: {},finish: {}} }
  local = []; envelopes = []; send_brackets = []; opcode_totals = Hash.new { |h,k| h[k] = Hash.new(0) }

  gpu.each do |die,data|
    nph = integer(data[:meta].fetch('nph'))
    data[:rows].group_by { |r| r['cta'] }.each do |cta,cta_rows|
      previous_node = nil; previous_time = nil; previous_publication = nil
      pending = Hash.new(0)
      cta_rows.group_by { |r| r['position'] }.sort.each do |position,rows|
        first = rows.first.fetch('start_ns'); total = Hash.new(0)
        total['token_transition'] = previous_publication ? first - previous_publication : 0
        need(total['token_transition'] >= 0, 'token starts before prior publication')
        if previous_node
          pending['token_transition'] += total['token_transition']
        else
          previous_node = add_node.call([die,cta,position,0,'start',first,nil,nil])
          starts << previous_node; previous_time = first
        end
        last_end = nil
        rows.each do |r|
          phase = r.fetch('phase'); flags = r.fetch('flags'); generation = r.fetch('exchange_generation')
          a,ws,we,bs,be,z = r.values_at('start_ns','wait_start_ns','wait_end_ns','body_start_ns','body_end_ns','end_ns')
          need(a.positive? && [a,ws,we,bs,be,z].each_cons(2).all? { |x,y| x <= y }, 'nonmonotonic phase')
          gap = last_end ? a-last_end : 0; need(gap >= 0, 'overlapping local phases')
          total['phase_gap'] += gap; pending['phase_gap'] += gap
          p0,p1 = r.values_at('ple_wait_start_ns','ple_wait_end_ns')
          need((p0.zero? && p1.zero?) || (bs <= p0 && p0 <= p1 && p1 <= be), 'invalid PLE wait')
          s0,s1 = r.values_at('send_before_ns','send_after_ns')
          if flags & 1 != 0
            need(s0.positive? && be <= s0 && s0 <= s1 && s1 <= z, 'invalid send bracket')
            send_brackets << s1-s0
          else
            need(s0.zero? && s1.zero?, 'send bracket without send')
          end
          need(generation >= 0, 'negative exchange generation')
          segments = [[a,ws,'dispatch_setup'],[ws,we,flags & 2 != 0 ? 'exchange_wait' : 'dispatch_no_wait'],
                      [we,bs,'opcode_dispatch']]
          segments += p0.positive? ? [[bs,p0,'body'],[p0,p1,'ple_wait'],[p1,be,'body']] : [[bs,be,'body']]
          segments += s0.positive? ? [[be,s0,'barrier_before_send'],[s0,s1,'send_publication_bracket'],[s1,z,'barrier_after_send']] : [[be,z,'barrier_send']]
          events = Hash.new { |h,t| h[t] = [] }
          events[we] << ['wait_end',generation,0] if flags & 2 != 0
          if flags & 4 == 0
            events[be] << ['grid_body_end',nil,1]
            events[z] << ['grid_end',nil,3]
          end
          events[s0] << ['send_before',generation+1,2] if flags & 1 != 0
          segments.map { |s| s[1] }.uniq.each do |t|
            segments.select { |s| s[1] == t }.each do |x,y,bucket|
              pending[bucket] += y-x; total[bucket] += y-x
              opcode_totals[[die,r['opcode']]][bucket] += y-x
            end
            events[t].sort_by { |e| e[2] }.each do |event,g,_rank|
              id = add_node.call([die,cta,position,phase,event,t,g,r['opcode']])
              add_edge.call(previous_node,id,t-previous_time,'local',pending.dup)
              pending.clear; previous_node = id; previous_time = t
              case event
              when 'send_before'
                need(!sends[g].key?([die,cta]), 'duplicate outgoing generation')
                sends[g][[die,cta]] = id
              when 'wait_end' then waits[g][[die,cta]] << id
              when 'grid_body_end' then grids[[die,position,phase]][:body][cta] = id
              when 'grid_end' then grids[[die,position,phase]][:finish][cta] = id
              end
            end
          end
          need(phase == nph-1 || r['publication_ns'].zero?, 'nonfinal publication')
          last_end = z
        end
        pub = rows.last.fetch('publication_ns')
        need(pub.positive? && pub >= last_end, 'missing final publication')
        tail = pub-last_end; total['publication_tail'] += tail; pending['publication_tail'] += tail
        id = add_node.call([die,cta,position,nph-1,'publication',pub,nil,nil])
        add_edge.call(previous_node,id,pub-previous_time,'local',pending.dup)
        pending.clear; previous_node = id; previous_time = pub
        duration = pub-first; partition = total.values.sum-total['token_transition']
        period = previous_publication ? pub-previous_publication : nil
        residual = duration-partition; period_residual = period && period-total.values.sum
        need(residual.zero? && (!period_residual || period_residual.zero?), 'nonzero arithmetic residual')
        local << {die: die,cta: cta,position: position,phases: nph,duration_ns: duration,
                  interpublication_ns: period,buckets_ns: total,residual_ns: residual,period_residual_ns: period_residual}
        previous_publication = pub
      end
      finishes << previous_node
    end
    data[:rows].group_by { |r| r['position'] }.sort.each do |position,rows|
      intervals = rows.group_by { |r| r['phase'] }.values.map { |rs| [rs.map { |r| r['start_ns'] }.min,rs.map { |r| r['end_ns'] }.max] }
      first = intervals.map(&:first).min; last = intervals.map(&:last).max
      pub = rows.map { |r| r['publication_ns'] }.max; covered = union_ns(intervals)
      gap = last-first-covered; tail = pub-last
      need(tail >= 0, 'negative envelope tail')
      envelopes << {die: die,position: position,ctas: ctas_by_die[die],duration_ns: pub-first,
                    phase_union_ns: covered,phase_overlap_ns: intervals.sum { |x,y| y-x }-covered,
                    phase_gap_ns: gap,publication_tail_ns: tail,residual_ns: pub-first-covered-gap-tail}
    end
  end

  # Preserve observed start skew among CTAs sharing one timer domain. Each
  # die root remains independent of the other devices' unknown epochs.
  roots = []
  starts.group_by { |id| nodes[id][0] }.each do |die,ids|
    first = ids.map { |id| nodes[id][5] }.min
    root = add_node.call([die,nil,positions.first,0,'die_start',first,nil,nil])
    roots << root
    ids.each do |id|
      skew = nodes[id][5]-first
      add_edge.call(root,id,skew,'local',{'start_skew'=>skew})
    end
  end
  starts = roots

  grids.each do |(die,pos,phase),group|
    expected = (0...ctas_by_die[die]).to_a
    need(group[:body].keys.sort == expected && group[:finish].keys.sort == expected, 'incomplete grid coverage')
    latest = group[:body].values.map { |id| nodes[id][5] }.max
    earliest = group[:finish].values.map { |id| nodes[id][5] }.min
    need(latest <= earliest + quantum, 'same-die grid ordering exceeds quantum')
    join = add_node.call([die,nil,pos,phase,'grid_join',nil,nil,nil])
    group[:body].each_value { |id| add_edge.call(id,join,0,'grid_arrival') }
    group[:finish].each_value { |id| add_edge.call(join,id,[0,nodes[id][5]-latest].min,'grid_release_quantum') }
  end
  participants = dies.flat_map { |d| (0...ctas_by_die[d]).map { |c| [d,c] } }.sort
  bounds = Array.new(dies.length) { Array.new(dies.length) }
  dies.each_index { |i| bounds[i][i] = 0 }; external = []; causal_count = 0
  waits.keys.sort.each do |generation|
    producers = sends[generation]; receivers = waits[generation]
    if producers.empty?
      external << generation; next
    end
    need(producers.keys.sort == participants, "generation #{generation}: incomplete producers")
    need(receivers.keys.sort == participants || options[:incomplete], "generation #{generation}: incomplete receivers")
    join = add_node.call([nil,nil,nil,nil,'exchange_join',nil,generation,nil])
    producers.each_value { |id| add_edge.call(id,join,0,'causal_send') }
    receivers.each_value { |ids| ids.each { |id| add_edge.call(join,id,-quantum,'causal_wait_quantum') } }
    causal_count += 1
    dies.each_with_index do |sender,i|
      latest = producers.filter_map { |(d,_c),id| nodes[id][5] if d == sender }.max
      dies.each_with_index do |receiver,j|
        earliest = receivers.filter_map { |(d,_c),ids| ids.map { |id| nodes[id][5] }.min if d == receiver }.min
        next unless earliest
        value = earliest-latest+quantum
        bounds[i][j] = bounds[i][j] ? [bounds[i][j],value].min : value
      end
    end
  end
  dies.length.times do |k|
    dies.length.times do |i|
      next unless bounds[i][k]
      dies.length.times do |j|
        next unless bounds[k][j]
        value = bounds[i][k]+bounds[k][j]
        bounds[i][j] = value if bounds[i][j].nil? || value < bounds[i][j]
      end
    end
  end
  need(dies.each_index.all? { |i| bounds[i][i] >= 0 }, 'negative clock-offset cycle: inspect generation/quantum/drift')
  queue = indegree.each_index.select { |id| indegree[id].zero? }
  distance = Array.new(nodes.length); predecessor = Array.new(nodes.length)
  starts.each { |id| distance[id] = 0 }; head = 0
  while head < queue.length
    id = queue[head]; head += 1
    edges[id].each do |target,weight,kind,buckets|
      if distance[id]
        value = distance[id]+weight
        if distance[target].nil? || value > distance[target]
          distance[target] = value; predecessor[target] = [id,weight,kind,buckets]
        end
      end
      indegree[target] -= 1; queue << target if indegree[target].zero?
    end
  end
  need(head == nodes.length, 'cycle in observed causal graph')
  last = finishes.max_by { |id| distance[id] || -1 }; chain = distance[last]
  need(chain, 'no causal path to final publication')
  witness = []; witness_buckets = Hash.new(0); allowance = 0; cursor = last
  while predecessor[cursor]
    prior,weight,kind,buckets = predecessor[cursor]
    buckets&.each { |k,v| witness_buckets[k] += v }
    allowance -= weight if kind.end_with?('_quantum')
    witness << {from: node_info(nodes[prior]),to: node_info(nodes[cursor]),weight_ns: weight,kind: kind,buckets_ns: buckets}
    cursor = prior
  end
  witness.reverse!
  need(witness_buckets.values.sum-allowance == chain, 'witness bucket/weight mismatch')
  first_by_die = gpu.transform_values { |d| d[:rows].map { |r| r['start_ns'] }.min }
  last_by_die = gpu.transform_values { |d| d[:rows].map { |r| r['publication_ns'] }.max }
  lower = 0; upper = 0; connected = true
  dies.each_with_index do |d,i|
    dies.each_with_index do |e,j|
      base = last_by_die[d]-first_by_die[e]
      if bounds[i][j] && bounds[j][i]
        lower = [lower,base-bounds[j][i]-quantum].max
        upper = [upper,base+bounds[i][j]+quantum].max
      else
        connected = false
      end
    end
  end
  lower = [0,lower,chain-quantum].max
  need(!connected || upper >= lower, 'causal bound exceeds feasible enclosing span')
  host = hosts.first; batches = {}
  %w[poll readback].each do |kind|
    intervals = host[:rows].map { |r| [r.fetch("#{kind}_start_ns"),r.fetch("#{kind}_end_ns")] }.uniq
    need(intervals.all? { |a,b| a.positive? && a <= b }, 'invalid host batch interval')
    batches[kind] = {count: intervals.length,summed_duration_ns: intervals.sum { |a,b| b-a },
                     union_duration_ns: union_ns(intervals),intervals_ns: intervals}
  end
  host_rows = host[:rows].map do |r|
    TraceChecks.ordered(r.transform_values(&:to_s),%w[observed_ns callback_start_ns callback_end_ns])
    {position: r['position'],callback_ns: r['callback_end_ns']-r['callback_start_ns'],
     ple_staging_ns: r['ple_staging_end_ns']-r['ple_staging_start_ns'],
     ple_read_ns: r['ple_read_end_ns']-r['ple_read_start_ns'],
     ple_ready_bracket_ns: r['ple_ready_ns']-r['ple_ready_start_ns'],raw_fields: r}
  end
  report = {
    format: 'flash_trace_reconciliation_v2',schema: 2,
    analysis_scope: {kind: options[:position] ? 'single_position' : 'capture_window',
                     selected_position: options[:position],original_input_metadata: original},
    lifecycle: numeric(reference),quantum_ns: quantum,dies: dies,ctas_by_die: ctas_by_die,positions: positions,
    local_cta_reconciliation: local,same_die_envelopes: envelopes,
    opcode_sampled_totals: opcode_totals.map { |(d,op),b| {die: d,opcode: op,buckets_ns: b} },
    host: {bounded_token_rows: host_rows,unique_successful_batches: batches,
           request_totals: {scope: 'whole request including prefix/outside bounded window',
                            counters: numeric(host[:meta].select { |k,_v| k.start_with?('request_poll_','request_readback_') })}},
    causal_analysis: {
      interpretation: 'Observed causal longest-chain lower bound; branch merges use max, not sum.',
      clock_assumption: 'Unknown constant per-die epoch and equal nominal timer rate over the bounded window; quantum does not cover arbitrary drift.',
      nodes: nodes.length,causal_generations: causal_count,grid_barrier_joins: grids.length,
      external_incoming_generations: external,outgoing_without_observed_wait: (sends.keys-waits.keys).sort,
      chain_sampled_ns_after_crossing_allowance: chain,makespan_lower_bound_ns: lower,
      enclosing_span_upper_bound_ns: connected ? upper : nil,
      unresolved_span_gap_upper_bound_ns: connected ? upper-lower : nil,
      relative_clock_offsets: dies.each_with_index.map { |d,i| {die: d,reference_die: dies.first,lower_ns: bounds[0][i] && -bounds[0][i],upper_ns: bounds[i][0]} },
      witness_local_buckets_ns: witness_buckets,witness_crossing_allowance_ns: allowance,witness: witness,
      send_publication_bracket_max_ns: send_brackets.max,
      limitations: [
        'Local residuals are exact integer arithmetic, not a claim of sub-quantum physical precision.',
        'Bodies sample thread 0; grid/barrier residency includes unsampled work. CTA durations overlap.',
        'Only send-before is a causal predecessor of remote wait-end; send-after can be later than receiver observation.',
        'The exact last producer and visibility/poll instants are unresolved within publication brackets.',
        'Prewindow producers are external. Single-position analysis has no previous-token publication sample.',
        'Enclosing-span bounds are conservative offset constraints; their gap is reported, not silently aligned away.',
        'Host and device timestamps are never subtracted. Request host totals and bounded successful batches have different scopes.'
      ]
    }
  }
  result = JSON.pretty_generate(report) + "\n"
  options[:out] ? File.write(options[:out],result) : puts(result)
rescue StandardError => e
  warn "trace_reconcile: #{e.class}: #{e.message}"; exit 1
end
