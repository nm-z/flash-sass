#!/usr/bin/env ruby
# gguf-list.rb FILE [PATTERN]: the metadata keys (scalars and array lengths) and the tensors (name, shape, type, bytes) of a GGUF file.
TYPES = %w[F32 F16 Q4_0 Q4_1 Q4_2 Q4_3 Q5_0 Q5_1 Q8_0 Q8_1 Q2_K Q3_K Q4_K Q5_K Q6_K Q8_K IQ2_XXS IQ2_XS IQ3_XXS IQ1_S IQ4_NL IQ3_S
           IQ2_S IQ4_XS I8 I16 I32 I64 F64 IQ1_M BF16].freeze
# bytes per block and elements per block, ggml's table
BLOCK = { 'F32' => [4, 1], 'F16' => [2, 1], 'BF16' => [2, 1], 'Q4_0' => [18, 32], 'Q4_1' => [20, 32], 'Q5_0' => [22, 32], 'Q5_1' => [24, 32],
          'Q8_0' => [34, 32], 'Q2_K' => [84, 256], 'Q3_K' => [110, 256], 'Q4_K' => [144, 256], 'Q5_K' => [176, 256], 'Q6_K' => [210, 256],
          'IQ4_NL' => [18, 32], 'IQ4_XS' => [136, 256], 'I32' => [4, 1] }.freeze
f = File.open(ARGV.fetch(0), 'rb')
pattern = ARGV[1] && Regexp.new(ARGV[1])
rd = ->(n) { f.read(n) }
u32 = -> { rd.(4).unpack1('V') }
u64 = -> { rd.(8).unpack1('Q<') }
str = -> { rd.(u64.()).force_encoding('UTF-8') }
scalar = lambda do |t|
  case t
  when 0 then rd.(1).unpack1('C') when 1 then rd.(1).unpack1('c') when 2 then rd.(2).unpack1('v') when 3 then rd.(2).unpack1('s<')
  when 4 then u32.() when 5 then rd.(4).unpack1('l<') when 6 then rd.(4).unpack1('e') when 7 then rd.(1).unpack1('C') != 0
  when 8 then str.() when 10 then u64.() when 11 then rd.(8).unpack1('q<') when 12 then rd.(8).unpack1('E')
  else raise "value type #{t}"
  end
end
raise 'not a GGUF file' unless rd.(4) == 'GGUF'
version = u32.()
nt = u64.(); nkv = u64.()
puts "version #{version}, #{nt} tensors, #{nkv} keys"
nkv.times do
  k = str.(); t = u32.()
  if t == 9
    et = u32.(); n = u64.()
    vals = Array.new(n) { scalar.(et) }
    shown = vals.first(8).map { |v| v.is_a?(String) && v.size > 40 ? "#{v[0, 40]}…" : v }
    puts "  #{k} = [#{n}] #{shown.inspect[0, 160]}" if !pattern || k.match?(pattern)
  else
    v = scalar.(t)
    puts "  #{k} = #{v.is_a?(String) && v.size > 120 ? "#{v[0, 120]}…" : v.inspect}" if !pattern || k.match?(pattern)
  end
end
total = 0
nt.times do
  name = str.(); nd = u32.(); dims = Array.new(nd) { u64.() }; t = u32.(); u64.()
  ty = TYPES[t] || t.to_s
  bs, be = BLOCK[ty]
  bytes = bs ? dims.inject(:*) / be * bs : 0
  total += bytes
  puts format('  %-44s %-22s %-7s %10.3f MB', name, dims.inspect, ty, bytes / 1e6) if !pattern || name.match?(pattern)
end
puts format('total %.3f GB', total / 1e9)
