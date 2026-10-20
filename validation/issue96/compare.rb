#!/usr/bin/env ruby
require 'json'
require 'digest'
abort 'usage: compare.rb CPU_REFERENCE GPU_EMBEDDINGS' unless ARGV.length == 2
a, b = ARGV.map { |p| File.binread(p).unpack('e*') }
raise 'dimension mismatch' unless a.length == b.length && a.length.positive? && a.length % 2560 == 0
raise 'non-finite embeddings' unless a.all?(&:finite?) && b.all?(&:finite?)
dot = aa = bb = error = maximum = 0.0
min_cosine = 1.0
max_relative = 0.0
(a.length / 2560).times do |row|
  rd = ra = rb = re = 0.0
  (row * 2560...(row + 1) * 2560).each do |i|
    x, y = a[i], b[i]
    rd += x * y; ra += x * x; rb += y * y; re += (x - y)**2
    maximum = [maximum, (x-y).abs].max
  end
  raise 'zero norm embedding' unless ra.positive? && rb.positive?
  min_cosine = [min_cosine, rd / Math.sqrt(ra * rb)].min
  max_relative = [max_relative, Math.sqrt(re / ra)].max
  dot += rd; aa += ra; bb += rb; error += re
end
puts JSON.generate(event: 'embedding_numerical_validation', tokens: a.length/2560,
                   global_cosine: dot/Math.sqrt(aa*bb), global_relative_l2: Math.sqrt(error/aa),
                   max_abs_error: maximum, min_token_cosine: min_cosine, max_token_relative_l2: max_relative,
                   reference_sha256: Digest::SHA256.file(ARGV[0]).hexdigest,
                   gpu_sha256: Digest::SHA256.file(ARGV[1]).hexdigest)
raise 'embedding numerical tolerance exceeded' unless min_cosine >= 0.999 && max_relative <= 0.02
