#!/usr/bin/env ruby
# dump-diff.rb A B: relative difference of two FLASH_DUMP_S stream dumps (die 0 of each), per hyper-connection lane
a, b = ARGV.map { |p| File.binread("#{p}.0").unpack('e*') }
raise 'size' unless a.size == b.size
4.times do |l|
  x = a[l * 2560, 2560]; y = b[l * 2560, 2560]
  num = x.zip(y).sum { |p, q| (p - q)**2 }; den = x.sum { |p| p * p }
  nan = (x + y).count { |v| v.nan? || v.infinite? }
  puts format('lane %d: rel L2 diff %.3e  |a|=%.3e  nonfinite=%d  a[0..3]=%s  b[0..3]=%s', l, Math.sqrt(num / (den.nonzero? || 1)), Math.sqrt(den), nan,
              x[0, 4].map { |v| format('%.4f', v) }.join(','), y[0, 4].map { |v| format('%.4f', v) }.join(','))
end
