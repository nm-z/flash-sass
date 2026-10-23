# Exhaustively compare each decoded byte, including adjacent-byte carries.
levels = [-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113]
65536.times do |selectors|
  lo = hi = expected = 0
  4.times do |byte|
    code = (selectors >> (4*byte)) & 15
    lo |= (code < 8 ? levels[code] & 255 : 255) << (8*byte)
    hi |= (code < 8 ? 0 : levels[code]+1) << (8*byte)
    expected |= (levels[code] & 255) << (8*byte)
  end
  got = ((lo & 0x7f7f7f7f) + hi) ^ 0x80808080
  abort "mismatch selectors=#{selectors}" unless got == expected
end
puts '65536 packed selectors: exact; no carries between bytes'
