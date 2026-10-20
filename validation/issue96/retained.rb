#!/usr/bin/env ruby
# Verify retained binary results from the completed full-image inference job.
require 'digest'
require 'json'
root = ARGV.fetch(0)
cases = [
  ['screenshot', 602, 'd9b7be627eb9828045805efc651e373127d29850acb11152a7d4c188f1beebab'],
  ['2048x2048', 4096, 'eb0a3449d67a8265c93ccea9b4206f2af4a2b0c9098e961ac908b8fc6d0c188f'],
  ['4096x4096', 16384, '807b537caf63c6e4e9c898e544e36b199f332814e435070f411cc31eb91ba93a']
]
cases.each_with_index do |(label, tokens, expected), index|
  path = File.join(root, "embeddings-#{index}.bin")
  bytes = File.size(path)
  raise 'retained tensor size mismatch' unless bytes == tokens * 2560 * 4
  sha = Digest::SHA256.file(path).hexdigest
  raise 'retained tensor changed from recorded inference' unless sha == expected
  count = 0
  File.open(path, 'rb') do |file|
    while (chunk = file.read(4 << 20))
      values = chunk.unpack('e*')
      raise 'non-finite retained output' unless values.all?(&:finite?)
      count += values.length
    end
  end
  puts JSON.generate(event: 'retained_embeddings_verified', original_job: 'b1c3a80a-0caa-482c-8ce1-4466da720162',
                     image: label, tokens: tokens, dimension: 2560, finite_values: count, bytes: bytes, sha256: sha)
end
