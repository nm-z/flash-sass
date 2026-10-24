require 'net/http'
require 'json'
require 'fileutils'
root=ARGV.fetch(0)
FileUtils.mkdir_p(root)
prompts=[
  'Why does a metal spoon feel colder than a wooden spoon in the same room? Explain in a short paragraph.',
  'Explain why a bicycle is easier to balance while moving than when stopped, in everyday language.',
  'How do clouds stay in the sky if water is heavier than air? Give a concise explanation.'
]
if ENV['FRESH_SET']=='2'
  prompts=[
    'Why does bread turn brown in a toaster? Explain briefly without equations.',
    'How can migrating birds navigate over long distances? Give a concise explanation.',
    'Why does a spinning ice skater turn faster when pulling their arms inward? Explain in everyday language.'
  ]
end
if ENV['FRESH_SET']=='3'
  prompts=[
    'Why can a large ship made of steel float while a small steel nail sinks? Explain briefly.',
    'How does yeast make bread dough rise? Give a short explanation in plain language.',
    'Why do mountains often have snow when nearby valleys are warm? Explain the main reasons concisely.'
  ]
end
if ENV['FRESH_SET']=='4'
  prompts=[
    'is it OK to drink water even if homeless people drink water?',
    'Why does a rainbow appear as an arc? Explain briefly in plain language.',
    'Why do leaves change color in autumn? Give a concise explanation.'
  ]
end
if ENV['FRESH_SET']=='5'
  paragraph='The workshop tracks three crates. The red crate contains eleven brass screws. The blue crate contains seventeen wooden pegs. The green crate contains twenty-three steel washers. Every delivery is checked against this inventory, and the counts remain unchanged during the inspection. '
  prompts=[
    'is it OK to drink water even if homeless people drink water?',
    'Why does a soap bubble show changing colors? Explain briefly in everyday language.',
    paragraph*30 + 'Based on the inventory above, how many wooden pegs are in the blue crate? Answer with the number.'
  ]
end
if ENV['FRESH_SET']=='6'
  prompts=[
    'is it OK to drink water even if homeless people drink water?',
    'Why does a pressure cooker cook food faster? Give a short explanation.',
    'How does a compass find north? Explain briefly in plain language.'
  ]
end
prompts=prompts.first(Integer(ENV['HTTP_LIMIT'])) if ENV['HTTP_LIMIT']
prompts.each_with_index do |prompt,index|
  origin=Process.clock_gettime(Process::CLOCK_MONOTONIC)
  rows=[];buffer='';status=nil
  Net::HTTP.start('127.0.0.1',18090,read_timeout:600,open_timeout:10) do |http|
    request=Net::HTTP::Post.new('/v1/chat/completions',{'Content-Type'=>'application/json'})
    body={messages:[{role:'user',content:prompt}],stream:true,max_tokens:Integer(ENV.fetch('HTTP_MAXTOK','256'))}
    body[:seed]=Integer(ENV['HTTP_SEED']) if ENV['HTTP_SEED']
    request.body=JSON.generate(body)
    http.request(request) do |response|
      status=response.code
      response.read_body do |chunk|
        buffer << chunk
        while (end_at=buffer.index("\n\n"))
          event=buffer.slice!(0,end_at+2)
          data=event.lines.filter_map{|l|l.delete_prefix('data:').strip if l.start_with?('data:')}.join("\n")
          next if data.empty?
          rows << {elapsed_ms:(Process.clock_gettime(Process::CLOCK_MONOTONIC)-origin)*1000,data:data=='[DONE]' ? data : JSON.parse(data)}
        end
      end
    end
  end
  record={prompt:prompt,status:status,wall_ms:(Process.clock_gettime(Process::CLOCK_MONOTONIC)-origin)*1000,events:rows}
  File.write(File.join(root,"request-#{index}.json"),JSON.pretty_generate(record))
  raise 'request failed' unless status=='200' && rows.last&.fetch(:data)=='[DONE]' && rows.none?{|r|r[:data].is_a?(Hash) && r[:data]['error']}
  timing=rows.filter_map{|r|r[:data]['timings'] if r[:data].is_a?(Hash)}.last
  puts JSON.generate(index:index,wall_ms:record[:wall_ms],timings:timing);STDOUT.flush
end
