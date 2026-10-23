#!/bin/bash
# vision_test.sh <image file> [question] [url]: one image question through flashd (OpenAI-style image_url data URL)
IMG=${1:?image file}; Q=${2:-Describe this image in two sentences.}; URL=${3:-http://127.0.0.1:18090}
case $IMG in *.png) MIME=image/png ;; *.gif) MIME=image/gif ;; *) MIME=image/jpeg ;; esac
REQ=$(jq -n --arg q "$Q" --arg url "data:$MIME;base64,$(base64 -w0 "$IMG")" '{messages:[{role:"user",content:[{type:"image_url",image_url:{url:$url}},{type:"text",text:$q}]}],max_tokens:200,temperature:0,chat_template_kwargs:{enable_thinking:false}}')
curl -s -m 1800 "$URL/v1/chat/completions" -H 'Content-Type: application/json' -d "$REQ" | jq -r '.error.message // ("answer: " + .choices[0].message.content), "usage: \(.usage)", "timings: \(.timings)"'
