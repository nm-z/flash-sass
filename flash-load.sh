#!/bin/bash
export PATH=$HOME/.lmstudio/bin:$PATH
echo "$(date +%T) LOAD START key=qwen3.8-flash-next"
lms load "qwen3.8-flash-next" --identifier flash --gpu max --context-length 8192 -y
echo "$(date +%T) LOAD EXIT rc=$?"
lms ps
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
