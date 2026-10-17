#!/usr/bin/env python3
import concurrent.futures as cf
import os, socket, struct, sys, tempfile, threading, time
sockpath=sys.argv[1]
def info():
 with socket.socket(socket.AF_UNIX) as s:
  s.connect(sockpath); s.sendall(b'info\n')
  with s.makefile('r') as f: return f.readline().split()
def ask(slot,ids,new=6,snap=0,sample=None,stop=None,image=None,barrier=None):
 with socket.socket(socket.AF_UNIX) as s:
  s.settimeout(120); s.connect(sockpath)
  commands=[]
  if sample is not None: commands.append('sample %d .7 8 .9'%sample)
  if stop is not None: commands.append('stop %d'%stop)
  if snap: commands.append('snap %d'%snap)
  if image: commands.append('img %s 1'%image)
  commands.append('chat_slot %d %d %s'%(slot,new,' '.join(map(str,ids))))
  if barrier: barrier.wait()
  s.sendall(('\n'.join(commands)+'\n').encode())
  tokens=[]; first=None
  with s.makefile('r') as f:
   for line in f:
    a=line.split()
    if a and a[0]=='error:': raise AssertionError(line)
    if a and a[0]=='tok':
     tokens.append(int(a[1])); first=first or time.monotonic()
    if a and a[0]=='done':
     assert int(a[1])==len(tokens),line
     return dict(tokens=tokens,ended=int(a[2]),resumed=int(a[6]),slot=int(a[8]),first=first,done=time.monotonic())
  raise AssertionError('engine closed without done')
def pair(a,b):
 gate=threading.Barrier(2)
 with cf.ThreadPoolExecutor(2) as pool:
  fa=pool.submit(ask,barrier=gate,**a); fb=pool.submit(ask,barrier=gate,**b)
  return fa.result(),fb.result()
i=info(); assert i[0]=='info' and len(i)>=5,i
cap,images,nslots,quantum=map(int,i[1:5]); assert nslots>=2,i
A=[1,2,3,4,5]; B=[65,66,67,68,69]
for sampling in (None,12345):
 a=dict(slot=0,ids=A,sample=sampling)
 b=dict(slot=1,ids=B,sample=None if sampling is None else 98765)
 serial=[ask(**a),ask(**b)]
 inter=pair(a,b)
 for n in range(2):
  assert inter[n]['tokens']==serial[n]['tokens'],(sampling,n,serial,inter)
  assert inter[n]['slot']==n
 a['snap']=2; b['snap']=2
 snap=pair(a,b); resumed=pair(a,b)
 for n in range(2):
  assert snap[n]['tokens']==serial[n]['tokens']
  assert resumed[n]['tokens']==serial[n]['tokens']
  assert resumed[n]['resumed']==2,resumed
 print('serial/interleaved/snapshot match, sample=',sampling,flush=True)
longids=[100+(x%100) for x in range(min(96,cap-8))]
a,b=pair(dict(slot=0,ids=longids,new=1),dict(slot=1,ids=B[:2],new=2))
assert b['done']<a['done'],(a,b)
print('short decode completed while other slot was prefilling',flush=True)
first=ask(0,A)['tokens'][0]
ended=ask(0,A,stop=first); assert ended['ended']==1 and len(ended['tokens'])==1,ended
with socket.socket(socket.AF_UNIX) as s:
 s.connect(sockpath); s.sendall(('chat_slot 0 1 '+' '.join(map(str,longids))+'\n').encode())
assert ask(1,B)['tokens']
assert ask(0,A)['tokens']
print('stop, cancelled request and slot reuse passed',flush=True)
if images:
 fd,path=tempfile.mkstemp(prefix='flash-sequence-image-',suffix='.bin')
 try:
  with os.fdopen(fd,'wb') as f:
   f.write(struct.pack('<5I',1,1,1,7,2560))
   f.write(struct.pack('<2560f',*([0.0]*2560)))
   f.write(struct.pack('<3I',0,1,2))
  a=dict(slot=0,ids=A,image=path); b=dict(slot=1,ids=B)
  baseline=[ask(**a),ask(**b)]; inter=pair(a,b)
  for n in range(2): assert baseline[n]['tokens']==inter[n]['tokens'],(baseline,inter)
  print('image/M-RoPE and plain-text slots isolated',flush=True)
 finally: os.unlink(path)
print('sequence_check: passed',flush=True)
