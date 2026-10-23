#!/usr/bin/env python3
import re,socket,sys
sock=socket.socket(socket.AF_UNIX);sock.settimeout(180);sock.connect(sys.argv[1]);f=sock.makefile('r')
def send(c):sock.sendall((c+'\n').encode())
def line():
 x=f.readline();assert x,'engine closed connection';return x.strip()
def control(mode='none',pos=0,head=0,after=0):
 send('ple_io_test %s %d %d %d'%(mode,pos,head,after));x=line();assert x.startswith('ok: ple_io_test '),x
def status():
 send('ple_io_test status');x=line();assert x.startswith('ple_io_test '),x
 return {k:int(v) for k,v in re.findall(r'(launches|failed|pos|head|snapshot)=(-?\d+)',x)}
def ask(ids,new=4,snap=0):
 if snap:
  send('snap %d'%snap);x=line();assert x.startswith('ok: snap '),x
 send('chat %d %s'%(new,' '.join(map(str,ids))));toks=[]
 while True:
  x=line()
  if x.startswith('tok '):toks.append(int(x.split()[1]))
  elif x.startswith('done '):return toks,x,None
  elif x.startswith('error:'):return toks,None,x
  else:raise AssertionError(x)
send('config');config=line();assert 'ple_disk=1' in config and 'layers=48/48' in config,config
send('info');cap=line().split();assert cap[0]=='info' and int(cap[1])>=192,cap
control();baseline,done,error=ask([10,11]);assert done and not error
for mode in ('partial','eintr'):
 control(mode,0,3,17);toks,done,error=ask([10,11]);assert done and not error and toks==baseline,(mode,error,toks);print(mode,'recovered to identical tokens',flush=True)
control();longbase,done,error=ask([10,11],190,snap=1);assert done and not error
for mode in ('eof','eio'):
 control(mode,0,3,17);before=status()['launches'];toks,done,error=ask([12,13]);assert not done and error and not toks,(toks,done,error);assert 'PLE I/O position 0 head 3' in error and 'bytes 17/90' in error,error;st=status();assert st['launches']==before and st['snapshot']==0,st
 control();toks,done,error=ask([10,11],4,snap=1);assert done and not error;assert status()['snapshot']==1
 control(mode,160,3,17);before=status()['launches'];toks,done,error=ask([10,11],190);assert not done and error,(done,error);assert 'PLE I/O position 160 head 3' in error and 'bytes 17/90' in error,error;assert toks==longbase[:len(toks)] and len(toks)==159,(len(toks),error);st=status();assert st['launches']>before and st['snapshot']==0,st
 control();recovered,done,error=ask([10,11],190);assert done and not error and recovered==longbase,(done,error);assert status()['failed']==0;print(mode,'prelaunch/midrun failure and healthy recovery passed',flush=True)
f.close();sock.close();print('PLE I/O GPU regression passed',flush=True)
