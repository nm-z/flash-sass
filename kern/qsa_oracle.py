#!/usr/bin/env python3
"""Pinned HF QSA semantics and independent numerical acceptance.
The local C twins are candidates, not the independent oracle.
Synthetic success does not establish checkpoint/model quality.
"""
from __future__ import annotations
import argparse, ast, ctypes as C, hashlib, json, math, os, pathlib, platform
import subprocess, sys, tempfile, types, urllib.request
import numpy as np
import torch
from torch import nn
ROOT=pathlib.Path(__file__).resolve().parent.parent
HF_COMMIT="8498ed425fa01bbee7d086efdf1a6634c2962aed"
HF_BLOB="946f5fd6e9ba21347654b1b2ca196e86b0f751b3"
HF_PATH="src/transformers/models/qwen4_exp/modeling_qwen4_exp.py"
HF_URL=f"https://raw.githubusercontent.com/huggingface/transformers/{HF_COMMIT}/{HF_PATH}"
FLAGS=["-std=gnu11","-O2","-fno-fast-math","-ffp-contract=off","-frounding-math","-Wall","-Wextra","-fPIC","-shared"]
LIMITS={"attention_abs":3e-5,"attention_nrms":3e-5,"kv_attention_abs":.025,"kv_attention_nrms":.025,"score_relative":.005,"logit_abs":.10,"logit_nrms":.01}
def require(ok,message):
    if not bool(ok): raise AssertionError(message)
def digest(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(8<<20),b""): h.update(chunk)
    return h.hexdigest()
def ptr(x): return np.asarray(x).ctypes.data_as(C.c_void_p)
def metrics(reference,candidate):
    reference=np.asarray(reference,np.float64); candidate=np.asarray(candidate,np.float64)
    require(reference.shape==candidate.shape and reference.size>0 and np.isfinite(reference).all() and np.isfinite(candidate).all(),"invalid numerical comparison")
    delta=candidate-reference; rms=float(np.sqrt(np.mean(delta*delta)))
    return {"max_abs":float(np.max(np.abs(delta))),"max_relative":float(np.max(np.abs(delta)/np.maximum(np.abs(reference),1e-8))),"rms":rms,"normalized_rms":rms/max(float(np.sqrt(np.mean(reference*reference))),1e-8)}
def build_bridge(tmp,report):
    cc=os.environ.get("CC","cc")
    sources=[ROOT/"kern/qsa_oracle_bridge.c",ROOT/"ref/qsa.c",ROOT/"ref/attn.c",ROOT/"ref/fmath.c"]
    so=pathlib.Path(tmp)/"qsa_oracle.so"
    command=[cc,*FLAGS,"-I"+str(ROOT/"ref"),*map(str,sources),"-o",str(so),"-lm"]
    subprocess.run(command,check=True)
    report["build"]={"command":command,"compiler":subprocess.check_output([cc,"--version"],text=True).splitlines()[0],"sources_sha256":{str(p.relative_to(ROOT)):digest(p) for p in sources},"image_sha256":digest(so)}
    lib=C.CDLL(str(so)); p,u=C.c_void_p,C.c_uint
    signatures={"qsaoracle_environment":([],C.c_int),"qsaoracle_set_rounding":([C.c_int],C.c_int),
    "qsaoracle_index":([p,u,p,p,p,p,p,p,p,p],C.c_int),"qsaoracle_quant":([p,u,p,p,p],C.c_int),
    "qsaoracle_attention":([p,p,p,u,u,p,p,p,p,p],C.c_int),"idx_select_gpu":([p,u,u,p],u),
    "idx_norm_rope_gpu":([p,p,C.c_float,p,p,p],None),"idx_scores_gpu":([p,p,u,p],None),
    "exp_gpu":([C.c_float],C.c_float),"rcp_gpu":([C.c_float],C.c_float),
    "qsa_f2h":([C.c_float],C.c_uint16),"qsa_h2f":([C.c_uint16],C.c_float)}
    for name,(args,result) in signatures.items(): f=getattr(lib,name); f.argtypes,f.restype=args,result
    require(lib.qsaoracle_environment()==1,"binary32 evaluation and FE_TONEAREST required")
    require(lib.qsaoracle_set_rounding(0)==0,"cannot set FE_DOWNWARD")
    rejected=lib.qsaoracle_environment()==0
    require(lib.qsaoracle_set_rounding(1)==0,"cannot restore rounding")
    require(rejected,"rounding guard accepted FE_DOWNWARD")
    return lib
def load_hf(path,report):
    if path: source=pathlib.Path(path).read_bytes()
    else:
        with urllib.request.urlopen(HF_URL,timeout=30) as response: source=response.read()
    blob=hashlib.sha1(b"blob "+str(len(source)).encode()+b"\0"+source).hexdigest()
    require(blob==HF_BLOB,f"HF source blob mismatch: {blob}")
    names={"Qwen4ExpTextRMSNorm","rotate_half","apply_rotary_pos_emb","Qwen4ExpTextQSAIndexer"}
    nodes=[n for n in ast.parse(source).body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names]
    require({n.name for n in nodes}==names,"pinned oracle definitions incomplete")
    cls=next(n for n in nodes if n.name=="Qwen4ExpTextQSAIndexer"); inserted=0
    for n in ast.walk(cls):
        if isinstance(n,(ast.For,ast.If)):
            body=[]
            for statement in n.body:
                if isinstance(statement,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="selected_block_indices" for t in statement.targets):
                    body+=ast.parse("self.oracle_scores = scores.detach().clone()\nself.oracle_pool = block_key_states.detach().clone()\nself.oracle_iq = q[batch_idx, query_idx].detach().clone()\n").body
                    inserted+=1
                body.append(statement)
            n.body=body
    require(inserted==1,"HF observation point changed")
    tree=ast.Module(body=ast.parse("from __future__ import annotations").body+nodes,type_ignores=[])
    env={"torch":torch,"nn":nn,"math":math}
    exec(compile(ast.fix_missing_locations(tree),HF_URL,"exec"),env)
    report["external_oracle"]={"repository":"huggingface/transformers","commit":HF_COMMIT,"blob":blob,"path":HF_PATH,"execution":"literal AST definitions; observation before topk","tie_policy":"backend dependent; admissible cutoff sets tested"}
    return env
def select(lib,scores,k=512):
    scores=np.ascontiguousarray(scores,np.float32); ids=np.empty(min(len(scores),k),np.uint32)
    count=lib.idx_select_gpu(ptr(scores),len(scores),k,ptr(ids)); require(count==len(ids),"candidate selection count"); return ids
def admissible(scores,ids,k=512):
    k=min(k,len(scores)); ids=np.asarray(ids,np.int64)
    if not k:return len(ids)==0
    cutoff=np.partition(scores,len(scores)-k)[len(scores)-k]
    return len(ids)==k and len(np.unique(ids))==k and bool(np.all(scores[ids]>=cutoff)) and set(np.flatnonzero(scores>cutoff)).issubset(set(ids.tolist()))
def margin(scores,k=512):
    if len(scores)<=k:return None
    ordered=np.sort(scores);return float(np.float64(ordered[-k])-np.float64(ordered[-k-1]))
def rope(n):
    theta=np.power(1e7,-2.*np.arange(32,dtype=np.float64)/64.)
    angles=np.arange(n,dtype=np.float64)[:,None]*theta[None,:]
    return np.ascontiguousarray(np.cos(angles),np.float32),np.ascontiguousarray(np.sin(angles),np.float32)
def make_hf(env):
    cfg=types.SimpleNamespace(indexer_n_heads=4,indexer_kv_heads=1,indexer_head_dim=128,indexer_budget=2048,indexer_compress_ratio=4,hidden_size=640,rms_norm_eps=1e-6)
    model=env["Qwen4ExpTextQSAIndexer"](cfg,0).eval();model.index_qk_proj=nn.Identity();return model
def hf_index(env,raw,query,offset_q,offset_k,cs,sn):
    model=make_hf(env)
    with torch.no_grad():
        model.q_layernorm.weight.copy_(torch.from_numpy(offset_q));model.k_layernorm.weight.copy_(torch.from_numpy(offset_k))
    class Cache:
        def update_indexer(self,keys,layer):return torch.from_numpy(raw)[None]
    projected=np.concatenate((query.reshape(-1),raw[-1])).astype(np.float32)
    cos=torch.from_numpy(np.concatenate((cs,cs),-1))[None];sin=torch.from_numpy(np.concatenate((sn,sn),-1))[None]
    mask=torch.ones((1,1,1,len(raw)),dtype=torch.bool)
    with torch.no_grad(): chosen=model(torch.from_numpy(projected)[None,None],(cos,sin),mask,Cache())[0,0,0].numpy()
    nv=len(raw)//4;blocks=chosen[:nv*4].reshape(nv,4)
    require(bool(np.all(blocks==blocks[:,:1])),"HF selected incomplete blocks")
    ids=np.flatnonzero(chosen[:nv*4:4]).astype(np.uint32)
    require(chosen.sum()==4*len(ids)+len(raw)%4 and bool(chosen[nv*4:].all()),"HF complete-block/tail contract")
    if nv:return model.oracle_scores.numpy().copy(),ids,model.oracle_pool.numpy().copy(),model.oracle_iq.numpy().copy()
    with torch.no_grad():iq=env["apply_rotary_pos_emb"](model.q_layernorm(torch.from_numpy(query)[None,None]),cos=cos[:,-1:,:],sin=sin[:,-1:,:],unsqueeze_dim=2)[0,0].numpy().copy()
    return np.empty(0,np.float32),ids,np.empty((0,128),np.float32),iq
def cutoff_tests(lib,outdir):
    rows=[];a,b,divisor=map(np.float32,(1.4142136573791504,1.41421377658844,11.313708305358887))
    require(a<b and np.float32(a/divisor)==np.float32(b/divisor),"FP32 division witness lost")
    fixtures=[("adjacent-low-id-a",[a,b]),("adjacent-low-id-b",[b,a]),("equal",[a,a]),("zero",[0.,0.])]
    for name,pair in fixtures:
        raw=np.concatenate((np.full(511,4.,np.float32),np.asarray(pair,np.float32),np.zeros(4,np.float32)))
        if name=="equal":raw.fill(a)
        if name=="zero":raw.fill(np.float32(0.))
        scaled=np.asarray(raw/divisor,np.float32);local,lower=select(lib,raw),select(lib,scaled)
        expected=np.lexsort((np.arange(len(raw)),-raw))[:512]
        require(np.array_equal(local,np.sort(expected)),"local lower-ID tie policy")
        hf=torch.topk(torch.from_numpy(scaled),512).indices.numpy()
        require(admissible(scaled,hf) and admissible(scaled,lower),"scaled cutoff admissibility")
        if name=="adjacent-low-id-a":require(512 in local and 511 not in local and 511 in lower and 512 not in lower,"scaling must change the deterministic cutoff")
        np.savez_compressed(outdir/("cutoff-"+name),raw=raw,raw_bits=raw.view(np.uint32),scaled=scaled,scaled_bits=scaled.view(np.uint32),local_ids=local,scaled_lower_ids=lower,hf_ids=hf)
        rows.append({"fixture":name,"raw_margin":margin(raw),"scaled_margin":margin(scaled),"overlap_local_scaled":len(set(local)&set(lower)),"overlap_local_hf":len(set(local)&set(hf))})
    return rows

def full_chain_cutoff(lib,env,outdir):
    n=2052;raw=np.zeros((n,128),np.float32)
    raw[:2044,64]=1.;raw[2044:2048,64]=1.;raw[2044:2048,66]=1.;raw[2048:,64]=1.;raw[2048:,65]=1.
    query=np.zeros((4,128),np.float32);query[0,64]=1.
    qn=np.ones(128,np.float32);qn[64]=np.float32(.0234375);kn=np.ones(128,np.float32)
    cs,sn=rope(n);iq=np.empty((4,128),np.float32);pool=np.empty((513,128),np.uint16);scores=np.empty(513,np.float32)
    def run(keys):return lib.qsaoracle_index(ptr(keys),n,ptr(query),ptr(qn),ptr(kn),ptr(cs),ptr(sn),ptr(iq),ptr(pool),ptr(scores))
    require(run(raw)==0,"full-chain cutoff setup");found=False
    for extra in np.linspace(.25,1.5,251,dtype=np.float32):
        query[0,67]=extra
        for tiny in (1e-9,2e-9,3e-9,4e-9):
            query[0,65]=np.float32(tiny)
            for h in range(4):lib.idx_norm_rope_gpu(ptr(query[h]),ptr(qn),1e-6,ptr(cs[-1]),ptr(sn[-1]),ptr(iq[h]))
            lib.idx_scores_gpu(ptr(iq),ptr(pool),513,ptr(scores))
            scaled=np.asarray(scores/np.float32(math.sqrt(128)),np.float32)
            if np.nextafter(scores[511],np.float32(np.inf))==scores[512] and scaled[511]==scaled[512] and np.min(scores[:511])>scores[512]:
                found=True;break
        if found:break
    require(found,"deterministic full-chain cutoff search exhausted");rows=[]
    for reverse in (False,True):
        keys=raw.copy()
        if reverse:keys[2044:2048]=raw[2048:2052];keys[2048:2052]=raw[2044:2048]
        require(run(keys)==0,"full-chain cutoff replay")
        scaled=np.asarray(scores/np.float32(math.sqrt(128)),np.float32);ids,lower=select(lib,scores),select(lib,scaled)
        hs,hi,hp,hq=hf_index(env,keys,query,qn-np.float32(1.),kn-np.float32(1.),cs,sn)
        require(admissible(hs,hi),"HF full-chain cutoff admissibility")
        require(scaled[511]==scaled[512] and 511 in lower and 512 not in lower,"scaled full-chain lower-ID tie")
        require((511 in ids)==reverse and (512 in ids)!=reverse,"unscaled full-chain strict order")
        np.savez_compressed(outdir/f"full-chain-cutoff-{int(reverse)}",raw_keys=keys,index_query=query,q_gamma=qn,k_gamma=kn,raw_score=scores.copy(),raw_bits=scores.view(np.uint32).copy(),scaled_score=scaled,scaled_bits=scaled.view(np.uint32),hf_score=hs,local_ids=ids,scaled_lower_ids=lower,hf_ids=hi,local_pool_bits=pool.copy(),hf_pool=hp,hf_iq=hq)
        rows.append({"reversed_ids":reverse,"raw_cutoff_bits":scores[511:513].view(np.uint32).tolist(),"scaled_cutoff_bits":scaled[511:513].view(np.uint32).tolist(),"raw_margin":margin(scores),"scaled_margin":margin(scaled),"hf_margin":margin(hs),"overlap":len(set(ids)&set(lower))})
    return rows
def index_domain_tests(lib):
    n=2052;raw=np.zeros((n,128),np.float32);query=np.zeros((4,128),np.float32)
    qn,kn=np.ones(128,np.float32),np.ones(128,np.float32);cs,sn=rope(n)
    iq=np.empty((4,128),np.float32);pool=np.empty((513,128),np.uint16);scores=np.empty(513,np.float32)
    def run():return lib.qsaoracle_index(ptr(raw),n,ptr(query),ptr(qn),ptr(kn),ptr(cs),ptr(sn),ptr(iq),ptr(pool),ptr(scores))
    require(run()==0,"zero full-chain fixture")
    require(bool(np.all(scores.view(np.uint32)==0)),"zero scores must be canonical +0")
    require(np.array_equal(select(lib,scores),np.arange(512,dtype=np.uint32)),"zero full-chain selection must use lower IDs")
    require(bool(np.all(iq==0)) and bool(np.all(pool.view(np.float16)==0)),"zero query/pool normalization")
    for bad in (np.nan,np.inf,-np.inf,1e5):
        raw[0,0]=np.float32(bad);require(run()!=0,"invalid raw-key domain accepted")
    raw[0,0]=0;query[0,0]=np.nan;require(run()!=0,"nonfinite query accepted")
    query[0,0]=0;qn[0]=np.inf;require(run()!=0,"nonfinite gamma accepted")
    qn[0]=1;cs[0,0]=np.nan;require(run()!=0,"nonfinite RoPE accepted")
    return {"zero_scores":"canonical +0","zero_selection":"IDs0..511","invalid_inputs":"rejected","mask":"contiguous causal prefix"}
def hf_bf16_probe(env,outdir):
    rng=np.random.default_rng(13579);n=20
    raw=torch.from_numpy(rng.normal(0,.5,(n,128)).astype(np.float32)).bfloat16()
    projected=torch.from_numpy(rng.normal(0,.5,(1,1,640)).astype(np.float32)).bfloat16()
    model=make_hf(env);seen=[]
    hook=model.k_layernorm.register_forward_pre_hook(lambda module,args:seen.append(args[0].detach().clone()))
    class Cache:
        def update_indexer(self,keys,layer):return raw[None]
    cs,sn=rope(n)
    cos=torch.from_numpy(np.concatenate((cs,cs),-1))[None].bfloat16()
    sin=torch.from_numpy(np.concatenate((sn,sn),-1))[None].bfloat16()
    with torch.no_grad():model(projected,(cos,sin),torch.ones((1,1,1,n),dtype=torch.bool),Cache())
    hook.remove();mean=raw.view(n//4,4,128).float().mean(1);expected=mean.bfloat16()
    require(len(seen)==1 and seen[0].dtype==torch.bfloat16 and torch.equal(seen[0],expected),"HF must cast the FP32 mean to raw dtype before RMS")
    require(bool(torch.any(expected.float()!=mean)),"BF16 mean probe must actually round")
    with torch.no_grad():
        normalized=model.k_layernorm(expected).float()
        direct=(expected.float()*torch.rsqrt(expected.float().square().mean(-1,keepdim=True)+1e-6)).bfloat16().float()
    require(torch.equal(normalized,direct),"HF gamma=1+offset convention")
    np.savez_compressed(outdir/"hf-bf16-boundary",mean_before_cast=mean.numpy(),mean_after_cast=expected.float().numpy(),normalized=normalized.numpy())
    return {"raw_dtype":"bfloat16","mean_accumulator":"float32","cast_before_RMS":True,"gamma":"1+stored_offset"}
def index_fixture(lib,env,n,seed,outdir):
    rng=np.random.default_rng(seed);raw=rng.normal(0,.5,(n,128)).astype(np.float32);query=rng.normal(0,.5,(4,128)).astype(np.float32)
    offsets=[torch.from_numpy(rng.normal(0,.1,128).astype(np.float32)).bfloat16().float().numpy() for _ in range(2)]
    oq,ok=offsets;qn,kn=np.float32(1.)+oq,np.float32(1.)+ok;cs,sn=rope(n);nv=n//4
    iq=np.empty((4,128),np.float32);pool=np.empty((nv,128),np.uint16);scores=np.empty(nv,np.float32)
    require(lib.qsaoracle_index(ptr(raw),n,ptr(query),ptr(qn),ptr(kn),ptr(cs),ptr(sn),ptr(iq),ptr(pool),ptr(scores))==0,"candidate indexer rejected fixture")
    hs,hi,hp,hq=hf_index(env,raw,query,oq,ok,cs,sn);ids=select(lib,scores)
    scaled=np.asarray(scores/np.float32(math.sqrt(128)),np.float32)
    require(admissible(hs,hi) and admissible(scores,ids),"full-chain selection invalid")
    error=metrics(hs,scaled) if nv else None;pool_only=hs
    if nv:
        pool_only=torch.relu(torch.from_numpy(hq)@torch.from_numpy(hp.astype(np.float16).astype(np.float32)).T).sum(0).numpy()/np.float32(math.sqrt(128))
        require(error["max_abs"]<=LIMITS["score_relative"]*max(float(np.max(np.abs(hs))),1e-8),"full-chain score error exceeds declared bound")
        e=error["max_abs"];gap=margin(hs)
        if nv>512:
            cutoff=np.partition(hs,nv-512)[nv-512]
            require(bool(np.all(hs[ids]>=cutoff-2*e)),"selected ID outside measured cutoff interval")
            mandatory=set(np.flatnonzero(hs>cutoff+2*e).tolist());require(mandatory.issubset(set(ids.tolist())),"mandatory ID missing outside cutoff interval")
        if gap is not None and gap>2*e:require(set(ids.tolist())==set(hi.tolist()),"selection differs across stable cutoff")
        require(metrics(hq,iq)["max_abs"]<1e-5,"gamma conversion/query normalization mismatch")
    np.savez_compressed(outdir/f"index-{n}",raw_score=scores,raw_bits=scores.view(np.uint32),scaled_score=scaled,scaled_bits=scaled.view(np.uint32),hf_score=hs,hf_bits=hs.view(np.uint32),pool_only_score=pool_only,local_ids=ids,hf_ids=hi,local_iq=iq,hf_iq=hq,local_pool_bits=pool,hf_pool=hp,index_query=query,stored_q_offset=oq,stored_k_offset=ok,full_q_gamma=qn,full_k_gamma=kn)
    return {"visible_length":n,"complete_blocks":nv,"tail":n%4,"selected_blocks":len(ids),"excluded_blocks":nv-len(ids),"seed":seed,"local_margin":margin(scores),"hf_margin":margin(hs),"overlap":len(set(ids)&set(hi)),"scaled_score_error":error,"raw_input_sha256":hashlib.sha256(raw.tobytes()).hexdigest()}
def quantize(lib,x):
    x=np.ascontiguousarray(x,np.float32);require(x.size%32==0,"quantization groups")
    q=np.empty(x.size,np.int8);s=np.empty(x.size//32,np.uint16);d=np.empty(x.size,np.float32)
    require(lib.qsaoracle_quant(ptr(x),x.size//32,ptr(q),ptr(s),ptr(d))==0,"candidate quantization rejected input")
    return q.reshape(x.shape),s,d.reshape(x.shape)
def quant_tests(lib):
    rows=[]
    for amplitude in (0.,1e-10,1e-6,1.,3000.,65504.*126.):
        x=np.linspace(-1,1,32,dtype=np.float32)*np.float32(amplitude);q,scale,deq=quantize(lib,x)
        g=max(np.max(np.abs(x)),np.float32(2.**-100));inv=np.float32(np.float32(1.)/g)*np.float32(127.)
        expected_q=np.rint(np.float32(x*inv)).astype(np.int8);expected_s=np.asarray(np.float32(g*np.float32(1./127.)),np.float16).view(np.uint16)
        require(np.array_equal(q,expected_q) and int(scale[0])==int(expected_s),"independent IEEE quantization formula")
        rows.append({"amplitude":amplitude,"scale_bits":int(scale[0]),"scale_underflow":bool(scale[0]==0),"error":metrics(x,deq)})
    x=np.zeros(32,np.float32);q,s,d=np.empty(32,np.int8),np.empty(1,np.uint16),np.empty(32,np.float32)
    for bad in (np.nan,np.inf,-np.inf,1e8):
        x[0]=bad;require(lib.qsaoracle_quant(ptr(x),1,ptr(q),ptr(s),ptr(d))!=0,"nonfinite/overflow quantization accepted")
    for value in (0.,-0.,2.**-24,2.**-25,65504.,65520.,np.inf,-np.inf):
        with np.errstate(over="ignore"):expected=int(np.asarray(np.float32(value),np.float16).view(np.uint16))
        require(lib.qsa_f2h(value)==expected,"FP16 conversion boundary")
    return rows
def local_attention(lib,query,key,value,gate,padding_fault=None):
    n=len(key);capacity=((n+8191)//8192)*8192
    kp=np.zeros((capacity,2,256),np.float32);vp=np.zeros_like(kp);kp[:n],vp[:n]=key,value
    if padding_fault is not None:vp[n,0,0]=padding_fault
    out=np.empty((24,256),np.float32);pm,pd=np.empty((24,8),np.float32),np.empty((24,8),np.float32);po=np.empty((24,8,256),np.float32)
    status=lib.qsaoracle_attention(ptr(query),ptr(kp),ptr(vp),n,capacity,ptr(gate),ptr(out),ptr(pm),ptr(pd),ptr(po))
    if padding_fault is not None:require(status!=0,"nonzero/nonfinite padding accepted");return None
    require(status==0,"candidate attention domain violation");return out,pm,pd,po
def independent_attention(query,key,value,gate):
    q=torch.from_numpy(query).double();k=torch.from_numpy(key).double().transpose(0,1).repeat_interleave(12,dim=0)
    v=torch.from_numpy(value).double().transpose(0,1).repeat_interleave(12,dim=0)
    probabilities=torch.softmax(torch.einsum("hd,hnd->hn",q,k)/16.,-1)
    return (torch.einsum("hn,hnd->hd",probabilities,v)*torch.sigmoid(torch.from_numpy(gate).double())).numpy()
def reduction_gate(error):
    require(error["max_abs"]<=LIMITS["attention_abs"] and error["normalized_rms"]<=LIMITS["attention_nrms"],"same-ID attention reduction error")
def attention_fixture(lib,n,seed,outdir):
    rng=np.random.default_rng(seed);query=rng.normal(0,.3,(24,256)).astype(np.float32)
    key=rng.normal(0,.3,(n,2,256)).astype(np.float32);value=rng.normal(0,.5,(n,2,256)).astype(np.float32);gate=rng.normal(0,2,(24,256)).astype(np.float32)
    for j in range(0,n,1024):key[j,:,0]+=np.float32(4.*(j//1024+1))
    exact=independent_attention(query,key,value,gate);got,pm,pd,po=local_attention(lib,query,key,value,gate)
    reduction=metrics(exact,got);reduction_gate(reduction)
    kq,ks,kd=quantize(lib,key);vq,vs,vd=quantize(lib,value)
    exact_quant=independent_attention(query,kd,vd,gate);quant_got,qpm,qpd,qpo=local_attention(lib,query,kd,vd,gate)
    quant_reduction=metrics(exact_quant,quant_got);reduction_gate(quant_reduction);loss=metrics(exact,exact_quant)
    require(loss["max_abs"]<=LIMITS["kv_attention_abs"] and loss["normalized_rms"]<=LIMITS["kv_attention_nrms"],"same-ID KV quantization error")
    if n%8192:
        local_attention(lib,query,key,value,gate,padding_fault=np.nan);local_attention(lib,query,key,value,gate,padding_fault=1.)
    np.savez_compressed(outdir/f"attention-{n}",query=query,key=key,value=value,gate=gate,reference=exact,candidate=got,quant_reference=exact_quant,quant_candidate=quant_got,kq=kq,ks=ks,vq=vq,vs=vs,pm=pm,pd=pd,po=po,quant_pm=qpm,quant_pd=qpd,quant_po=qpo)
    return {"keys":n,"seed":seed,"same_ids_reduction":reduction,"same_ids_kv_quantization":loss,"quantized_reduction":quant_reduction,"empty_partitions":int(np.sum(pd[0]==0)),"partition_maxima":pm[0].tolist()}
def selection_attention(lib,n,seed,outdir):
    with np.load(outdir/f"index-{n}.npz") as z:local_blocks,hf_blocks=z["local_ids"],z["hf_ids"]
    def token_ids(blocks):
        whole=(4*blocks[:,None]+np.arange(4,dtype=np.uint32)).reshape(-1);tail=np.arange((n//4)*4,n,dtype=np.uint32)
        return np.concatenate((whole,tail)).astype(np.uint32)
    local_ids,hf_ids=token_ids(local_blocks),token_ids(hf_blocks);dims=np.arange(512,dtype=np.float64).reshape(1,2,256)
    def kv(ids):
        phase=ids.astype(np.float64)[:,None,None]
        return (np.sin(phase*.013+dims*.071)*.3).astype(np.float32),(np.cos(phase*.021+dims*.047)*.5).astype(np.float32)
    lk,lv=kv(local_ids);hk,hv=kv(hf_ids);rng=np.random.default_rng(seed)
    query=rng.normal(0,.3,(24,256)).astype(np.float32);gate=rng.normal(0,1,(24,256)).astype(np.float32)
    external=independent_attention(query,hk,hv,gate);same_local=independent_attention(query,lk,lv,gate)
    _,_,kd=quantize(lib,lk);_,_,vd=quantize(lib,lv);same_quant=independent_attention(query,kd,vd,gate)
    got,pm,pd,po=local_attention(lib,query,kd,vd,gate);reduction=metrics(same_quant,got);reduction_gate(reduction)
    np.savez_compressed(outdir/f"sparse-output-{n}",local_token_ids=local_ids,hf_token_ids=hf_ids,query=query,gate=gate,external_selection_output=external,local_selection_output=same_local,local_quantized_output=same_quant,candidate_output=got,pm=pm,pd=pd,po=po)
    return {"visible_length":n,"attended_keys":len(local_ids),"excluded_tokens":n-len(local_ids),"selection_only_error":metrics(external,same_local),"same_selection_kv_error":metrics(same_local,same_quant),"reduction_error":reduction}
def elementary_tests(lib):
    xs=np.linspace(-87.,87.,17401,dtype=np.float32);got=np.asarray([lib.exp_gpu(float(x)) for x in xs],np.float64);reference=np.exp(xs.astype(np.float64))
    relative=float(np.max(np.abs(got/reference-1)))
    require(np.isfinite(got).all() and (got>0).all() and relative<2e-6,"polynomial exp error")
    require(lib.exp_gpu(-1000.)==lib.exp_gpu(-87.) and lib.exp_gpu(1000.)==lib.exp_gpu(87.),"exp clamp contract")
    require(math.copysign(1.,lib.rcp_gpu(-0.))<0 and math.isinf(lib.rcp_gpu(0.)),"signed-zero reciprocal")
    for x in map(np.float32,(1e-30,-1e-30,1.,3.,1e30)):
        require(np.float32(lib.rcp_gpu(float(x))).view(np.uint32)==np.float32(np.float32(1.)/x).view(np.uint32),"correctly rounded reciprocal")
    zeros=np.zeros(6144,np.float32);empty=np.zeros((0,2,256),np.float32);pm=np.zeros((24,8),np.float32);po=np.zeros((24,8,256),np.float32)
    require(lib.qsaoracle_attention(ptr(zeros),ptr(empty),ptr(empty),0,0,ptr(zeros),ptr(zeros),ptr(pm),ptr(pm),ptr(po))!=0,"empty decode accepted")
    return {"exp_max_relative_on_grid":relative,"exp_domain":"finite; clamp[-87,87]","padding":"finite zero over full partition sweep","empty_decode":"rejected","signed_zero_reciprocal":"IEEE sign retained"}
def model_evidence(args,report):
    if args.checkpoint_manifest:
        manifest=json.loads(pathlib.Path(args.checkpoint_manifest).read_text())
        require(manifest.get("repository")=="ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF" and manifest.get("revision")=="ed59f92082b1e93c0e96d60a8b11aab089b52f09","checkpoint repository/revision must be pinned")
        require(bool(manifest.get("shards")),"checkpoint shard hashes required");report["checkpoint"]=manifest
    if args.logits:
        require(args.checkpoint_manifest is not None,"paired logits need checkpoint manifest")
        with np.load(args.logits,allow_pickle=False) as z:
            rt,ct=z["reference_token_ids"],z["candidate_token_ids"];ref,got=z["reference_logits"],z["candidate_logits"]
            require(np.array_equal(rt,ct) and ref.ndim==2 and ref.shape[0]==len(rt) and len(rt)>1,"paired teacher-forced rows and identical IDs required")
            error=metrics(ref,got);require(error["max_abs"]<=LIMITS["logit_abs"] and error["normalized_rms"]<=LIMITS["logit_nrms"],"teacher-forced logit error")
            report["teacher_forced_logits"]={"tokens":len(rt),"error":error,"artifact_sha256":digest(args.logits)}
    report["model_conformance"]="PAIRED_LOGIT_GATE_PASSED; inspect source/conversion/fixture provenance" if args.logits else "NOT ESTABLISHED: independent checkpoint teacher-forced logits absent"
    if args.require_model:
        require(bool(args.logits),"--require-model needs paired logits")
        require(bool(report["checkpoint"].get("conversion_revision")),"--require-model needs pinned conversion source revision")
def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--out",default="qsa-oracle-results")
    parser.add_argument("--hf-source",help="offline pinned file; git blob verified");parser.add_argument("--full",action="store_true")
    parser.add_argument("--checkpoint-manifest");parser.add_argument("--logits",help="paired teacher-forced NPZ");parser.add_argument("--require-model",action="store_true")
    parser.add_argument("--cubin",help="record image SHA256; CPU does not execute it");args=parser.parse_args()
    outdir=pathlib.Path(args.out).resolve();outdir.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    report={"scope":"synthetic external numerical conformance; CUDA parity separate","limits":LIMITS,"python":platform.python_version(),"numpy":np.__version__,"torch":torch.__version__,"status":"RUNNING"}
    code=1
    try:
        with tempfile.TemporaryDirectory(prefix="qsa-oracle-") as tmp:
            lib=build_bridge(tmp,report);env=load_hf(args.hf_source,report)
            if args.cubin:report["assembled_image"]={"path":args.cubin,"sha256":digest(args.cubin)}
            report["cutoff"]=cutoff_tests(lib,outdir);report["full_chain_cutoff"]=full_chain_cutoff(lib,env,outdir)
            report["index_domain"]=index_domain_tests(lib);report["bf16_boundary"]=hf_bf16_probe(env,outdir)
            report["quantization"]=quant_tests(lib);report["elementary"]=elementary_tests(lib)
            lengths=[1,2047,2048,2049,2050,2051,2052,3100,3101,3102,3103,8000,8001,8002,8003]
            if args.full:lengths+=[65000+i for i in range(4)];lengths+=[262000+i for i in range(4)]
            report["indexer"],report["sparse_attention"]=[],[]
            for n in lengths:
                print(f"indexer/sparse fixture visible={n}",flush=True)
                report["indexer"].append(index_fixture(lib,env,n,100000+n,outdir))
                report["sparse_attention"].append(selection_attention(lib,n,200000+n,outdir))
            ns=[1,2,3,4,1023,1024,1025,1026,1027,2047,2048,2049,2050,2051,2052]
            if args.full:ns +=[8191,8192,8193]
            report["attention"]=[]
            for n in ns:
                print(f"same-ID attention fixture keys={n}",flush=True)
                report["attention"].append(attention_fixture(lib,n,300000+n,outdir))
            model_evidence(args,report);report["status"],code="PASS_SYNTHETIC",0
    except Exception as exc:
        report["status"]="FAIL";report["error"]=f"{type(exc).__name__}: {exc}";print(report["error"],file=sys.stderr)
    (outdir/"report.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(f"{report['status']}: {outdir/'report.json'}",flush=True);return code
if __name__=="__main__":sys.exit(main())
