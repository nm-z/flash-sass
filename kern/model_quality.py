#!/usr/bin/env python3
"""Freeze and compare teacher-forced model logit captures (NumPy required).
Raw IDs/vocabulary IDs: little-endian uint32; logits: little-endian float32
[positions,vocabulary]. Captures must have identical teacher-forced IDs.
No CPU instruction twin is automatically an independent reference.
"""
import argparse
import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import sys
import numpy as np

SCHEMA='flash-sass.frozen-logits.v1'
def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def dump_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')
def load_json(path):
    with Path(path).open() as f:return json.load(f)
def immutable(value):
    return isinstance(value,str) and bool(re.fullmatch(r'(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})',value))
def provenance_missing(meta):
    missing=[]
    if meta.get('teacher_forced') is not True:missing.append('teacher_forced=true')
    base=meta.get('base_model',{})
    if not base.get('id'):missing.append('base_model.id')
    if not immutable(base.get('revision')):missing.append('base_model.revision (immutable hash)')
    tok=meta.get('tokenizer',{})
    if not re.fullmatch('[0-9a-fA-F]{64}',str(tok.get('sha256',''))):missing.append('tokenizer.sha256 (canonical tokenizer identity)')
    ck=meta.get('checkpoint',{}).get('files',[])
    if not ck:missing.append('checkpoint.files')
    for i,f in enumerate(ck):
        if not re.fullmatch('[0-9a-fA-F]{64}',str(f.get('sha256',''))):missing.append(f'checkpoint.files[{i}].sha256')
        if not f.get('path') and not f.get('source_uri'):missing.append(f'checkpoint.files[{i}].path/source_uri')
    impl=meta.get('implementation',{})
    if not impl.get('name'):missing.append('implementation.name')
    if not immutable(impl.get('revision')):missing.append('implementation.revision (immutable hash)')
    if not meta.get('command'):missing.append('command')
    return missing

def freeze(args):
    ids=np.fromfile(args.ids,dtype='<u4')
    if not len(ids) or Path(args.ids).stat().st_size!=len(ids)*4:raise ValueError('IDs must be nonempty raw little-endian u32')
    V=args.vocab_size
    if V<1 or Path(args.logits).stat().st_size!=len(ids)*V*4:raise ValueError('logits size must be len(ids)*vocab_size*4')
    vocab=np.fromfile(args.vocab_ids,dtype='<u4') if args.vocab_ids else np.arange(V,dtype='<u4')
    if len(vocab)!=V or len(np.unique(vocab))!=V:raise ValueError('vocabulary IDs must be unique and match logit columns')
    if args.vocab_ids and Path(args.vocab_ids).stat().st_size!=V*4:raise ValueError('invalid vocabulary ID bytes')
    meta=load_json(args.metadata) if args.metadata else {}
    if not isinstance(meta,dict):raise ValueError('metadata must be a JSON object')
    if meta.get('kind') not in (None,args.kind):raise ValueError('metadata kind contradicts --kind')
    # Remote hashes are producer declarations, not local verification.
    for item in meta.get('checkpoint',{}).get('files',[]):
        path=Path(item['path']) if item.get('path') else None
        item['verified_local']=bool(path and path.is_file())
        if item['verified_local']:
            actual=digest(path)
            if item.get('sha256') and item['sha256'].lower()!=actual:raise ValueError(f'checkpoint hash mismatch: {path}')
            item['sha256']=actual;item['bytes']=path.stat().st_size
    missing=provenance_missing(meta)
    raw=None
    if args.raw_info:
        raw=load_json(args.raw_info)
        if raw.get('teacher_forced') is not True or raw.get('positions')!=len(ids) or raw.get('vocab_size')!=V:raise ValueError('raw capture metadata does not match frozen dimensions/teacher forcing')
    out=Path(args.out)
    out.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(args.ids,out/'ids.u32')
    shutil.copyfile(args.logits,out/'logits.f32')
    vocab.astype('<u4').tofile(out/'vocab_ids.u32')
    if raw is not None:dump_json(out/'raw_capture_info.json',raw)
    manifest={'schema':SCHEMA,'kind':args.kind,'positions':int(len(ids)),'vocab_size':V,'dtype':'<f4','byte_order':'little','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'metadata':meta,'missing_provenance':missing,'provenance_note':'Producer declarations and local file identities; not execution attestation.','artifacts':{}}
    for leaf in ('ids.u32','logits.f32','vocab_ids.u32'):
        f=out/leaf;manifest['artifacts'][leaf]={'bytes':f.stat().st_size,'sha256':digest(f)}
    if raw is not None:manifest['artifacts']['raw_capture_info.json']={'bytes':(out/'raw_capture_info.json').stat().st_size,'sha256':digest(out/'raw_capture_info.json')}
    dump_json(out/'manifest.json',manifest)
    print(json.dumps({'capture':str(out),'manifest_sha256':digest(out/'manifest.json'),'missing_provenance':missing,'quality_validation':False},indent=2))
    return 0

def read_capture(directory,kind):
    directory=Path(directory);m=load_json(directory/'manifest.json')
    if m.get('schema')!=SCHEMA or m.get('kind')!=kind:raise ValueError(f'{directory}: wrong capture schema/kind')
    if m.get('dtype')!='<f4' or m.get('byte_order')!='little':raise ValueError('capture must declare little-endian FP32')
    T,V=m.get('positions'),m.get('vocab_size')
    if not isinstance(T,int) or not isinstance(V,int) or T<1 or V<1:raise ValueError('invalid capture dimensions')
    for leaf,want in m['artifacts'].items():
        if Path(leaf).name!=leaf:raise ValueError('artifact must be a plain filename')
        f=directory/leaf
        if f.stat().st_size!=want['bytes'] or digest(f)!=want['sha256']:raise ValueError(f'frozen artifact changed: {f}')
    for leaf in ('ids.u32','logits.f32','vocab_ids.u32'):
        if leaf not in m['artifacts']:raise ValueError(f'missing frozen artifact {leaf}')
    if (directory/'ids.u32').stat().st_size!=T*4 or (directory/'logits.f32').stat().st_size!=T*V*4 or (directory/'vocab_ids.u32').stat().st_size!=V*4:raise ValueError('capture shape/byte mismatch')
    ids=np.fromfile(directory/'ids.u32',dtype='<u4');vocab=np.fromfile(directory/'vocab_ids.u32',dtype='<u4')
    if len(np.unique(vocab))!=V:raise ValueError('duplicate vocabulary IDs')
    logits=np.memmap(directory/'logits.f32',dtype='<f4',mode='r',shape=(T,V))
    return m,ids,vocab,logits

def log_softmax(x):
    shifted=x-float(x.max());return shifted-math.log(float(np.exp(shifted).sum(dtype=np.float64)))
def top_indices(values,k):
    cutoff=np.partition(values,len(values)-k)[len(values)-k]
    above=np.flatnonzero(values>cutoff)
    tied=np.flatnonzero(values==cutoff)[:k-len(above)]
    return np.concatenate((above,tied))
def finite_exp(x):
    try:return math.exp(x)
    except OverflowError:return None

def compare(args):
    cm,ci,cv,cl=read_capture(args.candidate,'candidate')
    rm,ri,rv,rl=read_capture(args.reference,'reference')
    if not np.array_equal(ci,ri):raise ValueError('captures do not have identical teacher-forced token IDs')
    co=np.argsort(cv);ro=np.argsort(rv);vocab=cv[co]
    if not np.array_equal(vocab,rv[ro]):raise ValueError('vocabulary ID sets differ; explicit compatible vocabulary captures are required')
    if not np.isin(ci,vocab).all():raise ValueError('input token IDs absent from vocabulary')
    T,V=cl.shape
    if rl.shape!=(T,V):raise ValueError('reference/candidate dimensions differ')
    cmeta=cm.get('metadata',{});rmeta=rm.get('metadata',{})
    missing={'candidate':provenance_missing(cmeta),'reference':provenance_missing(rmeta)}
    if rmeta.get('implementation',{}).get('independent_reference') is not True:missing['reference'].append('implementation.independent_reference=true')
    for parent,field in (('base_model','id'),('base_model','revision'),('tokenizer','sha256')):
        a=cmeta.get(parent,{}).get(field);b=rmeta.get(parent,{}).get(field)
        if a and b and a!=b:raise ValueError(f'incompatible {parent}.{field}')
    sq=abs_sum=ref_sq=center_sq=0.;max_abs=max_prob=0.;kl_sum=top1=overlap=cos_sum=0.;cos_count=0
    ref_nll=[];can_nll=[];ref_rank=[];can_rank=[];detail=[]
    K=min(args.top_k,V)
    for t in range(T):
        c=np.asarray(cl[t,co],dtype=np.float64);r=np.asarray(rl[t,ro],dtype=np.float64)
        if not np.isfinite(c).all() or not np.isfinite(r).all():
            report={'status':'failed_numerical_contract','position':t,'candidate_nonfinite':int((~np.isfinite(c)).sum()),'reference_nonfinite':int((~np.isfinite(r)).sum()),'quality_pass':False,'missing_provenance':missing}
            if args.out:dump_json(args.out,report)
            print(json.dumps(report,indent=2));return 3
        d=c-r;sq+=float(d@d);abs_sum+=float(np.abs(d).sum());ref_sq+=float(r@r);max_abs=max(max_abs,float(np.abs(d).max()))
        centered=(c-float(c.mean()))-(r-float(r.mean()));center_sq+=float(centered@centered)
        norm=math.sqrt(float(c@c)*float(r@r))
        if norm:cos_sum+=float(c@r)/norm;cos_count+=1
        lpc=log_softmax(c);lpr=log_softmax(r);pc=np.exp(lpc);pr=np.exp(lpr)
        kl=max(0.,float(pr@(lpr-lpc)));kl_sum+=kl;max_prob=max(max_prob,float(np.abs(pc-pr).max()))
        # Sorted token IDs give deterministic low-ID ties.
        cp=int(np.argmax(c));rp=int(np.argmax(r));same=cp==rp;top1+=same
        ctop=top_indices(c,K);rtop=top_indices(r,K);ov=len(set(ctop.tolist())&set(rtop.tolist()))/K;overlap+=ov
        row={'position':t,'input_id':int(ci[t]),'candidate_top1':int(vocab[cp]),'reference_top1':int(vocab[rp]),'top1_agrees':bool(same),'topk_overlap':ov,'reference_to_candidate_kl':kl}
        if t+1<T:
            target=int(ci[t+1]);j=int(np.searchsorted(vocab,target))
            if j==V or int(vocab[j])!=target:raise ValueError(f'next-token label {target} absent from vocabulary')
            rn=-float(lpr[j]);cn=-float(lpc[j]);ref_nll.append(rn);can_nll.append(cn)
            rr=1+int(np.count_nonzero(r>r[j]))+int(np.count_nonzero((r==r[j])&(vocab<target)))
            cr=1+int(np.count_nonzero(c>c[j]))+int(np.count_nonzero((c==c[j])&(vocab<target)))
            ref_rank.append(rr);can_rank.append(cr);row.update({'next_id':target,'reference_nll':rn,'candidate_nll':cn,'nll_increase':cn-rn,'reference_target_rank':rr,'candidate_target_rank':cr})
        detail.append(row)
    metrics={'positions':T,'vocabulary':V,'logit_rmse':math.sqrt(sq/(T*V)),'mean_absolute_logit_error':abs_sum/(T*V),'maximum_absolute_logit_error':max_abs,'relative_l2_logit_error':math.sqrt(sq/ref_sq) if ref_sq else None,'centered_logit_rmse':math.sqrt(center_sq/(T*V)),'mean_cosine_similarity':cos_sum/cos_count if cos_count else None,'mean_reference_to_candidate_kl':kl_sum/T,'maximum_absolute_probability_error':max_prob,'top1_agreement':top1/T,'topk':K,'mean_topk_overlap':overlap/T,'teacher_forced_labels':len(ref_nll)}
    if ref_nll:
        rn=sum(ref_nll)/len(ref_nll);cn=sum(can_nll)/len(can_nll)
        rp=finite_exp(rn);cp=finite_exp(cn)
        metrics.update({'reference_mean_nll':rn,'candidate_mean_nll':cn,'mean_nll_increase':cn-rn,'reference_perplexity':rp,'candidate_perplexity':cp,'perplexity_overflow':rp is None or cp is None,'reference_mean_target_rank':sum(ref_rank)/len(ref_rank),'candidate_mean_target_rank':sum(can_rank)/len(can_rank)})
    criteria=[]
    for flag,metric,sense in [('max_mean_kl','mean_reference_to_candidate_kl','max'),('max_rmse','logit_rmse','max'),('max_abs_logit','maximum_absolute_logit_error','max'),('min_top1_agreement','top1_agreement','min'),('max_mean_nll_increase','mean_nll_increase','max')]:
        limit=getattr(args,flag)
        if limit is not None:
            value=metrics.get(metric)
            ok=value is not None and (value<=limit if sense=='max' else value>=limit)
            criteria.append({'metric':metric,'sense':sense,'limit':limit,'actual':value,'passes':bool(ok)})
    incomplete=bool(missing['candidate'] or missing['reference'])
    failed=any(not c['passes'] for c in criteria)
    status='diagnostic_missing_provenance' if incomplete else 'failed_declared_limits' if failed else 'passed_declared_limits_unattested' if criteria else 'metrics_only'
    report={'schema':'flash-sass.logit-comparison.v1','status':status,'quality_pass':False if failed else None,'declared_limits_pass':None if not criteria else not failed,'independent_checkpoint_evidence':False,'provenance_status':'producer_declared_unattested','scope':'Numerical comparison of these supplied teacher-forced rows only; no claim about other prompts/tasks.','provenance_note':'Capture-producer declarations and hashed files are not execution attestation.','missing_provenance':missing,'candidate_manifest_sha256':digest(Path(args.candidate)/'manifest.json'),'reference_manifest_sha256':digest(Path(args.reference)/'manifest.json'),'metrics':metrics,'declared_criteria':criteria,'positions':detail}
    if args.out:dump_json(args.out,report)
    print(json.dumps(report,indent=2,allow_nan=False))
    if incomplete and not args.diagnostic:return 2
    return 3 if failed else 0

def nonnegative(text):
    value=float(text)
    if not math.isfinite(value) or value<0:raise argparse.ArgumentTypeError('must be finite and nonnegative')
    return value

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    f=sub.add_parser('freeze');f.add_argument('--kind',required=True,choices=['candidate','reference']);f.add_argument('--ids',required=True,help='raw little-endian u32 token IDs');f.add_argument('--logits',required=True,help='raw little-endian FP32 rows');f.add_argument('--vocab-size',required=True,type=int);f.add_argument('--vocab-ids',help='optional raw u32 token ID per logit column');f.add_argument('--metadata',help='producer JSON provenance; missing fields remain explicitly missing');f.add_argument('--raw-info',help='optional model_capture capture_info.json');f.add_argument('--out',required=True);f.set_defaults(fn=freeze)
    c=sub.add_parser('compare');c.add_argument('--candidate',required=True);c.add_argument('--reference',required=True);c.add_argument('--out');c.add_argument('--top-k',type=int,default=10);c.add_argument('--diagnostic',action='store_true',help='allow metrics with missing producer declarations; never establishes checkpoint quality')
    for flag in ['max-mean-kl','max-rmse','max-abs-logit','min-top1-agreement','max-mean-nll-increase']:c.add_argument('--'+flag,type=nonnegative)
    c.set_defaults(fn=compare)
    args=p.parse_args()
    if getattr(args,'top_k',1)<1:p.error('--top-k must be positive')
    if getattr(args,'min_top1_agreement',None) is not None and args.min_top1_agreement>1:p.error('--min-top1-agreement must be <=1')
    try:return args.fn(args)
    except (OSError,ValueError,KeyError,TypeError,AttributeError) as exc:
        print(json.dumps({'status':'invalid_capture','error':str(exc),'quality_pass':False}),file=sys.stderr);return 2
if __name__=='__main__':raise SystemExit(main())
