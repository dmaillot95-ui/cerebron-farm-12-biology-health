import json, os, time, subprocess
from pathlib import Path
ROLE=os.environ.get('ROLE','BIOLOGY_GENERALIST'); MODEL=os.environ.get('MODEL','huggingface-projects/llama-3.2-3B-Instruct'); TASK=os.environ.get('TASK','Analyze biology/health evidence.'); OUT=Path(os.environ.get('OUT','out')); OUT.mkdir(parents=True,exist_ok=True)
SYSTEM='''You are one role in CEREBRON Farm 12 Biology Health. REALITY>COHERENCE; CLAIM<=EVIDENCE; OBSERVATION!=CAUSATION; ASSOCIATION!=CLINICAL BENEFIT; IN_VITRO!=IN_VIVO; ANIMAL_MODEL!=HUMAN_EVIDENCE; SIMULATION!=EXPERIMENT; UNKNOWN remains UNKNOWN. Distinguish established, supported, uncertain, speculative. Do not invent citations, trials, guidelines or measurements. Do not provide personalized diagnosis or treatment.'''
prompt=f"{SYSTEM}\nROLE={ROLE}\nTASK={TASK}\nReturn EVIDENCE, REASONING, LIMITATIONS, FALSIFICATION, VERDICT."
def run(cmd,t=240): return subprocess.run(cmd,capture_output=True,text=True,timeout=t)
def payload(spec,prompt):
 d={}; used=False
 for p in spec.get('parameters',[]):
  n=p.get('name',''); l=n.lower(); req=bool(p.get('required')); default=p.get('default'); typ=(p.get('type') or {}).get('type')
  if l in {'message','prompt','text','query','input','instruction','user_message'}: d[n]=prompt; used=True
  elif l in {'chat_history','history','messages'}: d[n]=[]
  elif l in {'max_new_tokens','max_tokens','maximum_new_tokens'}: d[n]=700
  elif l=='temperature': d[n]=0.1
  elif l=='top_p': d[n]=0.9
  elif req and default is None:
   if typ=='string' and not used: d[n]=prompt; used=True
   else: return None
 return d if used else None
def invoke(space,prompt):
 i=run(['hf-gradio','info',space],120)
 if i.returncode: return False,'',{'stage':'info','error':(i.stderr or i.stdout)[-1200:]}
 try: api=json.loads(i.stdout)
 except Exception as e: return False,'',{'stage':'decode','error':repr(e)}
 pref=['/generate','/chat','/predict','/respond','/infer','/run']; eps=list(api.items()); eps.sort(key=lambda kv:(pref.index(kv[0]) if kv[0] in pref else 99,kv[0]))
 errs=[]
 for ep,spec in eps:
  pl=payload(spec,prompt)
  if pl is None: continue
  r=run(['hf-gradio','predict',space,ep,json.dumps(pl,ensure_ascii=False)],240)
  if r.returncode==0 and (r.stdout or '').strip(): return True,r.stdout.strip(),{'endpoint':ep}
  errs.append((r.stderr or r.stdout)[-700:])
 return False,'',{'stage':'predict','error':' | '.join(errs[-3:])}
ok,text,meta=invoke(MODEL,prompt)
result={'role':ROLE,'model':MODEL,'task':TASK,'status':'success' if ok else 'failed','epistemic_status':'UNREVIEWED_EXTERNAL_AGENT_OUTPUT' if ok else 'EXTERNAL_INFERENCE_FAILED','text':text if ok else '','meta':meta,'ts':time.time()}
(OUT/f'{ROLE}.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)); print(json.dumps({'role':ROLE,'status':result['status']}))
