import json, os, sys, time
from pathlib import Path

ROLE = os.environ.get('ROLE','BIOLOGY_GENERALIST')
MODEL = os.environ.get('MODEL','huggingface-projects/llama-3.2-3B-Instruct')
TASK = os.environ.get('TASK','Analyze the assigned biology/health question with strict evidence discipline.')
OUT = Path(os.environ.get('OUT','out'))
OUT.mkdir(parents=True, exist_ok=True)

SYSTEM = '''You are one role in CEREBRON Farm 12 Biology Health.
Rules: REALITY>COHERENCE; CLAIM<=EVIDENCE; OBSERVATION!=CAUSATION; ASSOCIATION!=CLINICAL BENEFIT; IN_VITRO!=IN_VIVO; ANIMAL_MODEL!=HUMAN_EVIDENCE; SIMULATION!=EXPERIMENT; UNKNOWN remains UNKNOWN. Distinguish established, supported, uncertain, speculative. Do not invent citations, trials, guidelines or measurements. Do not provide personalized diagnosis or treatment. Focus on evidence review, mechanisms, uncertainty, study design, replication and falsification.'''

prompt = f"{SYSTEM}\n\nROLE={ROLE}\nTASK={TASK}\nReturn concise sections: EVIDENCE, REASONING, LIMITATIONS, FALSIFICATION, VERDICT."
result = {'role': ROLE, 'model': MODEL, 'task': TASK, 'status': 'failed', 'text': '', 'ts': time.time()}
try:
    from gradio_client import Client
    c = Client(MODEL)
    try:
        text = c.predict(message=prompt, api_name='/chat')
    except Exception:
        text = c.predict(prompt, api_name='/predict')
    result['text'] = str(text)
    result['status'] = 'success'
except Exception as e:
    result['error'] = repr(e)

(OUT / f"{ROLE}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
print(json.dumps({'role': ROLE, 'status': result['status']}))
