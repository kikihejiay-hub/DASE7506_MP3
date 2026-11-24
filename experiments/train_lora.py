"""Train only Qwen3-0.6B LoRA A/B matrices on the official TRAIN split."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

root=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(root))
from mp3_sql.common import load_bank,read_jsonl
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
from peft import LoraConfig,get_peft_model
import psutil
import importlib.util
spec=importlib.util.spec_from_file_location('lora_prompt',root/'submission/model_agent.py')
prompt_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(prompt_module)

parser=argparse.ArgumentParser()
parser.add_argument('--steps',type=int,default=160)
parser.add_argument('--max-length',type=int,default=640)
parser.add_argument('--output',type=Path,default=root/'adapters/qwen-sql')
parser.add_argument('--seed',type=int,default=7506)
args=parser.parse_args()
torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.manual_seed(args.seed)
random.seed(args.seed)
tokenizer=AutoTokenizer.from_pretrained(os.environ['MP3_BASE_MODEL'],local_files_only=True,trust_remote_code=False)
base=AutoModelForCausalLM.from_pretrained(os.environ['MP3_BASE_MODEL'],local_files_only=True,trust_remote_code=False,
                                        use_safetensors=True,torch_dtype=torch.float32)
base.config.use_cache=False
model=get_peft_model(base,LoraConfig(r=8,lora_alpha=16,lora_dropout=.05,bias='none',
                                    target_modules=['q_proj','v_proj'],task_type='CAUSAL_LM'))
trainable=[(name,p) for name,p in model.named_parameters() if p.requires_grad]
assert trainable and all('lora_A.' in name or 'lora_B.' in name for name,p in trainable)
assert sum(p.numel() for p in model.parameters())<=1_000_000_000
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
model.enable_input_require_grads()
optimizer=torch.optim.AdamW([p for _,p in trainable],lr=2e-4,weight_decay=0.)
rows,manifest,folder=load_bank('train')
memory={r['id']:r for r in read_jsonl(folder/'train_memory.jsonl')}
trajectories={p['id']:p for p in json.loads((root/'outputs/train-candidates.json').read_text(encoding='utf-8'))['predictions']}
examples=[]
for row in rows:
    schema=manifest['databases'][row['db_id']]['schema']
    history=trajectories[row['id']]['trace']
    hints=next((e['observation'] for e in history if e['tool']=='repair_hint'),[])
    observation={'task':row,'issue_sql':row['issue_sql'],'example':row['example'],
                 'history':[e for e in history if e['tool']=='schema']+history[:1]}
    prompt=prompt_module.build_prompt(tokenizer,observation,hints[0] if hints else row['issue_sql'])
    gold=memory[row['id']]['gold_sql']
    target=tokenizer(gold+tokenizer.eos_token,add_special_tokens=False)['input_ids']
    if len(target)>args.max_length//2:continue
    prefix=tokenizer(prompt,add_special_tokens=False)['input_ids']
    # Keep question at the head and the generation header at the tail.
    room=args.max_length-len(target)
    if len(prefix)>room:prefix=prefix[:room-32]+prefix[-32:]
    ids=prefix+target
    examples.append((row['id'],ids,[-100]*len(prefix)+target))
random.shuffle(examples)
args.output.mkdir(parents=True,exist_ok=True)
records=root/'experiments/lora-training-log.jsonl'
start=time.perf_counter();cpu_start=time.process_time()
model.train()
with records.open('w',encoding='utf-8') as log:
    for step in range(args.steps):
        task_id,ids,labels=examples[step%len(examples)]
        before=time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        batch=torch.tensor([ids]);target=torch.tensor([labels])
        loss=model(input_ids=batch,attention_mask=torch.ones_like(batch),labels=target).loss
        loss.backward();torch.nn.utils.clip_grad_norm_([p for _,p in trainable],1.)
        optimizer.step()
        record={'step':step+1,'task_id':task_id,'loss':float(loss.detach()),'tokens':len(ids),
                'seconds':time.perf_counter()-before,'elapsed':time.perf_counter()-start,
                'cpu_seconds':time.process_time()-cpu_start,'rss':psutil.Process().memory_info().rss}
        log.write(json.dumps(record)+'\n');log.flush();print(json.dumps(record),flush=True)
    model.save_pretrained(args.output,safe_serialization=True)
    # PEFT writes a generated README; adapter export must contain exactly two files.
    provenance={'base_repo':'Qwen/Qwen3-0.6B','revision':'c1899de289a04d12100db370d81485cdf75e47ca',
                'split':'train','bank_sha256':hashlib.sha256((folder/'train.jsonl').read_bytes()).hexdigest(),
                'seed':args.seed,'steps':args.steps,'max_length':args.max_length,'rank':8,'alpha':16,
                'learning_rate':2e-4,'target_modules':['q_proj','v_proj'],
                'trainable_parameters':sum(p.numel() for _,p in trainable),
                'actual_parameters':sum(p.numel() for p in model.parameters()),
                'examples':len(examples),'eligible_training_ids':[e[0] for e in examples],
                'training_ids':[examples[i%len(examples)][0] for i in range(args.steps)],
                'seconds':time.perf_counter()-start,'cpu_seconds':time.process_time()-cpu_start}
    (root/'experiments/lora-provenance.json').write_text(json.dumps(provenance,indent=2)+'\n',encoding='utf-8')
