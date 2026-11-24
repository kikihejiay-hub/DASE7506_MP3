"""Run the unchanged official predictor/scorer and measure process resources."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import psutil

root=Path(__file__).resolve().parent.parent
parser=argparse.ArgumentParser()
parser.add_argument('--split',choices=['train','validation','test'],default='validation')
parser.add_argument('--tag',default='lora')
parser.add_argument('--method',default='submission/model_agent.py')
parser.add_argument('--workers',type=int,choices=[1,2,4],default=1)
parser.add_argument('--checkpoint',type=Path)
args=parser.parse_args()
os.environ['PYTHONUTF8']='1'
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['OMP_NUM_THREADS']='4'
os.environ['MKL_NUM_THREADS']='4'
os.environ['MP3_MODEL_LOGDIR']=str(root/'outputs'/f'{args.tag}-{args.split}-logs')
output=root/'outputs'/f'{args.tag}-{args.split}.json'
started=time.perf_counter()
command=[sys.executable,'-m','mp3_sql.predict','--split',args.split,'--workers',str(args.workers),
         '--method',args.method,'--output',str(output)]
if args.checkpoint:
    command.extend(['--checkpoint',str(args.checkpoint)])
process=subprocess.Popen(command,cwd=root)
seen_cpu={};peak_rss=0;peak_main_rss=0
while process.poll() is None:
    try:
        main=psutil.Process(process.pid)
        group=[main]+main.children(recursive=True)
        rss=0
        for p in group:
            try:
                info=p.memory_info();rss+=info.rss
                cpu=p.cpu_times();seen_cpu[p.pid]=cpu.user+cpu.system
                if p.pid==main.pid:peak_main_rss=max(peak_main_rss,getattr(info,'peak_wset',info.rss))
            except (psutil.NoSuchProcess,psutil.AccessDenied):pass
        peak_rss=max(peak_rss,rss)
    except psutil.NoSuchProcess:pass
    time.sleep(.2)
resources={'method':args.method,'workers':args.workers,'checkpoint':str(args.checkpoint) if args.checkpoint else None,
           'wall_seconds':time.perf_counter()-started,'sampled_cpu_seconds':sum(seen_cpu.values()),
           'sampled_peak_group_rss':peak_rss,'main_peak_rss':peak_main_rss,
           'sampling_interval_seconds':.2,'predictor_exit_code':process.returncode}
(root/'outputs'/f'{args.tag}-{args.split}-resources.json').write_text(json.dumps(resources,indent=2)+'\n',encoding='utf-8')
if process.returncode:raise SystemExit(process.returncode)
subprocess.run([sys.executable,'-m','mp3_sql.score','--split',args.split,'--predictions',str(output),
                '--output',str(root/'outputs'/f'{args.tag}-{args.split}-score.json')],cwd=root,check=True)
print(json.dumps(resources),flush=True)
