"""Serialized handoff: frozen configuration screen, then controlled routing trials."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'paper'))
from config_search.run_screen import process_identity


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait-pid',type=int)
    parser.add_argument('--pause-idle-orchestrator',type=int,action='append',default=[])
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args()
    out=ROOT/'runs/config_search/research_chain_v1';out.mkdir(parents=True,exist_ok=True)
    lock=(out/'chain.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    screen=ROOT/'runs/config_search/trl_strategy_v2'
    jobs=[dict(name='configuration_screen',command=[sys.executable,str(ROOT/'paper/config_search/run_screen.py'),
          '--out',str(screen),'--steps','256','--extend-steps','1024'],result=str(screen/'summary.json'))]
    for arm in ('vanilla','quantum','classical'):
        for routing in (['adaptive'] if arm=='vanilla' else ['adaptive','static','permuted']):
            dest=ROOT/f'runs/config_search/long_routing_v1/{arm}_{routing}_s1337'
            jobs.append(dict(name=f'{arm}_{routing}',command=[sys.executable,str(ROOT/'paper/config_search/routing_trial.py'),
                        '--arm',arm,'--routing',routing,'--out',str(dest),'--steps','1024','--seed','1337'],
                        result=str(dest/'result.json')))
    files=[ROOT/'28-long-routing-comparison.md',ROOT/'paper/config_search/routing_trial.py',
           ROOT/'paper/config_search/signal_router.py',ROOT/'paper/config_search/run_screen.py']
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    manifest=dict(jobs=jobs,source_hashes=hashes,automatic_promotion=False)
    previous=out/'manifest.json'
    if previous.exists() and json.loads(previous.read_text())!=manifest:
        raise ValueError('Existing chain manifest differs; use a separately registered chain')
    previous.write_text(json.dumps(manifest,indent=2)+'\n')
    if args.prepare_only:
        print('Prepared one existing screen plus seven 1024-step routing trials');return
    paused=[];child=None
    def log(message):
        with (out/'status').open('a') as f:f.write(time.strftime('%Y-%m-%dT%H:%M:%S%z')+' '+message+'\n')
        print(message,flush=True)
    def stop(*_):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        for pid in args.pause_idle_orchestrator:
            command=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
            if not any(n in command for n in ('physiq_orchestrator.sh','wmb_orchestrator.sh')):
                raise ValueError('Unexpected orchestrator')
            children=Path(f'/proc/{pid}/task/{pid}/children').read_text().split()
            if any(Path(f'/proc/{p}/comm').read_text().strip()!='sleep' for p in children):
                raise ValueError('Orchestrator is not idle')
            identity=process_identity(pid);os.kill(pid,signal.SIGSTOP);paused.append((pid,identity))
            log(f'HELD_IDLE_ORCHESTRATOR pid={pid} identity={identity}')
        if args.wait_pid:
            identity=process_identity(args.wait_pid);log(f'WAIT_EXISTING pid={args.wait_pid} identity={identity}')
            while identity is not None and process_identity(args.wait_pid)==identity:time.sleep(30)
        for job in jobs:
            for name,digest in hashes.items():
                if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=digest:
                    raise ValueError(f'Queued source changed: {name}')
            while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip():
                log('WAIT_GPU');time.sleep(30)
            log('START '+job['name'])
            with (out/(job['name']+'.log')).open('a') as stream:
                child=subprocess.Popen(job['command'],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
                rc=child.wait();child=None
            if rc:raise RuntimeError(f"{job['name']} failed with exit {rc}")
            result=json.loads(Path(job['result']).read_text())
            if job['name']=='configuration_screen':
                if len(result)!=22 or any(r['result']['status']!='complete' for r in result):
                    raise ValueError('Incomplete configuration screen')
            elif result['status']!='complete' or result['step']!=1024 or result['curves'][-1]['step']!=1024:
                raise ValueError('Incomplete routing trial')
            log('COMPLETED '+job['name'])
        log('COMPLETE_EXPLORATORY_CHAIN_NO_PROMOTION')
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try:child.wait(timeout=30)
            except subprocess.TimeoutExpired:child.kill();child.wait()
        for pid,identity in paused:
            if process_identity(pid)==identity:
                os.kill(pid,signal.SIGCONT);log(f'RESUMED_ORCHESTRATOR pid={pid}')


if __name__=='__main__':main()
