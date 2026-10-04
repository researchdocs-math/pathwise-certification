"""Replay archived evidence or regenerate experiments in an isolated run directory."""
from pathlib import Path
import argparse,subprocess,sys,shutil,json,hashlib
ROOT=Path(__file__).resolve().parents[1]
def copy_inputs(dest,archived):
 dest.mkdir(parents=True,exist_ok=True)
 for name in ['src','configs','data']:
  shutil.copytree(ROOT/name,dest/name,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
 (dest/'results').mkdir(exist_ok=True)
 if archived:
  shutil.copytree(ROOT/'results/banks',dest/'results/banks',dirs_exist_ok=True)
  for name in ['simulation_trials.csv.gz','simulation_config.json','environment.json']:
   shutil.copy2(ROOT/'results'/name,dest/'results'/name)
def run(dest,script,*args):
 subprocess.run([sys.executable,str(dest/'src'/script),*args],cwd=dest,check=True)
def paired(dest,script,device):
 handles=[];jobs=[]
 try:
  for label,families in [('cpu',['logistic','boosting']),('neural',['mlp'])]:
   h=(dest/f'{script}_{label}.log').open('w');handles.append(h)
   jobs.append(subprocess.Popen([sys.executable,str(dest/'src'/script),'--families',*families,'--device',device],cwd=dest,stdout=h,stderr=subprocess.STDOUT))
  codes=[p.wait() for p in jobs]
  if any(codes):raise RuntimeError(f'{script} failed: {codes}; inspect the run logs.')
 finally:
  for h in handles:h.close()
def verify(dest):
 import pandas as pd
 count=0
 for ref in sorted((ROOT/'results/analysis').glob('*')):
  if ref.name.endswith(('.csv','.csv.gz')):
   actual=dest/'results/analysis'/ref.name
   pd.testing.assert_frame_equal(pd.read_csv(ref),pd.read_csv(actual),check_exact=False,rtol=1e-9,atol=1e-11)
   count+=1
 print(f'Verified {count} numerical analysis files against the reference results.')
def main():
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['replay','full'],default='replay');p.add_argument('--workdir',type=Path);p.add_argument('--device',choices=['mps','cpu'],default='mps');p.add_argument('--verify',action='store_true');a=p.parse_args()
 dest=(a.workdir or ROOT/'reruns'/a.mode).resolve()
 if dest==ROOT:raise ValueError('Choose a separate work directory.')
 if dest.exists():raise FileExistsError(f'Choose a new work directory: {dest}')
 copy_inputs(dest,a.mode=='replay')
 if a.mode=='full':
  paired(dest,'run_benchmarks.py',a.device)
  paired(dest,'run_coordinate_paths.py',a.device)
  run(dest,'run_simulations.py','--repetitions','5000','--workers','4')
 run(dest,'analyse.py');run(dest,'additional_analysis.py')
 if a.verify:verify(dest)
 print('Completed:',dest)
if __name__=='__main__':main()
