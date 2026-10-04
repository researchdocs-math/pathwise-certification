"""Download the eight public UCI benchmark tables and retain provenance."""
from pathlib import Path
import urllib.request,zipfile,io,json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
SOURCES={
'breast':(17,'breast+cancer+wisconsin+diagnostic','wdbc.data'),
'ionosphere':(52,'ionosphere','ionosphere.data'),
'sonar':(151,'connectionist+bench+sonar+mines+vs+rocks','sonar.all-data'),
'banknote':(267,'banknote+authentication','data_banknote_authentication.txt'),
'spambase':(94,'spambase','spambase.data'),
'wine_red':(186,'wine+quality','winequality-red.csv'),
'wine_white':(186,'wine+quality','winequality-white.csv'),
'magic':(159,'magic+gamma+telescope','magic04.data')}
def main():
 out=ROOT/'data';out.mkdir(exist_ok=True);records={}
 cache={}
 for name,(ident,slug,member) in SOURCES.items():
  url=f'https://archive.ics.uci.edu/static/public/{ident}/{slug}.zip'
  if ident not in cache:
   raw=urllib.request.urlopen(url,timeout=90).read();cache[ident]=raw
   (out/f'uci_{ident}.zip').write_bytes(raw)
  raw=cache[ident];z=zipfile.ZipFile(io.BytesIO(raw));full=next(n for n in z.namelist() if n.endswith(member))
  txt=z.read(full).decode('utf-8-sig').strip()
  lines=[s.split(';' if name.startswith('wine') else ',') for s in txt.splitlines()]
  if name.startswith('wine'):features=[s.strip('"') for s in lines.pop(0)];a=np.array(lines,float);X=a[:,:-1];y=(a[:,-1]>=6).astype(int)
  elif name=='breast':a=np.array(lines);X=a[:,2:].astype(float);y=(a[:,1]=='B').astype(int);features=[f'feature_{j}' for j in range(X.shape[1])]
  else:
   a=np.array(lines);X=a[:,:-1].astype(float)
   if name=='sonar':y=(a[:,-1]=='R').astype(int)
   elif name in ('magic','ionosphere'):y=(a[:,-1]=='g').astype(int)
   else:y=a[:,-1].astype(float).astype(int)
   features=[f'feature_{j}' for j in range(X.shape[1])]
  if len(features)>X.shape[1]:features=features[:X.shape[1]]
  np.savez_compressed(out/(name+'.npz'),X=X,y=y,features=np.array(features))
  records[name]={'uci_id':ident,'source':url,'source_sha256':hashlib.sha256(raw).hexdigest(),'member':full,'rows':len(y),'features':X.shape[1],'positive_rate':float(y.mean()),'license':'CC BY 4.0; original data ownership retained'}
  print(name,X.shape,flush=True)
 (out/'provenance.json').write_text(json.dumps(records,indent=2)+'\n')
if __name__=='__main__':main()
