"""Check every packaged file against its SHA-256 digest."""
from pathlib import Path
import hashlib
ROOT=Path(__file__).resolve().parents[1]
def main():
 count=0
 for line in (ROOT/'checksums.sha256').read_text().splitlines():
  expected,name=line.split('  ',1);p=ROOT/name
  actual=hashlib.sha256(p.read_bytes()).hexdigest()
  if actual!=expected:raise RuntimeError('Checksum mismatch: '+name)
  count+=1
 print(f'Validated {count} file checksums.')
if __name__=='__main__':main()
