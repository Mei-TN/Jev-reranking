"""Bind the unchanged artifact identities to an explicitly supplied media directory."""
import argparse,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--media-directory',type=Path,required=True);args=p.parse_args()
root=Path(__file__).resolve().parent
assert not (root/'BUILD_STARTED.json').exists(), 'Existing build must be preserved'
assert not (root/'SPEC.json').exists(), 'Use a fresh extracted directory'
s=json.loads((root/'SPEC.template.json').read_text())
media=args.media_directory.expanduser().resolve(strict=True)
for row in s['wheels']+[s['python_archive'],s['pillow_overlay']]+s['isolated_system_libraries']:
    path=media/row['filename']
    assert path.is_file(), str(path)
    row['source_file']=str(path)
payload=(json.dumps(s,indent=2)+'\n').encode()
(root/'SPEC.json').write_bytes(payload)
(root/'SPEC.sha256').write_text(hashlib.sha256(payload).hexdigest()+'\n')
print('Prepared site-specific paths; no installation or import was executed.')
