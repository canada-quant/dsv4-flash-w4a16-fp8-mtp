"""Second-pass rename fixing compressor/indexer nesting + kv_norm + position_bias."""
import json
from pathlib import Path
from safetensors import safe_open
from safetensors.torch import save_file

art = Path('/scratch/weights/w4a16-fp8-mtp-smoke')
idx_p = art / 'model.safetensors.index.json'
idx = json.load(open(idx_p))
wm = idx['weight_map']

# Order matters: longer/specific patterns FIRST
RENAMES = [
    ('.attn.compressor.indexer.wq_b.',           '.attn.indexer.wq_b.'),
    ('.attn.compressor.indexer.weights_proj.',   '.attn.indexer.weights_proj.'),
    ('.attn.compressor.indexer.kv_norm.',        '.attn.indexer.compressor.norm.'),
    ('.attn.compressor.indexer.position_bias',   '.attn.indexer.compressor.ape'),
    ('.attn.compressor.indexer.wgate.',          '.attn.indexer.compressor.wgate.'),
    ('.attn.compressor.indexer.wkv.',            '.attn.indexer.compressor.wkv.'),
    ('.attn.compressor.kv_norm.',                '.attn.compressor.norm.'),
    ('.attn.compressor.position_bias',           '.attn.compressor.ape'),
]

def rename(k):
    for old, new in RENAMES:
        if old in k:
            k = k.replace(old, new)
    return k

renames_map = {}
for k in wm:
    nk = rename(k)
    if nk != k:
        renames_map[k] = nk

shards = {}
for old, new in renames_map.items():
    shards.setdefault(wm[old], []).append((old, new))

print(f'Total renames: {len(renames_map)}, across {len(shards)} shards', flush=True)

for sh, rns in sorted(shards.items()):
    sp = art / sh
    print(f'patching {sh}: {len(rns)} renames', flush=True)
    tensors = {}
    with safe_open(sp, framework='pt') as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k)
    rm = dict(rns)
    new_tensors = {rm.get(k, k): v for k, v in tensors.items()}
    save_file(new_tensors, str(sp))

new_wm = {renames_map.get(k, k): v for k, v in wm.items()}
idx['weight_map'] = new_wm
json.dump(idx, open(idx_p, 'w'), indent=2)
print('DONE', flush=True)
