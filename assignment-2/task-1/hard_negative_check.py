"""What do the hardest negatives look like in this dataset? (proxy experiment)

No embedding is trained (Task 1 is theoretical), so an ImageNet-pretrained
ResNet-18 stands in for the network at the start of training (the backbone the
design fine-tunes). Crops are sampled photo-first (one crop of the class per
training photo), WITHOUT the 24 px rule, so we can see whether tiny crops are
over-represented among the hardest negatives. Batches are 6 classes x 8 crops.
For every anchor-positive pair we record three negatives: the hardest (closest),
a random semi-hard one (d(a,p) < d(a,n) < d(a,p)+m), and a random one.
"""
import argparse, csv, json, random
from pathlib import Path
import numpy as np, torch, torchvision
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
A1 = HERE.parents[1] / 'assignment-1'
ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--data', default=str(A1 / 'data/processed'), help='prepared split (images/ and labels/ with train/)')
ap.add_argument('--mismatches', default=str(A1 / 'reports/name-label-mismatches.csv'))
ap.add_argument('--out', default=str(HERE / 'figures'))
args = ap.parse_args()
DATA, MISMATCH, OUT = Path(args.data), Path(args.mismatches), Path(args.out); OUT.mkdir(exist_ok=True)
NAMES = ['biodegradable', 'cardboard', 'glass', 'metal', 'paper', 'plastic']
PER_CLASS, K, BATCHES, M, SEED = 500, 8, 400, 0.2, 0
rng = random.Random(SEED); torch.manual_seed(SEED); torch.set_num_threads(4)

suspect = {r['image'] for r in csv.DictReader(open(MISMATCH)) if r['split'] == 'train'}
imgs = {p.stem: p for p in (DATA / 'images/train').iterdir()}
boxes = {}
for f in (DATA / 'labels/train').glob('*.txt'):
    rows = [list(map(float, l.split())) for l in f.read_text().splitlines() if l.strip()]
    boxes[f.stem] = [r for r in rows if r[3] > 0 and r[4] > 0]

pool = []  # photo first, then one crop of the class
for c in range(6):
    photos = [s for s, b in boxes.items() if any(int(r[0]) == c for r in b)]
    for s in rng.sample(photos, min(PER_CLASS, len(photos))):
        r = rng.choice([r for r in boxes[s] if int(r[0]) == c])
        pool.append(dict(stem=s, cls=c, box=r[1:], suspect=imgs[s].name in suspect))

tf = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
net = torchvision.models.resnet18(weights=tf); net.fc = torch.nn.Identity(); net.eval()
prep = tf.transforms()

def crop(p):
    im = Image.open(imgs[p['stem']]).convert('RGB'); W, H = im.size
    x, y, w, h = p['box']
    p['side'] = min(w * W, h * H)
    return im.crop((int((x - w / 2) * W), int((y - h / 2) * H), max(int((x + w / 2) * W), int((x - w / 2) * W) + 1),
                    max(int((y + h / 2) * H), int((y - h / 2) * H) + 1)))

crops = [crop(p) for p in pool]
with torch.no_grad():
    E = torch.cat([net(torch.stack([prep(c) for c in crops[i:i + 64]])) for i in range(0, len(crops), 64)])
E = torch.nn.functional.normalize(E, dim=1).numpy()
cls = np.array([p['cls'] for p in pool]); side = np.array([p['side'] for p in pool])
sus = np.array([p['suspect'] for p in pool])
by_cls = [np.where(cls == c)[0] for c in range(6)]

picked = {'hardest': [], 'semi-hard': [], 'random': []}; examples = []
for b in range(BATCHES):
    idx = np.concatenate([rng.sample(list(by_cls[c]), K) for c in range(6)])
    D = np.sqrt(np.clip(2 - 2 * E[idx] @ E[idx].T, 0, None))
    for i in range(len(idx)):
        neg = np.where(cls[idx] != cls[idx[i]])[0]
        for j in np.where(cls[idx] == cls[idx[i]])[0]:
            if j == i: continue
            dap, dn = D[i, j], D[i, neg]
            h = neg[np.argmin(dn)]
            semi = neg[(dn > dap) & (dn < dap + M)]
            picked['hardest'].append(idx[h]); picked['random'].append(idx[rng.choice(list(neg))])
            if len(semi): picked['semi-hard'].append(idx[rng.choice(list(semi))])
            if b < 6 and j == np.where(cls[idx] == cls[idx[i]])[0][-1] and i % 8 == 0 and len(semi):
                examples.append((idx[i], idx[j], idx[h], idx[rng.choice(list(semi))], dap, D[i, h]))

res = {'pool_crops': len(pool), 'batches': BATCHES, 'K': K, 'margin': M,
       'base_rate_below_24px': round(float((side < 24).mean()), 4),
       'base_rate_suspect_label_image': round(float(sus.mean()), 4)}
for k, v in picked.items():
    v = np.array(v)
    res[k] = dict(n=int(len(v)), below_24px=round(float((side[v] < 24).mean()), 4),
                  below_16px=round(float((side[v] < 16).mean()), 4),
                  median_side_px=round(float(np.median(side[v])), 1),
                  suspect_label_image=round(float(sus[v].mean()), 4))
print(json.dumps(res, indent=2)); (OUT / 'hard_negative_check.json').write_text(json.dumps(res, indent=2))

# figure: anchor | positive | hardest negative | semi-hard negative
T, rows = 150, examples[:10]
fig = Image.new('RGB', (4 * T + 50, len(rows) * (T + 34) + 30), 'white'); d = ImageDraw.Draw(fig)
for c, t in enumerate(['anchor', 'positive', 'HARDEST negative', 'semi-hard negative']):
    d.text((10 + c * (T + 12), 8), t, fill='black')
for r, (a, p, h, s, dap, dah) in enumerate(rows):
    for c, q in enumerate((a, p, h, s)):
        X, Y = 10 + c * (T + 12), 26 + r * (T + 34)
        fig.paste(crops[q].resize((T, T), Image.NEAREST), (X, Y))
        tag = f"{NAMES[cls[q]]} {side[q]:.0f}px" + (' !label' if sus[q] else '')
        d.text((X, Y + T + 2), tag, fill='red' if (c == 2 and (side[q] < 24 or sus[q])) else 'black')
fig.save(OUT / 'hardest_vs_semihard.png')
