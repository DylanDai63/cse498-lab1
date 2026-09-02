# Build a small YOLO detection dataset from Open Images v5/v7 (val+test splits)
# Classes: Tomato /m/07j87, Toilet paper /m/09gtd, Bowl /m/04kkgm
# Selection: drop IsGroupOf/IsDepiction, prefer images with 1-4 target boxes,
#            cap per-class image counts, download from official S3, resize to 640, emit YOLO labels.
import csv, os, random, io, sys
import urllib.request
from collections import defaultdict
from PIL import Image

random.seed(42)
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "dataset")
CLASSES = {"/m/07j87": ("tomato", 0)}
CAPS = {"tomato": 50}

# rows per image: {(split,img): [(cls_id, xmin,xmax,ymin,ymax), ...]}
imgs = defaultdict(list)
bad = set()  # images containing group/depiction boxes of our classes -> skip entirely
for fname, split in [("targets.csv", "validation"), ("targets-test.csv", "test")]:
    with open(os.path.join(HERE, fname), newline="", encoding="utf-8") as f:
        for r in csv.reader(f):
            img, label = r[0], r[2]
            xmin, xmax, ymin, ymax = map(float, r[4:8])
            group, depiction = r[10], r[11]
            if label not in CLASSES: continue
            key = (split, img)
            if group == "1" or depiction == "1":
                bad.add(key); continue
            imgs[key].append((CLASSES[label][1], xmin, xmax, ymin, ymax))

cand = {k: v for k, v in imgs.items() if k not in bad and 1 <= len(v) <= 6}
# bucket by dominant class (fewest-first so toilet_paper keeps its scarce images)
by_class = defaultdict(list)
for k, v in cand.items():
    cls = min(set(c for c, *_ in v))  # arbitrary but deterministic dominant pick
    names = {0: "tomato"}
    # dominant = most frequent class in image
    cnt = defaultdict(int)
    for c, *_ in v: cnt[c] += 1
    dom = names[max(cnt, key=lambda c: (cnt[c], -c))]
    by_class[dom].append(k)

chosen = []
for cls, keys in sorted(by_class.items()):
    random.shuffle(keys)
    chosen += [(cls, k) for k in keys[: CAPS[cls]]]
print("selected per class:", {c: sum(1 for cc, _ in chosen if cc == c) for c in CAPS})

for sub in ["train/images", "train/labels", "valid/images", "valid/labels"]:
    os.makedirs(os.path.join(OUT, sub), exist_ok=True)

random.shuffle(chosen)
n_valid = max(1, round(len(chosen) * 0.15))
splits = ["valid"] * n_valid + ["train"] * (len(chosen) - n_valid)

ok, fail = 0, 0
for (cls, (split, img)), dest in zip(chosen, splits):
    url = f"https://open-images-dataset.s3.amazonaws.com/{split}/{img}.jpg"
    try:
        data = urllib.request.urlopen(url, timeout=60).read()
        im = Image.open(io.BytesIO(data)).convert("RGB")
        im.thumbnail((640, 640))
        im.save(os.path.join(OUT, dest, "images", f"{img}.jpg"), quality=88)
        with open(os.path.join(OUT, dest, "labels", f"{img}.txt"), "w") as f:
            for c, xmin, xmax, ymin, ymax in imgs[(split, img)]:
                cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
                w, h = xmax - xmin, ymax - ymin
                f.write(f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
        ok += 1
    except Exception as e:
        print("FAIL", img, e); fail += 1

with open(os.path.join(OUT, "data.yaml"), "w") as f:
    f.write("train: train/images\nval: valid/images\nnc: 1\nnames: [tomato]\n")
print(f"downloaded ok={ok} fail={fail}; dataset at {OUT}")
