"""
============================================================
ABLATION_BRANCHES - Studi di ablazione con riaddestramento (palmo e dorso)
============================================================
Protocollo (quello tipico dei paper di biometria):
  * stessi fold PER SOGGETTO per tutte le varianti (confronto appaiato);
  * i fold CAMBIANO a ogni ripetizione (seed + rep): le ripetizioni sono quindi
    una vera ripetizione dello split e non solo del seed di training;
  * iperparametri FISSI (nessuna selezione di epoca/checkpoint sul test fold);
  * test su soggetti mai visti, EER / TAR@FAR=1% / d' per ogni fold;
  * differenze vs configurazione completa con IC95% e test t appaiato corretto
    per CV ripetuta (Nadeau & Bengio, 2003) + Wilcoxon + correzione di Holm.

Per renderlo fattibile (pochi training lunghi disponibili) tutti i backbone sono
CONGELATI e i loro output sono in cache; si addestrano solo le teste:
  palmo : ViT/DINOv2 (cache npz esistente) + MobileNetV3 ROI congelato (cache creata
          qui) + CNN FRIT (addestrata, canali FRIT in memmap int8 creato qui)
  dorso : Swin (cache npz esistente) + MobileNetV3 nocche congelato (cache creata qui
          o --knuckle_cache) + attention pooling addestrato
ATTENZIONE: nel modello principale la MobileNet ROI/nocche e' in fine-tuning e il
training usa augmentation; qui no. La configurazione "full" di QUESTO script e' il
riferimento delle ablazioni (non va confrontata col nested CV principale).

STUDI (--study)
  components    tutti i sottoinsiemi di branch (full, leave-one-out, branch singolo)
                + fusione a livello di score dei branch singoli (early vs late fusion)
  frit_channels (solo palmo) sottoinsiemi dei 6 canali FRIT dentro il modello completo
  loss          margine ArcFace 0 (=softmax coseno) / 0.15 / 0.30 / 0.45

ESEMPI
  python ablation_branches.py palm  --data_dir dataset_preprocessed --hand_cache vit_embeds.npz \
         --study components --folds 5 --repeats 3 --out_dir ablation_branches_out
  python ablation_branches.py dorsal --data_dir dataset_preprocessed --hand_cache swin_embeds.npz \
         --study components --folds 5 --repeats 3
  python ablation_branches.py palm  --data_dir dataset_preprocessed --hand_cache vit_embeds.npz \
         --study frit_channels
  # solo sui soggetti di sviluppo (escludendo gli held-out):
  ... --subjects_file dev_subjects.json [--subjects_key train]

Il file va messo nella stessa cartella di palm_core.py / dorsal_core.py.
============================================================
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import itertools
import json
import time
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
FRIT_CH = 6
FRIT_SETS = {
    "all6": (0, 1, 2, 3, 4, 5),
    "ridgelet_only": (0, 1),
    "classical_only": (2, 3, 4, 5),
    "gray_only": (2,),
    "no_gray": (0, 1, 3, 4, 5),
}
LOSS_MARGINS = [0.0, 0.15, 0.30, 0.45]


# ============================================================
# 1) CACHE DELLE FEATURE
# ============================================================
def load_npz_cache(path):
    d = np.load(path, allow_pickle=True)
    paths, embeds = d["paths"], d["embeds"]
    return {str(p): embeds[i] for i, p in enumerate(paths)}


def save_npz_cache(path, cache):
    keys = list(cache.keys())
    np.savez(path, paths=np.array(keys, dtype=object), embeds=np.stack([cache[k] for k in keys]))


@torch.no_grad()
def mnv3_features(paths, device, size=224, batch=64):
    """MobileNetV3-Large ImageNet congelato -> feature 960-d (stesse del branch originale)."""
    from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
    net = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT).to(device).eval()
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    out = []
    for i in range(0, len(paths), batch):
        imgs = []
        for p in paths[i:i + batch]:
            im = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
            if im.shape[:2] != (size, size):
                im = cv2.resize(im, (size, size), interpolation=cv2.INTER_AREA)
            imgs.append(torch.from_numpy(im).permute(2, 0, 1))
        x = torch.stack(imgs).to(device).float().div_(255.0)
        x = (x - mean) / std
        out.append(net.avgpool(net.features(x)).flatten(1).cpu().numpy().astype(np.float16))
        print(f"    mnv3 [{min(i + batch, len(paths))}/{len(paths)}]", flush=True)
    return np.concatenate(out) if out else np.zeros((0, 960), np.float16)


def cached_mnv3(paths, cache_file, device):
    cache = load_npz_cache(cache_file) if Path(cache_file).exists() else {}
    keys = [str(Path(p).resolve()) for p in paths]
    missing = sorted({k for k in keys if k not in cache})
    if missing:
        print(f"  MobileNetV3 congelato: estrazione di {len(missing)} immagini -> {cache_file}")
        feats = mnv3_features(missing, device)
        cache.update({k: feats[i] for i, k in enumerate(missing)})
        save_npz_cache(cache_file, cache)
    return np.stack([cache[k] for k in keys]).astype(np.float16)


def _frit_worker(task):
    paths, size = task
    from palm_core import compute_frit_channels
    out = np.zeros((len(paths), FRIT_CH, size, size), np.int8)
    for i, p in enumerate(paths):
        ch = compute_frit_channels(cv2.imread(str(p)))          # (6, H, W) in [-1, 1]
        if ch.shape[1:] != (size, size):
            ch = np.stack([cv2.resize(c, (size, size), interpolation=cv2.INTER_AREA) for c in ch])
        out[i] = np.round(np.clip(ch, -1, 1) * 127).astype(np.int8)
    return out


def build_frit_memmap(samples, K, size, cache_dir, workers):
    """Canali FRIT di tutte le nocche, quantizzati a int8 (~3-4 GB per ~5k campioni) su disco."""
    path, meta = cache_dir / "palm_frit_int8.npy", cache_dir / "palm_frit_int8.json"
    key = hashlib.md5(("\n".join(str(s["hand_path"]) for s in samples) + f"|{K}|{size}").encode()).hexdigest()
    if path.exists() and meta.exists() and json.load(open(meta)).get("key") == key:
        return np.load(path, mmap_mode="r")
    print(f"  FRIT: calcolo {len(samples)} campioni x {K} nocche -> {path}")
    arr = np.lib.format.open_memmap(path, mode="w+", dtype=np.int8,
                                    shape=(len(samples), K, FRIT_CH, size, size))
    tasks = [([str(p) for p in s["knuckle_paths"][:K]], size) for s in samples]
    if workers <= 0:
        for i, task in enumerate(tasks):
            arr[i, :len(task[0])] = _frit_worker(task)
            if i % 200 == 0:
                print(f"    FRIT [{i}/{len(tasks)}]", flush=True)
    else:
        with Pool(workers) as pool:
            for i, out in enumerate(pool.imap(_frit_worker, tasks, chunksize=4)):
                arr[i, :len(out)] = out
                if i % 200 == 0:
                    print(f"    FRIT [{i}/{len(tasks)}]", flush=True)
    arr.flush()
    del arr
    json.dump({"key": key}, open(meta, "w"))
    return np.load(path, mmap_mode="r")


# ============================================================
# 2) TABELLA DEI CAMPIONI
# ============================================================
def parse_side(hand_path, subject):
    name = Path(hand_path).name
    for tail in ("_palm_hand.png", "_dorsal_hand.png"):
        if name.endswith(tail):
            name = name[: -len(tail)]
    mid = name[len(subject) + 1:] if name.startswith(subject + "_") else name
    return mid.rsplit("_", 1)[0] if "_" in mid else mid


def load_subjects_file(path, key=None):
    p = Path(path)
    if p.suffix.lower() == ".json":
        obj = json.load(open(p, "r", encoding="utf-8"))
        if isinstance(obj, dict):
            if key is None:
                raise SystemExit(f"{p} e' un dict: specifica --subjects_key tra {list(obj.keys())}")
            obj = obj[key]
        return [str(s) for s in obj]
    return [ln.strip() for ln in open(p, "r", encoding="utf-8") if ln.strip()]


def build_table(args, device):
    cache_dir = Path(args.out_dir) / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    subjects = load_subjects_file(args.subjects_file, args.subjects_key) if args.subjects_file else None
    core = importlib.import_module(f"{args.domain}_core")
    if args.domain == "palm":
        ds = core.PalmBiometricDataset(args.data_dir, subject_ids=subjects, train=False)
    else:
        ds = core.DorsalBiometricDataset(args.data_dir, subject_ids=subjects, train=False)
    S = ds.samples
    K = ds.n_knuckles_max
    print(f"Campioni: {len(S)} | soggetti: {len({s['subject'] for s in S})} | nocche max: {K}")

    hc = load_npz_cache(args.hand_cache)
    keys = [str(s["hand_path"].resolve()) for s in S]
    miss = [k for k in keys if k not in hc]
    if miss:
        raise SystemExit(
            f"--hand_cache non contiene {len(miss)}/{len(keys)} immagini (es. {miss[0]}).\n"
            f"Per '{args.domain}' serve la cache "
            f"{'ViT/DINOv2 dei *_palm_hand.png' if args.domain == 'palm' else 'Swin dei *_dorsal_hand.png'}.")
    t = {"hand": np.stack([hc[k] for k in keys]).astype(np.float32),
         "subj": np.array([s["subject"] for s in S]),
         "hid": np.array([f"{s['subject']}|{parse_side(s['hand_path'], s['subject'])}" for s in S]),
         "K": K, "core": core}
    mask = np.zeros((len(S), K), np.float32)
    for i, s in enumerate(S):
        mask[i, :min(K, len(s["knuckle_paths"]))] = 1.0
    t["mask"] = mask

    if args.domain == "palm":
        t["roi"] = cached_mnv3([s["roi_path"] for s in S], cache_dir / "palm_roi_mnv3.npz",
                               device).astype(np.float32)
        t["knuckle"] = build_frit_memmap(S, K, core.cfg.PALM_KNUCKLE_SIZE[0], cache_dir, args.num_workers)
    else:
        if args.knuckle_cache:
            kc = load_npz_cache(args.knuckle_cache)
        else:
            allp = sorted({str(p.resolve()) for s in S for p in s["knuckle_paths"][:K]})
            kc_file = cache_dir / "dorsal_knuckle_mnv3.npz"
            feats = cached_mnv3(allp, kc_file, device)
            kc = {p: feats[i] for i, p in enumerate(allp)}
        kf = np.zeros((len(S), K, 960), np.float16)
        for i, s in enumerate(S):
            for j, p in enumerate(s["knuckle_paths"][:K]):
                kf[i, j] = kc[str(p.resolve())]
        t["knuckle"] = kf
    return t


# ============================================================
# 3) MODELLO ABLATO (stesse teste dei modelli originali, branch opzionali)
# ============================================================
class Ablated(nn.Module):
    def __init__(self, domain, core, branches, tbl, n_chans, d=128, emb=256):
        super().__init__()
        self.br = list(branches)
        if "hand" in self.br:
            self.hand = nn.Sequential(nn.Linear(tbl["hand"].shape[1], 512), nn.GELU(),
                                      nn.Dropout(0.2), nn.Linear(512, d))
        if "roi" in self.br:
            self.roi = nn.Sequential(nn.Linear(tbl["roi"].shape[1], 512), nn.Hardswish(),
                                     nn.Dropout(0.2), nn.Linear(512, d))
        if "knuckle" in self.br:
            if domain == "palm":
                self.knuckle = core.KnuckleFRITBranch(tbl["K"], d, in_channels=n_chans)
            else:
                self.knuckle = core.KnuckleMobileNetBranch(tbl["K"], d, pretrained=False,
                                                           freeze_backbone=True)
        self.fusion = nn.Sequential(nn.Linear(d * len(self.br), 512), nn.BatchNorm1d(512),
                                    nn.ReLU(inplace=True), nn.Dropout(0.3), nn.Linear(512, emb))

    def forward(self, b):
        parts = []
        if "hand" in self.br:
            parts.append(self.hand(b["hand"]))
        if "roi" in self.br:
            parts.append(self.roi(b["roi"]))
        if "knuckle" in self.br:
            parts.append(self.knuckle(b["knuckle"], b["mask"]))
        return F.normalize(self.fusion(torch.cat(parts, 1)), p=2, dim=1)


class Feed:
    """Costruisce i batch dalle feature in cache (GPU) e dalla memmap FRIT (disco)."""

    def __init__(self, tbl, domain, dev, chans):
        self.domain, self.dev, self.chans = domain, dev, list(chans)
        self.hand = torch.from_numpy(tbl["hand"]).to(dev)
        self.roi = torch.from_numpy(tbl["roi"]).to(dev) if "roi" in tbl else None
        self.mask = torch.from_numpy(tbl["mask"]).to(dev)
        self.k = tbl["knuckle"]
        if domain == "dorsal":
            self.k = torch.from_numpy(np.ascontiguousarray(self.k)).to(dev)

    def get(self, idx):
        b = {"hand": self.hand[idx], "mask": self.mask[idx]}
        if self.roi is not None:
            b["roi"] = self.roi[idx]
        if self.domain == "palm":
            x = np.asarray(self.k[np.asarray(idx)])[:, :, self.chans]
            b["knuckle"] = torch.from_numpy(np.ascontiguousarray(x)).to(self.dev).float().div_(127.0)
        else:
            b["knuckle"] = self.k[idx].float()
        return b


def train_and_embed(tbl, feed, tr_idx, te_idx, spec, args, seed, dev):
    core, domain = tbl["core"], args.domain
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    subj_tr = tbl["subj"][tr_idx]
    _, y_np = np.unique(subj_tr, return_inverse=True)
    y = torch.from_numpy(y_np).long().to(dev)
    model = Ablated(domain, core, spec["branches"], tbl, len(spec["chans"])).to(dev)
    head = core.ArcMarginHead(256, int(y_np.max()) + 1, scale=args.scale, margin=spec["margin"]).to(dev)
    params = [p for p in model.parameters() if p.requires_grad] + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs))
    for _ in range(args.epochs):
        model.train(); head.train()
        perm = rng.permutation(len(tr_idx))
        for i in range(0, len(perm), args.batch):
            bi = perm[i:i + args.batch]
            if len(bi) < 2:
                continue
            emb = model(feed.get(tr_idx[bi]))
            loss = F.cross_entropy(head(emb, y[torch.from_numpy(bi).to(dev)]),
                                   y[torch.from_numpy(bi).to(dev)])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
        sched.step()
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(te_idx), 128):
            out.append(model(feed.get(te_idx[i:i + 128])).cpu().numpy())
    return np.concatenate(out)


# ============================================================
# 4) METRICHE
# ============================================================
def metrics(gen, imp):
    if len(gen) == 0 or len(imp) == 0:
        return {"eer": float("nan"), "tar@far1e-2": float("nan"), "dprime": float("nan")}
    gs, is_ = np.sort(gen), np.sort(imp)
    th = np.linspace(-1, 1, 4001)
    far = 1.0 - np.searchsorted(is_, th, side="left") / len(is_)
    frr = np.searchsorted(gs, th, side="left") / len(gs)
    i = int(np.argmin(np.abs(far - frr)))
    thr = np.quantile(imp, 0.99)
    return {"eer": float((far[i] + frr[i]) / 2),
            "tar@far1e-2": float((gen >= thr).mean()),
            "dprime": float((gen.mean() - imp.mean()) / np.sqrt(0.5 * (gen.var() + imp.var()) + 1e-12))}


def eval_sim(sim, subj, hid):
    """genuine 'hand' = stesso soggetto e stessa mano; 'subject' = stesso soggetto (come compute_eer).
    Impostori = soggetti diversi (uguali nei due casi)."""
    _, sc = np.unique(subj, return_inverse=True)
    _, hc = np.unique(hid, return_inverse=True)
    iu = np.triu_indices(sim.shape[0], 1)
    s = sim[iu]
    same_s, same_h = sc[iu[0]] == sc[iu[1]], hc[iu[0]] == hc[iu[1]]
    imp = s[~same_s]
    return {"hand": metrics(s[same_h], imp), "subject": metrics(s[same_s], imp)}


# ============================================================
# 5) STATISTICA
# ============================================================
def holm(p):
    p = np.asarray(p, float)
    order, adj, run = np.argsort(p), np.empty(len(p)), 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(p) - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj


def corrected_paired(d, test_over_train):
    """t test appaiato corretto per CV ripetuta (Nadeau & Bengio 2003) + IC95%."""
    from scipy import stats
    J, m = len(d), float(np.mean(d))
    v = float(np.var(d, ddof=1)) if J > 1 else 0.0
    se = np.sqrt((1.0 / J + test_over_train) * v)
    if se < 1e-12:
        return m, (m, m), float("nan")
    t = m / se
    tc = stats.t.ppf(0.975, J - 1)
    return m, (m - tc * se, m + tc * se), float(2 * stats.t.sf(abs(t), J - 1))


def summarize(rows, ref, K):
    from scipy import stats
    variants = list(dict.fromkeys(r["variant"] for r in rows))
    per = {v: {(r["rep"], r["fold"]): r for r in rows if r["variant"] == v} for v in variants}
    out = []
    for v in variants:
        keys = sorted(per[v])
        e = np.array([per[v][k]["eer"] for k in keys]) * 100
        s = {"variant": v, "n": len(keys), "eer_mean": e.mean(), "eer_std": e.std(ddof=1) if len(e) > 1 else 0.0,
             "eer_hand": np.mean([per[v][k]["eer_hand"] for k in keys]) * 100,
             "eer_subject": np.mean([per[v][k]["eer_subject"] for k in keys]) * 100,
             "tar": np.mean([per[v][k]["tar@far1e-2"] for k in keys]) * 100,
             "dprime": np.mean([per[v][k]["dprime"] for k in keys])}
        if v != ref and ref in per:
            common = sorted(set(keys) & set(per[ref]))
            d = np.array([per[v][k]["eer"] - per[ref][k]["eer"] for k in common]) * 100
            m, ci, p = corrected_paired(d, 1.0 / max(1, K - 1))
            try:
                pw = float(stats.wilcoxon(d).pvalue)
            except ValueError:
                pw = float("nan")
            s.update({"delta": m, "ci_lo": ci[0], "ci_hi": ci[1], "p_corr": p, "p_wilcoxon": pw,
                      "worse_than_ref": f"{int((d > 0).sum())}/{len(d)}"})
        out.append(s)
    cmp_idx = [i for i, s in enumerate(out) if "p_corr" in s and not np.isnan(s["p_corr"])]
    if cmp_idx:
        adj = holm([out[i]["p_corr"] for i in cmp_idx])
        for i, a in zip(cmp_idx, adj):
            out[i]["p_holm"] = float(a)
    return out


def print_summary(summ, ref, title):
    print("\n" + "=" * 118 + f"\n{title}\n" + "=" * 118)
    print(f"{'variante':34s} {'EER % (media±std)':20s} {'TAR@1%':>8s} {'d-prime':>8s} "
          f"{'ΔEER pp [IC95%]':>26s} {'p_corr':>7s} {'p_holm':>7s} {'peggio':>7s}")
    for s in summ:
        base = (f"{s['variant']:34s} {s['eer_mean']:6.2f} ± {s['eer_std']:5.2f}      "
                f"{s['tar']:7.2f}% {s['dprime']:8.3f} ")
        if "delta" in s:
            base += (f"{s['delta']:+7.2f} [{s['ci_lo']:+6.2f},{s['ci_hi']:+6.2f}]  "
                     f"{s['p_corr']:7.3f} {s.get('p_holm', float('nan')):7.3f} {s['worse_than_ref']:>7s}")
        else:
            base += f"{'(riferimento)' if s['variant'] == ref else '':>26s}"
        print(base)
    print("ΔEER > 0 = peggio del riferimento. p_corr: t appaiato corretto (Nadeau-Bengio); "
          "p_holm: Holm sui confronti del riferimento.")


def write_latex(summ, ref, path):
    L = ["\\begin{tabular}{lcccc}", "\\toprule",
         "Configurazione & EER (\\%) & TAR@FAR=1\\% (\\%) & $\\Delta$EER (pp) & $p_{Holm}$ \\\\", "\\midrule"]
    for s in summ:
        dl = f"{s['delta']:+.2f} [{s['ci_lo']:+.2f}, {s['ci_hi']:+.2f}]" if "delta" in s else "--"
        ph = f"{s['p_holm']:.3f}" if "p_holm" in s else "--"
        L.append(f"{s['variant'].replace('_', ' ')} & {s['eer_mean']:.2f} $\\pm$ {s['eer_std']:.2f} & "
                 f"{s['tar']:.1f} & {dl} & {ph} \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    Path(path).write_text("\n".join(L), encoding="utf-8")


# ============================================================
# 6) SPECIFICHE DELLE VARIANTI
# ============================================================
def make_specs(domain, study, margin):
    brs = ["hand", "roi", "knuckle"] if domain == "palm" else ["hand", "knuckle"]
    allc = FRIT_SETS["all6"]
    if study == "components":
        specs = []
        for r in range(len(brs), 0, -1):
            for comb in itertools.combinations(brs, r):
                name = "full" if len(comb) == len(brs) else "+".join(comb)
                specs.append({"name": name, "branches": comb, "chans": allc, "margin": margin})
        return specs, "full"
    if study == "frit_channels":
        if domain != "palm":
            raise SystemExit("frit_channels vale solo per il palmo")
        return [{"name": f"frit_{n}", "branches": tuple(brs), "chans": c, "margin": margin}
                for n, c in FRIT_SETS.items()], "frit_all6"
    if study == "loss":
        return [{"name": f"margin_{m:.2f}", "branches": tuple(brs), "chans": allc, "margin": m}
                for m in LOSS_MARGINS], "margin_0.30"
    raise SystemExit(f"study sconosciuto: {study}")


# ============================================================
# 7) ESECUZIONE
# ============================================================
def run(args):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tbl = build_table(args, dev)
    core = tbl["core"]
    specs, ref = make_specs(args.domain, args.study, args.margin)
    if args.variants:
        specs = [s for s in specs if s["name"] in args.variants or s["name"] == ref]
    names = [s["name"] for s in specs]
    brs = {"palm": {"hand", "roi", "knuckle"}, "dorsal": {"hand", "knuckle"}}[args.domain]
    singles = {s["branches"][0]: s["name"] for s in specs if len(s["branches"]) == 1}
    do_sf = args.study == "components" and set(singles) == brs
    expected = names + (["score_fusion"] if do_sf else [])

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{args.domain}_{args.study}"
    jl = out / f"rows_{tag}.jsonl"
    rows = [json.loads(l) for l in open(jl, encoding="utf-8")] if jl.exists() else []
    cnt = {}
    for r in rows:
        cnt.setdefault((r["rep"], r["fold"]), set()).add(r["variant"])
    complete = {k for k, v in cnt.items() if v >= set(expected)}
    rows = [r for r in rows if (r["rep"], r["fold"]) in complete]
    with open(jl, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    log = open(jl, "a", encoding="utf-8")

    feed = Feed(tbl, args.domain, dev, FRIT_SETS["all6"])
    subjects = sorted(set(tbl["subj"].tolist()))
    t0 = time.time()
    for rep in range(args.repeats):
        for fold, trs, tes in core.kfold_subject_splits(subjects, args.folds, seed=args.seed + rep):
            if (rep, fold) in complete:
                print(f"  rep {rep} fold {fold}: gia' completato, salto")
                continue
            tr = np.where(np.isin(tbl["subj"], trs))[0]
            te = np.where(np.isin(tbl["subj"], tes))[0]
            sims = {}
            for sp in specs:
                feed.chans = list(sp["chans"])
                emb = train_and_embed(tbl, feed, tr, te, sp, args, args.seed + 1000 * rep + fold, dev)
                sim = emb @ emb.T
                if sp["name"] in singles.values():
                    sims[sp["name"]] = sim
                res = eval_sim(sim, tbl["subj"][te], tbl["hid"][te])
                r = {"study": args.study, "rep": rep, "fold": fold, "variant": sp["name"],
                     "n_train_subjects": len(trs), "n_test_subjects": len(tes),
                     "eer_hand": res["hand"]["eer"], "eer_subject": res["subject"]["eer"],
                     **res[args.identity]}
                rows.append(r); log.write(json.dumps(r) + "\n"); log.flush()
                print(f"  rep {rep} fold {fold} {sp['name']:28s} EER={100 * r['eer']:.2f}% "
                      f"TAR@1%={100 * r['tar@far1e-2']:.1f}%  [{time.time() - t0:.0f}s]", flush=True)
            if do_sf:
                sim = np.mean([sims[n] for n in singles.values()], axis=0)
                res = eval_sim(sim, tbl["subj"][te], tbl["hid"][te])
                r = {"study": args.study, "rep": rep, "fold": fold, "variant": "score_fusion",
                     "n_train_subjects": len(trs), "n_test_subjects": len(tes),
                     "eer_hand": res["hand"]["eer"], "eer_subject": res["subject"]["eer"],
                     **res[args.identity]}
                rows.append(r); log.write(json.dumps(r) + "\n"); log.flush()
                print(f"  rep {rep} fold {fold} {'score_fusion':28s} EER={100 * r['eer']:.2f}%", flush=True)
    log.close()

    summ = summarize(rows, ref, args.folds)
    print_summary(summ, ref, f"ABLAZIONE [{tag}] - {args.folds}-fold per soggetto x {args.repeats} rip., "
                             f"identita'={args.identity}, test su soggetti mai visti")
    if do_sf:
        print("score_fusion = media dei coseni dei modelli a branch singolo (late fusion), "
              "da confrontare con 'full' (early fusion a livello di feature).")
    json.dump(summ, open(out / f"summary_{tag}.json", "w", encoding="utf-8"), indent=2, default=float)
    write_latex(summ, ref, out / f"summary_{tag}.tex")
    print(f"\nSalvati: {jl.name}, summary_{tag}.json, summary_{tag}.tex in {out}")


def main():
    ap = argparse.ArgumentParser(description="Ablazioni con riaddestramento (palmo/dorso)")
    ap.add_argument("domain", choices=["palm", "dorsal"])
    ap.add_argument("--data_dir", default="./dataset_preprocessed")
    ap.add_argument("--hand_cache", required=True,
                    help="npz di precompute_vit_embeddings.py (palm) / precompute_swin_embeddings.py (dorsal)")
    ap.add_argument("--knuckle_cache", default=None, help="solo dorsal: npz nocche gia' calcolato (opzionale)")
    ap.add_argument("--out_dir", default="./ablation_branches_out")
    ap.add_argument("--study", choices=["components", "frit_channels", "loss"], default="components")
    ap.add_argument("--variants", nargs="+", default=None, help="filtra i nomi delle varianti (il riferimento resta)")
    ap.add_argument("--subjects_file", default=None)
    ap.add_argument("--subjects_key", default=None)
    ap.add_argument("--identity", choices=["hand", "subject"], default="hand",
                    help="genuine = stessa mano (default, piu' rigoroso) o stesso soggetto (come compute_eer)")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--margin", type=float, default=0.30)
    ap.add_argument("--scale", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_workers", type=int, default=2, help="processi per il calcolo FRIT (0 = nessuno)")
    run(ap.parse_args())


if __name__ == "__main__":
    main()