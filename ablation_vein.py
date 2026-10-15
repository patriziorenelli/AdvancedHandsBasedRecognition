"""
============================================================
ABLATION_VEIN - Studi di ablazione SENZA training (stream DORSO)
============================================================
Lavora direttamente sull'output di preProcessing.py (dorsal_hand.png, 224x224,
gia' canonicalizzato L/R) e riusa la stessa logica di enhance_veins() di
dorsal_core.py, resa parametrica. Nessun MLP, nessun training.

STUDI
  A) inputs   : Swin-Tiny congelato (768-d) + coseno, per le varianti di input
                raw | clahe | green | frangi | full
  B) sweep    : sensibilita' ai parametri di enhance_veins (un parametro alla
                volta): weight, clip CLAHE, sigma Frangi, normalizzazione
                (max / p99 / p95 / none). Backbone: Swin-Tiny congelato + coseno
  C) ncc      : baseline handcrafted senza modelli: NCC tra mappe
                gray | clahe_gray | green | vessel (con eventuale blur)
  D) mlp      : testa MLP + ArcFace (come DorsalTextureBranch) addestrata sopra gli
                embedding Swin congelati gia' in cache, con K-fold PER SOGGETTO
                (open-set). Stessi fold per tutte le varianti -> confronto appaiato.

COMANDI
  # verifica che la versione parametrica coincida con dorsal_core.enhance_veins
  python ablation_vein.py selftest --data_dir dataset_preprocessed

  # immagini di esempio (evidenza qualitativa del preprocessing)
  python ablation_vein.py examples --data_dir dataset_preprocessed --n 6

  # A) varianti di input con Swin congelato
  python ablation_vein.py inputs --data_dir dataset_preprocessed --out_dir ablation_out

  # B) sweep parametri enhance_veins
  python ablation_vein.py sweep --data_dir dataset_preprocessed --out_dir ablation_out

  # C) handcrafted NCC
  python ablation_vein.py ncc --data_dir dataset_preprocessed --out_dir ablation_out

  # D) MLP addestrata su Swin congelato (usa la cache di A/B, nessuna nuova estrazione
  #    per le varianti raw|clahe|green|frangi|full)
  python ablation_vein.py mlp --data_dir dataset_preprocessed --out_dir ablation_out

Opzioni utili (tutti i comandi di studio):
  --subjects_file held_out.json   usa solo quei soggetti (json lista, txt, o json dict + --subjects_key)
  --max_subjects 100              sottoinsieme casuale riproducibile (--seed)
  --identity subject|hand         genuine = stesso soggetto (default, come compute_eer)
                                  oppure stessa mano (soggetto + hand_side)
  --num_workers 2                 processi per CLAHE/Frangi (0 = nessun processo figlio)
  --n_sub 20                      sub-campionamenti (80% soggetti) per mean/std/IC dell'EER
============================================================
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import multiprocessing as mp
import sys
import time
from collections import namedtuple
from pathlib import Path

import cv2
import numpy as np
from skimage.filters import frangi

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SWIN_NAME = "swin_tiny_patch4_window7_224"

# Parametri di default = quelli di dorsal_core.enhance_veins
DEF_CLIP = 3.0
DEF_SIGMAS = (1, 2, 3, 4)
DEF_WEIGHT = 0.55
DEF_NORM = "max"

# kind: raw | clahe | green | frangi | full
Spec = namedtuple("Spec", ["kind", "clip", "sigmas", "weight", "norm"])


def make_spec(kind, clip=DEF_CLIP, sigmas=DEF_SIGMAS, weight=DEF_WEIGHT, norm=DEF_NORM):
    return Spec(kind, float(clip), tuple(int(s) for s in sigmas), float(weight), str(norm))


# ============================================================
# 1) ENHANCEMENT PARAMETRICO (replica di dorsal_core.enhance_veins)
# ============================================================
def clahe_rgb(img_rgb, clip):
    """CLAHE sul canale L (Lab). clip <= 0 -> nessuna CLAHE."""
    if clip <= 0:
        return img_rgb
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    L, a, b = cv2.split(lab)
    L = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(8, 8)).apply(L)
    return cv2.cvtColor(cv2.merge([L, a, b]), cv2.COLOR_LAB2RGB)


def vesselness_raw(contrast_rgb, sigmas):
    """Frangi sul canale verde invertito (vene = strutture scure). Output NON normalizzato."""
    green = contrast_rgb[:, :, 1].astype(np.float64) / 255.0
    return frangi(1.0 - green, sigmas=tuple(sigmas), black_ridges=False)


def normalize_vesselness(v, norm):
    """max: v/max (originale) | pXX: v/percentile XX, clippato a 1 | none: clip(v, 0, 1)."""
    if norm == "max":
        ref = float(v.max())
    elif norm.startswith("p"):
        ref = float(np.percentile(v, float(norm[1:])))
    elif norm == "none":
        ref = 1.0
    else:
        raise ValueError(f"norm sconosciuta: {norm}")
    if ref > 1e-8:
        v = v / ref
    return np.clip(v, 0.0, 1.0)


def render(img_rgb, sp, clahe_cache, vess_cache):
    """Ritorna uint8 HxWx3 RGB per la specifica sp. Le cache evitano di rifare CLAHE/Frangi
    quando piu' specifiche condividono (clip) o (clip, sigmas)."""
    if sp.kind == "raw":
        return img_rgb
    if sp.clip not in clahe_cache:
        clahe_cache[sp.clip] = clahe_rgb(img_rgb, sp.clip)
    contrast = clahe_cache[sp.clip]
    if sp.kind == "clahe":
        return contrast
    if sp.kind == "green":
        return np.ascontiguousarray(np.repeat(contrast[:, :, 1:2], 3, axis=2))

    vkey = (sp.clip, sp.sigmas)
    if vkey not in vess_cache:
        vess_cache[vkey] = vesselness_raw(contrast, sp.sigmas)
    v = normalize_vesselness(vess_cache[vkey], sp.norm)
    v_u8 = (v * 255).astype(np.uint8)
    v_rgb = cv2.cvtColor(v_u8, cv2.COLOR_GRAY2RGB)
    if sp.kind == "frangi":
        return v_rgb
    if sp.kind == "full":
        out = cv2.addWeighted(contrast, 1.0 - sp.weight * 0.5, v_rgb, sp.weight * 0.5, 0)
        return np.clip(out, 0, 255).astype(np.uint8)
    raise ValueError(f"kind sconosciuto: {sp.kind}")


# ============================================================
# 2) DATASET (stessa logica di DorsalBiometricDataset, senza torch)
# ============================================================
def list_subjects(data_dir):
    return sorted(d.name for d in Path(data_dir).iterdir() if d.is_dir())


def load_subjects_file(path, key=None):
    p = Path(path)
    if p.suffix.lower() == ".json":
        obj = json.load(open(p, "r", encoding="utf-8"))
        if isinstance(obj, dict):
            if key is None:
                raise ValueError(f"{p} e' un dict, specifica --subjects_key tra: {list(obj.keys())}")
            obj = obj[key]
        return [str(s) for s in obj]
    return [ln.strip() for ln in open(p, "r", encoding="utf-8") if ln.strip()]


def collect_samples(data_dir, subjects, identity="subject", require_knuckles=True):
    data_dir = Path(data_dir)
    samples = []
    for sid in subjects:
        sdir = data_dir / sid
        if not sdir.is_dir():
            continue
        for meta_path in sorted(sdir.glob("*_metadata.json")):
            meta = json.load(open(meta_path, "r", encoding="utf-8"))
            if not meta.get("is_dorsal", False):
                continue
            base = meta_path.name.replace("_metadata.json", "")
            hand = sdir / f"{base}_dorsal_hand.png"
            if not hand.exists():
                continue
            if require_knuckles:
                kn = [p for p in sdir.glob(f"{base}_dorsal_*.png") if p.name != hand.name]
                if not kn:
                    continue
            ident = sid if identity == "subject" else f"{sid}|{meta.get('hand_side', '')}"
            samples.append({"path": str(hand.resolve()), "subject": sid, "identity": ident})
    return samples


def select_samples(args):
    all_subjects = list_subjects(args.data_dir)
    if args.subjects_file:
        wanted = set(load_subjects_file(args.subjects_file, args.subjects_key))
        subjects = [s for s in all_subjects if s in wanted]
        missing = wanted - set(subjects)
        if missing:
            print(f"[WARN] {len(missing)} soggetti del file non trovati in data_dir")
    else:
        subjects = all_subjects
    if args.max_subjects and len(subjects) > args.max_subjects:
        rng = np.random.RandomState(args.seed)
        subjects = sorted(rng.choice(subjects, args.max_subjects, replace=False).tolist())

    samples = collect_samples(args.data_dir, subjects, args.identity, require_knuckles=True)
    if not samples:
        raise SystemExit("Nessun campione dorsale trovato.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "subjects_used.json", "w", encoding="utf-8") as f:
        json.dump(sorted({s["subject"] for s in samples}), f, indent=2)

    ids = sorted({s["identity"] for s in samples})
    id2i = {k: i for i, k in enumerate(ids)}
    subs = sorted({s["subject"] for s in samples})
    sub2i = {k: i for i, k in enumerate(subs)}
    labels = np.array([id2i[s["identity"]] for s in samples])
    groups = np.array([sub2i[s["subject"]] for s in samples])
    print(f"Campioni: {len(samples)} | soggetti: {len(subs)} | identita' (genuine): {len(ids)} "
          f"[identity={args.identity}]")
    return samples, labels, groups


# ============================================================
# 3) ENGINE DI RENDERING (multiprocessing con coda limitata)
# ============================================================
_SPECS = None


def _init_worker(specs):
    global _SPECS
    _SPECS = specs
    cv2.setNumThreads(1)


def _render_path(path):
    img = cv2.imread(path)
    if img is None:
        raise RuntimeError(f"Immagine non leggibile: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cc, vc = {}, {}
    return [render(img, sp, cc, vc) for sp in _SPECS]


def iter_rendered(paths, specs, num_workers, chunk):
    """Genera, per chunk di immagini, la lista [per immagine][per spec] -> uint8 RGB.
    Double-buffering: al massimo 2 chunk in RAM (evita la saturazione vista con le code illimitate)."""
    chunks = [paths[i:i + chunk] for i in range(0, len(paths), chunk)]
    if num_workers <= 0:
        _init_worker(specs)
        for c in chunks:
            yield [_render_path(p) for p in c]
        return
    ctx = mp.get_context("spawn")
    with ctx.Pool(num_workers, initializer=_init_worker, initargs=(specs,)) as pool:
        pending = pool.map_async(_render_path, chunks[0], chunksize=2)
        for i in range(len(chunks)):
            cur = pending.get()
            if i + 1 < len(chunks):
                pending = pool.map_async(_render_path, chunks[i + 1], chunksize=2)
            yield cur


def _fit224(a):
    return a if a.shape[:2] == (224, 224) else cv2.resize(a, (224, 224), interpolation=cv2.INTER_AREA)


# ============================================================
# 4) FEATURE: Swin-Tiny congelato (768-d) con cache su disco
# ============================================================
def swin_features(paths, specs, args):
    import torch
    import timm

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = timm.create_model(SWIN_NAME, pretrained=True, num_classes=0).to(dev).eval()
    for p in model.parameters():
        p.requires_grad = False
    mean = torch.tensor(IMAGENET_MEAN, device=dev).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=dev).view(1, 3, 1, 1)

    feats = [[] for _ in specs]
    done, t0 = 0, time.time()
    for rendered in iter_rendered(paths, specs, args.num_workers, args.batch_size):
        for k in range(len(specs)):
            arr = np.stack([_fit224(r[k]) for r in rendered])
            x = torch.from_numpy(arr).to(dev).permute(0, 3, 1, 2).float() / 255.0
            x = (x - mean) / std
            with torch.inference_mode():
                f = model(x)
            feats[k].append(f.float().cpu().numpy())
        done += len(rendered)
        if (done // args.batch_size) % 10 == 0:
            print(f"  [{done}/{len(paths)}] {time.time() - t0:.0f}s", flush=True)
    return [np.concatenate(f, axis=0) for f in feats]


def get_swin_embeddings(samples, configs, args):
    """configs: lista di (tag, Spec). Ritorna {tag: ndarray (N, 768)}. Cache in --cache_dir."""
    paths = [s["path"] for s in samples]
    fp = hashlib.sha1("|".join(paths).encode()).hexdigest()[:10]
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    embs, missing = {}, []
    for tag, sp in configs:
        f = cache_dir / f"swin_{fp}_{hashlib.sha1(repr(sp).encode()).hexdigest()[:10]}.npy"
        if f.exists() and not args.no_cache:
            embs[tag] = np.load(f)
            print(f"  cache: {tag}")
        else:
            missing.append((tag, sp, f))
    if missing:
        print(f"Estrazione Swin per {len(missing)} configurazioni: {[m[0] for m in missing]}")
        feats = swin_features(paths, [m[1] for m in missing], args)
        for (tag, sp, f), arr in zip(missing, feats):
            np.save(f, arr)
            embs[tag] = arr
    return embs


# ============================================================
# 5) METRICHE (EER esatto su tutte le coppie genuine)
# ============================================================
def collect_pair_scores(E, labels, groups, max_impostor=3_000_000, seed=0, block=512):
    """E: (N, D) con norma unitaria -> prodotto scalare = coseno (o NCC).
    Genuine: stessa identita' (tutte le coppie). Impostor: identita' diverse (sub-campionate se troppe).
    Coppie stesso soggetto / mano diversa (identity=hand) vengono ESCLUSE (ne' genuine ne' impostor)."""
    rng = np.random.RandomState(seed)
    N = len(E)
    counts = np.bincount(labels)
    n_gen_tot = int((counts * (counts - 1) // 2).sum())
    # coppie escluse (stesso gruppo, identita' diversa)
    gl = {}
    for g, l in zip(groups, labels):
        gl.setdefault(g, {}).setdefault(l, 0)
        gl[g][l] += 1
    n_excl = 0
    for d in gl.values():
        c = np.array(list(d.values()))
        tot = int(c.sum())
        n_excl += (tot * (tot - 1) // 2) - int((c * (c - 1) // 2).sum())
    n_imp_tot = N * (N - 1) // 2 - n_gen_tot - n_excl
    p = min(1.0, max_impostor / max(n_imp_tot, 1))

    gen, imp = [], []
    for i0 in range(0, N, block):
        i1 = min(N, i0 + block)
        S = E[i0:i1] @ E[i0:].T
        rows = np.arange(i0, i1)[:, None]
        cols = np.arange(i0, N)[None, :]
        upper = cols > rows
        same_l = labels[i0:i1, None] == labels[None, i0:]
        same_g = groups[i0:i1, None] == groups[None, i0:]
        g_mask = upper & same_l
        i_mask = upper & ~same_l & ~same_g
        gen.append(S[g_mask])
        s_imp = S[i_mask]
        if p < 1.0:
            s_imp = s_imp[rng.rand(len(s_imp)) < p]
        imp.append(s_imp)
    return np.concatenate(gen), np.concatenate(imp)


def eer_metrics(gen, imp):
    if len(gen) == 0 or len(imp) == 0:
        return None
    scores = np.concatenate([gen, imp])
    is_gen = np.concatenate([np.ones(len(gen), bool), np.zeros(len(imp), bool)])
    order = np.argsort(-scores, kind="stable")
    is_gen = is_gen[order]
    tar = np.cumsum(is_gen) / len(gen)
    far = np.cumsum(~is_gen) / len(imp)
    frr = 1.0 - tar
    idx = int(np.argmin(np.abs(far - frr)))

    def tar_at(f):
        m = far <= f
        return float(tar[m].max()) if m.any() else 0.0

    pooled = np.sqrt((gen.var() + imp.var()) / 2.0)
    return {
        "eer": float((far[idx] + frr[idx]) / 2.0),
        "eer_threshold": float(scores[order][idx]),
        "tar@far1e-2": tar_at(1e-2),
        "tar@far1e-3": tar_at(1e-3),
        "dprime": float((gen.mean() - imp.mean()) / pooled) if pooled > 0 else 0.0,
        "gen_mean": float(gen.mean()),
        "imp_mean": float(imp.mean()),
        "n_genuine": int(len(gen)),
        "n_impostor": int(len(imp)),
    }


def l2norm(X):
    X = X.astype(np.float32)
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)


def evaluate_embeddings(E, labels, groups, args):
    E = l2norm(E)
    gen, imp = collect_pair_scores(E, labels, groups, seed=args.seed)
    res = eer_metrics(gen, imp)
    if res is None:
        raise SystemExit("Coppie genuine/impostor insufficienti (servono >=2 campioni per identita').")
    if args.n_sub > 0:
        rng = np.random.RandomState(args.seed + 1)
        ug = np.unique(groups)
        k = max(2, int(round(0.8 * len(ug))))
        eers = []
        for _ in range(args.n_sub):
            keep = rng.choice(ug, k, replace=False)
            m = np.isin(groups, keep)
            _, new_l = np.unique(labels[m], return_inverse=True)
            g_s, i_s = collect_pair_scores(E[m], new_l, groups[m], max_impostor=500_000,
                                           seed=int(rng.randint(1 << 30)))
            r = eer_metrics(g_s, i_s)
            if r:
                eers.append(r["eer"])
        if eers:
            res.update({"eer_sub_mean": float(np.mean(eers)), "eer_sub_std": float(np.std(eers)),
                        "eer_sub_ci_lo": float(np.percentile(eers, 2.5)),
                        "eer_sub_ci_hi": float(np.percentile(eers, 97.5))})
    return res


# ============================================================
# 6) OUTPUT
# ============================================================
def print_table(rows, cols, title):
    print("\n" + "=" * 100 + f"\n{title}\n" + "=" * 100)
    widths = [max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in cols]
    print(" | ".join(c.ljust(w) for c, w in zip(cols, widths)))
    print("-+-".join("-" * w for w in widths))
    for r in rows:
        print(" | ".join(str(r.get(c, "")).ljust(w) for c, w in zip(cols, widths)))


def fmt_rows(rows):
    out = []
    for r in rows:
        o = dict(r)
        for k in ("eer", "eer_sub_mean", "eer_sub_std", "eer_sub_ci_lo", "eer_sub_ci_hi",
                  "tar@far1e-2", "tar@far1e-3"):
            if k in o and isinstance(o[k], float):
                o[k] = f"{100 * o[k]:.2f}%"
        for k in ("dprime", "eer_threshold", "gen_mean", "imp_mean"):
            if k in o and isinstance(o[k], float):
                o[k] = f"{o[k]:.3f}"
        if "eer_sub_mean" in o:
            o["eer_sub"] = f"{o['eer_sub_mean']} +/- {o['eer_sub_std']}"
            o["ci95"] = f"[{o['eer_sub_ci_lo']}, {o['eer_sub_ci_hi']}]"
        out.append(o)
    return out


def save_results(rows, out_dir, name):
    out_dir = Path(out_dir)
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(out_dir / f"{name}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    with open(out_dir / f"{name}.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print(f"\nSalvato: {out_dir / (name + '.csv')} / .json")


# ============================================================
# 7) STUDIO A - varianti di input (Swin congelato + coseno)
# ============================================================
def cmd_inputs(args):
    samples, labels, groups = select_samples(args)
    configs = [(v, make_spec(v)) for v in args.variants]
    embs = get_swin_embeddings(samples, configs, args)

    rows = []
    for tag, _ in configs:
        res = evaluate_embeddings(embs[tag], labels, groups, args)
        rows.append({"study": "A_inputs", "variant": tag, "feat_dim": int(embs[tag].shape[1]),
                     "n_samples": len(samples), **res})
        print(f"{tag:8s} EER = {100 * res['eer']:.2f}%")
    ref = next((r["eer"] for r in rows if r["variant"] == "raw"), None)
    for r in rows:
        r["delta_vs_raw_pp"] = round(100 * (r["eer"] - ref), 3) if ref is not None else ""
    cols = ["variant", "eer", "eer_sub", "ci95", "delta_vs_raw_pp", "tar@far1e-2", "dprime", "n_genuine"]
    print_table(fmt_rows(rows), [c for c in cols if c in fmt_rows(rows)[0]],
                "A) Swin-Tiny congelato, coseno su 768-d (nessun training)")
    save_results(rows, args.out_dir, "A_inputs")


# ============================================================
# 8) STUDIO B - sensibilita' ai parametri di enhance_veins
# ============================================================
SWEEPS = {
    "weight": [0.0, 0.25, 0.40, 0.55, 0.70, 0.85, 1.0],
    "clip": [0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0],            # 0.0 = nessuna CLAHE
    "sigma": [(1,), (2,), (3,), (4,), (1, 2), (1, 2, 3), (1, 2, 3, 4), (2, 3, 4), (1, 2, 3, 4, 5, 6)],
    "norm": ["max", "p99", "p95", "none"],
}
SWEEP_DEFAULT = {"weight": DEF_WEIGHT, "clip": DEF_CLIP, "sigma": DEF_SIGMAS, "norm": DEF_NORM}


def _label(param, v):
    return "-".join(map(str, v)) if param == "sigma" else str(v)


def cmd_sweep(args):
    samples, labels, groups = select_samples(args)

    entries = []   # (sweep, value_label, is_default, spec)
    for param in args.sweeps:
        for v in SWEEPS[param]:
            kw = dict(clip=DEF_CLIP, sigmas=DEF_SIGMAS, weight=DEF_WEIGHT, norm=DEF_NORM)
            kw[{"sigma": "sigmas"}.get(param, param)] = v
            entries.append((param, _label(param, v), v == SWEEP_DEFAULT[param], make_spec("full", **kw)))

    unique = {}   # spec -> tag (calcolo una sola volta per configurazione identica)
    for _, _, _, sp in entries:
        unique.setdefault(sp, f"cfg{len(unique):02d}")
    configs = [(tag, sp) for sp, tag in unique.items()]
    if args.with_refs:
        configs = [("ref_raw", make_spec("raw")), ("ref_clahe", make_spec("clahe"))] + configs
    print(f"Configurazioni uniche da estrarre: {len(configs)}")
    embs = get_swin_embeddings(samples, configs, args)

    cache = {}
    def ev(tag):
        if tag not in cache:
            cache[tag] = evaluate_embeddings(embs[tag], labels, groups, args)
        return cache[tag]

    rows = []
    if args.with_refs:
        for tag, name in (("ref_raw", "raw"), ("ref_clahe", "clahe (solo)")):
            rows.append({"study": "B_sweep", "sweep": "reference", "value": name, "default": "",
                         **ev(tag)})
    for param, lab, is_def, sp in entries:
        rows.append({"study": "B_sweep", "sweep": param, "value": lab,
                     "default": "*" if is_def else "", **ev(unique[sp])})

    cols = ["sweep", "value", "default", "eer", "eer_sub", "ci95", "tar@far1e-2", "dprime"]
    fr = fmt_rows(rows)
    print_table(fr, [c for c in cols if c in fr[0]],
                "B) Sensibilita' ai parametri di enhance_veins (* = default). Swin congelato + coseno")

    print("\nRiepilogo di stabilita' (un parametro alla volta, EER in punti percentuali):")
    summary = []
    for param in args.sweeps:
        e = np.array([100 * r["eer"] for r in rows if r["sweep"] == param])
        d = next(100 * r["eer"] for r in rows if r["sweep"] == param and r["default"] == "*")
        s = {"study": "B_summary", "sweep": param, "eer_default": round(d, 3), "eer_min": round(e.min(), 3),
             "eer_max": round(e.max(), 3), "range_pp": round(e.max() - e.min(), 3),
             "std_pp": round(float(e.std()), 3)}
        summary.append(s)
        print(f"  {param:7s} default={s['eer_default']:.2f}  min={s['eer_min']:.2f}  "
              f"max={s['eer_max']:.2f}  range={s['range_pp']:.2f}  std={s['std_pp']:.2f}")
    save_results(rows + summary, args.out_dir, "B_sweep")


# ============================================================
# 9) STUDIO C - handcrafted NCC senza modelli
# ============================================================
def ncc_vectors(maps, blur):
    """maps: (N, s, s) float32 -> (N, s*s) con media zero e norma 1 (prodotto scalare = NCC globale)."""
    out = np.empty((len(maps), maps.shape[1] * maps.shape[2]), np.float32)
    for i, m in enumerate(maps):
        if blur > 0:
            m = cv2.GaussianBlur(m, (0, 0), blur)
        v = m.reshape(-1) - m.mean()
        out[i] = v / max(float(np.linalg.norm(v)), 1e-12)
    return out


def cmd_ncc(args):
    samples, labels, groups = select_samples(args)
    paths = [s["path"] for s in samples]
    map_specs = [("gray", make_spec("raw")), ("clahe_gray", make_spec("clahe")),
                 ("green", make_spec("green")), ("vessel", make_spec("frangi"))]
    specs = [sp for _, sp in map_specs]
    S = args.ncc_size
    maps = [[] for _ in specs]
    done = 0
    for rendered in iter_rendered(paths, specs, args.num_workers, args.batch_size):
        for r in rendered:
            for k in range(len(specs)):
                g = cv2.cvtColor(r[k], cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
                maps[k].append(cv2.resize(g, (S, S), interpolation=cv2.INTER_AREA))
        done += len(rendered)
        print(f"  [{done}/{len(paths)}]", flush=True)

    rows = []
    for (name, _), mk in zip(map_specs, maps):
        mk = np.stack(mk)
        for b in args.ncc_blur:
            E = ncc_vectors(mk, b)
            res = evaluate_embeddings(E, labels, groups, args)   # E gia' unitario -> prodotto = NCC
            rows.append({"study": "C_ncc", "map": name, "blur_sigma": b, "size": S, **res})
            print(f"{name:11s} blur={b:<4} EER = {100 * res['eer']:.2f}%")
    cols = ["map", "blur_sigma", "eer", "eer_sub", "ci95", "tar@far1e-2", "dprime", "n_genuine"]
    fr = fmt_rows(rows)
    print_table(fr, [c for c in cols if c in fr[0]],
                f"C) NCC globale tra mappe ({S}x{S}), nessun modello neurale")
    save_results(rows, args.out_dir, "C_ncc")


# ============================================================
# 9b) STUDIO D - testa MLP + ArcFace su Swin congelato, K-fold per soggetto
# ============================================================
# Replica la testa addestrabile di DorsalTextureBranch (768 -> 512 -> GELU -> Dropout 0.2 -> out_dim),
# ArcFace (margine 0.30, scala 30), AdamW lr 3e-4 wd 1e-4, cosine, clip grad 5.0, 60 epoche.
# Il backbone e' congelato: si lavora sugli embedding Swin in cache (nessun Frangi/Swin a runtime).
# Valutazione: EER sui soggetti del fold di test, MAI visti in training (open-set).
PRESETS = {
    "raw": make_spec("raw"),
    "clahe": make_spec("clahe"),
    "green": make_spec("green"),
    "frangi": make_spec("frangi"),
    "full": make_spec("full"),
    # varianti con scale di Frangi piu' grandi (sigma=1 escluso). Se "full_s*" proviene dallo sweep B
    # l'embedding e' gia' in cache; "frangi_s*" richiede una nuova estrazione.
    "frangi_s3": make_spec("frangi", sigmas=(3,)),
    "frangi_s234": make_spec("frangi", sigmas=(2, 3, 4)),
    "full_s3": make_spec("full", sigmas=(3,)),
    "full_s234": make_spec("full", sigmas=(2, 3, 4)),
}


def subject_folds(groups, n_folds, seed):
    """K-fold deterministico PER SOGGETTO (stessa logica di dorsal_core.kfold_subject_splits).
    Yield: (indice fold da 1, maschera booleana dei campioni di test)."""
    ug = np.array(sorted(np.unique(groups)))
    if len(ug) < n_folds:
        raise SystemExit(f"Servono almeno {n_folds} soggetti, trovati {len(ug)}")
    rng = np.random.RandomState(seed)
    rng.shuffle(ug)
    sizes = np.full(n_folds, len(ug) // n_folds, dtype=int)
    sizes[: len(ug) % n_folds] += 1
    cur = 0
    for f, sz in enumerate(sizes, start=1):
        yield f, np.isin(groups, ug[cur:cur + sz])
        cur += sz


def train_mlp_head(Xtr, ytr, args, seed):
    """Addestra MLP + ArcFace sugli embedding Swin. Ritorna una funzione embed(X) -> ndarray L2-normalizzato."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    n_cls = int(ytr.max()) + 1
    mlp = nn.Sequential(nn.Linear(Xtr.shape[1], 512), nn.GELU(), nn.Dropout(0.2),
                        nn.Linear(512, args.out_dim)).to(dev)
    W = nn.Parameter(torch.randn(n_cls, args.out_dim, device=dev) * 0.01)
    params = list(mlp.parameters()) + [W]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs))

    X = torch.from_numpy(np.ascontiguousarray(Xtr, dtype=np.float32)).to(dev)
    y = torch.from_numpy(ytr.astype(np.int64)).to(dev)
    gen = torch.Generator().manual_seed(seed)
    n = len(X)
    for _ in range(args.epochs):
        mlp.train()
        perm = torch.randperm(n, generator=gen).to(dev)
        for i in range(0, n, args.train_batch):
            idx = perm[i:i + args.train_batch]
            emb = F.normalize(mlp(X[idx]), p=2, dim=1)
            cos = F.linear(emb, F.normalize(W, p=2, dim=1)).clamp(-1 + 1e-7, 1 - 1e-7)
            target = torch.cos(torch.acos(cos) + args.margin)
            one_hot = torch.zeros_like(cos).scatter_(1, y[idx].view(-1, 1), 1.0)
            logits = (one_hot * target + (1.0 - one_hot) * cos) * args.scale
            loss = F.cross_entropy(logits, y[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
        sched.step()
    mlp.eval()

    @torch.no_grad()
    def embed(Xe):
        xe = torch.from_numpy(np.ascontiguousarray(Xe, dtype=np.float32)).to(dev)
        return F.normalize(mlp(xe), p=2, dim=1).cpu().numpy()

    return embed


def cmd_mlp(args):
    samples, labels, groups = select_samples(args)
    names = list(dict.fromkeys(args.variants))
    embs = get_swin_embeddings(samples, [(n, PRESETS[n]) for n in names], args)

    rows = []
    pooled = {(n, r): ([], []) for n in names for r in range(args.repeats)}
    t0 = time.time()
    for rep in range(args.repeats):
        for fold, te in subject_folds(groups, args.folds, args.seed):
            tr = ~te
            _, ytr = np.unique(labels[tr], return_inverse=True)
            _, yte = np.unique(labels[te], return_inverse=True)
            gte = groups[te]
            for n in names:
                X = embs[n]
                # riferimento: coseno sul Swin congelato, stesso fold di test
                gf, imf = collect_pair_scores(l2norm(X[te]), yte, gte, seed=args.seed)
                frozen = eer_metrics(gf, imf)["eer"]
                # testa addestrata
                embed = train_mlp_head(X[tr], ytr, args, seed=args.seed + 1000 * rep + fold)
                gen, imp = collect_pair_scores(embed(X[te]), yte, gte, seed=args.seed)
                m = eer_metrics(gen, imp)
                pooled[(n, rep)][0].append(gen)
                pooled[(n, rep)][1].append(imp)
                rows.append({"study": "D_mlp", "variant": n, "rep": rep, "fold": fold,
                             "n_test_subjects": int(len(np.unique(gte))), "eer": m["eer"],
                             "tar@far1e-2": m["tar@far1e-2"], "dprime": m["dprime"],
                             "eer_frozen": frozen})
                print(f"  rep {rep} fold {fold} {n:12s} EER={100 * m['eer']:.2f}%  "
                      f"(frozen {100 * frozen:.2f}%)  [{time.time() - t0:.0f}s]", flush=True)

    # ---- riepilogo per variante
    def per_fold(n, key):
        return {(r["rep"], r["fold"]): r[key] for r in rows if r["variant"] == n}

    ref = per_fold("raw", "eer") if "raw" in names else None
    summary = []
    for n in names:
        e = np.array(list(per_fold(n, "eer").values()))
        pooled_eer = float(np.mean([eer_metrics(np.concatenate(pooled[(n, r)][0]),
                                                np.concatenate(pooled[(n, r)][1]))["eer"]
                                    for r in range(args.repeats)]))
        s = {"study": "D_summary", "variant": n, "eer_fold_mean": float(e.mean()),
             "eer_fold_std": float(e.std()), "eer_pooled": pooled_eer,
             "eer_frozen_mean": float(np.mean(list(per_fold(n, "eer_frozen").values()))),
             "tar@far1e-2": float(np.mean(list(per_fold(n, "tar@far1e-2").values()))),
             "dprime": float(np.mean(list(per_fold(n, "dprime").values())))}
        if ref is not None:
            cur = per_fold(n, "eer")
            d = np.array([100 * (cur[k] - ref[k]) for k in cur])
            s.update({"delta_vs_raw_pp_mean": float(d.mean()), "delta_vs_raw_pp_std": float(d.std()),
                      "folds_better_than_raw": f"{int((d < 0).sum())}/{len(d)}"})
        summary.append(s)

    tab = []
    for s in summary:
        t = {"variant": s["variant"],
             "eer (fold mean +/- std)": f"{100 * s['eer_fold_mean']:.2f}% +/- {100 * s['eer_fold_std']:.2f}%",
             "eer_pooled": f"{100 * s['eer_pooled']:.2f}%",
             "eer_frozen": f"{100 * s['eer_frozen_mean']:.2f}%",
             "tar@far1e-2": f"{100 * s['tar@far1e-2']:.2f}%",
             "dprime": f"{s['dprime']:.3f}"}
        if "delta_vs_raw_pp_mean" in s:
            t["delta_vs_raw_pp"] = f"{s['delta_vs_raw_pp_mean']:+.2f} +/- {s['delta_vs_raw_pp_std']:.2f}"
            t["folds_better"] = s["folds_better_than_raw"]
        tab.append(t)
    print_table(tab, list(tab[0].keys()),
                f"D) MLP+ArcFace su Swin congelato, {args.folds}-fold per soggetto "
                f"(x{args.repeats} ripetizioni), test su soggetti mai visti")
    print("\n(eer_frozen = coseno sul Swin congelato negli stessi fold di test, riferimento senza training)")
    save_results(rows + summary, args.out_dir, "D_mlp")


# ============================================================
# 10) EXAMPLES + SELFTEST
# ============================================================
def cmd_examples(args):
    samples, _, _ = select_samples(args)
    rng = np.random.RandomState(args.seed)
    idx = rng.choice(len(samples), min(args.n, len(samples)), replace=False)
    specs = [make_spec(v) for v in ["raw", "clahe", "green", "frangi", "full"]]
    names = ["raw", "clahe", "green", "frangi", "full"]
    out = Path(args.out_dir) / "examples"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in idx:
        img = cv2.cvtColor(cv2.imread(samples[i]["path"]), cv2.COLOR_BGR2RGB)
        cc, vc = {}, {}
        tiles = []
        for n, sp in zip(names, specs):
            t = render(img, sp, cc, vc).copy()
            cv2.putText(t, n, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1, cv2.LINE_AA)
            tiles.append(t)
        panel = np.concatenate(tiles, axis=1)
        name = Path(samples[i]["path"]).stem
        cv2.imwrite(str(out / f"{name}_panel.png"), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out / f"{name}_full.png"), cv2.cvtColor(tiles[-1], cv2.COLOR_RGB2BGR))
        rows.append(panel)
    print(f"Salvati {len(idx)} pannelli in {out}")


def cmd_selftest(args):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from dorsal_core import enhance_veins   # dorsal_core.py nella stessa cartella
    samples, _, _ = select_samples(args)
    rng = np.random.RandomState(args.seed)
    idx = rng.choice(len(samples), min(args.n, len(samples)), replace=False)
    worst = 0
    for i in idx:
        img = cv2.cvtColor(cv2.imread(samples[i]["path"]), cv2.COLOR_BGR2RGB)
        a = enhance_veins(img)
        b = render(img, make_spec("full"), {}, {})
        worst = max(worst, int(np.abs(a.astype(int) - b.astype(int)).max()))
    print(f"Differenza massima per pixel tra dorsal_core.enhance_veins e la versione parametrica: {worst}")
    print("OK: identiche." if worst == 0 else "ATTENZIONE: differenze presenti.")


# ============================================================
# 11) CLI
# ============================================================
def main():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data_dir", default="./dataset_preprocessed")
    common.add_argument("--out_dir", default="./ablation_out")
    common.add_argument("--cache_dir", default="./ablation_out/cache")
    common.add_argument("--no_cache", action="store_true")
    common.add_argument("--subjects_file", default=None)
    common.add_argument("--subjects_key", default=None)
    common.add_argument("--max_subjects", type=int, default=0)
    common.add_argument("--identity", choices=["subject", "hand"], default="subject")
    common.add_argument("--seed", type=int, default=42)
    common.add_argument("--batch_size", type=int, default=32)
    common.add_argument("--num_workers", type=int, default=2)
    common.add_argument("--n_sub", type=int, default=20,
                        help="sub-campionamenti (80%% soggetti) per mean/std/IC95 dell'EER; 0 = disattivato")

    ap = argparse.ArgumentParser(description="Ablazioni senza training - stream dorso")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("inputs", parents=[common], help="A) varianti di input, Swin congelato + coseno")
    p.add_argument("--variants", nargs="+", default=["raw", "clahe", "green", "frangi", "full"],
                   choices=["raw", "clahe", "green", "frangi", "full"])
    p.set_defaults(fn=cmd_inputs)

    p = sub.add_parser("sweep", parents=[common], help="B) sensibilita' parametri enhance_veins")
    p.add_argument("--sweeps", nargs="+", default=["weight", "clip", "sigma", "norm"],
                   choices=list(SWEEPS.keys()))
    p.add_argument("--with_refs", type=lambda s: s.lower() in ("1", "true", "yes"), default=True)
    p.set_defaults(fn=cmd_sweep)

    p = sub.add_parser("ncc", parents=[common], help="C) baseline handcrafted NCC")
    p.add_argument("--ncc_size", type=int, default=64)
    p.add_argument("--ncc_blur", type=float, nargs="+", default=[0.0, 1.5, 3.0],
                   help="sigma del blur gaussiano (px a ncc_size) applicato prima dell'NCC")
    p.set_defaults(fn=cmd_ncc)

    p = sub.add_parser("mlp", parents=[common],
                       help="D) MLP+ArcFace su Swin congelato (cache), K-fold per soggetto")
    p.add_argument("--variants", nargs="+", default=["raw", "clahe", "green", "frangi", "full"],
                   choices=list(PRESETS))
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--repeats", type=int, default=1, help="ripetizioni con seed di training diversi")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--train_batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--out_dim", type=int, default=128)
    p.add_argument("--margin", type=float, default=0.30)
    p.add_argument("--scale", type=float, default=30.0)
    p.set_defaults(fn=cmd_mlp)

    p = sub.add_parser("examples", parents=[common], help="salva pannelli raw|clahe|green|frangi|full")
    p.add_argument("--n", type=int, default=6)
    p.set_defaults(fn=cmd_examples)

    p = sub.add_parser("selftest", parents=[common], help="confronta con dorsal_core.enhance_veins")
    p.add_argument("--n", type=int, default=10)
    p.set_defaults(fn=cmd_selftest)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()