#!/usr/bin/env python3
"""Train a skip-gram Word2Vec model on text8 and export compact static files
for the in-browser demo (assets/w2v/{vocab.json,vectors.bin,meta.json}).

Usage:
    python3 w2v-training/train.py [--data-dir DIR] [--out-dir assets/w2v] [--epochs 10]

Requires: gensim>=4, numpy.  text8 (~31 MB gz, 100 MB raw, ~17M tokens of
cleaned English Wikipedia, March 2006) is downloaded into --data-dir if missing.
"""
import argparse, gzip, json, logging, multiprocessing, os, re, shutil, time, urllib.request
import numpy as np
from gensim.models import Word2Vec

TEXT8_URL = "https://github.com/RaRe-Technologies/gensim-data/releases/download/text8/text8.gz"
TOKEN_RE = re.compile(r"^[a-z][a-z'\-]*$")  # alphabetic tokens only (no punctuation/digit-only)
KEEP_SINGLE = {"a", "i"}


def ensure_text8(data_dir):
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, "text8")
    if not os.path.exists(path):
        gz = path + ".gz"
        print("Downloading", TEXT8_URL)
        urllib.request.urlretrieve(TEXT8_URL, gz)
        with gzip.open(gz, "rb") as src, open(path, "wb") as dst:
            shutil.copyfileobj(src, dst)
        os.remove(gz)
    return path


def load_sentences(path, chunk=1000):
    """text8 is one long line; split into pseudo-sentences of `chunk` tokens."""
    with open(path, encoding="utf-8") as f:
        words = f.read().lower().split()
    words = [w for w in words if TOKEN_RE.match(w)]
    return [words[i:i + chunk] for i in range(0, len(words), chunk)], len(words)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.path.expanduser("~/.cache/w2v-text8"))  # outside the repo
    ap.add_argument("--out-dir", default=os.path.join(here, "..", "assets", "w2v"))
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--min-count", type=int, default=5)
    ap.add_argument("--negative", type=int, default=10)
    ap.add_argument("--top", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)

    sentences, n_tokens = load_sentences(ensure_text8(args.data_dir))
    print(f"tokens: {n_tokens:,}")

    params = dict(sg=1, vector_size=100, window=5, negative=args.negative,
                  epochs=args.epochs, min_count=args.min_count)
    t0 = time.time()
    model = Word2Vec(sentences, workers=multiprocessing.cpu_count(), sample=1e-4,
                     seed=args.seed, **params)
    print(f"trained in {time.time() - t0:.0f}s, full vocab {len(model.wv):,}")

    # index_to_key is sorted by descending frequency; drop 1-letter noise (except a/i)
    words = [w for w in model.wv.index_to_key if len(w) > 1 or w in KEEP_SINGLE][:args.top]
    vecs = np.stack([model.wv[w] for w in words]).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    q = np.clip(np.round(vecs * 127), -127, 127).astype(np.int8)

    os.makedirs(args.out_dir, exist_ok=True)
    q.tofile(os.path.join(args.out_dir, "vectors.bin"))
    with open(os.path.join(args.out_dir, "vocab.json"), "w") as f:
        json.dump(words, f, separators=(",", ":"))
    meta = {"dim": 100, "count": len(words), "scale": 127, "normalized": True,
            "corpus": "text8 (first 100 MB of cleaned English Wikipedia, March 2006)",
            "tokens": n_tokens, "params": params, "trained": time.strftime("%Y-%m-%d")}
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("exported", len(words), "words to", os.path.abspath(args.out_dir))


if __name__ == "__main__":
    main()
