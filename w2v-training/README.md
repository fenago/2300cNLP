# Word2Vec training (for the in-browser embedding demo)

`train.py` trains a skip-gram Word2Vec model with gensim on **text8** (the first
100 MB of cleaned English Wikipedia, March 2006; ~17M tokens) and exports the
compact static files the site loads:

| file | contents |
|---|---|
| `assets/w2v/vocab.json` | JSON array of the top 20,000 words, most frequent first |
| `assets/w2v/vectors.bin` | raw `Int8Array`, row-major `count x dim`; each vector L2-normalized, then `round(v*127)` clipped to [-127,127] |
| `assets/w2v/meta.json` | dim, count, scale, corpus, token count, training params, date |

In the browser, cosine similarity between two rows is just their int8 dot
product divided by the product of their norms (≈ 127² since rows are normalized).

## Rerun

```bash
pip install gensim numpy
python3 w2v-training/train.py      # downloads text8 to ~/.cache/w2v-text8 on first run
# options: --epochs 5 --negative 10 --min-count 5 --top 20000 --out-dir assets/w2v
```

The corpus is fetched from the gensim-data GitHub release
(`RaRe-Technologies/gensim-data`, tag `text8`). The default `--data-dir` (`~/.cache/w2v-text8`) is outside the
repo, so the 100 MB corpus is never committed. Training takes
~3.5 min for 5 epochs / ~7 min for 10 epochs on 4 cores.

Note: text8 spells out digits ("one", "nine", "zero"), so those words are very
frequent, and it contains no punctuation or capitalization.
