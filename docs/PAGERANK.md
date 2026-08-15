# Filtering Algorithm: PageRank-based Risk Aggregation

How QGuard turns per-question yes-probabilities into a single harmful/unharmful decision. Paper reference: Sec 3.4 (algorithm) and Sec 4.1.3 (actual edge-weight values used) of [PAPER.md](PAPER.md) (arXiv:2506.12299). Code: [`qguard/graph.py`](../qguard/graph.py), wired in [`qguard/pipeline.py`](../qguard/pipeline.py).

## Why not just average?

Each guard question gets a `yes_prob` from the MLLM. The naive baseline is `mean(yes_prob) > 0.5`. The paper reports this underperforms (Sec 4.3.2) because it treats every question independently and ignores two structural signals:

- **Question overlap** — questions in the same group often probe the same underlying risk; several of them lighting up together is a stronger signal than the same average spread thin across unrelated questions.
- **Group relationships** — a risk detected in one category (e.g. hate speech) can be relevant context for another (e.g. threats).

QGuard encodes both as a weighted graph and lets PageRank aggregate them.

## Graph construction

Two node types:

- **Question nodes** — one per guard question
- **Group nodes** — one per category (`General Toxic`, `Toxic Prompt`, `Core Harmfulness Detection`, `Additional Nuanced Questions` by default)

Three edge types:

| Edge | Weight | Meaning |
|---|---|---|
| question → its group | `yes_prob` (clamped to [0,1]) | how strongly this question fired |
| group ↔ group | `similarity(g_i, g_j)` if known, else `0.1` (Sec 3.4 formula); **fixed at `1.0`** in the actual experiments (Sec 4.1.3) — no similarity table is implemented here, so this repo always uses the fixed value | cross-category leakage of risk |
| question ↔ question (same group) | constant, `0.3` (Sec 3.4 and 4.1.3 agree) | overlap between questions probing similar risk |

This builds a directed, weighted graph — questions feed their group, and groups/questions leak weight to their peers.

## PageRank

PageRank models a random walker moving along edges, weighted by edge weight; a node's score is its steady-state visiting probability — how much "traffic" it accumulates once the walk settles.

$$PR(v) = (1-d) + d\sum_{u \in \text{In}(v)} \frac{w_{uv}\,PR(u)}{\sum_{z \in \text{Out}(u)} w_{uz}}$$

`d` is the damping factor (0.85 default in `networkx`). This is computed via `nx.pagerank(G, weight="weight")` — internally, power iteration until convergence, which is exactly the "risk propagating through the graph" step: a node's importance depends on how important its incoming neighbors are, recursively.

## Risk score

$$\text{Risk Score} = \sum_{n \in V} PR(n) \times \sum_{(n \to m) \in E} w_{nm}$$

For every node, multiply its PageRank by its total outgoing edge weight, then sum over all nodes. A node scores high when it both (a) receives a lot of propagated importance and (b) forwards a lot of weight onward — i.e. risk that's both concentrated and actively spreading. Compare to `threshold` (default `0.50`): `risk > threshold` → `harmful`.

## Two knobs

- `group_coupling` (default `1.0`, matches paper Sec 4.1.3) — how much risk leaks between different categories. The `0.1` value that appears in the Sec 3.4 formula for $w_{g_ig_j}$ is only the no-similarity fallback in the general algorithm description; Sec 4.1.3 states the actual experiments fix every group↔group edge at `1.0`, and `graph.py`'s default reflects that.
- `intra_group_q_coupling` (default `0.3`, matches paper Sec 4.1.3) — how much overlap is assumed between questions in the same group.

Set both to `0` and the graph degenerates to independent question→group stars — PageRank then behaves close to a weighted average, which is the ablation baseline the paper compares against.

## One implementation detail worth knowing

Both group↔group and same-group question↔question edges are added via `itertools.combinations`, i.e. `(a, b)` only, not `(b, a)` — since the graph is a `DiGraph`, these are one-directional edges, not mutual pairs. The paper's phrasing ("we add **a** directed edge between them") matches this, so the one-directional design is intentional, not a bug — but easy to misread as symmetric coupling if you're modifying `graph.py`. A `symmetric_coupling` flag now exists to run the bidirectional variant for comparison; empirically it shifts the risk-score scale substantially (F1-optimal θ moves from ~0.89 to ~2.6) and roughly halves Financial_Advice recall, so it is not a drop-in improvement over the paper's one-directional design.

## Fixed: group↔group edge direction used to be non-deterministic

`groups` — the set of category names feeding `itertools.combinations(groups, 2)` for the group↔group edges — used to be built as a bare Python `set`. Iterating a `set` of strings is ordered by hash, and CPython randomizes string hashing per process (`PYTHONHASHSEED`) unless pinned. That meant which group landed as `g1` vs `g2` in every pair — and therefore the entire directed-edge structure between groups — changed **on every process run**, even for the exact same `yes_prob` input and the exact same `group_coupling`/`intra_group_q_coupling` values. `risk_score` was not reproducible across runs; re-running the identical row three times produced three different scores (verified 2026-08-18, swings up to ~0.07 on a [0.8, 1.4]-wide score band — enough to flip the F1-optimal θ from a degenerate "predict everything harmful" point to a normal interior threshold, or back).

Fixed by sorting `groups` before iterating (`sorted({...})`) so edge direction is fixed by name, not by process-local hash state. With the fix, re-scoring `assets/mmsafety_full` at the paper's own Sec 4.1.3 settings (`group_coupling=1.0`, `intra_group_q_coupling=0.3`, one-directional, `alpha=0.85`) gives a non-degenerate F1-optimal θ=0.8947 (AUROC 0.8888, precision 0.8265, recall 0.7881) and a Financial_Advice recall of **0.2335 — an exact match to the paper's own reported multi-modal figure**. The `risk_score` values already stored in `assets/mmsafety_full/*.jsonl` predate this fix and should be treated as one arbitrary (and, on this run, unusually bad) draw rather than the ground truth; re-derive with `--group_coupling 1.0` (which forces a rescore through the fixed code) rather than trusting the stored field as-is.
