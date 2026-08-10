# Filtering Algorithm: PageRank-based Risk Aggregation

How QGuard turns per-question yes-probabilities into a single harmful/unharmful decision. Paper reference: Sec 3.4 of [PAPER.md](PAPER.md) (arXiv:2506.12299). Code: [`qguard/graph.py`](../qguard/graph.py), wired in [`qguard/pipeline.py`](../qguard/pipeline.py).

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
| group ↔ group | `similarity(g_i, g_j)` if known, else `0.1` | cross-category leakage of risk |
| question ↔ question (same group) | constant, `0.3` | overlap between questions probing similar risk |

This builds a directed, weighted graph — questions feed their group, and groups/questions leak weight to their peers.

## PageRank

PageRank models a random walker moving along edges, weighted by edge weight; a node's score is its steady-state visiting probability — how much "traffic" it accumulates once the walk settles.

$$PR(v) = (1-d) + d\sum_{u \in \text{In}(v)} \frac{w_{uv}\,PR(u)}{\sum_{z \in \text{Out}(u)} w_{uz}}$$

`d` is the damping factor (0.85 default in `networkx`). This is computed via `nx.pagerank(G, weight="weight")` — internally, power iteration until convergence, which is exactly the "risk propagating through the graph" step: a node's importance depends on how important its incoming neighbors are, recursively.

## Risk score

$$\text{Risk Score} = \sum_{n \in V} PR(n) \times \sum_{(n \to m) \in E} w_{nm}$$

For every node, multiply its PageRank by its total outgoing edge weight, then sum over all nodes. A node scores high when it both (a) receives a lot of propagated importance and (b) forwards a lot of weight onward — i.e. risk that's both concentrated and actively spreading. Compare to `threshold` (default `0.50`): `risk > threshold` → `harmful`.

## Two knobs

- `group_coupling` (default `0.1`) — how much risk leaks between different categories. Paper's actual experiments use `1.0` for group↔group edges (Sec 4.1.3) rather than the `0.1` code default — worth confirming which value a given run should use before comparing against reported numbers.
- `intra_group_q_coupling` (default `0.3`) — how much overlap is assumed between questions in the same group.

Set both to `0` and the graph degenerates to independent question→group stars — PageRank then behaves close to a weighted average, which is the ablation baseline the paper compares against.

## One implementation detail worth knowing

Both group↔group and same-group question↔question edges are added via `itertools.combinations`, i.e. `(a, b)` only, not `(b, a)` — since the graph is a `DiGraph`, these are one-directional edges, not mutual pairs. The paper's phrasing ("we add **a** directed edge between them") matches this, so it's intentional, not a bug — but easy to misread as symmetric coupling if you're modifying `graph.py`.
