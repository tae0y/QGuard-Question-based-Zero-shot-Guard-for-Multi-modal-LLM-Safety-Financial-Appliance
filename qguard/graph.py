from typing import Iterable, Dict, List, Set
import itertools
import networkx as nx

def build_graph_from_results(
    question_results: Iterable[Dict],
    group_by_category: Dict[str, str],
    group_coupling: float = 1.0,  # paper Sec 4.1.3; the 0.1 in Eq. for w_{g_i g_j} is the no-similarity fallback
    intra_group_q_coupling: float = 0.3,
    symmetric_coupling: bool = False,  # paper adds "a directed edge" (one-way); True adds both directions
) -> nx.DiGraph:
    G = nx.DiGraph()

    for item in question_results:
        q = item["question"]
        p = float(item["yes_prob"])
        g = group_by_category.get(q, "Unknown")
        G.add_node(q, type="question")
        G.add_node(g, type="group")
        G.add_edge(q, g, weight=max(0.0, min(1.0, p)))

    # list(dict.fromkeys(...)), not a bare set: iterating a set of str is
    # ordered by hash, and CPython randomizes str hashing per process
    # (PYTHONHASHSEED) unless fixed. With one-directional edges, that flipped
    # which of every {g1, g2} pair got the edge on every single run -- the
    # graph, and therefore risk_score, was not reproducible across process
    # invocations for identical input. dict preserves insertion order
    # regardless of PYTHONHASHSEED, so this keeps groups in first-appearance
    # order from question_results instead of imposing sorted()'s alphabetical
    # order -- deterministic either way, but this one doesn't invent an order
    # the data didn't have. NOTE: this changes risk_score numerically from the
    # sorted() baseline (see docs -- Financial_Advice recall 0.2335 -> 0.2695
    # on mmsafety_full); re-run sweep_theta.py / sweep_graph_structure.py and
    # update the documented baseline before trusting downstream results.
    groups: List[str] = list(dict.fromkeys(
        group_by_category.get(it["question"], "Unknown") for it in question_results
    ))
    for g1, g2 in itertools.combinations(groups, 2):
        G.add_edge(g1, g2, weight=group_coupling)
        if symmetric_coupling:
            G.add_edge(g2, g1, weight=group_coupling)

    qs_by_group: Dict[str, List[str]] = {}
    for it in question_results:
        g = group_by_category.get(it["question"], "Unknown")
        qs_by_group.setdefault(g, []).append(it["question"])
    for g, qs in qs_by_group.items():
        for q1, q2 in itertools.combinations(qs, 2):
            G.add_edge(q1, q2, weight=intra_group_q_coupling)
            if symmetric_coupling:
                G.add_edge(q2, q1, weight=intra_group_q_coupling)

    return G


def pagerank_risk_score(G: nx.DiGraph, alpha: float = 0.85) -> float:
    if len(G) == 0:
        return 0.0
    pr = nx.pagerank(G, weight="weight", alpha=alpha)
    score = 0.0
    for n in G.nodes:
        out_w = sum(d.get("weight", 0.0) for _, _, d in G.edges(n, data=True))
        score += pr[n] * out_w
    return float(score)
