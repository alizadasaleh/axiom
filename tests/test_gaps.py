"""Tests for axiom.gaps — community detection, labels, centroids, gap scoring."""
from __future__ import annotations

import math

import numpy as np

from axiom import config, gaps, graph as graphmod
from tests.conftest import SEED_VECTORS


def test_detect_communities_splits_two_clusters(seeded_conn):
    g = graphmod.load_graph(seeded_conn)
    node2c = gaps.detect_communities(g, seed=42)
    # All 8 corpus papers assigned; A* and B* land in different communities.
    assert set(node2c) == {"A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4"}
    assert len({node2c["A2"], node2c["A3"], node2c["A4"]}) == 1
    assert len({node2c["B2"], node2c["B3"], node2c["B4"]}) == 1
    assert node2c["A2"] != node2c["B2"]


def test_community_labels_drop_generic(seeded_conn):
    g = graphmod.load_graph(seeded_conn)
    node2c = gaps.detect_communities(g, seed=42)
    labels = gaps.community_labels(seeded_conn, node2c)
    all_labels = {lbl for lbls in labels.values() for lbl in lbls}
    # "Computer science" is in every community → generic → dropped as a label.
    assert "Computer science" not in all_labels
    assert "Information retrieval" in all_labels or "Machine translation" in all_labels


def test_centroids_are_means(seeded_conn):
    g = graphmod.load_graph(seeded_conn)
    node2c = gaps.detect_communities(g, seed=42)
    centroids = gaps.community_centroids(node2c, SEED_VECTORS)
    for cid, c in centroids.items():
        # every seed vector is [1,0,0] or [1,1,0]; the mean keeps x==1.
        assert math.isclose(c[0], 1.0)


def test_cosine_zero_vector():
    assert gaps._cosine(np.zeros(3), np.array([1.0, 0.0, 0.0])) == 0.0


def test_analyze_gap_score_formula(seeded_conn):
    g = graphmod.load_graph(seeded_conn)
    analysis = gaps.analyze(g, seeded_conn, SEED_VECTORS, min_community_size=4)
    assert len(analysis.gaps) == 1
    cand = analysis.gaps[0]
    # centroids [1,0,0] vs [1,1,0] → cosine 1/sqrt(2); single A1->B1 bridge.
    assert math.isclose(cand.semantic_similarity, 1 / math.sqrt(2), abs_tol=1e-6)
    assert cand.inter_citations == 1
    assert math.isclose(cand.gap_score, cand.semantic_similarity / (1 + 1), abs_tol=1e-6)


def test_analyze_respects_min_community_size(seeded_conn):
    g = graphmod.load_graph(seeded_conn)
    # Both clusters are size 4; require 5 → no candidate pairs survive.
    analysis = gaps.analyze(g, seeded_conn, SEED_VECTORS, min_community_size=5)
    assert analysis.gaps == []


def test_analyze_empty_graph(empty_conn):
    import networkx as nx
    analysis = gaps.analyze(nx.DiGraph(), empty_conn, {})
    assert analysis.gaps == []
    assert analysis.communities == {}


# --- OD17 composite G-score --------------------------------------------------
def test_components_in_unit_range(seeded_conn):
    g = graphmod.load_graph(seeded_conn)
    cand = gaps.analyze(g, seeded_conn, SEED_VECTORS, min_community_size=4).gaps[0]
    assert set(cand.components) == {"similarity", "disconnection", "velocity", "authority"}
    for name, val in cand.components.items():
        assert 0.0 <= val <= 1.0, name
    assert 0.0 <= cand.g_score <= 1.0


def test_g_score_matches_weighted_components(seeded_conn, monkeypatch):
    weights = {"similarity": 0.5, "disconnection": 0.2, "velocity": 0.2, "authority": 0.1}
    monkeypatch.setattr(config, "load_calibration",
                        lambda: {"weights": weights, "tau": 0.5})
    g = graphmod.load_graph(seeded_conn)
    cand = gaps.analyze(g, seeded_conn, SEED_VECTORS, min_community_size=4).gaps[0]
    expected = sum(weights[k] * cand.components[k] for k in weights)
    assert math.isclose(cand.g_score, expected, abs_tol=1e-9)


def test_meets_threshold_follows_tau(seeded_conn, monkeypatch):
    weights = {"similarity": 0.4, "disconnection": 0.3, "velocity": 0.2, "authority": 0.1}

    monkeypatch.setattr(config, "load_calibration",
                        lambda: {"weights": weights, "tau": 0.0})
    g = graphmod.load_graph(seeded_conn)
    cand = gaps.analyze(g, seeded_conn, SEED_VECTORS, min_community_size=4).gaps[0]
    assert cand.meets_threshold is True        # τ=0 → everything passes
    g_val = cand.g_score

    monkeypatch.setattr(config, "load_calibration",
                        lambda: {"weights": weights, "tau": g_val + 0.01})
    cand2 = gaps.analyze(g, seeded_conn, SEED_VECTORS, min_community_size=4).gaps[0]
    assert cand2.meets_threshold is False       # τ just above g_score → fails
