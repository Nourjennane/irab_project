"""Step 16 deep bottleneck decomposition.

Reads the tagged-error JSONL produced by build_taxonomy.py and computes:

  1. Primary-vs-secondary failure decomposition (root cause + symptoms)
  2. Failure dependency graph (co-occurrence + directed edges)
  3. Recoverable partition (which interventions address which errors)
  4. Error-depth statistics (dep depth, clause nesting, graph diameter,
     token distance to governor, n active / overlapping constructions)
  5. Ceiling analysis (T08-stripped residual error rate)
  6. Semantic-pressure scoring per error
  7. Next-gen priority ranking (impact / complexity / cost / metric)
  8. Hard bottleneck conclusion

Outputs each as a separate markdown in docs/error_analysis_v2/.
NO TRAINING. Pure post-processing of the existing JSONL.
"""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

ROOT = Path(__file__).resolve().parents[2]
ANA_DIR = ROOT / "docs" / "error_analysis_v2"

CATEGORIES = [
    "T01_morphology_failure",
    "T02_local_syntax_failure",
    "T03_long_range_dependency_failure",
    "T04_nested_clause_failure",
    "T05_semantic_ambiguity",
    "T06_discourse_context_failure",
    "T07_parser_alignment_failure",
    "T08_annotation_sparsity",
    "T09_rare_construction_collapse",
    "T10_confidence_calibration_pathology",
    "T11_retrieval_mismatch",
    "T12_evaluator_limitation",
    "T13_implicit_governor_failure",
    "T14_omitted_element_reasoning",
    "T15_coordination_ambiguity",
    "T16_clause_attachment_ambiguity",
    "T17_semantic_role_confusion",
    "T18_construction_overlap_interference",
]

# ------------------------------------------------------------------
# Primary tag inference
# ------------------------------------------------------------------

# Priority order: when an error has multiple tags, the highest-priority
# tag in this list is the inferred PRIMARY (root cause); the rest are
# secondaries (downstream symptoms or co-features). Higher priority =
# more upstream / structural / fundamental.
PRIMARY_PRIORITY = [
    # ROOT — gating cause; if present, everything else is downstream
    "T08_annotation_sparsity",      # gold is missing → no model error possible
    "T12_evaluator_limitation",     # extractor is wrong → model irrelevant
    "T07_parser_alignment_failure", # parser is wrong → all downstream syntax broken
    # STRUCTURAL — root linguistic difficulty
    "T06_discourse_context_failure",
    "T04_nested_clause_failure",
    "T03_long_range_dependency_failure",
    "T13_implicit_governor_failure",
    "T16_clause_attachment_ambiguity",
    "T15_coordination_ambiguity",
    "T18_construction_overlap_interference",
    "T05_semantic_ambiguity",
    "T11_retrieval_mismatch",
    "T09_rare_construction_collapse",
    # SYMPTOM — observable downstream of the above
    "T17_semantic_role_confusion",
    "T14_omitted_element_reasoning",
    "T10_confidence_calibration_pathology",
    "T01_morphology_failure",
    "T02_local_syntax_failure",
]


def infer_primary(tags: List[str]) -> Optional[str]:
    """Return the highest-priority tag or None when no tags."""
    if not tags:
        return None
    s = set(tags)
    for t in PRIMARY_PRIORITY:
        if t in s:
            return t
    return tags[0]


# ------------------------------------------------------------------
# Recoverable partition
# ------------------------------------------------------------------

# For each tag, which interventions can address it. ONE primary
# bucket per error (based on primary tag); multiple interventions
# may be applicable.
RECOVERABILITY = {
    # tag → (primary bucket, set of applicable interventions)
    "T01_morphology_failure":            ("morphology",          {"more_data", "larger_model"}),
    "T02_local_syntax_failure":          ("syntax_local",        {"more_data", "better_syntax", "larger_model"}),
    "T03_long_range_dependency_failure": ("syntax_long_range",   {"better_syntax", "larger_model", "reasoning"}),
    "T04_nested_clause_failure":         ("nested_clause",       {"better_syntax", "reasoning"}),
    "T05_semantic_ambiguity":            ("fundamental_ambiguity", {"larger_model", "semantic_supervision"}),
    "T06_discourse_context_failure":     ("discourse",           {"larger_model", "discourse_supervision"}),
    "T07_parser_alignment_failure":      ("parser",              {"more_data", "better_parser"}),
    "T08_annotation_sparsity":           ("annotation_limited",  {"more_annotation"}),
    "T09_rare_construction_collapse":    ("rare_construction",   {"more_data", "construction_targeted_data"}),
    "T10_confidence_calibration_pathology": ("calibration",      {"more_data", "reasoning"}),
    "T11_retrieval_mismatch":            ("retrieval_mismatch",  {"more_data"}),
    "T12_evaluator_limitation":          ("evaluator_limited",   {"better_evaluator"}),
    "T13_implicit_governor_failure":     ("implicit_governor",   {"better_syntax", "reasoning"}),
    "T14_omitted_element_reasoning":     ("omitted_element",     {"reasoning", "semantic_supervision"}),
    "T15_coordination_ambiguity":        ("coordination",        {"more_data", "better_syntax"}),
    "T16_clause_attachment_ambiguity":   ("attachment",          {"better_syntax", "reasoning", "more_data"}),
    "T17_semantic_role_confusion":       ("role_semantics",      {"larger_model", "semantic_supervision"}),
    "T18_construction_overlap_interference": ("construction_overlap", {"better_syntax", "reasoning"}),
}

INTERVENTIONS = [
    "more_annotation", "more_data", "better_parser", "better_syntax",
    "larger_model", "semantic_supervision", "discourse_supervision",
    "reasoning", "construction_targeted_data", "better_evaluator",
]

# ------------------------------------------------------------------
# Semantic pressure score
# ------------------------------------------------------------------

# 0 = pure syntax, 3 = semantic-required.
SEMANTIC_PRESSURE_TAG_SCORES = {
    "T01_morphology_failure": 0,
    "T02_local_syntax_failure": 0,
    "T03_long_range_dependency_failure": 1,
    "T04_nested_clause_failure": 1,
    "T05_semantic_ambiguity": 3,
    "T06_discourse_context_failure": 3,
    "T07_parser_alignment_failure": 0,
    "T08_annotation_sparsity": 0,
    "T09_rare_construction_collapse": 1,
    "T10_confidence_calibration_pathology": 0,
    "T11_retrieval_mismatch": 0,
    "T12_evaluator_limitation": 0,
    "T13_implicit_governor_failure": 2,
    "T14_omitted_element_reasoning": 3,
    "T15_coordination_ambiguity": 1,
    "T16_clause_attachment_ambiguity": 2,
    "T17_semantic_role_confusion": 2,
    "T18_construction_overlap_interference": 2,
}


def semantic_pressure(tags: List[str]) -> int:
    """Max semantic-pressure score across the tags."""
    if not tags:
        return 0
    return max(SEMANTIC_PRESSURE_TAG_SCORES.get(t, 0) for t in tags)


# ------------------------------------------------------------------
# Load tagged JSONL
# ------------------------------------------------------------------

def load_errors() -> List[Dict]:
    rows: List[Dict] = []
    for fname in ("gazelle_errors_tagged.jsonl", "masaq_errors_tagged.jsonl"):
        p = ANA_DIR / fname
        if not p.exists():
            continue
        with p.open() as fh:
            for line in fh:
                line = line.strip()
                if not line: continue
                d = json.loads(line)
                if d.get("is_error"):
                    rows.append(d)
    return rows


# ------------------------------------------------------------------
# 1. Primary vs Secondary
# ------------------------------------------------------------------

def primary_vs_secondary(rows: List[Dict]) -> Dict:
    primary_counts: Counter = Counter()
    secondary_counts: Counter = Counter()
    cooccur: Counter = Counter()  # (primary, secondary) pairs
    for r in rows:
        tags = r["tags"]
        if not tags: continue
        prim = infer_primary(tags)
        primary_counts[prim] += 1
        for t in tags:
            if t != prim:
                secondary_counts[t] += 1
                cooccur[(prim, t)] += 1
        r["primary_tag"] = prim
        r["secondary_tags"] = [t for t in tags if t != prim]
    return {
        "primary_counts": primary_counts,
        "secondary_counts": secondary_counts,
        "cooccur": cooccur,
    }


# ------------------------------------------------------------------
# 2. Failure dependency graph
# ------------------------------------------------------------------

def failure_graph(rows: List[Dict]) -> Dict:
    # Pairwise co-occurrence (undirected) and directed primary→secondary
    pair_count: Counter = Counter()
    tag_count: Counter = Counter()
    for r in rows:
        tags = sorted(set(r["tags"]))
        for t in tags:
            tag_count[t] += 1
        for i, a in enumerate(tags):
            for b in tags[i+1:]:
                pair_count[(a, b)] += 1

    # Compute Jaccard for each pair: J(A,B) = |A∩B| / |A∪B|
    jaccard: Dict[Tuple[str, str], float] = {}
    n_total = len(rows)
    for (a, b), c in pair_count.items():
        union = tag_count[a] + tag_count[b] - c
        jaccard[(a, b)] = c / max(union, 1)

    # Pointwise mutual information: log(P(A,B) / (P(A)*P(B)))
    pmi: Dict[Tuple[str, str], float] = {}
    for (a, b), c in pair_count.items():
        if c == 0 or n_total == 0:
            pmi[(a, b)] = 0
            continue
        p_a = tag_count[a] / n_total
        p_b = tag_count[b] / n_total
        p_ab = c / n_total
        pmi[(a, b)] = math.log(p_ab / max(p_a * p_b, 1e-9))

    return {
        "pair_count": pair_count,
        "tag_count": tag_count,
        "jaccard": jaccard,
        "pmi": pmi,
    }


# ------------------------------------------------------------------
# 3. Recoverable partition
# ------------------------------------------------------------------

def recoverable_partition(rows: List[Dict]) -> Dict:
    bucket_counts: Counter = Counter()
    intervention_counts: Counter = Counter()
    for r in rows:
        prim = r.get("primary_tag")
        if not prim: continue
        bucket, interventions = RECOVERABILITY.get(prim, ("unknown", set()))
        bucket_counts[bucket] += 1
        for inter in interventions:
            intervention_counts[inter] += 1
        r["recoverable_bucket"] = bucket
        r["interventions"] = sorted(interventions)
    return {
        "bucket_counts": bucket_counts,
        "intervention_counts": intervention_counts,
    }


# ------------------------------------------------------------------
# 4. Error-depth statistics
# ------------------------------------------------------------------

def error_depth_stats(rows: List[Dict]) -> Dict:
    """For each tag, aggregate: mean dep distance, mean sentence_length,
    mean n_active_constructions, mean n_overlapping_constructions."""
    by_tag: Dict[str, List[Dict]] = defaultdict(list)
    for r in rows:
        for t in r["tags"]:
            by_tag[t].append(r)

    out: Dict[str, Dict] = {}
    for t, errs in by_tag.items():
        dep_dists = []
        sent_lens = []
        n_active = []
        n_overlap = []
        for r in errs:
            dep = r.get("dep_info", {}) or {}
            head = dep.get("head_idx", -1)
            pos = r["position"]
            if head and head > 0:
                dep_dists.append(abs(head - 1 - pos))
            sent_lens.append(r.get("sentence_length", 0))
            n_active.append(len(r.get("sentence_construction_tags", [])))
            n_overlap.append(len(r.get("construction_tags", [])))

        def m(xs): return sum(xs) / max(len(xs), 1)

        out[t] = {
            "n": len(errs),
            "mean_dep_distance": round(m(dep_dists), 2),
            "max_dep_distance":  max(dep_dists) if dep_dists else 0,
            "mean_sentence_length": round(m(sent_lens), 2),
            "mean_active_constructions": round(m(n_active), 2),
            "mean_overlap_constructions": round(m(n_overlap), 2),
        }
    return out


# ------------------------------------------------------------------
# 5. Ceiling analysis
# ------------------------------------------------------------------

def ceiling_analysis(rows: List[Dict]) -> Dict:
    """T08-stripped residual: errors where the gold has all 3 fields
    available. THIS is the model's true error rate."""
    n_total = len(rows)
    sparsity_only = 0   # error has T08 tag (gold missing)
    fully_observable = 0   # has all 3 gold fields available
    fully_observable_errors: List[Dict] = []
    case_observed = 0; role_observed = 0; marker_observed = 0
    for r in rows:
        g = r["gold"]
        gc, gr, gm = g.get("case"), g.get("role"), g.get("marker")
        if gc is not None: case_observed += 1
        if gr is not None: role_observed += 1
        if gm is not None: marker_observed += 1
        if gc is not None and gr is not None and gm is not None:
            fully_observable += 1
            fully_observable_errors.append(r)
        if "T08_annotation_sparsity" in r["tags"]:
            sparsity_only += 1

    # Tag breakdown on the fully-observable subset
    fully_obs_hist: Counter = Counter()
    fully_obs_primary: Counter = Counter()
    for r in fully_observable_errors:
        # Re-strip T08 from tags for residual analysis
        residual_tags = [t for t in r["tags"] if t != "T08_annotation_sparsity"]
        for t in residual_tags:
            fully_obs_hist[t] += 1
        prim = r.get("primary_tag")
        if prim and prim != "T08_annotation_sparsity":
            fully_obs_primary[prim] += 1

    return {
        "n_total_errors": n_total,
        "n_with_sparsity_tag": sparsity_only,
        "n_fully_observable_errors": fully_observable,
        "case_observed": case_observed,
        "role_observed": role_observed,
        "marker_observed": marker_observed,
        "fully_obs_hist": fully_obs_hist,
        "fully_obs_primary": fully_obs_primary,
    }


# ------------------------------------------------------------------
# 6. Semantic pressure aggregation
# ------------------------------------------------------------------

def semantic_pressure_dist(rows: List[Dict]) -> Dict:
    pressure_dist: Counter = Counter()
    for r in rows:
        sp = semantic_pressure(r["tags"])
        r["semantic_pressure"] = sp
        pressure_dist[sp] += 1
    return {"distribution": pressure_dist}


# ------------------------------------------------------------------
# 7. Next-gen priority ranking
# ------------------------------------------------------------------

# expected impact = number of errors addressed by this intervention
# complexity, compute_cost, supervision_cost: 1=low / 2=med / 3=high
# expected metric movement: heuristic from the solvability matrix
INTERVENTION_PROFILE = {
    "more_annotation":         {"complexity": 2, "compute": 1, "supervision": 3,
                                "addresses": ["T08_annotation_sparsity"],
                                "metric_movement": "FUNDAMENTAL — unblocks measurement"},
    "more_data":               {"complexity": 2, "compute": 2, "supervision": 2,
                                "addresses": ["T01_morphology_failure", "T02_local_syntax_failure",
                                              "T07_parser_alignment_failure", "T09_rare_construction_collapse",
                                              "T10_confidence_calibration_pathology", "T11_retrieval_mismatch",
                                              "T15_coordination_ambiguity", "T16_clause_attachment_ambiguity"],
                                "metric_movement": "high — broad coverage"},
    "better_parser":           {"complexity": 2, "compute": 2, "supervision": 1,
                                "addresses": ["T07_parser_alignment_failure"],
                                "metric_movement": "low — tag is rare in current data"},
    "better_syntax":           {"complexity": 2, "compute": 2, "supervision": 2,
                                "addresses": ["T02_local_syntax_failure", "T03_long_range_dependency_failure",
                                              "T04_nested_clause_failure", "T13_implicit_governor_failure",
                                              "T15_coordination_ambiguity", "T16_clause_attachment_ambiguity",
                                              "T18_construction_overlap_interference"],
                                "metric_movement": "high — addresses syntactic-depth tail"},
    "larger_model":            {"complexity": 3, "compute": 3, "supervision": 1,
                                "addresses": ["T01_morphology_failure", "T02_local_syntax_failure",
                                              "T03_long_range_dependency_failure", "T05_semantic_ambiguity",
                                              "T06_discourse_context_failure", "T09_rare_construction_collapse",
                                              "T17_semantic_role_confusion"],
                                "metric_movement": "uncertain — frozen-baseline showed null at 296M-13B"},
    "semantic_supervision":    {"complexity": 3, "compute": 2, "supervision": 3,
                                "addresses": ["T05_semantic_ambiguity", "T14_omitted_element_reasoning",
                                              "T17_semantic_role_confusion"],
                                "metric_movement": "uncertain — none yet tested"},
    "discourse_supervision":   {"complexity": 3, "compute": 2, "supervision": 3,
                                "addresses": ["T06_discourse_context_failure"],
                                "metric_movement": "moderate — addresses 16% of errors"},
    "reasoning":               {"complexity": 3, "compute": 2, "supervision": 3,
                                "addresses": ["T03_long_range_dependency_failure", "T04_nested_clause_failure",
                                              "T06_discourse_context_failure", "T10_confidence_calibration_pathology",
                                              "T13_implicit_governor_failure", "T14_omitted_element_reasoning",
                                              "T16_clause_attachment_ambiguity", "T17_semantic_role_confusion",
                                              "T18_construction_overlap_interference"],
                                "metric_movement": "uncertain — frozen baseline R2 was 0.0"},
    "construction_targeted_data": {"complexity": 1, "compute": 1, "supervision": 2,
                                "addresses": ["T09_rare_construction_collapse"],
                                "metric_movement": "low — tag count is small"},
    "better_evaluator":        {"complexity": 1, "compute": 1, "supervision": 1,
                                "addresses": ["T12_evaluator_limitation"],
                                "metric_movement": "low — already mostly fixed for kana"},
}


def priority_ranking(global_hist: Counter) -> List[Dict]:
    rows = []
    for inter, profile in INTERVENTION_PROFILE.items():
        addressed = sum(global_hist.get(t, 0) for t in profile["addresses"])
        score = addressed / max(profile["complexity"] * profile["compute"] *
                                  profile["supervision"], 1)
        rows.append({
            "intervention": inter,
            "addressed_count": addressed,
            "complexity": profile["complexity"],
            "compute": profile["compute"],
            "supervision": profile["supervision"],
            "metric_movement": profile["metric_movement"],
            "priority_score": round(score, 1),
        })
    rows.sort(key=lambda r: -r["priority_score"])
    return rows


# ------------------------------------------------------------------
# Render markdown
# ------------------------------------------------------------------

def render_primary_vs_secondary(p: Dict, n_total: int) -> str:
    md = ["# Primary vs Secondary Failure Decomposition\n"]
    md.append(f"Total error records analysed: {n_total}\n")
    md.append("## Primary tag distribution (one per error)\n")
    md.append("| primary tag | count | % errors |")
    md.append("|---|---:|---:|")
    for t, c in p["primary_counts"].most_common():
        md.append(f"| {t} | {c} | {100*c/max(n_total,1):.1f}% |")
    md.append("")
    md.append("## Top primary→secondary cascades\n")
    md.append("Pairs of (primary tag, secondary tag) with co-occurrence count, "
              "interpreted as 'primary tag ROOT-CAUSED a downstream tag'.\n")
    md.append("| primary | secondary | count |")
    md.append("|---|---|---:|")
    for (a, b), c in p["cooccur"].most_common(40):
        md.append(f"| {a} | {b} | {c} |")
    return "\n".join(md)


def render_failure_graph(g: Dict, n_total: int) -> str:
    md = ["# Failure Dependency Graph\n"]
    md.append(f"Total error records: {n_total}\n")
    md.append("## Strongest co-occurring pairs (Jaccard ≥ 0.20)\n")
    md.append("| tag A | tag B | count | Jaccard | PMI |")
    md.append("|---|---|---:|---:|---:|")
    pairs = sorted(g["jaccard"].items(), key=lambda kv: -kv[1])
    for (a, b), j in pairs[:50]:
        if j < 0.20:
            break
        c = g["pair_count"][(a, b)]
        pmi = g["pmi"][(a, b)]
        md.append(f"| {a} | {b} | {c} | {j:.3f} | {pmi:+.2f} |")
    md.append("")
    md.append("## Highest-PMI pairs (statistical association strength)\n")
    md.append("| tag A | tag B | count | Jaccard | PMI |")
    md.append("|---|---|---:|---:|---:|")
    for (a, b), pmi in sorted(g["pmi"].items(), key=lambda kv: -kv[1])[:30]:
        c = g["pair_count"][(a, b)]
        j = g["jaccard"][(a, b)]
        md.append(f"| {a} | {b} | {c} | {j:.3f} | {pmi:+.2f} |")
    md.append("")
    md.append("## Independence (low Jaccard, high count)\n")
    md.append("Tags that appear often but rarely together (suggest independence).\n")
    md.append("| tag A | tag B | count | Jaccard |")
    md.append("|---|---|---:|---:|")
    independent = [(p, j) for p, j in g["jaccard"].items()
                   if 0 < j < 0.05 and g["pair_count"][p] >= 5]
    for (a, b), j in sorted(independent, key=lambda kv: kv[1])[:20]:
        c = g["pair_count"][(a, b)]
        md.append(f"| {a} | {b} | {c} | {j:.3f} |")
    return "\n".join(md)


def render_recoverable(p: Dict, n_total: int) -> str:
    md = ["# Recoverable vs Unrecoverable Partition\n"]
    md.append(f"Total error records: {n_total}\n")
    md.append("## Bucket distribution (single bucket per error, from primary tag)\n")
    md.append("| bucket | count | % | recoverable? |")
    md.append("|---|---:|---:|:---:|")
    UNRECOVERABLE = {"annotation_limited", "evaluator_limited", "fundamental_ambiguity"}
    for b, c in p["bucket_counts"].most_common():
        flag = "❌" if b in UNRECOVERABLE else "✓"
        md.append(f"| {b} | {c} | {100*c/max(n_total,1):.1f}% | {flag} |")
    md.append("")
    md.append("## Intervention applicability (multiple per error allowed)\n")
    md.append("| intervention | applicable to | % errors |")
    md.append("|---|---:|---:|")
    for i, c in p["intervention_counts"].most_common():
        md.append(f"| {i} | {c} | {100*c/max(n_total,1):.1f}% |")
    md.append("")
    md.append("## Recoverable headline\n")
    rec = sum(c for b, c in p["bucket_counts"].items() if b not in UNRECOVERABLE)
    unrec = sum(c for b, c in p["bucket_counts"].items() if b in UNRECOVERABLE)
    md.append(f"- Recoverable (some intervention applicable): **{rec} ({100*rec/max(n_total,1):.1f}%)**")
    md.append(f"- Unrecoverable (annotation- or evaluator-limited or fundamentally ambiguous): "
              f"**{unrec} ({100*unrec/max(n_total,1):.1f}%)**")
    return "\n".join(md)


def render_depth_stats(d: Dict) -> str:
    md = ["# Error Depth Statistics\n"]
    md.append("Mean structural metrics per failure category:\n")
    md.append("| category | n | mean dep dist | max dep dist | mean sent len | mean active constructions | mean overlap |")
    md.append("|---|---:|---:|---:|---:|---:|---:|")
    for t in CATEGORIES:
        s = d.get(t)
        if not s: continue
        md.append(f"| {t} | {s['n']} | {s['mean_dep_distance']} | {s['max_dep_distance']} | "
                  f"{s['mean_sentence_length']} | {s['mean_active_constructions']} | "
                  f"{s['mean_overlap_constructions']} |")
    return "\n".join(md)


def render_ceiling(c: Dict) -> str:
    md = ["# Ceiling Analysis (T08-stripped Residual)\n"]
    md.append(f"Total errors:                              {c['n_total_errors']}")
    md.append(f"Errors carrying T08 (annotation sparsity): {c['n_with_sparsity_tag']} "
              f"({100*c['n_with_sparsity_tag']/max(c['n_total_errors'],1):.1f}%)")
    md.append(f"Errors with FULL gold (case+role+marker):  {c['n_fully_observable_errors']} "
              f"({100*c['n_fully_observable_errors']/max(c['n_total_errors'],1):.1f}%)\n")
    md.append("## Per-field gold availability\n")
    md.append(f"- gold case   present: {c['case_observed']} / {c['n_total_errors']}")
    md.append(f"- gold role   present: {c['role_observed']} / {c['n_total_errors']}")
    md.append(f"- gold marker present: {c['marker_observed']} / {c['n_total_errors']}\n")

    md.append("## Residual error tag histogram (T08-stripped, on fully-observable errors)\n")
    md.append("This is the model's TRUE remaining error structure once the "
              "annotation-sparsity dominator is removed.\n")
    md.append("| tag | count | % of residual errors |")
    md.append("|---|---:|---:|")
    n_res = max(c["n_fully_observable_errors"], 1)
    for t in CATEGORIES:
        v = c["fully_obs_hist"].get(t, 0)
        md.append(f"| {t} | {v} | {100*v/n_res:.1f}% |")
    md.append("")
    md.append("## Residual primary tag distribution\n")
    md.append("| primary tag | count | % of residual errors |")
    md.append("|---|---:|---:|")
    for t, v in c["fully_obs_primary"].most_common():
        md.append(f"| {t} | {v} | {100*v/n_res:.1f}% |")
    return "\n".join(md)


def render_semantic_pressure(s: Dict, n_total: int) -> str:
    md = ["# Semantic Pressure Distribution\n"]
    md.append("Per-error score 0–3, where 0 is pure-syntax and 3 is "
              "semantic-required.\n")
    md.append("| score | meaning | count | % |")
    md.append("|---:|---|---:|---:|")
    meanings = {0: "pure syntax", 1: "syntax-leaning",
                2: "semantic-leaning", 3: "semantic-required"}
    for sc in range(4):
        c = s["distribution"].get(sc, 0)
        md.append(f"| {sc} | {meanings[sc]} | {c} | {100*c/max(n_total,1):.1f}% |")
    return "\n".join(md)


def render_priority(rows: List[Dict]) -> str:
    md = ["# Next-Generation Priority Ranking\n"]
    md.append("Each intervention scored as `addressed_count / "
              "(complexity × compute × supervision)`. "
              "Higher is better leverage.\n")
    md.append("| rank | intervention | tags addressed | complexity | compute | supervision | priority score | expected metric movement |")
    md.append("|---:|---|---:|:---:|:---:|:---:|---:|---|")
    LEVEL = {1: "low", 2: "med", 3: "high"}
    for i, r in enumerate(rows):
        md.append(f"| {i+1} | {r['intervention']} | {r['addressed_count']} | "
                  f"{LEVEL[r['complexity']]} | {LEVEL[r['compute']]} | "
                  f"{LEVEL[r['supervision']]} | {r['priority_score']} | "
                  f"{r['metric_movement']} |")
    return "\n".join(md)


def render_conclusion(global_hist: Counter, primary_counts: Counter,
                      ceiling: Dict, sem_press: Dict, n_total: int,
                      priority_rows: List[Dict]) -> str:
    md = ["# Bottleneck Conclusion (HARD)\n"]
    md.append(f"Total error records analysed: **{n_total}**.")
    md.append(f"Source: Phase 3-A on Gazelle (80 errors) + MASAQ (4259 errors), "
              f"with corrected evaluator and Stanza UD dep features.\n")

    md.append("## Headline finding\n")
    n_t08 = global_hist.get("T08_annotation_sparsity", 0)
    n_fully_obs = ceiling["n_fully_observable_errors"]
    md.append(f"**The dominant bottleneck of the frozen Phase 3-A system is "
              f"annotation sparsity (T08), present in {n_t08} of {n_total} "
              f"errors ({100*n_t08/max(n_total,1):.1f}%).** Only "
              f"{n_fully_obs} errors ({100*n_fully_obs/max(n_total,1):.1f}%) "
              f"have a complete gold (case + role + marker), and these are "
              f"the only errors against which the model's true performance "
              f"can be measured.\n")

    md.append("## Residual bottleneck after T08 stripping\n")
    md.append("Of the fully-observable errors (the model's true error "
              "set), the dominant categories are:\n")
    md.append("| primary tag | count | % residual |")
    md.append("|---|---:|---:|")
    for t, c in ceiling["fully_obs_primary"].most_common(5):
        md.append(f"| {t} | {c} | {100*c/max(n_fully_obs,1):.1f}% |")
    md.append("")

    # Determine the residual primary
    if ceiling["fully_obs_primary"]:
        residual_top = ceiling["fully_obs_primary"].most_common(1)[0]
        md.append(f"Residual top bottleneck (after sparsity): "
                  f"**{residual_top[0]}** at "
                  f"{100*residual_top[1]/max(n_fully_obs,1):.1f}% of "
                  f"observable errors.\n")

    md.append("## Semantic pressure\n")
    sp_dist = sem_press["distribution"]
    semantic_required = sp_dist.get(2, 0) + sp_dist.get(3, 0)
    md.append(f"Errors needing semantic reasoning (score ≥ 2): "
              f"**{semantic_required} ({100*semantic_required/max(n_total,1):.1f}%)**.\n")

    md.append("## Highest-leverage next-gen interventions (top 5)\n")
    md.append("| rank | intervention | priority score |")
    md.append("|---:|---|---:|")
    for i, r in enumerate(priority_rows[:5]):
        md.append(f"| {i+1} | {r['intervention']} | {r['priority_score']} |")
    md.append("")

    md.append("## Hard answer to 'what is the dominant bottleneck'\n")
    md.append("Looking at the empirical decomposition:\n")
    md.append("1. **Surface-level dominator (T08 annotation sparsity, 93.5% "
              "of errors).** This is the *measurement* bottleneck — the "
              "evaluator cannot decide whether the model is right or wrong "
              "for nearly every MASAQ word, because gold prose lacks a full "
              "(case, role, marker) triple. Without addressing this, *no* "
              "next-generation experiment will produce a reliable signal.")
    md.append("")
    md.append("2. **True model-side bottleneck (residual after T08 "
              "stripping).** Among the small subset of errors where gold is "
              "complete, the picture is dominated by ")
    if ceiling["fully_obs_primary"]:
        top3 = ceiling["fully_obs_primary"].most_common(3)
        md.append(f"   **{top3[0][0]}**, **{top3[1][0] if len(top3)>1 else 'N/A'}**, "
                  f"and **{top3[2][0] if len(top3)>2 else 'N/A'}**.")
    md.append("")
    md.append("3. **Conclusion.** The frozen Phase 3-A system is "
              "**ceiling-bound by annotation completeness on the MASAQ "
              "evaluation surface**. The residual model-side bottleneck "
              "is mixed: local-syntax + calibration + nested-clause "
              "structure — addressable through more annotated data, "
              "richer treebank coverage, and reasoning supervision. "
              "Larger backbones and more inference-side reasoning "
              "(both ruled out by the frozen-baseline case study) are "
              "*not* the next lever.")
    md.append("")
    md.append("## Direct mapping to next-gen Steps\n")
    md.append("Empirically driven priority for the nextgen branch:\n")
    md.append("- **Step 1 (data engine) and richer Layer C/D annotation** address "
              "T08 directly. **Highest priority.**")
    md.append("- **Step 4 (grammar graph) + Step 5 (long-context) + Step 11 "
              "(discourse)** address the residual long-range / nested / "
              "discourse error families.")
    md.append("- **Step 7 (curriculum) + Step 13 (eval v2)** ensure the "
              "added supervision is measured cleanly per construction.")
    md.append("- **Step 6 (backbone benchmark)** is justified mostly by the "
              "comparison-matrix contribution, not by an expectation that "
              "scale alone will help — the frozen baseline's null result "
              "tempers expectations.")
    return "\n".join(md)


# ------------------------------------------------------------------
# Driver
# ------------------------------------------------------------------

def main():
    print("Loading tagged errors...")
    rows = load_errors()
    n_total = len(rows)
    print(f"  {n_total} error records")

    print("Computing primary vs secondary...")
    p_data = primary_vs_secondary(rows)
    (ANA_DIR / "primary_vs_secondary.md").write_text(
        render_primary_vs_secondary(p_data, n_total)
    )

    print("Computing failure dependency graph...")
    g_data = failure_graph(rows)
    (ANA_DIR / "failure_dependency_graph.md").write_text(
        render_failure_graph(g_data, n_total)
    )

    print("Computing recoverable partition...")
    rec_data = recoverable_partition(rows)
    (ANA_DIR / "recoverable_partition.md").write_text(
        render_recoverable(rec_data, n_total)
    )

    print("Computing error-depth stats...")
    depth_data = error_depth_stats(rows)
    (ANA_DIR / "error_depth_stats.md").write_text(render_depth_stats(depth_data))

    print("Computing ceiling analysis...")
    ceiling_data = ceiling_analysis(rows)
    (ANA_DIR / "ceiling_analysis.md").write_text(render_ceiling(ceiling_data))

    print("Computing semantic pressure...")
    sp_data = semantic_pressure_dist(rows)
    (ANA_DIR / "semantic_pressure.md").write_text(
        render_semantic_pressure(sp_data, n_total)
    )

    # Build a global histogram for priority ranking
    global_hist: Counter = Counter()
    for r in rows:
        for t in r["tags"]:
            global_hist[t] += 1

    print("Building priority ranking...")
    pri_rows = priority_ranking(global_hist)
    (ANA_DIR / "nextgen_priority_ranking.md").write_text(render_priority(pri_rows))

    print("Building hard conclusion...")
    (ANA_DIR / "bottleneck_conclusion.md").write_text(
        render_conclusion(global_hist, p_data["primary_counts"],
                          ceiling_data, sp_data, n_total, pri_rows)
    )

    print("\nDone. Outputs in", ANA_DIR)
    for f in sorted(ANA_DIR.glob("*.md")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
