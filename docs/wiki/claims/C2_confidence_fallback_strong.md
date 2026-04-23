---
type: claim
node_id: claim:C2
title: Confidence-Based Fallback is a Strong Baseline
status: supported
confidence: high
evidence_type: empirical
created_at: 2026-04-07T21:31:00Z
updated_at: 2026-04-09T00:00:00Z
tags: [baseline, fallback, confidence]
---

# Claim: Confidence-Based Fallback is a Strong Baseline

## Statement

A simple confidence-based fallback mechanism (`if q_graph < tau: use semantic, else: use graph`) outperforms learned routing on this benchmark.

## Evidence Status

✅ **SUPPORTED** — Empirically validated (2026-04-08)

## Supporting Evidence

### Empirical Results (exp:confidence_fallback)

- Confidence fallback (tau=0.9): **F1=0.798**
- Learned router: **F1=0.788**
- **Gain: +1.0% F1** over learned router

### Slice Analysis

- Graph hurts on degree_gt20 and disagreement slices
- Graph only helps on degree_6_20 slice
- Target slice (N=22): F1=0.533 for both semantic and fallback — insufficient

### Important Limitation

Fallback alone is insufficient for the hardest cases:
- Fallback disagree-slice F1: ~0.43
- Propagation repair (rw4) disagree-slice F1: **0.5881**
- **Repair beats fallback by +0.16 on disagreement nodes**

## Tested By

- exp:confidence_fallback (M2 diagnostics, seed 42)

## Connections

[AUTO-GENERATED from graph/edges.jsonl — do not edit manually]

## Implications

- ✅ Confirms routing bottleneck is "when to trust graph" not "how to fuse"
- ✅ Supports selective abstention direction
- ⚠️ But fallback alone is not enough — propagation-level repair needed for hardest cases
