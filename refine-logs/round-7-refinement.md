# Round 7 Refinement

**Round**: 7
**Review source**: `refine-logs/round-7-review.md` (overall 9.0 nightmare, REVISE)
**Headline revision**: one targeted IMPORTANT fix (C5 compound centrality gate) + one MINOR fix (u_edge component ablation). No structural changes. This is a surgical patch, not a restructure.

---

## Problem Anchor (verbatim from Round 0)

[unchanged]

## Anchor Check

- Original bottleneck still targeted: yes.
- Both Round-7 items are compatible with the anchor — they tighten EQC as the dominant primitive by (a) restoring the v5 sprawl-containment property for mechanism centrality, and (b) applying the same necessity-testing discipline to the commodity modifier.
- Reviewer suggestions rejected as drift: none.

## Simplicity Check

- Dominant contribution unchanged: EQC single primitive. The compound C5 gate **strengthens** the "EQC rewrite+gate is the mechanism" framing, not weakens it.
- No new components. No new trainable parameters. No new system stages.
- Component **ablation discipline extended** from `s_composite` (Block-S2) to `u_edge` (new Block-S4) — symmetric rigor.

---

## Changes Made (Round 7)

### 1. C5 Compound Centrality Gate (IMPORTANT — restores v5 property)

**Reviewer said**: "v6's softened C5 (`CI excludes 0` + `0.25·seed_std`) regresses v5 sprawl-containment. Fix: CI excludes 0 **AND** `(M-ro − M-base) ≥ max(0.5pp, β × (M-rr − M-base))` for pre-registered β."

**My self-audit**: accepting my own v6 compromise was wrong. Switching fixed fraction 0.5 to `0.25·seed_std` traded arbitrariness-of-threshold for **loss of centrality meaning**. When `seed_std` is small (which is exactly the favorable case), `0.25·seed_std` can be trivially satisfied by tiny rewrite effects. The reviewer caught that the v5 guarantee was structurally about "fraction of combined lift", not "statistical presence of any lift". I collapsed two different things into one.

**Fix (compound gate with pre-registered β)**:

```
C5 (v7 — compound centrality gate):

  (i)  95% paired-bootstrap CI on (M-ro − M-base) excludes 0       ← significance
  AND
  (ii) (M-ro − M-base) ≥ max(0.5pp,  β · (M-rr − M-base))           ← centrality
       where β = 0.4   (pre-registered; 0.4 ∈ [0.25, 0.5] chosen as
                        stronger than the reviewer's 0.25 minimum
                        and looser than v5's raw 0.5)

  BOTH conditions must hold on BOTH TwiBot-20 AND TwiBot-22.
```

Why `β = 0.4` (not 0.25 min, not 0.5 v5):
- **0.25** is the reviewer's minimal centrality floor — passing at this level means rewrite is 25%+ of combined effect, which is "nontrivial but still possibly secondary".
- **0.5** is v5's original round number — means rewrite is majority, which is strongly central but was flagged as arbitrary.
- **0.4** is pre-registered to fall inside `[0.25, 0.5]` and to stand between "nontrivial" and "majority". It still **requires rewrite to account for ≥ 40% of the combined lift**, which preserves the "not peripheral tweak" property. Pre-registration makes the choice auditable.
- **0.5pp absolute floor** guards against pathological cases where the combined gain itself is tiny and a relative threshold becomes meaningless.

Report: decomposition table `(M-ro − M-base)`, `(M-re − M-base)`, `(M-rr − M-base)` with 95% CIs, alongside β-threshold status per dataset.

### 2. u_edge Component Ablation — Block-S4 (MINOR)

**Reviewer said**: "Add u_edge leave-one-component-out (affinity / conflict / relation prior) mirroring Block-S2, to preempt 'heuristic stacking moved from q(v) to u_edge'."

**Fix**: add **Block-S4** (modifier-component necessity), parallel to Block-S2 (composite-term necessity).

```
Block-S4 (Support claim S4, NEW):
  Variants:
    U1: u_edge = α_aff · affinity only
    U2: u_edge = -α_conf · conflict only (with + sign correction so it enters consistently)
    U3: u_edge = α_rel · relation prior only
    U4: u_edge = affinity + conflict (no relation prior)
    U5: u_edge = affinity + relation prior (no conflict)
    U6: u_edge = conflict + relation prior (no affinity)
    U-full: v1 main (all three).

  Claim S4: U-full slice F1 ≥ each leave-one-out variant by ≥ 1 seed-std on at
             least one of TwiBot-20 / TwiBot-22; OTHERWISE the un-necessary
             component is dropped in the final method.

  Decision rule: mirror Block-S2 — terms not improving by 1 seed-std on any
                 primary dataset are REMOVED from the paper's final method
                 (not kept in appendix "for completeness").
```

Keeps the "single dominant primitive + commodity u_edge" narrative airtight. If, say, `relation_prior` doesn't move slice F1 by 1 seed-std, it's removed — ensures u_edge stays minimally sufficient.

### 3. No other changes

v6 package (EQC rename, 3-term composite, s_npi, external stability guard, pre-registered grid, BR-public tier, citation downgrade, unit tests → artifact doc, all other fixes) stands as-is. The two items above are strictly additive/corrective.

---

## Revised Claim Map v7

| Claim | Status v7 | Falsification |
|---|---|---|
| C1 | unchanged from v6 | T4 ≤ T4-NS in 0.010 → EQC-S; BR-soft wins both → refinement-of-prior-art |
| C2 | unchanged from v6 | G1 degrades ≥ 0.1pp OR G2 no degradation → redesign/drop |
| C3 | unchanged from v6 | Opposite-sign slice gain → relation-type sensitivity limitation |
| C4 | unchanged from v6 | δ = max(0.005, 1.5·seed_std(L0)); L2 − L0 > δ significantly → LLM is mechanism |
| **C5 (v7)** | **compound gate** | **CI excludes 0 AND (M-ro − M-base) ≥ max(0.5pp, 0.4·(M-rr − M-base)) on BOTH datasets** — else demote Stage-5/6 |
| C6 | unchanged from v6 | Any of `s_tg / s_npi` not improving slice F1 by 1 seed-std on any primary dataset → remove from final method |
| **S4 (NEW)** | **mirrors Block-S2 for u_edge** | Any of `affinity / conflict / relation_prior` not improving slice F1 by 1 seed-std → remove |
| S1 | unchanged | wall-clock overhead ≤ 2×; unit tests pass |
| S3 | unchanged | R-hard-delete ≥ R-soft on both datasets + camouflage slice → narrative change |

---

## Expected Round 8 Score

- Venue Readiness 8.6 → ~ 9.3 (C5 centrality restored; u_edge symmetric ablation addressed).
- Contribution Quality 8.7 → ~ 9.2 (sprawl-containment property restored).
- Overall 9.0 → **target 9.3+**, READY at nightmare difficulty.

---

## No Pushback This Round

Both Round-7 items are surgical corrections of my own v6 compromises. I had no reason to push back: the C5 compound gate is exactly the right resolution (CI grounded significance + pre-registered effect-size centrality), and the u_edge Block-S4 is the natural symmetric discipline.
