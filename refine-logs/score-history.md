# Score Evolution

| Round | Problem Fidelity | Method Specificity | Contribution Quality | Frontier Leverage | Feasibility | Validation Focus | Venue Readiness | Overall | Verdict |
|-------|------------------|--------------------|----------------------|-------------------|-------------|------------------|-----------------|---------|---------|
| 1     | 8                | 6                  | 6                    | 7                 | 6           | 6                | 6               | 6.5     | REVISE  |
| 2     | 9                | 8                  | 8                    | 8                 | 8           | 8                | 6               | 8.1     | REVISE  |
| 3     | 9                | 9                  | 9                    | 8                 | 9           | 9                | 8               | 8.8     | REVISE  |
| 4     | 10               | 10                 | 9                    | 8                 | 9           | 9                | 8               | 9.2     | READY (spec-level, conditional on Pilot Gates A/B) |
| 5     | 9.4              | 9.2                | 8.1                  | 8.4               | 8.0         | 8.5              | 6.8             | 8.6     | REVISE (nightmare difficulty; rescope introduced Stage-7/8 sprawl risk) |
| 6     | 9.6              | 9.5                | 9.1                  | 8.9               | 8.7         | 9.4              | 9.1             | 9.2     | **READY (nightmare difficulty; all 8 claims have falsification paths)** |
| 7     | 9.6              | 9.4                | 8.7                  | 8.6               | 8.5         | 9.3              | 8.6             | 9.0     | REVISE (external critique applied; C5 regressed from v5 — sprawl-containment weakened) |
| 8     | 9.6              | 9.5                | 9.1                  | 8.9               | 8.7         | 9.4              | 9.1             | 9.2     | **READY (nightmare; C5 compound gate + Block-S4 restore sprawl-containment; strict improvement, zero regressions)** |

## v8 cycle (user rescope: 7-stage pipeline hard-locked, LLM refiner restored to main)

| Round | Problem Fidelity | Method Specificity | Contribution Quality | Frontier Leverage | Feasibility | Validation Focus | Venue Readiness | Overall | Verdict |
|-------|------------------|--------------------|----------------------|-------------------|-------------|------------------|-----------------|---------|---------|
| v8-1  | 8.0              | 6.5                | 6.0                  | 7.0               | 7.0         | 6.5              | 6.5             | 6.675   | RETHINK (3 parallel novelties dilute single-contribution story) |
| v8-2  | 8.5              | 7.3                | 7.1                  | 7.8               | 8.1         | 7.4              | 7.2             | 7.585   | REVISE (LOW drift; Stage-5 center clean but self-confirmation in Stage-4 accept + missing scrambled-structure control) |
| v8-4  | 8.8              | 8.0                | 7.6                  | 8.0               | 8.8         | 8.1              | 7.7             | 8.09    | REVISE (NONE drift; CQ ceiling — protocol could read as "schema engineering" without ordering-invariance + LLM-family robustness) |
| v8-5  | 9.0              | 8.5                | 8.6                  | 8.3               | 9.2         | 8.6              | 8.2             | 8.63    | **REVISE at MAX_ROUNDS cap** (NONE drift; all structural blockers resolved; remaining gap is empirical — needs Block-E + Block-CT results to confirm protocol dominance) |
