# Figures

This directory stores figure assets for the `NLPCC 2026` draft. The current
task prepares a reproducible figure package only; it does not integrate new
figures into `sections/*.tex`.

## Current Compiled-Manuscript Status

The compiled manuscript currently uses the framework figure and the experiment
figures for component ablation, hyperparameter sensitivity, and qualitative
case study. The figure notes below document the active assets and their
provenance.

## Prepared Figure Package

### `Architecture_fit.pdf`

- role: framework figure candidate
- source script: `gen_architecture_fit.py`
- editable/manual source: `Architecture.vsdx` exists, but this generated PDF was
  refreshed from the Python fallback because Visio export was not automated
- message: `Feature Encoder -> Hybrid-Order Relation Learner -> Bot Detector`
- must-read modules: `Risk Aggregator`, `High-Order Learner`, routed residual
  activation, non-routed bypass
- caption skeleton: Framework of the selective hybrid-order bot detector. The
  feature encoder produces node input and low-order relation representations;
  the hybrid-order learner aggregates error risk and constructs high-order
  support evidence; the detector applies high-order residual correction only to
  routed hard nodes while non-routed nodes bypass the branch.

### Main result tables

- role: reproducible main-result table assets
- source script: `gen_main_results_tables.py`
- data file: `data/main_results.csv`
- active manuscript table: `data/main_results_twibot20_compact_table.tex`
- additional table exports:
  - `data/main_results_twibot20_full_table.tex`
  - `data/main_results_twibot22_compact_table.tex`
- active manuscript scope: TwiBot-20 comparison against representative
  feature-based, graph-based, multimodal, reliability-oriented, and
  graph-distillation baselines
- active manuscript columns: `Acc.`, `Macro-P`, and `Macro-F1`
- full model on TwiBot-20: `K=8`, fanout 64, routed budget 10%, Acc. 0.8893,
  Macro-F1 0.8878
- current boundary: the full-model Macro-Precision is not recorded in the
  active result summary, so the Ours row reports `--` for `Macro-P`
- TwiBot-22 boundary: sampled TwiBot-22 baseline rows are recorded in
  `data/main_results.csv` and exported as a compact table, but they are not
  used as a main comparison for the proposed method until corresponding Ours
  results are available
- caption skeleton: Main results on TwiBot-20. The full model uses `K=8`,
  fanout 64, and a routed budget of 10%.

### `twibot20_component_ablation.pdf`

- role: component ablation asset
- source script: `gen_twibot20_component_ablation.py`
- data file: `data/twibot20_component_ablation.csv`
- table export: `data/twibot20_component_ablation_table.tex`
- plotted metrics: `Acc.`, `F1`
- full model: `K=8`, fanout 64, routed budget 10%, Acc. 0.8893, F1 0.8878
- traceability: values are recorded in the CSV from the current experiment
  summary provided for the manuscript revision; `w/o LM supervised fine-tuning`
  is kept in the table export as `--` and omitted from the plot because no
  result is currently reported
- caption skeleton: Component ablation on TwiBot-20. The full model uses
  `K=8`, fanout 64, and a routed budget of 10%.

### `twibot20_seed1_hparam_sensitivity.pdf`

- role: hyperparameter/sensitivity asset
- source script: `gen_twibot20_hparam_sensitivity.py`
- data file: `data/twibot20_hparam_sensitivity.csv`
- table export: `data/twibot20_hparam_sensitivity_table.tex`
- plotted panels: `K`, fanout, and routed budget
- full model: `K=8`, fanout 64, routed budget 10%, Acc. 0.8893, F1 0.8878
- traceability: values are recorded in the CSV from the current experiment
  summary provided for the manuscript revision
- caption skeleton: Hyperparameter sensitivity on TwiBot-20. The dashed line
  and star marker indicate the full-model setting: `K=8`, fanout 64, and a
  routed budget of 10%.

### `casestudy.pdf`

- role: BotUmc-style single-node case-study asset
- editable target: `casestudy.vsdx`
- current status: `casestudy.vsdx` is generated as a native-shape Visio package;
  fallback `casestudy.pdf`, `casestudy.png`, and `casestudy.svg` are also
  generated
- data-prep script: `prepare_casestudy_node11654.py`
- fallback drawing script: `gen_casestudy_fallback.py`
- Visio rebuild script: `gen_casestudy_visio.ps1`
- package rebuild script: `gen_casestudy_vsdx_package.py`
- data file: `data/casestudy_node11654_evidence.json`
- source files:
  `datasets/TwiBot-20/node_new.json`,
  `datasets/TwiBot-20/norm_user_text_new.json`,
  `data/case_study_selected_nodes.csv`,
  `data/case_study_support_nodes.csv`
- selected case: routed correction `seed=2,node=11654`
- target user: `u113490630` / `@PeterBotte`
- evidence pattern: low-order/base predicts `Human` for a `Bot` node; KNN
  support is bot-dominant (`B B B B B H B B`); routed-only final prediction is
  `Bot`
- interpretation boundary: qualitative mechanism illustration only, not
  statistical proof
- caption skeleton: BotUmc-style case study for a routed hard node on
  TwiBot-20. The target account is misclassified by low-order evidence, while
  the high-order support set is bot-dominant and the routed-only residual
  branch corrects the final prediction.

Visio automation was attempted but failed in the current session with
`HRESULT: 0x80070520` (`A specified logon session does not exist`). To avoid
shipping a raster-only fake Visio file, `gen_casestudy_vsdx_package.py`
generates `casestudy.vsdx` directly as editable Visio XML shapes. Package
validation confirms 75 shapes and no `visio/media` full-page reference image.
The COM script remains available for regenerating/exporting through Visio in a
normal interactive desktop session.

## Reproducibility

Run the generation scripts with the project experiment environment:

```powershell
& 'D:\Anaconda\envs\llmbot\python.exe' 'G:\Research\BotDetection\NLPCC\paper\figures\gen_architecture_fit.py'
& 'D:\Anaconda\envs\llmbot\python.exe' 'G:\Research\BotDetection\NLPCC\paper\figures\gen_main_results_tables.py'
& 'D:\Anaconda\envs\llmbot\python.exe' 'G:\Research\BotDetection\NLPCC\paper\figures\gen_twibot20_component_ablation.py'
& 'D:\Anaconda\envs\llmbot\python.exe' 'G:\Research\BotDetection\NLPCC\paper\figures\gen_twibot20_hparam_sensitivity.py'
& 'D:\Anaconda\envs\llmbot\python.exe' 'G:\Research\BotDetection\NLPCC\paper\figures\prepare_casestudy_node11654.py'
& 'D:\Anaconda\envs\llmbot\python.exe' 'G:\Research\BotDetection\NLPCC\paper\figures\gen_casestudy_fallback.py'
& 'D:\Anaconda\envs\llmbot\python.exe' 'G:\Research\BotDetection\NLPCC\paper\figures\gen_casestudy_vsdx_package.py'
```

When Visio COM is available, regenerate the editable Visio source and exports
through Visio with:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File 'G:\Research\BotDetection\NLPCC\paper\figures\gen_casestudy_visio.ps1' `
  -DataPath 'G:\Research\BotDetection\NLPCC\paper\figures\data\casestudy_node11654_evidence.json' `
  -VsdxPath 'G:\Research\BotDetection\NLPCC\paper\figures\casestudy.vsdx' `
  -OutputDir 'G:\Research\BotDetection\NLPCC\paper\figures' `
  -ExportFormats png,svg,pdf
```

The system Python does not include the required `torch` and `pandas` packages
for all scripts, so use the `llmbot` environment for reproducibility.

## Inactive / Archived Assets

The following files are retained for provenance but are not part of the current
compiled paper unless explicitly reintroduced:

- `framework_overview_selective_residual_detector.pdf`
- `framework_overview_selective_residual_detector.svg`
- `framework_overview_risk_gated_residual.pdf`
- `framework_overview_risk_gated_residual.svg`
- `case_study_selective_evidence.pdf`
- `case_study_selective_evidence.png`
- `inactive/risk_gate_mechanism_and_evidence.pdf`
- `inactive/fig2a_gate_mechanism.pdf`
- `inactive/fig2b_routed_vs_nonrouted_conflict.pdf`
- `inactive/fig2c_support_ablation.pdf`

These assets came from earlier manuscript states and should not be used as the
active method figure queue.

## Style Rules

- vector PDF output for paper figures
- PNG previews may be generated for visual inspection only
- no global titles inside figures; use LaTeX captions
- serif font aligned with the LNCS paper
- colors should remain distinguishable in grayscale through position, labels,
  and outlines
- do not imply all-node soft gating, LLM graph editing, or unrestricted KNN
  propagation in the active framework figure
