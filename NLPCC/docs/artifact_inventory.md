# NLPCC Artifact Inventory

This inventory records local NLPCC artifacts that are intentionally not tracked.

## Excluded Artifact Classes

| Class | Local examples | Reason | Required action |
| --- | --- | --- | --- |
| Compiled manuscripts | `G:\Research\BotDetection\paper\NLPCC\main.pdf`, `reference_doi.pdf`, `main_*_update.pdf` | Generated paper outputs | Rebuild from `NLPCC/paper/main.tex` or inspect local source path |
| TeX build products | `*.log`, `*.aux`, `*.bbl`, `*.blg`, `*.fls`, `*.fdb_latexmk` | Reproducible build output | Do not track; regenerate during local compile |
| Preview/rendered pages | `_preview_*`, `_pdf_preview`, `rendered_pages` | Visual QA cache | Do not track; regenerate from PDFs |
| Portable TeX runtime | `.texenv/` | Large local environment | Recreate locally; do not commit |
| Figure PDF/PNG exports | `figures/Architecture.pdf`, `figures/casestudy.pdf`, `figures/*.png` | Generated from tracked scripts or editable sources | Regenerate or inspect local source path |
| Historical duplicate drafts | `main_round*.pdf`, `main_*_update.*` | Revision snapshots | Keep only in local archive unless promoted by a future plan |

## Active Generated Figure Exports Referenced By LaTeX

| LaTeX reference | Local source path | Tracked source | Regeneration note |
| --- | --- | --- | --- |
| `figures/Architecture.pdf` | `G:\Research\BotDetection\paper\NLPCC\figures\Architecture.pdf` | `NLPCC/paper/figures/Architecture.vsdx`, `NLPCC/paper/figures/gen_architecture_fit.py` | Regenerate through Visio or Python fallback |
| `figures/feature encoder.pdf` | `G:\Research\BotDetection\paper\NLPCC\figures\feature encoder.pdf` | `NLPCC/paper/figures/feature encoder.vsdx`, `NLPCC/paper/figures/feature encoder.svg` | Regenerate from editable figure source |
| `figures/twibot20_seed1_hparam_sensitivity.pdf` | `G:\Research\BotDetection\paper\NLPCC\figures\twibot20_seed1_hparam_sensitivity.pdf` | `NLPCC/paper/figures/gen_twibot20_hparam_sensitivity.py`, `NLPCC/paper/figures/data/twibot20_hparam_sensitivity.csv` | Run the tracked generation script in the LLMbot environment |
| `figures/casestudy.pdf` | `G:\Research\BotDetection\paper\NLPCC\figures\casestudy.pdf` | `NLPCC/paper/figures/casestudy.vsdx`, `NLPCC/paper/figures/gen_casestudy_vsdx_package.py` | Regenerate through Visio or the package builder |
