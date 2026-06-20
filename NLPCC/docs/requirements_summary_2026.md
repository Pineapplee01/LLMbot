# NLPCC 2026 Requirements Summary

Last updated: 2026-06-16

This file is a writing-time summary of the official NLPCC 2026 submission
requirements that materially affect manuscript preparation.

## Official sources

- Call for Papers: `https://tcci.ccf.org.cn/conference/2026/calls/`
- Submission portal: `https://openreview.net/group?id=ccf.org/NLPCC/2026/Conference`

## Hard requirements for manuscript writing

- Venue: The 15th CCF International Conference on Natural Language Processing
  and Chinese Computing (NLPCC 2026)
- Review mode: double-blind
- Submission language: English only
- Proceedings: Springer LNAI / LNCS style
- Format: standard Springer style sheets
- Submission file: PDF
- Maximum length: 12 pages, including references and appendices

## Double-blind requirements that affect the draft

- Do not include author names or affiliations anywhere in the submission.
- Do not include funding acknowledgments.
- Do not acknowledge collaborators, group members, or colleagues.
- The submitted PDF/file name should not reveal authorship.
- Refer to the authors' own prior work in the third person.
- Violating anonymity rules can lead to rejection without review.

## Important dates

All deadlines are listed as 23:59 Beijing Time on the official site.

- Paper submission deadline: June 20, 2026 (extended)
- Paper rebuttal start: July 10, 2026
- Paper rebuttal deadline: July 15, 2026
- Notification: August 4, 2026
- Camera-ready deadline: August 15, 2026
- Tutorials: November 3, 2026
- Main conference: November 4-5, 2026

## Topic fit for the current paper

The current social-bot paper fits multiple listed NLPCC topics, especially:

- Computational Social Science and Social Media
- Large Language Models
- Machine Learning for NLP
- Interpretability and Analysis of Models for NLP
- NLP Applications

## Direct implications for `paper-write`

The `paper-write` skill defaults do not match this venue. When using the skill,
override its assumptions with the following:

- Do not use ICLR/NeurIPS/ICML style assumptions.
- Do not use the skill's default 9-page main-body budget.
- Use the existing local LNCS template under `paper/NLPCC/`.
- Treat references and appendices as part of the 12-page limit.
- Keep the author block anonymous.

## Local template status

The local directory `paper/NLPCC/` already contains the needed LNCS materials:

- `llncs.cls`
- `splncs04.bst`
- `main.tex`
- `sections/*.tex`

The current local draft compiles successfully with the LNCS template and is
already within the page budget. Future writing should preserve this template
instead of switching to another venue format.

## Writing budget guidance

Because NLPCC counts references and appendices inside the 12-page limit, the
main body must stay compact.

Recommended budget:

- Main body target: about 8-9 pages
- References: about 1.5-2 pages
- Appendix: at most about 0.5-1 page

This means only the strongest comparable results should stay in the main text.
Exploratory or single-seed analyses should move to the appendix unless they are
strictly necessary for the narrative.
