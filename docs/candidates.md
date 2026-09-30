# Candidate and Weakly Related References

This document records papers, directions, and implementation ideas discussed during planning that are not currently strong mainline references. Strong mainline references are recorded in `docs/reference.md` and are not duplicated here as entries.

## Boundary Triage: LM -> GNN -> Refined Local Evidence Graph -> LLM

Date: 2026-05-16.

Current local validation also points to a simpler router abstraction: post-hoc calibration + ranking on frozen GNN outputs. Relation/direction bins remain diagnostic or ablation channels unless they consistently beat scalar calibration baselines on the selected task.

### User Boundary Decisions

Updated from user clarification on 2026-05-16:

- **Final LLM role is fixed.** The LLM is an enhancer, not the final predictor, and it should not change labels. It consumes refined local graph evidence and produces an embedding/evidence representation for a downstream discriminator.
- **Downstream discriminator is fixed for v1.** The downstream consumer is a node discriminator for bot/human prediction. Edge-role discriminators remain references or later ablations, not the first final decision head.
- **Refined graph consumer is branch-specific.** Route A keeps the refined graph as LLM-facing local evidence. Route B, if GCL becomes the mainline, also makes the refined graph or its views GNN/GCL-facing as part of contrastive training. Different GNN backbones may require different view construction or adapter rules.
- **Split/dataset boundary.** The research split follows TwiBot. Exact dataset version and canonical split files should follow the repo's TwiBot protocol before any experiment claim.
- **Structural candidate generator action space and scoring target are fixed for v1.** The candidate generator is an edge-action proposer with binary `add/delete` actions. The first scoring target is edge utility rather than a generic reliability label. If it is learned, training/tuning must follow the official TwiBot split. Test labels remain evaluation-only.
- **Bot-human heterophily branch handling.** In the GCL branch, first follow a BotBR-inspired prune-retain contrastive framing: contrast a homophily/reliability-filtered view with an original or evidence-retaining view. In the LLM local evidence graph branch, heterophily is an evidence-role presentation issue, so the first comparison should cover `post_delete`, `retained_provenance`, and `role_annotated` prompt variants.
- **GCL status.** GCL is a mainline candidate, not ruled out and not merely a postscript. Its first candidate formulation is homo-hetero contrast. It does not need to prove that the homo-hetero view is unique to refined local evidence graphs; promotion can be based on experimental effectiveness against close baselines.

### Problem Boundary Draft

The current line should not be framed as direct migration of GAugLLM. GAugLLM is a TAG/GCL augmentation paper: its useful part for this project is the two-step pattern, "structural proposal + semantic verification", not its graph-contrastive objective or its generic TAG-oriented structural candidate heuristic.

The safer research boundary is:

> For hard or structurally unreliable nodes in social bot detection, construct or refine a small local evidence graph from bot-domain graph signals, semantic signals, and uncertainty signals, then use an LLM as a bounded evidence/embedding enhancer rather than a label-changing predictor.

This preserves the project identity as local, budgeted, and reversible evidence refinement instead of turning the paper into full graph structure learning, generic TAG augmentation, or direct LLM prediction work.

### Consequences For The Three Proposed Routes

1. **GCL route.** Do not reject GCL solely because social-bot GCL papers exist. GCL is a mainline candidate; the practical boundary question is whether homo-hetero contrast is empirically effective under the official TwiBot split and close bot-detection baselines.
2. **Structural candidate generator correction.** This is the best near-term repair for the GAugLLM mismatch. The candidate generator should propose binary `add/delete` edge actions using bot-detection-aware signals: relation type, direction, ego degree regime, local heterophily, edge confidence, GNN uncertainty, LM-GNN disagreement, and residual-risk or conformal quality signals.
3. **Local evidence graph + LLM route.** This is the cleanest narrative for the current `LM -> GNN -> refined local evidence graph -> LLM` line. The LLM should consume curated evidence artifacts, not raw full ego graphs, and should be described as an enhancer that returns an embedding/evidence representation, not as a final label predictor.

### Candidate Literature Triage

| Route | Paper | What it supports | Boundary risk | Candidate action |
|-------|-------|------------------|---------------|------------------|
| GAugLLM migration boundary | [GAugLLM, 2024](https://arxiv.org/abs/2406.11945) | Two-stage "structural/textual commonality" edge augmentation for TAG contrastive learning. | Its objective is self-supervised GCL on general TAGs, not social bot detection or post-hoc local evidence refinement. | Borrow only the propose-then-verify operator pattern. Replace the candidate generator with bot-domain signals. |
| GCL route | [BotSCL, 2024](https://arxiv.org/abs/2306.07478) | Bot-domain contrastive learning under heterophily; explicitly treats bot-human edges as camouflage risk. | High overlap if this project simply adds supervised contrastive loss over bot/human labels. | Use as a boundary reference: a GCL branch needs a distinct local-evidence or candidate-validation view, not just class-aware neighbor contrast. |
| GCL route | [SEBot, KDD 2024](https://openreview.net/forum?id=BjobM0Iwb4) | Social-bot multi-view contrastive learning using structural entropy and graph/subgraph views. | High overlap if the contribution becomes graph/subgraph multi-view CL for social bots. | Use as evidence that GCL is legitimate in bot detection, but also that novelty must come from a different view-construction hypothesis. |
| GCL and reliability boundary | [BotBR, SIGIR 2025](https://doi.org/10.1145/3726302.3729908) | Direct bot-domain evidence that edge reliability and homophily-based consistency contrastive learning are strong. | Very high overlap with binary edge reliability plus contrastive graph learning. | Treat as boundary-setting baseline; avoid binary reliable/unreliable deletion as the headline novelty. |
| Bot-specific structural candidate generator | [BECE, TNNLS 2024/2025](https://ieeexplore.ieee.org/document/10530431/) | Edge confidence evaluation for social bot graphs; unreliable edges arise from bot camouflage with genuine users. | If copied directly, the method becomes another edge-confidence filtering model. | Use as evidence that candidates should be edge-level and bot-domain-aware; preserve suspicious edges as evidence instead of only filtering them. |
| Bot-specific structural candidate generator | [RABot, AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/view/37127) | Reinforcement-guided graph augmentation for imbalanced and noisy bot detection; includes edge filtering during message passing. | Training-time RL augmentation and filtering may pull the project away from post-hoc, local, reversible refinement. | Consider as a candidate-generator inspiration if deterministic relation/direction/uncertainty heuristics fail. |
| Bot-specific LM+GNN support | [LGB, 2024](https://arxiv.org/abs/2406.08762) | Direct social-bot evidence that LM semantics help where graph links are sparse, while GNNs remain useful for connected nodes. | It is late fusion of LM and GNN, not graph refinement or evidence prompting. | Use to justify regime-specific semantic rescue for sparse or weak-support nodes. |
| Bot-specific LM+graph support | [BotLGT, Neurocomputing 2025](https://www.sciencedirect.com/science/article/pii/S0925231225021253) | LLM-derived semantic embeddings plus motif-enhanced structural encodings for social bot detection. | Global graph transformer architecture, not local evidence graph refinement. | Use as weak support that semantic and structural signals are complementary in bot detection. |
| Local evidence graph + LLM | [GLANCE, ICLR 2026](https://openreview.net/forum?id=oODFyykHF5) | Node-aware router decides when to invoke an LLM; reported gains are concentrated on nodes where GNNs struggle, including heterophilous nodes. | It refines predictions with LLM assistance, not graph repair. | Strong support for selective, budgeted LLM calls after hard-node routing, but not evidence that the graph itself was repaired. |
| Local evidence graph + LLM | [LOGIN, 2024/2025](https://arxiv.org/abs/2405.13902) | Uses concise prompts for spotted nodes carrying semantic and topological information, with LLM-as-consultant framing. | It is an interactive training framework; not post-hoc local refinement. | Borrow prompt compactness and consultant role; avoid claiming LOGIN reproduction. |
| Evidence serialization | [GraphText, 2023](https://arxiv.org/abs/2310.01089) | Converts graph structure and attributes into text sequences for LLM consumption. | Raw graph serialization can exceed budget and may overclaim graph reasoning. | Serialize curated local evidence artifacts only. |
| LLM graph-prompt caution | [Huang et al., TMLR 2024](https://openreview.net/forum?id=L2jRavXRxs) | Warns that LLMs may process graph prompts as contextual text rather than faithful graph structure. | Undercuts strong claims that the final LLM truly reasons over topology. | Frame prompts as evidence context, not proof of LLM structural reasoning. |
| LLM-guided structure repair | [LLM4RGNN, KDD 2025](https://arxiv.org/abs/2408.08685) | LLM-guided robust graph structure inference identifies malicious edges and missing important edges. | Adversarial robustness on TAG-style graphs, not social bot local evidence graphs. | Use as support for LLM-assisted edge plausibility, with bot-domain generator replacement. |
| Taxonomy support | [Graph Meets LLM survey, IJCAI 2024](https://www.ijcai.org/proceedings/2024/898) | Distinguishes LLM roles as enhancer, predictor, and alignment component. | Survey only. | Use taxonomy to keep this project in the LLM-as-enhancer lane unless explicitly changed. |

### Follow-Up From Two LLM-on-Graph Surveys

This note was added after reading two local PDF anchors provided on 2026-05-16:

- Chen et al., 2024, **Exploring the Potential of Large Language Models (LLMs) in Learning on Graphs**.
- Wu et al., ICML 2025, **When Do LLMs Help With Node Classification? A Comprehensive Analysis**.

#### Verified Observations From The Two Anchors

1. **LLM role taxonomy matters.** Chen et al. split graph usage into LLM-as-enhancer and LLM-as-predictor. This project should not blur the final LLM role: an enhancer/verifier line has different claims and evaluation from an LLM-as-final-classifier line.
2. **Naive structure prompts are unstable.** Chen et al. show that adding 2-hop neighborhood information can help some datasets but can also hurt when heterophily misleads the LLM. Their PubMed case study shows a structure-aware prompt following neighbor summary toward the wrong class while a structure-ignorant prompt is correct.
3. **LLM confidence is not reliable just because the model states it.** Chen et al. report that asking LLMs for confidence often yields overconfident outputs, so this project should rely on external calibration, consistency, or downstream validation rather than raw self-reported LLM confidence.
4. **LLMs help more when graph structure is less informative.** Wu et al. report across LLMNodeBed that LLM-as-Encoder has clearer gains over LM encoders on less informative graphs, especially heterophilic graphs. This supports targeting graph-corruption, weak-support, and heterophily regimes rather than claiming uniform global gains.
5. **Prompt design and learning paradigm are first-class variables.** Wu et al. evaluate encoder, explainer, predictor, direct-inference, and graph-foundation-model paradigms under supervised, semi-supervised, and zero-shot settings. This warns against comparing a local evidence prompt against only weak baselines.

#### Local Evidence Graph + LLM Literature Summary

| Paper | Key assumption | Boundary | Inspiration for this project |
|-------|----------------|----------|------------------------------|
| Chen et al., 2024, Exploring LLMs in Learning on Graphs | LLMs can enhance textual attributes or act as predictors, but graph structure must be translated into LLM-compatible text. | TAG node classification; not social bot detection; structure prompts can suffer from heterophily. | A local evidence graph should be curated and summarized, not dumped. The final LLM should be evaluated under a role-specific protocol. |
| Wu et al., ICML 2025, LLMNodeBed | LLMs add most value when text signal compensates for weak graph-label information. | Benchmark study; not a graph repair method; social datasets are general social/web graphs, not TwiBot-style bot graphs. | Use LLM only on routed regimes where graph evidence is weak or conflicting; report gains by regime, not only global F1. |
| [TAPE, ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/1766d75b077b66457040e4661771aec5-Abstract-Conference.html) | LLM explanations can become useful text features for downstream TAG representation learning. | LLM explanations enrich node text; it does not build a local evidence graph or repair edges. | If using LLM rationales, store them as evidence features/provenance rather than treating them as direct labels. |
| [GraphText, 2023](https://arxiv.org/abs/2310.01089) | Graphs can be translated into text sequences using graph-syntax trees. | General graph-to-text reasoning; can be verbose and not domain-calibrated. | Supports graph-to-text serialization, but this project should serialize only selected local evidence and relation/direction metadata. |
| [Huang et al., TMLR 2024](https://openreview.net/forum?id=L2jRavXRxs) | LLM gains from graph prompts may come from label-relevant neighbor text rather than true structural reasoning. | TAG prompt analysis; not bot detection. | Do not claim the LLM "understands" the graph. Claim it uses curated local evidence context; test ablations that remove relation labels, neighbor text, and structural fields. |
| [LLaGA, ICML 2024](https://arxiv.org/abs/2402.08170) | LLMs need graph structures converted into structure-aware token sequences or embeddings. | General graph assistant; requires model/projector machinery beyond this project v1. | Useful as a high-end reference for graph-to-LLM adaptation; likely too heavy for the current local evidence route. |
| [GraphGPT, SIGIR 2024](https://arxiv.org/abs/2310.13023) | Instruction tuning can align LLMs with graph structural knowledge. | Requires graph instruction tuning and graph encoders; not local post-hoc evidence refinement. | Use as a comparison boundary: our lightweight route should not claim full graph instruction-tuning capability. |
| [LLM4RGNN, KDD 2025](https://arxiv.org/abs/2408.08685) | LLMs can help identify malicious edges and missing important edges for robust graph structure inference. | Adversarial TAG robustness, not social bot graphs; uses robust structure inference rather than evidence prompting. | Supports LLM-assisted edge plausibility as a concept, but candidate generation must be bot-domain adapted and split-safe. |

Evidence status:

- **Verified from local PDFs.** Chen et al. explicitly separate LLM-as-enhancer and LLM-as-predictor, test 2-hop/neighborhood summaries, and warn about heterophilous neighbors and direct LLM confidence prompts. Wu et al. report LLMNodeBed results across 10 homophilic and 4 heterophilic datasets and show that learning paradigm, homophily, and prompt design change the value of LLMs.
- **Verified from external paper pages.** TAPE, GraphText, Huang et al., LLaGA, GraphGPT, LOGIN, GLANCE, and LLM4RGNN support graph-to-text, LLM-as-explainer/enhancer/predictor, selective routing, or LLM-guided edge plausibility.
- **Inference for this project.** These works support a curated local evidence graph and bounded LLM verifier/enhancer. They do not by themselves prove that a social-bot local evidence graph will outperform LM+GNN baselines; that claim requires regime-level experiments on TwiBot/MGTAB-style datasets.

#### Additional Support For LLM-Facing Local Evidence Graphs

These references are especially relevant after fixing the LLM role as enhancer and the downstream consumer as a discriminator.

| Paper | What it supports | Boundary | Migration to this project |
|-------|------------------|----------|---------------------------|
| [TAPE, ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/1766d75b077b66457040e4661771aec5-Abstract-Conference.html) | LLM-generated explanations can be translated into informative features for downstream GNNs. | Explanations are generated from node text/prediction prompts, not from a refined bot local evidence graph. | Strongest direct support for "LLM as feature/embedding enhancer, discriminator handles final prediction." |
| [ENGINE, IJCAI 2024](https://www.ijcai.org/proceedings/2024/634) | LLM encoders can be integrated with GNNs for textual graphs using efficient side structures, caching, and early exit. | It improves textual encoding and joint LM-GNN efficiency, not local graph evidence refinement. | Supports the engineering boundary that LLM-enhanced embeddings can feed downstream graph/discriminator modules without making the LLM the final predictor. |
| [OFA, ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/57faf5642eb06e0602b95f6aa989b38a-Abstract-Conference.html) | Nodes and edges can be described in natural language and encoded into a shared embedding space; nodes-of-interest standardize task-specific graph inputs. | General graph foundation model; not social bot detection and not LLM evidence enhancement. | Supports representing a target-centered local evidence graph with typed node/edge descriptions instead of raw adjacency dumps. |
| [G-Retriever, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/efaf1c9726648c8ba363a5c927440529-Abstract-Conference.html) | Relevant graph parts can be retrieved under context-budget pressure; subgraph retrieval helps reduce hallucination and scale to large textual graphs. | Graph QA/RAG setting; its output is generative text rather than an embedding for a discriminator. | Supports selecting a compact, relevant local evidence graph before LLM enhancement rather than feeding a full ego graph. |
| [RAGraph, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/34d6c7090bc5af0b96aeaf92fa074899-Abstract-Conference.html) | Retrieval-augmented graph learning can enrich GNN/foundation-model context through retrieved graph vectors and message-passing prompting. | Retrieves external toy graphs, not LLM-curated bot local evidence. | Useful for the GCL/GNN branch: refined or retrieved evidence graphs can be consumed by graph encoders, but this is not direct evidence for LLM prompting. |
| [LLM-GNN, ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/862819c227b16f9af64dd6ad6cfdf275-Abstract-Conference.html) | Selective LLM invocation can improve GNN training under a budget by annotating small node subsets. | It uses LLM annotations/pseudo-labels, which conflicts with the current "LLM does not change label" boundary. | Borrow only the budgeted selection and downstream-GNN handoff idea; do not migrate pseudo-labeling as the main role. |
| [Exploring the Potential of LLMs for Heterophilic Graphs, NAACL 2025](https://arxiv.org/abs/2408.14134) | Fine-tunes an LLM-enhanced edge discriminator to identify homophilic and heterophilic edges from node text, then uses LLM-guided edge reweighting for GNN propagation. | General heterophilic TAGs, not social bot detection; it uses homo/hetero edge supervision and edge reweighting, not a local evidence graph. | Strong support for a discriminator that consumes LLM-enhanced edge/evidence representations and separates homo/hetero edge roles. |
| [EG-GCN, AAAI 2025](https://ojs.aaai.org/index.php/AAAI/article/view/34087) | Uses an edge discriminator to split neighborhoods into homophilic and heterophilic parts, then aggregates them separately. | Non-LLM heterophily GNN; not social bot detection. | Useful architectural reference for the downstream discriminator or GNN branch: homo/hetero evidence can be consumed by separate channels instead of collapsed. |
| [Graph Pattern Comprehension Benchmark, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/9316da9c25ab559ba678f2fe52217a64-Abstract-Conference.html) | LLM graph-pattern performance depends on how graph data is formatted and aligned with pretrained knowledge. | Benchmark/capability study, not a bot detector. | Supports typed, compact, relation-aware evidence serialization and ablations over formatting choices. |

Why the LLM branch still needs a heterophily policy:

- This is not because the LLM branch must solve graph heterophily as a GCL objective. It is because the refined local evidence graph can contain both supportive homophilic edges and suspicious bot-human heterophilic edges. If they are serialized identically, the LLM enhancer and downstream discriminator can conflate "useful support" with "camouflage/suspicious evidence."
- The LLM branch should not automatically copy the GCL pruning rule. For LLM evidence enhancement, bot-human heterophily can be serialized as suspicious evidence if relation type, direction, confidence, and provenance are explicit.
- A claim-safe design should compare at least three LLM evidence prompt variants: post-delete refined graph, retained-with-role-annotation graph, and evidence-preserving graph with explicit add/delete provenance. These are prompt/evidence presentations, not extra candidate-generator actions.
- Any discriminator trained on LLM-enhanced embeddings must be evaluated against a no-LLM refined-graph embedding baseline and a raw ego-graph prompt/embedding baseline.

#### Structural Prompt Design Migration

The refined local evidence graph should be treated as a compact, structured prompt object, not as a raw ego graph dump. The goal is to make the LLM produce a useful embedding/evidence representation for the downstream discriminator, not to ask the LLM for a label.

| Paper | Structural prompt lesson | Boundary | Migration to refined local evidence graph |
|-------|--------------------------|----------|-------------------------------------------|
| [Talk Like a Graph, ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/bf72f65f30eedf5d48da6980ee02b589-Abstract-Conference.html) | Graph encoding method, graph task, and graph shape all change LLM graph reasoning performance. | Synthetic/general graph reasoning, not social bot detection. | Prompt format is an experimental variable. Compare edge-list, adjacency-list, natural-language relation triples, and compact JSON-like schemas on the same refined graph. |
| [Let Your Graph Do the Talking, 2024](https://arxiv.org/abs/2402.05862) | Structured data should be encoded into sequential forms that preserve relationships and are easy for LLMs to parse. | General structured-data encoding; no bot-specific claim. | Use a stable schema with explicit `target`, `nodes`, `edges`, `action_log`, and `evidence_budget` sections rather than prose-only prompts. |
| [GraphText, 2023](https://arxiv.org/abs/2310.01089) | Graph-syntax trees bridge graph data and language-model input. | General graph-to-text bridge; can be verbose. | Use a small graph-syntax-tree style representation for the local evidence graph: target node, 1-hop/2-hop evidence nodes, relation labels, direction, and action provenance. |
| [Huang et al., TMLR 2024](https://openreview.net/forum?id=L2jRavXRxs) | LLMs may use neighbor text as contextual paragraphs rather than faithful topology; label-relevant phrases can dominate structural reasoning. | TAG prompt analysis, not bot detection. | Keep graph structure explicit but do not overclaim structural reasoning. Add ablations that remove relation labels, direction, node text, or add/delete provenance. |
| [Structure Guided Prompt, EMNLP 2024](https://aclanthology.org/2024.emnlp-main.528/) | Explicit graph-like structure can scaffold multi-step reasoning over complex text. | Reasoning over text-derived graphs, not graph node classification. | Use sectioned prompts that separate evidence extraction from edge-action interpretation: first list facts/relations, then ask for an embedding-oriented evidence summary. |
| [GraphWiz, 2024](https://arxiv.org/abs/2402.16029) and [InstructGraph, Findings ACL 2024](https://aclanthology.org/2024.findings-acl.801/) | Graph-specific instruction formats and code-like verbalizers improve LLM handling of graph problems. | Instruction-tuned graph reasoning/generation, not a lightweight enhancer. | Borrow the code-like graph verbalizer idea, but avoid claiming graph-instruction-tuned capability unless such a model is actually used. |

Prompt object migration proposal:

- **Target block.** Target account id, available text/profile summary, base LM/GNN prediction metadata, and uncertainty/risk signal.
- **Node evidence block.** Only selected local evidence nodes under a fixed budget; include source text snippets or compact semantic summaries.
- **Edge action block.** For each candidate action, record `action=add` or `action=delete`, relation type, direction, original/predicted confidence, and proposal source.
- **Homophily/heterophily block.** Mark whether an edge is estimated homophilic, heterophilic, or unknown; this is an evidence role for the discriminator, not an LLM label prediction.
- **Output contract.** Ask the LLM for an embedding/evidence representation or compact evidence summary to be embedded, not for the final bot/human label.
- **Ablation plan.** Compare raw ego graph serialization, refined graph without action provenance, refined graph with add/delete provenance, and refined graph with homo/hetero roles.

#### Structure Candidate Adaptation: Assumptions And Boundaries

Here, **candidate generator** means the component that proposes binary edge actions before final refinement: `add` an edge or `delete` an edge. In a GAugLLM-style pattern, it is the "structural proposal" stage before semantic/LLM validation. In this project it should not be the final bot classifier; it is only a proposal mechanism for how the refined local evidence graph should change its edge set.

The candidate generator should not be "manual features in disguise" and should not be a direct transplant of a TAG similarity heuristic. The literature suggests the following research constraints:

| Literature family | Key assumption | Boundary / failure mode | Project-level implication |
|-------------------|----------------|--------------------------|---------------------------|
| General GSL: [LDS](https://arxiv.org/abs/1903.11960), [IDGL](https://arxiv.org/abs/2006.13009), [Pro-GNN](https://arxiv.org/abs/2005.10203) | Useful graph structure can be learned as edge probabilities, similarity metrics, or robust low-rank/sparse/smooth structures. | Often global, training-time, homophily/smoothness-biased, and may not preserve suspicious heterophily as evidence. | Use probabilistic candidate scoring and accept/rollback thinking, but avoid full-graph reconstruction and avoid assuming feature similarity means reliability. |
| Robust/noisy graph learning: [NRGNN](https://arxiv.org/abs/2201.00232), Pro-GNN | Noisy edges harm message passing; denoising or removing noisy edges can recover useful propagation. | Many methods define clean edges through label smoothness or feature smoothness, which conflicts with bot-human camouflage evidence. | Use this family as support for edge deletion as one binary action, while keeping separate evidence-prompt variants that can still show deleted/suspicious-edge provenance to the LLM. |
| Bot edge confidence: [BECE](https://ieeexplore.ieee.org/document/10530431/) | Advanced bots create unreliable edges with genuine users; edge confidence is a direct bot-domain object. | BECE is reliability filtering; copying it risks becoming a binary edge-confidence detector. | Use edge-level confidence as a domain premise, but make the project boundary "evidence role and local verification", not just edge deletion. |
| Bot heterophily/GCL: [BotSCL](https://arxiv.org/abs/2306.07478), [SEBot](https://openreview.net/forum?id=BjobM0Iwb4) | Bot-human heterophily and adversarial behavior make indiscriminate message passing harmful. | They already define bot-specific contrastive/heterophily handling. | Candidate generation must distinguish "harmful for propagation" from "useful suspicious evidence". |
| LLM-guided edge plausibility: [GAugLLM](https://arxiv.org/abs/2406.11945), [LLM4RGNN](https://arxiv.org/abs/2408.08685) | Structure candidates should be semantically checked; text can help validate or repair graph edges. | TAG/adversarial settings; structural proposal stage is not bot-specific. | Keep propose-then-verify, but make proposal relation/direction/regime-aware for bot detection. |

Candidate-generator design should therefore be chosen after deciding the research boundary. The action space is fixed to binary `add/delete`, while the scoring rule remains open. Plausible non-final scoring options include: (a) deterministic add/delete gates with relation/direction/regime constraints; (b) learned edge-action utility trained only on official train/validation-tune splits; or (c) hybrid add/delete proposals from GNN uncertainty plus LM semantic validation.

Design constraints before choosing a candidate generator:

- Avoid fixed user-behavior thresholds or metadata recipes as the main method; those would look like manual feature engineering rather than graph refinement.
- Avoid directly copying TAG similarity or same-label-edge proposal rules; social bot graphs include camouflage and heterophilic edges that can be harmful for propagation while useful as evidence.
- Keep proposal signals split-safe: train labels may be used only through training-time learned components or calibration, while validation/test labels must not leak into edge proposal or LLM prompts.
- Preserve provenance for every `add` or `delete` edge action so the local evidence graph remains auditable and reversible.

#### GCL Literature: What It Does And Does Not Prove

Existing evidence does **not** prove that using GCL would make the project unnovel. It proves a narrower point: if GCL is adopted, novelty must come from the semantics of the views and the social-bot failure regime, not from merely adding a contrastive loss.

| Paper | Key assumption | Boundary | Implication |
|-------|----------------|----------|-------------|
| [GraphCL, NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/hash/3fe230348e9a12c13120749e3f9fa4cd-Abstract.html) | Augmented graph views should preserve task semantics while perturbing nodes, edges, attributes, or subgraphs. | General graph representation learning; augmentation priors are domain-sensitive. | A bot GCL branch must define which local evidence perturbations preserve bot-detection semantics. |
| [GCA, WWW 2021](https://github.com/CRIPAC-DIG/GCA) | Adaptive augmentation can preserve more important topology/features than uniform random dropping. | Still general-purpose and often relies on centrality/feature-importance priors. | Supports learned/adaptive view generation, but bot-specific relation/direction semantics remain necessary. |
| [BGRL, ICLR 2022](https://openreview.net/pdf?id=0UXT6PpRpW) | Bootstrap/self-supervised graph representation can scale without many negative pairs. | It is representation pretraining, not local evidence verification. | If compute is a concern, GCL does not have to be negative-pair-heavy; but it still needs meaningful views. |
| [GraphACL, NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/file/3430bcc30cdaabd0bf6c5d0c31bda67c-Paper-Conference.pdf) | Many GCL methods rely on prefabricated augmentations and homophily assumptions; augmentation-free/asymmetric designs can handle heterophily better. | General homophily/heterophily benchmarks, not bot detection. | Strong support that naive GCL views are not enough; a bot route should be heterophily-aware. |
| [Graph Contrastive Learning under Heterophily via Graph Filters, UAI 2024](https://proceedings.mlr.press/v244/yang24a.html) | Homophilic and heterophilic subgraphs require different filters/views. | General heterophily, not social bot camouflage. | Supports separating propagation-friendly and suspicious/evidence-bearing relations instead of treating all views as same semantics. |
| [BotSCL](https://arxiv.org/abs/2306.07478), [SEBot](https://openreview.net/forum?id=BjobM0Iwb4), [BotDCGC](https://www.sciencedirect.com/science/article/pii/S0950705124003253) | Social-bot GCL can exploit heterophily, structural entropy, or unsupervised clustering. | These already occupy important parts of bot-domain GCL. | GCL remains possible, but must be framed as local-evidence-graph contrast or validation, not generic bot contrastive learning. |

GCL branch boundary after user clarification:

- The first GCL candidate should be **homo-hetero contrast**: a homophily/reliability-pruned view versus a heterophily/evidence-retaining view.
- BotBR is the closest boundary reference: it motivates pruning/downweighting unreliable propagation while retaining an original graph signal for consistency-style learning.
- Backbone-specific handling is expected. R-GCN/HGT-style backbones can keep relation/direction channels explicit; homogeneous GCN/GAT/SAGE-style backbones may need precomputed view graphs, edge weights, or relation-collapsed adapters.
- GCL does not need to prove that its homo-hetero view is unique to refined local evidence graphs. It should be promoted if experiments show reliable gains over close baselines such as BotBR-style reliability contrast, BotSCL-style heterophily contrast, and SEBot-style multi-view contrast under the official TwiBot split.

### Working Takeaway

The strongest current boundary is not "GAugLLM for bot detection" and not generic "social-bot GCL". It is:

> Use bot-domain, TwiBot split-safe candidate generation to build a refined local evidence graph for hard regimes; use the LLM as a bounded evidence/embedding enhancer; keep GCL as a candidate mainline if homo-hetero contrast is experimentally effective.

This keeps GCL available without letting it become a generic contrastive add-on, and it turns the GAugLLM mismatch into a principled candidate-generator replacement problem rather than an ad-hoc patch.

### Resolved And Unclear Points Before Mainline Promotion

Resolved:

1. **Final LLM role.** The LLM does not change labels and is not the final predictor. It is an enhancer over refined local evidence.
2. **Dataset/split boundary.** Experiments should use TwiBot splits under the repo protocol.
3. **LLM enhancer output target.** The LLM outputs an embedding/evidence representation to a downstream discriminator.
4. **Refined graph consumer.** In the LLM route, the refined graph is LLM-facing local evidence. In the GCL route, the refined graph or its views are also consumed by GNN/GCL, with backbone-specific schemes allowed.
5. **Candidate generator action space, scoring target, and split.** The structural candidate generator is a binary `add/delete` edge-action proposer. The first scoring target is edge utility. If learned, it follows the official TwiBot split; test labels are evaluation-only.
6. **GCL first formulation.** The first GCL mainline candidate is homo-hetero contrast.
7. **GCL heterophily handling.** For GCL, bot-human heterophily follows a BotBR-inspired prune-retain contrastive framing.
8. **LLM-branch suspicious heterophily prompt variants.** For LLM local evidence graphs, compare `post_delete`, `retained_provenance`, and `role_annotated` prompt variants rather than choosing one before experiments.
9. **Downstream discriminator design.** The first downstream head is a node discriminator for final bot/human prediction. Edge-role heads remain reference mechanisms or later ablations.

Still unclear:

1. **Concrete edge utility proxy.** The v1 target is edge utility, but the first operational proxy still needs selection: validation loss delta, post-action discriminator margin, calibrated risk reduction, or contrastive-view utility.
2. **LLM enhancer output representation.** The first implementation still needs to choose dense embedding, structured evidence vector, or hybrid summary-plus-embedding.
3. **GCL promotion threshold.** The uniqueness requirement is removed; the remaining question is what experimental margin, slice improvement, and baseline set are sufficient to call the homo-hetero contrast route effective.

## Candidate Route: Post-hoc Calibration + Ranking

- **Why considered.** The current local evidence favors the simplest useful abstraction: freeze the detector, calibrate its posterior post-hoc, and rank nodes by risk or reject score.
- **Why not mainline.** This route can underuse graph-local structure if localized bins keep proving useful, but those bins should be added as ablations or sensitivity controls rather than becoming a new predictor.
- **Upgrade condition.** Promote graph-localized or relation-conditioned bins only if they consistently beat the scalar route on hard-node lift, AUPRC, and budget curves.
- **Difference from mainline references.** It is a ranking/proxy view over frozen outputs; graph structure enters only through optional localized bins, not as a separate classifier.

## Candidate Route: LM-side IB-EDL Calibration / Cascade Disagreement

- **Why considered.** SimTeG-style training already produces finetuned LM hidden states before the GNN. A calibrated LM-side uncertainty head could expose whether the text evidence alone is uncertain before graph propagation.
- **Why not mainline.** The LM and GNN are not independent views in the current pipeline: the GNN consumes LM hidden states, and both components are trained in the same bot-human label space. Any LM-GNN mismatch is therefore a cascade diagnostic, not the same text-graph disagreement studied by CTGL, SKETCH, or GAugLLM.
- **Upgrade condition.** Promote only if a split-safe `LM_IB_EDL_diagnostic` improves hard-node lift, AUPRC, and risk-coverage over LM entropy, LM temperature scaling, simple LM-GNN JSD, and the GNN-side conformal router.
- **Difference from strong references.** The mainline router remains graph-calibrated and post-hoc ranking-driven. IB-EDL is only a candidate LM uncertainty source or disagreement diagnostic, not a new bot classifier, not an LLM/GNN fusion predictor, and not an epistemic guarantee.

### Candidate Literature

#### Li et al., ICLR 2025, Calibrating LLMs with Information-Theoretic Evidential Deep Learning

- **Motivation.** Large language models can be confidently wrong, so downstream systems need calibrated confidence estimates rather than raw softmax probabilities.
- **Problem analysis.** The paper treats calibration as both an uncertainty modeling problem and a representation compression problem: an over-informative or poorly regularized head can preserve misleading evidence.
- **Mechanism / method.** It combines information bottleneck regularization with evidential deep learning so the model emits Dirichlet-style evidence while discouraging unhelpful information in the calibrated representation.
- **Result / evidence.** The reported evidence supports improved calibration behavior for LLM confidence estimation relative to simpler confidence baselines.
- **Boundary.** The paper targets LLM calibration. It does not prove that an LM-side evidential head is an independent modality once the downstream GNN consumes the same LM hidden states.
- **Migration to this project.** Use only as `LM_IB_EDL_diagnostic`: fit an LM-side uncertainty head from finetuned LM hidden states or LM logits, output expected probability, Dirichlet strength, vacuity, dissonance, entropy, and LM prediction, then compare these with GNN-side conformal uncertainty.
- **Links.** Paper: https://openreview.net/forum?id=YcML3rJl0N ; official repo: https://github.com/sandylaker/ib-edl

#### Sensoy et al., NeurIPS 2018, Evidential Deep Learning to Quantify Classification Uncertainty

- **Motivation.** Standard neural classifiers often expose only point probabilities, which are poor uncertainty objects when evidence is weak or conflicting.
- **Problem analysis.** Softmax confidence conflates class preference with evidence strength; a model can be confident even when it has little reliable support.
- **Mechanism / method.** The classifier predicts non-negative evidence parameters for a Dirichlet distribution over class probabilities, allowing uncertainty measures such as total evidence and vacuity.
- **Result / evidence.** The paper shows that evidential classification can represent uncertainty without requiring explicit Bayesian sampling at inference time.
- **Boundary.** EDL uncertainty is not automatically reliable under dataset shift or post-hoc graph changes, and later work questions whether EDL always yields meaningful epistemic uncertainty.
- **Migration to this project.** Borrow the evidence-output interface for an LM-side diagnostic head. Do not claim that EDL alone certifies hard nodes or replaces conformal calibration.
- **Links.** Paper: https://papers.nips.cc/paper/7580-evidential-deep-learning-to-quantify-classification-uncertainty ; official repo: not verified

#### Alemi et al., ICLR 2017, Deep Variational Information Bottleneck

- **Motivation.** Deep models can learn task-relevant representations that discard nuisance information by explicitly optimizing a compression-prediction tradeoff.
- **Problem analysis.** Without bottleneck pressure, learned representations may preserve spurious information that helps training likelihood but hurts robust uncertainty and transfer.
- **Mechanism / method.** The method uses a variational approximation to the information bottleneck objective, balancing label prediction against mutual-information compression.
- **Result / evidence.** The work establishes a practical deep-learning route to information bottleneck training.
- **Boundary.** It is not a calibration method by itself and does not address graph-structured node classification.
- **Migration to this project.** Use as the conceptual source for bottleneck regularization inside an LM-side calibration ablation, not as a main router mechanism.
- **Links.** Paper: https://openreview.net/forum?id=HyxQzBceg ; official repo: not verified

#### Guo et al., ICML 2017, On Calibration of Modern Neural Networks

- **Motivation.** Modern neural networks can have high accuracy while producing poorly calibrated probabilities.
- **Problem analysis.** Architectural and training choices can increase confidence miscalibration, so probability quality must be evaluated separately from accuracy.
- **Mechanism / method.** The paper formalizes calibration metrics such as ECE and demonstrates temperature scaling as a simple post-hoc calibration baseline.
- **Result / evidence.** Temperature scaling is shown to be a strong and simple calibration baseline for many classifiers.
- **Boundary.** It provides scalar probability calibration, not hard-node ranking, graph conformal prediction sets, or relation-conditioned uncertainty.
- **Migration to this project.** Treat LM temperature scaling as the required baseline before claiming that LM-side IB-EDL adds value.
- **Links.** Paper: https://arxiv.org/abs/1706.04599 ; official repo: not verified

#### Desai and Durrett, EMNLP 2020, Calibration of Pre-trained Transformers

- **Motivation.** Pretrained transformers can be accurate but miscalibrated, and their calibration behavior differs across in-domain and out-of-domain settings.
- **Problem analysis.** Pretraining and fine-tuning change confidence behavior; transformer calibration cannot be assumed from classifier accuracy.
- **Mechanism / method.** The paper evaluates calibration of pretrained transformer classifiers across tasks and studies standard calibration interventions.
- **Result / evidence.** It shows that pretrained transformers often improve calibration compared with non-pretrained models but still require careful evaluation, especially under distribution shift.
- **Boundary.** This is text-classifier calibration, not graph-aware uncertainty and not LM-GNN disagreement.
- **Migration to this project.** Use it to justify measuring LM-side ECE, Brier, NLL, and OOD-like hard-node behavior before using LM uncertainty in router analysis.
- **Links.** Paper: https://arxiv.org/abs/2003.07892 ; official repo: https://github.com/shreydesai/calibration

#### Kong et al., EMNLP 2020, Calibrated Language Model Fine-Tuning for In- and Out-of-Distribution Data

- **Motivation.** Fine-tuned language models can become overconfident, especially when test data differ from the fine-tuning distribution.
- **Problem analysis.** Standard fine-tuning optimizes label likelihood but may not preserve calibrated uncertainty under in-distribution and out-of-distribution evaluation.
- **Mechanism / method.** The paper proposes calibration-aware fine-tuning strategies for language models.
- **Result / evidence.** It reports improved calibration behavior while maintaining competitive classification performance.
- **Boundary.** It calibrates the LM classifier, not a graph-conditioned posterior, and does not solve the dependence between LM features and downstream GNN predictions.
- **Migration to this project.** Use as an LM-side calibration baseline or training reference if `LM_IB_EDL_diagnostic` is implemented.
- **Links.** Paper: https://arxiv.org/abs/2010.11506 ; repo: https://github.com/Lingkai-Kong/Calibrated-BERT-Fine-Tuning

#### Kim et al., Findings of EACL 2023, Bag of Tricks for In-Distribution Calibration of Pretrained Transformers

- **Motivation.** Transformer calibration can often be improved by practical training and post-hoc choices without changing the full architecture.
- **Problem analysis.** Calibration quality is sensitive to implementation details, and complex calibration methods should be compared against strong simple recipes.
- **Mechanism / method.** The paper studies a set of practical calibration tricks for pretrained transformer classifiers.
- **Result / evidence.** It provides evidence that simple transformer-specific calibration choices can be strong in-distribution baselines.
- **Boundary.** It does not address graph propagation, relation-conditioned uncertainty, or post-rewrite calibration.
- **Migration to this project.** Use as a warning: IB-EDL must beat strong LM calibration baselines, not only raw softmax or entropy.
- **Links.** Paper: https://arxiv.org/abs/2302.06690 ; official repo: not verified

#### Jiang et al., TACL 2021/2022, How Can We Know When Language Models Know?

- **Motivation.** Language models need reliable confidence estimates so downstream systems can decide when to trust or abstain.
- **Problem analysis.** Confidence from language models can fail even when answers look plausible, so calibration must be evaluated against task correctness.
- **Mechanism / method.** The paper studies calibration and confidence estimation for language model question answering.
- **Result / evidence.** It shows that language model confidence can be calibrated and evaluated with task-level correctness signals.
- **Boundary.** The task is language QA rather than social-bot node classification, and it does not provide graph-side calibration.
- **Migration to this project.** Borrow the evaluation stance: LM-side uncertainty must be judged by whether it improves hard-node ranking and abstention curves, not by interpretability alone.
- **Links.** Paper: https://transacl.org/index.php/tacl/article/view/2865 ; repo: https://github.com/jzbjyb/lm-calibration

#### Shen et al., NeurIPS 2024, Are Uncertainty Quantification Capabilities of Evidential Deep Learning a Mirage?

- **Motivation.** Evidential deep learning is widely used for uncertainty, but its claimed uncertainty properties need stress testing.
- **Problem analysis.** EDL may appear to quantify epistemic uncertainty while failing under certain data or training conditions.
- **Mechanism / method.** The paper critically analyzes and empirically evaluates whether EDL uncertainty behaves as expected.
- **Result / evidence.** It raises caution that EDL uncertainty should not be treated as a guaranteed epistemic signal.
- **Boundary.** This is primarily a critique and diagnostic reference, not a construction of a better social-bot router.
- **Migration to this project.** Use as a guardrail: any LM-side IB-EDL branch must be described as a candidate diagnostic and validated against strong baselines before entering mainline claims.
- **Links.** Paper: https://arxiv.org/abs/2402.06160 ; official repo: not verified

### Migration Design for This Project

- **Candidate mode.** If implemented, use a separate `LM_IB_EDL_diagnostic` ablation. It should consume finetuned LM hidden states or LM logits and should not consume GNN posterior during fitting.
- **Outputs.** Record LM expected probability, Dirichlet strength, vacuity, dissonance, entropy, and LM prediction.
- **Disagreement diagnostics.** Compare with the GNN-side conformal router using `LM argmax != GNN argmax`, `JSD(p_LM_EDL, p_GNN)`, `LM confident / GNN uncertain`, and `LM uncertain / GNN confident`.
- **Evaluation.** Judge the branch by hard-node error lift, AUPRC, risk-coverage, ECE, Brier score, and NLL. Test labels are evaluation-only and must not be used for LM calibration tuning.
- **Claim boundary.** This branch is not independent text-graph disagreement, not a new bot classifier, not an LLM/GNN fusion predictor, and does not claim that EDL uncertainty itself has a reliable epistemic guarantee.

## Candidate Route: Full GCL/GSL After Rewriting

- **Why considered.** Modified graphs can be used for graph contrastive learning or graph structure learning after local rewriting.
- **Why not mainline.** It risks shifting the paper from post-hoc ego refinement into a broad training-time graph learning method.
- **Upgrade condition.** Only after local evidence artifacts prove useful should a separate branch test whether refined graphs improve GCL/GSL.
- **Difference from strong references.** Mainline keeps refined edges as local, budgeted, and reversible artifacts; GCL/GSL would train on modified views and needs a separate claim.

## Candidate Route: Learned Edge Role Head

- **Why considered.** A learned modifier could reduce hand-coded edge rules and better imitate structural-text coupling.
- **Why not mainline.** It introduces another trainable component and may become the real predictor unless carefully controlled.
- **Upgrade condition.** Promote only if deterministic/soft utility variants fail and counterfactual supervision can be split-safe and computationally feasible.
- **Difference from strong references.** Mainline currently treats role labels as artifact/provenance rather than the final method.

## Candidate Route: Four-Way Edge Roles

- **Why considered.** `supportive / suspicious / neutral / uncertain` roles preserve evidence that binary reliability deletion would remove.
- **Why not mainline.** The bins are easy to overfit as hand-crafted semantics without strong empirical support.
- **Upgrade condition.** Upgrade if ablations show role-aware evidence improves hard-node slices beyond continuous utility and binary reliability baselines.
- **Difference from strong references.** Strong social-bot reliability work motivates edge confidence, but not necessarily four fixed evidence roles.

## Candidate Route: Direct LLM Prediction

- **Why considered.** Direct LLM labels from evidence prompts are easy to inspect and may be strong for selected hard nodes.
- **Why not mainline.** It would shift the contribution from graph-quality-guided refinement to LLM prediction and weaken graph-method claims.
- **Upgrade condition.** Keep only as diagnostic baseline. Promote only if the project intentionally changes to an LLM-predictor paper.
- **Difference from strong references.** Mainline follows LLM-as-enhancer / embedding-refiner framing.

## Candidate Route: Full-Graph Rewrite

- **Why considered.** Full graph structure learning could globally repair noisy social graphs.
- **Why not mainline.** It is expensive, less auditable, and too close to general graph structure learning rather than local hard-node refinement.
- **Upgrade condition.** Consider only after local ego refinement has clear benefits and full-graph compute/rollback risks are addressed.
- **Difference from strong references.** Mainline borrows soft/probabilistic edge thinking but keeps the operation local and post-hoc.

## Candidate Route: Binary Reliability Headline

- **Why considered.** Binary reliable/unreliable edges are intuitive and align with social-bot unreliable-edge literature.
- **Why not mainline.** This direction is already crowded in social bot detection, and bot-human heterophily may be evidence rather than noise.
- **Upgrade condition.** Use only as a baseline or ablation, not as the main novelty claim.
- **Difference from strong references.** Mainline should not erase suspicious edges; it should reduce propagation while preserving evidence.

## Candidate Social-Bot Baselines and Context Papers

These papers may help appendix positioning, but they are not direct mainline mechanisms for the current pipeline. BotRGCN, BotRGT/BotHeterogeneity, TwiBot-22, and higher-order relation papers have been promoted to `docs/reference.md` because they support the relation-aware graph context that can be used as an ablation or secondary calibration channel, not the headline router.

- **BotSCL / social-bot contrastive learning.** Useful contrastive baseline context; not the current local post-hoc path.
- **BotMoE / multimodal expert fusion.** Useful multimodal fusion baseline context; not evidence-ego graph refinement.
- **RF-GNN / ensemble graph methods.** Useful robustness baseline context; not the current estimator-rewriter mechanism.
- **Dynamic bot graph methods such as BotDGT.** Useful future extension if temporal graphs enter scope; not current static ego refinement.

## Candidate TAG / LLM-Graph Methods Outside Mainline

- **LLM-as-GNN / graph vocabulary learning.** Interesting for foundation-style TAG models, but it changes the base architecture instead of refining a frozen detector.
- **LLM-generated text-attributed graphs.** Useful if the project needs synthetic node descriptions, but not the current social-bot ego repair path.
- **GraphRAG / knowledge-graph prompting systems.** Useful conceptual context for evidence serialization, but usually not node classification on social bot graphs.

## Candidate Engineering Work

- **Split `trainer.py`, `estimators.py`, and `operators.py`.** Needed for maintainability, but should be a separate interface-preserving refactor.
- **Delete or archive `LLMbot/baseline/` and `LLMbot/code/`.** Needed for repository hygiene, but separate from method implementation.
- **Remote GPU guide refresh.** Current remote notes may still point to deprecated paths; update only when the new root-mainline remote workflow is tested.
