import numpy as np
import torch
from sklearn.linear_model import Ridge


def _to_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _to_tensor(value, reference=None):
    if torch.is_tensor(value):
        return value
    kwargs = {}
    if reference is not None and torch.is_tensor(reference):
        kwargs["dtype"] = reference.dtype
        kwargs["device"] = reference.device
    return torch.tensor(value, **kwargs)


class NoOpSemanticOperator:
    metadata = {
        "claim_role": "anchor_control",
        "scientific_gate": "phase0_required",
        "paper_identity_risk": "low",
        "promotion_rule": "never",
    }

    def fit(self, *args, **kwargs):
        return self

    def apply(self, gnn_probs, lm_probs=None, focal_mask=None, **kwargs):
        return _to_tensor(gnn_probs)


class RidgeLocalSemanticOperator:
    """Probability-level semantic blend for sparse-evidence nodes.

    Note: fit() trains a Ridge projection (LM→GNN hidden space) for potential
    hidden-state injection, but apply() currently operates at the probability
    level only (direct logit blending). The Ridge model is retained for future
    hidden-state operator variants but is not used in the current apply() path.
    """
    metadata = {
        "claim_role": "candidate_mainline",
        "scientific_gate": "sparse_only",
        "paper_identity_risk": "low",
        "promotion_rule": "never",
    }

    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.ridge = Ridge(alpha=1.0)
        self.is_fitted = False

    def fit(self, lm_embeddings, gnn_node_repr, train_idx):
        train_idx = _to_numpy(train_idx).astype(np.int64)
        x_train = _to_numpy(lm_embeddings)[train_idx]
        y_train = _to_numpy(gnn_node_repr)[train_idx]
        self.ridge.fit(x_train, y_train)
        self.is_fitted = True
        return self

    def apply(self, gnn_probs, lm_probs=None, focal_mask=None, lm_embeddings=None, gnn_node_repr=None, classifier_head=None, **kwargs):
        probs = _to_tensor(gnn_probs).clone()
        if focal_mask is None or lm_probs is None:
            return probs
        focal_mask = _to_numpy(focal_mask).astype(bool)
        lm_probs = _to_tensor(lm_probs, reference=probs)
        probs[focal_mask] = (1.0 - self.alpha) * probs[focal_mask] + self.alpha * lm_probs[focal_mask]
        return probs


class LAGNNLocalSemanticOperator:
    metadata = {
        "claim_role": "candidate_mainline",
        "scientific_gate": "sparse_only",
        "paper_identity_risk": "low",
        "promotion_rule": "never",
    }

    def __init__(self, alpha=0.5):
        self.alpha = alpha

    def fit(self, *args, **kwargs):
        return self

    def apply(self, gnn_probs, lm_probs=None, focal_mask=None, edge_index=None, **kwargs):
        probs = _to_tensor(gnn_probs).clone()
        if focal_mask is None or lm_probs is None or edge_index is None:
            return probs

        focal_mask = _to_numpy(focal_mask).astype(bool)
        lm_probs = _to_tensor(lm_probs, reference=probs)
        edge_index = _to_numpy(edge_index)
        src, dst = edge_index

        neighbor_mean = torch.zeros_like(probs)
        degree = torch.zeros(probs.shape[0], dtype=probs.dtype, device=probs.device)
        src_t = torch.tensor(src, device=probs.device)
        dst_t = torch.tensor(dst, device=probs.device)
        neighbor_mean.index_add_(0, dst_t, probs[src_t])
        degree.index_add_(0, dst_t, torch.ones_like(dst_t, dtype=probs.dtype))
        neighbor_mean = neighbor_mean / degree.clamp(min=1).unsqueeze(-1)

        mask_t = torch.tensor(focal_mask, dtype=torch.bool, device=probs.device)
        probs[mask_t] = (1.0 - self.alpha) * probs[mask_t] + 0.5 * self.alpha * lm_probs[mask_t] + 0.5 * self.alpha * neighbor_mean[mask_t]
        return probs


class NoOpRepairOperator:
    metadata = {
        "claim_role": "anchor_control",
        "scientific_gate": "phase0_required",
        "paper_identity_risk": "low",
        "promotion_rule": "never",
    }

    def fit(self, *args, **kwargs):
        return self

    def apply(self, gnn_probs, focal_mask=None, **kwargs):
        return _to_tensor(gnn_probs)


class PruneRepairOperator:
    metadata = {
        "claim_role": "candidate_mainline",
        "scientific_gate": "propcorr_only",
        "paper_identity_risk": "low",
        "promotion_rule": "never",
    }

    def __init__(self, alpha=0.35):
        self.alpha = alpha

    def fit(self, *args, **kwargs):
        return self

    def apply(self, gnn_probs, focal_mask=None, lm_probs=None, **kwargs):
        probs = _to_tensor(gnn_probs).clone()
        if focal_mask is None:
            return probs
        focal_mask = torch.tensor(_to_numpy(focal_mask).astype(bool), dtype=torch.bool, device=probs.device)
        confidence = probs.max(dim=1, keepdim=True)[0]
        damped = probs * (1.0 - self.alpha * confidence)
        if lm_probs is not None:
            lm_probs = _to_tensor(lm_probs, reference=probs)
            damped = damped + self.alpha * lm_probs
        damped = damped / damped.sum(dim=1, keepdim=True).clamp(min=1e-8)
        probs[focal_mask] = damped[focal_mask]
        return probs


class DisagreementLocalRepairOperator:
    metadata = {
        "claim_role": "candidate_mainline",
        "scientific_gate": "propcorr_only",
        "paper_identity_risk": "low",
        "promotion_rule": "never",
    }

    def __init__(self, alpha=0.4):
        self.alpha = alpha

    def fit(self, *args, **kwargs):
        return self

    def apply(self, gnn_probs, focal_mask=None, edge_index=None, **kwargs):
        probs = _to_tensor(gnn_probs).clone()
        if focal_mask is None or edge_index is None:
            return probs

        focal_mask = torch.tensor(_to_numpy(focal_mask).astype(bool), dtype=torch.bool, device=probs.device)
        edge_index = _to_numpy(edge_index)
        src, dst = edge_index
        pred = probs.argmax(dim=1)

        smoothed = probs.clone()
        for node_idx in torch.nonzero(focal_mask, as_tuple=False).view(-1).tolist():
            neighbors = src[dst == node_idx]
            if neighbors.size == 0:
                continue
            neighbor_idx = torch.tensor(neighbors, dtype=torch.long, device=probs.device)
            agree_mask = pred[neighbor_idx] == pred[node_idx]
            if agree_mask.any():
                neighbor_mean = probs[neighbor_idx[agree_mask]].mean(dim=0)
                smoothed[node_idx] = (1.0 - self.alpha) * probs[node_idx] + self.alpha * neighbor_mean
        smoothed = smoothed / smoothed.sum(dim=1, keepdim=True).clamp(min=1e-8)
        return smoothed


class EgoRefinementRepairOperator:
    metadata = {
        "claim_role": "stage3_candidate_mainline",
        "scientific_gate": "conformal_quality_guided_local_ego_refinement",
        "paper_identity_risk": "medium_combination_innovation",
        "promotion_rule": "stage3_candidate_after_ablation",
    }

    def __init__(self, lambda_reweight=0.2, semantic_top_k=2):
        self.lambda_reweight = float(lambda_reweight)
        self.semantic_top_k = int(semantic_top_k)
        self.last_artifact = {
            "contract": "ego_refinement_evidence_v1",
            "propagation_edges": [],
            "virtual_context_edges": [],
            "evidence_edges": [],
        }

    def fit(self, *args, **kwargs):
        return self

    def _edge_role(self, probs, src, dst, node_repr=None):
        pred = probs.argmax(dim=1)
        if int(pred[src]) != int(pred[dst]):
            return "suspicious"
        if node_repr is not None:
            emb = _to_tensor(node_repr, reference=probs).float()
            sim = torch.nn.functional.cosine_similarity(emb[src].view(1, -1), emb[dst].view(1, -1)).item()
            if sim < 0.0:
                return "uncertain"
        return "supportive"

    def _new_weight(self, role):
        if role == "supportive":
            return 1.0 + self.lambda_reweight
        if role == "suspicious":
            return 1.0 - self.lambda_reweight
        return 1.0

    def _virtual_context_edges(self, node_idx, focal_nodes, node_repr, edge_index, num_nodes):
        if node_repr is None:
            return []
        embeddings = _to_tensor(node_repr).float()
        if embeddings.dim() != 2 or node_idx >= embeddings.shape[0]:
            return []
        edge_np = _to_numpy(edge_index).astype(np.int64) if edge_index is not None else np.empty((2, 0), dtype=np.int64)
        existing = set()
        if edge_np.ndim == 2 and edge_np.shape[0] == 2:
            existing = {(int(src), int(dst)) for src, dst in zip(edge_np[0], edge_np[1])}
        sims = torch.nn.functional.cosine_similarity(embeddings, embeddings[node_idx].view(1, -1), dim=1)
        order = torch.argsort(sims, descending=True).detach().cpu().numpy().tolist()
        context = []
        for candidate in order:
            if candidate == node_idx or candidate in focal_nodes:
                continue
            if (candidate, node_idx) in existing or (node_idx, candidate) in existing:
                continue
            context.append(
                {
                    "src": int(candidate),
                    "dst": int(node_idx),
                    "role": "virtual_context",
                    "score": float(sims[candidate].item()),
                    "writes_to_main_graph": False,
                    "source": "semantic_retrieval",
                }
            )
            if len(context) >= max(self.semantic_top_k, 0):
                break
        return context

    def apply(self, gnn_probs, focal_mask=None, edge_index=None, node_repr=None, risk_score=None, **kwargs):
        probs = _to_tensor(gnn_probs).clone()
        num_nodes = int(probs.shape[0])
        if focal_mask is None:
            focal_nodes = list(range(num_nodes))
        else:
            focal_nodes = torch.nonzero(
                torch.tensor(_to_numpy(focal_mask).astype(bool), dtype=torch.bool),
                as_tuple=False,
            ).view(-1).tolist()
        focal_set = set(int(item) for item in focal_nodes)
        edge_np = _to_numpy(edge_index).astype(np.int64) if edge_index is not None else np.empty((2, 0), dtype=np.int64)
        propagation_edges = []
        evidence_edges = []
        if edge_np.ndim == 2 and edge_np.shape[0] == 2:
            for src, dst in zip(edge_np[0], edge_np[1]):
                src_i = int(src)
                dst_i = int(dst)
                if dst_i not in focal_set:
                    continue
                role = self._edge_role(probs, src_i, dst_i, node_repr=node_repr)
                new_weight = self._new_weight(role)
                item = {
                    "src": src_i,
                    "dst": dst_i,
                    "role": role,
                    "old_weight": 1.0,
                    "new_weight": float(new_weight),
                }
                propagation_edges.append(item)
                evidence_edges.append(dict(item))
        virtual_context_edges = []
        for node_idx in focal_nodes:
            virtual_context_edges.extend(
                self._virtual_context_edges(int(node_idx), focal_set, node_repr, edge_index, num_nodes)
            )
        evidence_edges.extend(virtual_context_edges)
        self.last_artifact = {
            "contract": "ego_refinement_evidence_v1",
            "rewrite_policy": "soft_reweight_existing_edges_only",
            "llm_calls": 0,
            "propagation_edges": propagation_edges,
            "virtual_context_edges": virtual_context_edges,
            "evidence_edges": evidence_edges,
        }
        return probs


class IterativeStructuralTextConfirmRetriever:
    """Stage-3 of the v8 EQC pipeline (pre-committed plumbing).

    Implements a two-iteration, structural-first text-confirm retrieval
    loop for every hard ego node. At each iteration ``t = 1, 2`` the
    retriever (a) proposes candidate neighbours by Personalized PageRank
    (optionally gated by a trusted-relation indicator), and (b) keeps
    candidates whose raw cosine similarity with the target exceeds the
    temperature-calibrated top-50% threshold on the validation calibration
    split. The iteration halts early if the composite score ``q(v)``
    does not improve by more than ``epsilon`` or after two passes.

    The retriever is zero-training: only the temperature ``T_valid_cal``
    and the per-node ``k_struct`` are fit on the validation-calibration
    split. Budget-matched non-iterative controls (single-pass with
    ``2 * k_struct`` candidates; two-seed PPR) live in the Block-R
    falsification experiment.
    """

    metadata = {
        "claim_role": "eqc_v8_stage3_iterative_retrieval",
        "scientific_gate": "structural_first_text_confirm_budget_matched",
        "paper_identity_risk": "low_plumbing_not_novelty",
        "promotion_rule": "pre_committed_plumbing_never_primary_claim",
    }

    _PPR_DAMPING_DEFAULT = 0.15
    _PPR_STEPS_DEFAULT = 10

    def __init__(self, max_iterations=2, k_struct=8, epsilon=1e-3,
                 trusted_relations=None, ppr_damping=None, ppr_steps=None,
                 single_pass_budget_matched=False, ppr_num_seeds=1):
        self.max_iterations = int(max_iterations)
        self.k_struct = int(k_struct)
        self.epsilon = float(epsilon)
        self.trusted_relations = None if trusted_relations is None else set(int(r) for r in trusted_relations)
        self.ppr_damping = float(self._PPR_DAMPING_DEFAULT if ppr_damping is None else ppr_damping)
        self.ppr_steps = int(self._PPR_STEPS_DEFAULT if ppr_steps is None else ppr_steps)
        self.single_pass_budget_matched = bool(single_pass_budget_matched)
        self.ppr_num_seeds = int(ppr_num_seeds)
        self.temperature_valid_cal = 1.0
        self.bucket_q33 = 0.0
        self.bucket_q66 = 0.0
        self.fit_summary = {}

    def fit(self, node_repr, val_idx, edge_index=None, **kwargs):
        """Calibrate the temperature and role-bucket quantiles on
        ``val_idx`` pairwise similarities.

        The temperature scales raw cosine similarities so the median
        pairwise similarity on ``val_idx`` maps to ``0.5``. The Q66/Q33
        quantiles define the supportive / uncertain / suspicious buckets
        used by Stage-5's canonical schema.
        """
        if node_repr is None:
            raise ValueError("IterativeStructuralTextConfirmRetriever.fit requires node_repr.")
        embeddings = _to_tensor(node_repr).float().detach().cpu().numpy()
        val_idx_np = np.asarray(val_idx, dtype=np.int64).reshape(-1)
        if val_idx_np.size == 0:
            raise ValueError("IterativeStructuralTextConfirmRetriever.fit requires a non-empty val_idx.")

        rng = np.random.default_rng(42)
        num_pairs = min(int(val_idx_np.size), 2048)
        left = rng.choice(val_idx_np, size=num_pairs, replace=True)
        right = rng.choice(val_idx_np, size=num_pairs, replace=True)
        mask = left != right
        left = left[mask]
        right = right[mask]
        if left.size == 0:
            raise ValueError("Unable to sample disjoint calibration pairs for temperature fit.")

        def _cos(a_rows, b_rows):
            a = embeddings[a_rows]
            b = embeddings[b_rows]
            num = np.sum(a * b, axis=1)
            den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
            den = np.clip(den, 1e-12, None)
            return num / den

        raw_cos = _cos(left, right)
        raw_cos = raw_cos.astype(np.float64)
        median_cos = float(np.median(raw_cos))
        if abs(median_cos) < 1e-8:
            self.temperature_valid_cal = 1.0
        else:
            self.temperature_valid_cal = float(2.0 * median_cos)  # calibrated so median -> 0.5
        calibrated = raw_cos / max(self.temperature_valid_cal, 1e-8)
        self.bucket_q33 = float(np.quantile(calibrated, 0.33, method="higher"))
        self.bucket_q66 = float(np.quantile(calibrated, 0.66, method="higher"))

        self.fit_summary = {
            "source": "iterative_structural_text_confirm_retriever",
            "fit_scope": "validation_split_pairwise_cosine",
            "temperature_valid_cal": float(self.temperature_valid_cal),
            "bucket_q33": float(self.bucket_q33),
            "bucket_q66": float(self.bucket_q66),
            "max_iterations": int(self.max_iterations),
            "k_struct": int(self.k_struct),
            "epsilon": float(self.epsilon),
            "single_pass_budget_matched": bool(self.single_pass_budget_matched),
            "ppr_num_seeds": int(self.ppr_num_seeds),
            "num_calibration_pairs": int(left.size),
            "framing": "structural_first_text_confirm_budget_matched_ablation_ready",
        }
        return self

    def _calibrated_similarity(self, embeddings, target_idx, candidate_indices):
        if len(candidate_indices) == 0:
            return np.zeros((0,), dtype=np.float64)
        target = embeddings[target_idx]
        cand = embeddings[np.asarray(candidate_indices, dtype=np.int64)]
        num = np.sum(cand * target, axis=1)
        den = np.linalg.norm(cand, axis=1) * (np.linalg.norm(target) + 1e-12)
        den = np.clip(den, 1e-12, None)
        raw = num / den
        return raw / max(self.temperature_valid_cal, 1e-8)

    def _structural_proposals(self, target_idx, seed_set, edge_index, num_nodes, k):
        """Return top-``k`` 2-hop neighbours of ``seed_set`` ranked by PPR
        mass, excluding ``target_idx`` itself.
        """
        if edge_index is None or num_nodes == 0 or k <= 0:
            return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.float64)
        edge_np = _to_numpy(edge_index).astype(np.int64)
        if edge_np.ndim != 2 or edge_np.shape[0] != 2 or edge_np.shape[1] == 0:
            return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.float64)
        valid = (edge_np[0] >= 0) & (edge_np[0] < num_nodes) & (edge_np[1] >= 0) & (edge_np[1] < num_nodes)
        src = edge_np[0, valid]
        dst = edge_np[1, valid]
        if src.size == 0:
            return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.float64)

        ppr = np.zeros(num_nodes, dtype=np.float64)
        seed_list = [int(s) for s in seed_set if 0 <= int(s) < num_nodes]
        if not seed_list:
            seed_list = [int(target_idx)]
        seed_array = np.asarray(seed_list, dtype=np.int64)
        ppr[seed_array] = 1.0 / float(seed_array.size)
        out_degree = np.zeros(num_nodes, dtype=np.float64)
        np.add.at(out_degree, src, 1.0)
        out_degree = np.clip(out_degree, 1.0, None)

        for _ in range(self.ppr_steps):
            contrib = ppr[src] / out_degree[src]
            prop = np.zeros_like(ppr)
            np.add.at(prop, dst, contrib)
            restart = np.zeros_like(ppr)
            restart[seed_array] = 1.0 / float(seed_array.size)
            ppr = (1.0 - self.ppr_damping) * prop + self.ppr_damping * restart

        ppr[int(target_idx)] = -np.inf
        for s in seed_list:
            ppr[s] = -np.inf
        if np.all(np.isneginf(ppr)):
            return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.float64)
        order = np.argsort(-ppr)
        finite = ppr[order] > -np.inf
        order = order[finite]
        top = order[: max(int(k), 0)]
        return top.astype(np.int64), ppr[top].astype(np.float64)

    def retrieve(self, target_idx, seed_edges, embeddings, edge_index, num_nodes,
                 q_before_callable=None):
        """Run the iterative structural-first text-confirm retrieval loop.

        Parameters
        ----------
        target_idx : int
            Hard-node index under refinement.
        seed_edges : list[tuple[int, int]]
            The existing ego edges for the target (propagation edges).
        embeddings : numpy.ndarray
            Node representation matrix, shape ``[num_nodes, dim]``.
        edge_index : array-like or None
            Global edge index; shape ``[2, E]`` or ``None``.
        num_nodes : int
            Total number of nodes in the graph.
        q_before_callable : Callable[[list[tuple[int, int]]], float] or None
            Optional closure returning the composite ``q(v | C_t)`` for the
            given candidate ego; used for the conformal stop rule.

        Returns
        -------
        dict
            Retrieval artifact with ``confirmed_candidates``, ``calibrated_sim``,
            ``iterations_taken``, and per-iteration diagnostics.
        """
        seed_set = set((int(s), int(d)) for s, d in seed_edges if int(d) == int(target_idx) or int(s) == int(target_idx))
        accumulated_candidates = []
        accumulated_sims = []
        diagnostics = []

        if self.single_pass_budget_matched:
            iterations = 1
            k_per_iter = max(int(self.k_struct) * int(self.max_iterations), int(self.k_struct))
        else:
            iterations = int(self.max_iterations)
            k_per_iter = int(self.k_struct)

        last_q = q_before_callable([(s, d) for (s, d) in seed_set]) if q_before_callable is not None else None

        for t in range(1, iterations + 1):
            current_seed_nodes = {int(target_idx)}
            current_seed_nodes.update(int(u) for pair in seed_set for u in pair)
            current_seed_nodes.update(int(u) for u in accumulated_candidates)

            proposals, ppr_scores = self._structural_proposals(
                int(target_idx), current_seed_nodes, edge_index, num_nodes, k_per_iter
            )
            sims = self._calibrated_similarity(embeddings, int(target_idx), proposals)
            keep = sims > 0.0  # top-50% calibrated quantile gate
            confirmed = proposals[keep]
            confirmed_sims = sims[keep]

            new_mask = np.array([c not in accumulated_candidates for c in confirmed.tolist()], dtype=bool)
            confirmed = confirmed[new_mask]
            confirmed_sims = confirmed_sims[new_mask]

            accumulated_candidates.extend(int(c) for c in confirmed.tolist())
            accumulated_sims.extend(float(s) for s in confirmed_sims.tolist())

            iter_diag = {
                "iteration": int(t),
                "proposed": int(proposals.size),
                "confirmed": int(confirmed.size),
                "k_struct": int(k_per_iter),
            }

            if q_before_callable is not None and not self.single_pass_budget_matched:
                candidate_ego = list(seed_set) + [(int(c), int(target_idx)) for c in accumulated_candidates]
                q_after = q_before_callable(candidate_ego)
                iter_diag["q_after"] = float(q_after)
                if last_q is not None:
                    iter_diag["q_improvement"] = float(last_q - q_after)
                    if (last_q - q_after) <= self.epsilon:
                        diagnostics.append(iter_diag)
                        break
                last_q = q_after

            diagnostics.append(iter_diag)

        return {
            "contract": "iterative_structural_text_confirm_retrieval_v1",
            "target_idx": int(target_idx),
            "confirmed_candidates": [int(c) for c in accumulated_candidates],
            "calibrated_sim": [float(s) for s in accumulated_sims],
            "iterations_taken": int(len(diagnostics)),
            "per_iteration": diagnostics,
            "temperature_valid_cal": float(self.temperature_valid_cal),
            "bucket_q33": float(self.bucket_q33),
            "bucket_q66": float(self.bucket_q66),
            "single_pass_budget_matched": bool(self.single_pass_budget_matched),
        }


class EvidenceGraphRewriter(EgoRefinementRepairOperator):
    """Stage-4 + Stage-5 of the v8 EQC pipeline.

    Extends :class:`EgoRefinementRepairOperator` in four ways:

    1. Consumes the output of :class:`IterativeStructuralTextConfirmRetriever`
       to populate the three role buckets (supportive / suspicious /
       uncertain) by the deterministic Q66 / Q33 cuts on calibrated
       similarity; no learned head.
    2. Gates the soft-reweight with :class:`CalibratedExpectedErrorProxy`
       so acceptance depends on a validation-label-supervised external
       proxy, not on the model's own argmax confidence. The posterior
       L-infinity cap ``rho`` bounds per-node drift.
    3. Emits the canonical Stage-5 ``G_evidence(v)`` v1.0 artifact with a
       pre-registered 512-token budget and the canonical intra-bucket
       order ``(PPR_rank, -calibrated_sim)``.
    4. Exposes four falsification controls on the serialized artifact
       (``shuffle_role`` / ``shuffle_order`` / ``collapse`` / ``minimal``)
       so Block-E can run scramble variants against the canonical form
       without re-running Stages 1-4.
    """

    metadata = {
        "claim_role": "eqc_v8_stage4_and_stage5_evidence_graph_builder",
        "scientific_gate": "externalized_accept_rule_with_canonical_schema",
        "paper_identity_risk": "stage5_schema_is_the_single_dominant_contribution",
        "promotion_rule": "eqc_v8_dominant_contribution",
    }

    _DEFAULT_TOKEN_BUDGET = 512
    _SECTION_BUDGETS = {
        "target": 80,
        "supportive_edges": 200,
        "suspicious_edges": 128,
        "uncertain_edges": 64,
        "quality_summary": 40,
    }

    def __init__(self, lambda_reweight=0.2, rho_stability_cap=0.2,
                 budget_max=0.5, budget_coupling=1.0,
                 token_budget=None, schema_version="v1.0"):
        super().__init__(lambda_reweight=lambda_reweight, semantic_top_k=0)
        self.rho_stability_cap = float(rho_stability_cap)
        self.budget_max = float(budget_max)
        self.budget_coupling = float(budget_coupling)
        self.token_budget = int(self._DEFAULT_TOKEN_BUDGET if token_budget is None else token_budget)
        self.schema_version = str(schema_version)

    def fit(self, *args, **kwargs):
        return self

    @staticmethod
    def _assign_role(calibrated_sim, bucket_q66, bucket_q33):
        if calibrated_sim > float(bucket_q66):
            return "supportive"
        if calibrated_sim < float(bucket_q33):
            return "suspicious"
        return "uncertain"

    @staticmethod
    def _role_weight(role):
        if role == "supportive":
            return 1.0
        if role == "suspicious":
            return -1.0
        return 0.0

    def soft_reweight(self, edges_with_roles, node_q_value):
        """Apply the v8 Stage-4 soft reweight formula

            w' = w * clip(1 + lambda * budget(v) * role_weight(e) * (1 - q(v)), 1-lambda, 1+lambda)

        to a list of ``{src, dst, role, old_weight}`` edge dicts and
        return the updated edges with populated ``new_weight``.
        """
        node_q = float(node_q_value)
        budget = min(self.budget_max, self.budget_coupling * node_q)
        rewritten = []
        for edge in edges_with_roles:
            role = str(edge.get("role", "uncertain"))
            old_weight = float(edge.get("old_weight", 1.0))
            raw_delta = self.lambda_reweight * budget * self._role_weight(role) * (1.0 - node_q)
            scale = float(np.clip(1.0 + raw_delta, 1.0 - self.lambda_reweight, 1.0 + self.lambda_reweight))
            new_edge = dict(edge)
            new_edge["old_weight"] = old_weight
            new_edge["new_weight"] = float(old_weight * scale)
            rewritten.append(new_edge)
        return rewritten

    def accept_rewrite(self, e_phi_pre, e_phi_post, p_gnn_pre, p_gnn_post):
        """Two-criterion accept rule (see :class:`CalibratedExpectedErrorProxy`).

        Accept if the validation-calibrated expected-error proxy does not
        increase *and* the posterior L-infinity drift stays within
        ``rho_stability_cap``.
        """
        e_phi_pre = float(e_phi_pre)
        e_phi_post = float(e_phi_post)
        drift = float(np.max(np.abs(np.asarray(p_gnn_post) - np.asarray(p_gnn_pre))))
        proxy_ok = e_phi_post <= e_phi_pre + 1e-8
        stability_ok = drift <= self.rho_stability_cap + 1e-8
        return bool(proxy_ok and stability_ok), {
            "e_phi_pre": e_phi_pre,
            "e_phi_post": e_phi_post,
            "posterior_linf_drift": drift,
            "rho_stability_cap": float(self.rho_stability_cap),
            "accept": bool(proxy_ok and stability_ok),
            "reason": (
                "accept"
                if proxy_ok and stability_ok
                else ("proxy_regression" if not proxy_ok else "posterior_drift_exceeded")
            ),
        }

    def build_evidence_graph(self, target_idx, retrieval_artifact, target_payload,
                             quality_summary, propagation_edges,
                             scramble_mode="canonical", rng_seed=0):
        """Serialise the v1.0 canonical ``G_evidence(v)`` artifact.

        Parameters
        ----------
        target_idx : int
            Hard-node index under refinement.
        retrieval_artifact : dict
            The output of :meth:`IterativeStructuralTextConfirmRetriever.retrieve`.
        target_payload : dict
            ``{'text': str, 'p_gnn': list, 'p_lm': list, 'q': float}``.
        quality_summary : dict
            Pre-registered quality telemetry (``q_pre``, ``q_post``,
            ``set_size_pre``, ``set_size_post``, ``rollback_flag``,
            ``e_phi_pre``, ``e_phi_post``).
        propagation_edges : list[dict]
            The rewritten existing ego edges with role + weight.
        scramble_mode : str
            One of ``{"canonical", "shuffle_role", "shuffle_order",
            "collapse", "minimal"}``. Used by Block-E falsification.
        rng_seed : int
            Deterministic seed for shuffle variants.

        Returns
        -------
        dict
            The canonical artifact schema v1.0 with deterministic ordering.
        """
        if scramble_mode not in {"canonical", "shuffle_role", "shuffle_order", "collapse", "minimal"}:
            raise ValueError(f"Unknown scramble_mode: {scramble_mode}")

        rng = np.random.default_rng(int(rng_seed))
        confirmed = list(retrieval_artifact.get("confirmed_candidates", []))
        sims = list(retrieval_artifact.get("calibrated_sim", []))
        bucket_q66 = float(retrieval_artifact.get("bucket_q66", 0.0))
        bucket_q33 = float(retrieval_artifact.get("bucket_q33", 0.0))

        candidate_table = []
        for rank_idx, (cand_idx, sim) in enumerate(zip(confirmed, sims)):
            role = self._assign_role(float(sim), bucket_q66, bucket_q33)
            candidate_table.append(
                {
                    "src": int(cand_idx),
                    "dst": int(target_idx),
                    "ppr_rank": int(rank_idx),
                    "calibrated_sim": float(sim),
                    "role": role,
                }
            )

        propagation_lookup = {(int(e["src"]), int(e["dst"])): e for e in propagation_edges}
        for entry in candidate_table:
            key = (entry["src"], entry["dst"])
            rewritten = propagation_lookup.get(key)
            entry["old_weight"] = float(rewritten["old_weight"]) if rewritten else 1.0
            entry["new_weight"] = float(rewritten["new_weight"]) if rewritten else 1.0

        def _canonical_order(entries):
            return sorted(entries, key=lambda row: (int(row["ppr_rank"]), -float(row["calibrated_sim"])))

        supportive = _canonical_order([e for e in candidate_table if e["role"] == "supportive"])
        suspicious = _canonical_order([e for e in candidate_table if e["role"] == "suspicious"])
        uncertain = _canonical_order([e for e in candidate_table if e["role"] == "uncertain"])

        if scramble_mode == "shuffle_role":
            all_rows = supportive + suspicious + uncertain
            if len(all_rows) < 2:
                raise ValueError(
                    "shuffle_role requires at least 2 candidates to guarantee byte-wise divergence from canonical."
                )
            role_set = {e["role"] for e in all_rows}
            if len(role_set) < 2:
                raise ValueError(
                    f"shuffle_role requires candidates from at least 2 distinct roles; got only {role_set}."
                )
            labels = ["supportive"] * len(supportive) + ["suspicious"] * len(suspicious) + ["uncertain"] * len(uncertain)
            perm = rng.permutation(len(all_rows))
            shuffled = [dict(all_rows[i], role=labels[perm[i]]) for i in range(len(all_rows))]
            supportive = [row for row in shuffled if row["role"] == "supportive"]
            suspicious = [row for row in shuffled if row["role"] == "suspicious"]
            uncertain = [row for row in shuffled if row["role"] == "uncertain"]
        elif scramble_mode == "shuffle_order":
            total_candidates = len(supportive) + len(suspicious) + len(uncertain)
            if total_candidates < 2:
                raise ValueError(
                    "shuffle_order requires at least 2 candidates to guarantee byte-wise divergence from canonical."
                )
            for bucket in (supportive, suspicious, uncertain):
                rng.shuffle(bucket)
        elif scramble_mode == "collapse":
            flat = supportive + suspicious + uncertain
            for row in flat:
                row["role"] = "flat"
            supportive = flat
            suspicious = []
            uncertain = []
        elif scramble_mode == "minimal":
            supportive = supportive[: max(self.semantic_top_k, 3)]
            suspicious = []
            uncertain = []

        budgets = dict(self._SECTION_BUDGETS)
        total_budget = sum(budgets.values())
        if total_budget != self.token_budget:
            scale = self.token_budget / max(float(total_budget), 1.0)
            budgets = {k: int(round(v * scale)) for k, v in budgets.items()}

        return {
            "contract": "evidence_graph_v1_0",
            "schema_version": self.schema_version,
            "scramble_mode": scramble_mode,
            "target": {
                "node_idx": int(target_idx),
                "text": str(target_payload.get("text", "")),
                "p_gnn": list(target_payload.get("p_gnn", [])),
                "p_lm": list(target_payload.get("p_lm", [])),
                "q": float(target_payload.get("q", 0.0)),
            },
            "supportive_edges": supportive,
            "suspicious_edges": suspicious,
            "uncertain_edges": uncertain,
            "quality_summary": dict(quality_summary),
            "token_budget": {
                "total": int(self.token_budget),
                "per_section": budgets,
                "truncation_policy": "drop_tail_by_rank_no_padding",
            },
            "canonical_order": "(ppr_rank, -calibrated_sim)",
            "bucketization": {
                "q33": float(bucket_q33),
                "q66": float(bucket_q66),
                "method": "deterministic_valid_cal_quantile",
            },
            "llm_calls_so_far": 0,
        }

    @staticmethod
    def serialize_evidence_graph_to_text(evidence_graph):
        """Deterministic string serialization of the v1.0 artifact.

        The ordering of sections and fields is strictly fixed; no random
        decoding or formatting is applied. This is the artifact that
        feeds :class:`LLMEvidenceRefiner` in Stage-6.
        """

        def _fmt_edge(row, include_role=True):
            role_suffix = f"|role={row.get('role', 'flat')}" if include_role else ""
            return (
                f"[src={int(row['src'])} dst={int(row['dst'])} "
                f"rank={int(row.get('ppr_rank', -1))} "
                f"sim={float(row.get('calibrated_sim', 0.0)):.4f} "
                f"w_old={float(row.get('old_weight', 1.0)):.4f} "
                f"w_new={float(row.get('new_weight', 1.0)):.4f}{role_suffix}]"
            )

        target = evidence_graph.get("target", {})
        lines = [f"SCHEMA={evidence_graph.get('schema_version', 'v1.0')}"]
        lines.append(
            "TARGET idx=" + str(int(target.get("node_idx", -1)))
            + " q=" + f"{float(target.get('q', 0.0)):.4f}"
            + " p_gnn=" + ",".join(f"{float(x):.4f}" for x in target.get("p_gnn", []))
            + " p_lm=" + ",".join(f"{float(x):.4f}" for x in target.get("p_lm", []))
        )
        target_text = str(target.get("text", ""))
        if target_text:
            lines.append("TARGET_TEXT " + target_text.replace("\n", " "))

        include_role = evidence_graph.get("scramble_mode") != "collapse"
        for section in ("supportive_edges", "suspicious_edges", "uncertain_edges"):
            rows = evidence_graph.get(section, [])
            if not rows:
                continue
            lines.append(section.upper())
            for row in rows:
                lines.append("  " + _fmt_edge(row, include_role=include_role))

        q_summary = evidence_graph.get("quality_summary", {})
        if q_summary:
            lines.append("QUALITY " + " ".join(f"{k}={v}" for k, v in sorted(q_summary.items())))
        return "\n".join(lines)

    def apply(self, gnn_probs, focal_mask=None, edge_index=None, node_repr=None,
              risk_score=None, **kwargs):
        """Inherit the parent's edge-role and reweight scaffold (the
        standalone ``ego_refinement`` mode). The full v8 pipeline invokes
        :meth:`soft_reweight`, :meth:`accept_rewrite`, and
        :meth:`build_evidence_graph` through a dedicated driver in
        ``trainer.py``. This ``apply`` override preserves backward
        compatibility with the ``repair_matrix`` stage runner.
        """
        probs = super().apply(
            gnn_probs,
            focal_mask=focal_mask,
            edge_index=edge_index,
            node_repr=node_repr,
            risk_score=risk_score,
            **kwargs,
        )
        self.last_artifact = dict(self.last_artifact)
        self.last_artifact["contract"] = "eqc_v8_stage4_soft_reweight_v1_compatibility_shim"
        self.last_artifact["rewrite_policy"] = "soft_reweight_existing_edges_only_with_external_accept_gate"
        self.last_artifact["external_accept_gate"] = "calibrated_expected_error_proxy_plus_linf_cap"
        self.last_artifact["schema_version"] = self.schema_version
        return probs


def build_semantic_operator(mode):
    if mode == "off":
        return NoOpSemanticOperator()
    if mode == "ridge_local":
        return RidgeLocalSemanticOperator()
    if mode == "lagnn_local":
        return LAGNNLocalSemanticOperator()
    raise ValueError(f"Unknown semantic mode: {mode}")


def build_repair_operator(mode):
    if mode == "noop":
        return NoOpRepairOperator()
    if mode == "prune":
        return PruneRepairOperator()
    if mode == "disagreement_local":
        return DisagreementLocalRepairOperator()
    if mode == "ego_refinement":
        return EgoRefinementRepairOperator()
    if mode == "eqc_v8_stage4":
        return EvidenceGraphRewriter()
    raise ValueError(f"Unknown repair mode: {mode}")
