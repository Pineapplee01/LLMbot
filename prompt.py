"""Prompt builders for GLANCE-style caches and prompt-expert explanation runs.

This module centralizes prompt text so precompute can focus on evidence
assembly, fallback handling, sidecars, and encoder execution.
"""

from __future__ import annotations


def _compact_whitespace(text):
    if not isinstance(text, str):
        return ""
    return " ".join(text.replace("\n", " ").split()).strip()


def _format_bool(value):
    return "yes" if bool(value) else "no"


def _normalize_prompt_style(prompt_style):
    return str(prompt_style or "default").strip().lower()


def _is_botsay_style(prompt_style):
    return _normalize_prompt_style(prompt_style) == "botsay"


def _botsay_task_intro(view_name):
    mapping = {
        "tweet": (
            "The following task focuses on evaluating whether a Twitter user is a bot or human "
            "with the user's tweets."
        ),
        "graph": (
            "The following task focuses on evaluating whether a Twitter user is a bot or human "
            "with the help of the user's followers and followings."
        ),
        "profile": (
            "The following task focuses on evaluating whether a Twitter user is a bot or human "
            "with the help of the user's self-written description and metadata."
        ),
        "conflict": (
            "The following task focuses on evaluating whether a Twitter user is a bot or human "
            "with the help of the user's profile, tweets, followers, and followings."
        ),
    }
    return mapping[view_name]


def _botsay_adaptation_rules():
    return [
        "You should output the label first and explanation after.",
        "Use only the provided evidence for this view.",
        "Do not invent missing labels, metadata fields, or neighbor information.",
    ]


def _summary_generation_system(prompt_style="default"):
    if _is_botsay_style(prompt_style):
        return (
            "You are helping evaluate whether a Twitter user is a bot or a human. "
            "Use only the provided evidence. "
            "Output the label first and explanation after. "
            "Do not invent missing evidence."
        )
    return (
        "You are an analyst for social-media bot detection. "
        "Use only the provided evidence from the current view. "
        "Fill the requested schema with source-grounded cues. "
        "Do not give a final account-level label, score, or recommendation."
    )


def _evidence_card_generation_system():
    return (
        "You extract source-grounded evidence cards for social-media bot detection. "
        "Separate suspicious cues, organic cues, ambiguity, and benign explanations. "
        "Do not output a final account-level label, probability, confidence score, or recommendation."
    )


def _summary_prompt_rules():
    return [
        "Return only the requested fields in the given order.",
        "Use short bullet phrases under the evidence fields; if a field has no evidence, write '- None observed'.",
        "Assess only this view and do not output a final account-level label, score, or recommendation.",
        "Make uncertainty explicit when the evidence is sparse, mixed, or explainable by benign behavior.",
        "Set LOCAL_LEANING to exactly one of: bot_like | human_like | inconclusive.",
        "Write RATIONALE as 1-2 short evidence-grounded sentences about this view only.",
    ]


def _summary_output_fields():
    return [
        "OUTPUT_FIELDS:",
        "OBSERVED_SUSPICIOUS_CUES:",
        "OBSERVED_ORGANIC_OR_BENIGN_CUES:",
        "AMBIGUITY_OR_MISSING_EVIDENCE:",
        "LOCAL_LEANING: bot_like | human_like | inconclusive",
        "RATIONALE:",
    ]


def _evidence_card_prompt_rules():
    return [
        "Return only the requested evidence-card fields.",
        "Use short bullet phrases grounded in the provided profile, tweets, and directed neighbors.",
        "Do not output a final account-level label, probability, confidence score, recommendation, or model-correction instruction.",
        "Do not mention false positives, false negatives, base models, or whether a downstream model should change its prediction.",
        "Separate observed bot-like cues, human-like cues, relation ambiguity, possible benign explanations, evidence coverage, evidence consistency, and source support.",
        "If evidence is sparse, noisy, ambiguous, or explainable by benign social behavior, state that explicitly.",
    ]


def _expert_embedding_prompt(component_name, explanation_text, evidence_schema="summary"):
    label_map = {
        "ego": "profile-based",
        "graph_following": "following-side social-role",
        "graph_follower": "follower-side audience",
        "tweet": "tweet-behavior",
        "conflict": "cross-view conflict",
    }
    explanation_text = _compact_whitespace(explanation_text)
    component_label = label_map.get(component_name, component_name.replace("_", " "))
    schema_label = "structured evidence card" if str(evidence_schema) == "structured_evidence_card" else "evidence summary"
    query_tag = "EVIDENCE_CARD" if str(evidence_schema) == "structured_evidence_card" else "SUMMARY"
    return (
        f"Instruct: Encode the {component_label} {schema_label} for downstream correction reasoning.\n"
        f"Query: [{query_tag}] {explanation_text} </END>"
    )


def _botsay_label_explanation_fields():
    return [
        "Output format:",
        "Label: bot or human",
        "Explanation: concise evidence-grounded explanation",
    ]


def _tweet_explain_generation_prompt(
    profile_cue,
    tweet_behavior_summary,
    tweet_samples_block,
    evidence_schema="summary",
    prompt_style="default",
):
    structured = str(evidence_schema) == "structured_evidence_card"
    botsay = _is_botsay_style(prompt_style)
    user = "\n".join(
        [
            (
                "Extract a structured evidence card for this account's posting behavior from the provided tweet evidence."
                if structured
                else (
                    _botsay_task_intro("tweet")
                    if botsay
                    else "Explain how this account posts content for bot detection."
                )
            ),
            (
                "Focus on observed posting cues, ambiguity, benign explanations, and evidence coverage."
                if structured
                else (
                    " ".join(_botsay_adaptation_rules())
                    if botsay
                    else "Focus on repetition, promotion intensity, conversationality, topical consistency, retweet behavior, and automation cues. Use the schema to record suspicious cues, benign or human-like cues, uncertainty, and a local leaning."
                )
            ),
            *(_evidence_card_prompt_rules() if structured else ([] if botsay else _summary_prompt_rules())),
            "Target user profile:" if botsay and not structured else "PROFILE_CUE:",
            profile_cue,
            "Tweet behavior summary:" if botsay and not structured else "TWEET_BEHAVIOR_SUMMARY:",
            tweet_behavior_summary,
            tweet_samples_block.replace("TWEET_SAMPLES:", "Target user tweets:") if botsay and not structured else tweet_samples_block,
            *(
                [
                    *(_botsay_label_explanation_fields() if botsay else _summary_output_fields()),
                ]
                if not structured
                else [
                    "OUTPUT_FIELDS:",
                    "OBSERVED_BOT_LIKE_CUES:",
                    "OBSERVED_HUMAN_LIKE_CUES:",
                    "RELATION_AMBIGUITY:",
                    "POSSIBLE_BENIGN_EXPLANATION:",
                    "EVIDENCE_COVERAGE: rich | partial | sparse",
                    "EVIDENCE_CONSISTENCY: consistent | mixed | conflicting",
                    "SUPPORTED_BY_SOURCE: yes | partly | weak",
                ]
            ),
        ]
    )
    return {
        "system": _evidence_card_generation_system() if structured else _summary_generation_system(prompt_style=prompt_style),
        "user": user,
        "prompt_role": "tweet_explainer_botsay" if botsay and not structured else "tweet_explainer",
    }


def _graph_explain_generation_prompt(
    direction_name,
    ego_profile_cue,
    neighbor_cards,
    summary,
    evidence_schema="summary",
    prompt_style="default",
):
    structured = str(evidence_schema) == "structured_evidence_card"
    botsay = _is_botsay_style(prompt_style)
    if direction_name == "following":
        if structured:
            task_line = (
                "Extract a structured evidence card for who this account chooses to follow in this directed view. "
                "Treat high similarity among followed accounts as ambiguous unless supported by additional source-grounded cues."
            )
        else:
            task_line = (
                _botsay_task_intro("graph")
                if botsay
                else "Fill the schema for who this account chooses to follow and what that implies for bot detection. "
                "Focus on social role, affiliation, fandom, coordination, promotion, and whether the following pattern looks organic or strategic."
            )
        heading = "The target user follows these users:" if botsay and not structured else "FOLLOWING_NEIGHBORS:"
    else:
        if structured:
            task_line = (
                "Extract a structured evidence card for who follows this account in this directed view. "
                "Treat suspicious or similar followers as ambiguous because they may reflect amplification, purchased audience, or victimization."
            )
        else:
            task_line = (
                _botsay_task_intro("graph")
                if botsay
                else "Fill the schema for who follows this account and what that implies for bot detection. "
                "Focus on audience type, amplification pattern, credibility, coordination, and whether the follower side looks organic or suspicious."
            )
        heading = "These users follow the target user:" if botsay and not structured else "FOLLOWER_NEIGHBORS:"
    user = "\n".join(
        [
            task_line,
            (
                "Sparse neighborhoods are a limitation, not by themselves proof of automation."
                if not structured
                else ""
            ),
            *(
                _evidence_card_prompt_rules()
                if structured
                else (_botsay_adaptation_rules() if botsay else _summary_prompt_rules())
            ),
            "Target user:" if botsay and not structured else "EGO_PROFILE_CUE:",
            ego_profile_cue,
            "Relation summary:" if botsay and not structured else "DIRECTIONAL_SUMMARY:",
            f"count_{direction_name}: {int(summary['count'])}",
            f"has_{direction_name}: {_format_bool(summary['count'] > 0)}",
            f"selected_{direction_name}: {int(summary['selected_count'])}",
            f"reciprocal_ratio_{direction_name}_selected: {float(summary['reciprocal_ratio_selected']):.2f}",
            heading,
            *(neighbor_cards or ["- None"]),
            *(
                [
                    *(_botsay_label_explanation_fields() if botsay else _summary_output_fields()),
                ]
                if not structured
                else [
                    "OUTPUT_FIELDS:",
                    "OBSERVED_BOT_LIKE_CUES:",
                    "OBSERVED_HUMAN_LIKE_CUES:",
                    "RELATION_AMBIGUITY:",
                    "POSSIBLE_BENIGN_EXPLANATION:",
                    "EVIDENCE_COVERAGE: rich | partial | sparse",
                    "EVIDENCE_CONSISTENCY: consistent | mixed | conflicting",
                    "SUPPORTED_BY_SOURCE: yes | partly | weak",
                ]
            ),
        ]
    )
    user = "\n".join(line for line in user.splitlines() if line != "")
    return {
        "system": _evidence_card_generation_system() if structured else _summary_generation_system(prompt_style=prompt_style),
        "user": user,
        "prompt_role": f"{direction_name}_graph_explainer_botsay" if botsay and not structured else f"{direction_name}_graph_explainer",
    }


def _conflict_explain_generation_prompt(
    profile_cue_summary,
    tweet_explanation,
    graph_following_explanation,
    graph_follower_explanation,
    mismatch_hints,
    evidence_schema="summary",
    prompt_style="default",
):
    structured = str(evidence_schema) == "structured_evidence_card"
    botsay = _is_botsay_style(prompt_style)
    user = "\n".join(
        [
            (
                "Extract a structured cross-view evidence card across the profile, tweet, following, and follower views."
                if structured
                else (
                    _botsay_task_intro("conflict")
                    if botsay
                    else "Fill the schema for whether the profile, tweet behavior, following pattern, and follower audience support or contradict each other for bot detection."
                )
            ),
            (
                "Treat disagreement between views as evidence ambiguity unless the provided source text directly supports a stronger claim."
                if structured
                else (
                    " ".join(_botsay_adaptation_rules())
                    if botsay
                    else "Focus on cross-view agreement, contradiction, and uncertainty. Do not convert sparse evidence alone into automation evidence."
                )
            ),
            *(_evidence_card_prompt_rules() if structured else ([] if botsay else _summary_prompt_rules())),
            "Target user profile:" if botsay and not structured else "PROFILE_CUE_SUMMARY:",
            profile_cue_summary,
            "Tweet evidence:" if botsay and not structured else "TWEET_EXPERT_SUMMARY:",
            tweet_explanation or "None",
            "Following-side evidence:" if botsay and not structured else "FOLLOWING_EXPERT_SUMMARY:",
            graph_following_explanation or "None",
            "Follower-side evidence:" if botsay and not structured else "FOLLOWER_EXPERT_SUMMARY:",
            graph_follower_explanation or "None",
            "Cross-view hints:" if botsay and not structured else "MISMATCH_HINTS:",
            *(mismatch_hints or ["- None"]),
            *(
                [
                    *(_botsay_label_explanation_fields() if botsay else _summary_output_fields()),
                ]
                if not structured
                else [
                    "OUTPUT_FIELDS:",
                    "OBSERVED_BOT_LIKE_CUES:",
                    "OBSERVED_HUMAN_LIKE_CUES:",
                    "RELATION_AMBIGUITY:",
                    "POSSIBLE_BENIGN_EXPLANATION:",
                    "EVIDENCE_COVERAGE: rich | partial | sparse",
                    "EVIDENCE_CONSISTENCY: consistent | mixed | conflicting",
                    "SUPPORTED_BY_SOURCE: yes | partly | weak",
                ]
            ),
        ]
    )
    return {
        "system": _evidence_card_generation_system() if structured else _summary_generation_system(prompt_style=prompt_style),
        "user": user,
        "prompt_role": "conflict_explainer_botsay" if botsay and not structured else "conflict_explainer",
    }


def _ego_explain_generation_prompt(profile_card, evidence_schema="summary", prompt_style="default"):
    structured = str(evidence_schema) == "structured_evidence_card"
    botsay = _is_botsay_style(prompt_style)
    user = "\n".join(
        [
            (
                "Extract a structured evidence card from the profile card using identity presentation, reach/activity cues, and missing or unusual profile fields."
                if structured
                else (
                    _botsay_task_intro("profile")
                    if botsay
                    else "Fill the schema for what the profile presentation implies for bot detection."
                )
            ),
            (
                "Focus on observed profile cues, ambiguity, benign explanations, and evidence coverage."
                if structured
                else (
                    " ".join(_botsay_adaptation_rules())
                    if botsay
                    else "Focus on identity consistency, account setup completeness, reach/activity cues, and unusual or missing fields. Treat sparse profile evidence as a limitation, not proof of automation."
                )
            ),
            *(_evidence_card_prompt_rules() if structured else ([] if botsay else _summary_prompt_rules())),
            "Target user:" if botsay and not structured else "PROFILE_CARD:",
            profile_card,
            *(
                [
                    *(_botsay_label_explanation_fields() if botsay else _summary_output_fields()),
                ]
                if not structured
                else [
                    "OUTPUT_FIELDS:",
                    "OBSERVED_BOT_LIKE_CUES:",
                    "OBSERVED_HUMAN_LIKE_CUES:",
                    "RELATION_AMBIGUITY:",
                    "POSSIBLE_BENIGN_EXPLANATION:",
                    "EVIDENCE_COVERAGE: rich | partial | sparse",
                    "EVIDENCE_CONSISTENCY: consistent | mixed | conflicting",
                    "SUPPORTED_BY_SOURCE: yes | partly | weak",
                ]
            ),
        ]
    )
    return {
        "system": _evidence_card_generation_system() if structured else _summary_generation_system(prompt_style=prompt_style),
        "user": user,
        "prompt_role": "profile_explainer_botsay" if botsay and not structured else "profile_explainer",
    }


def _graph_prompt(direction_name, ego_profile_brief, neighbor_cards, total_count, reciprocal_count):
    heading = "FOLLOWING_NEIGHBORS" if direction_name == "following" else "FOLLOWER_NEIGHBORS"
    if direction_name == "following":
        instruct = (
            "Instruct: Encode who this account chooses to follow and what that implies about "
            "social role, coordination, fandom, promotion, or organic behavior."
        )
    else:
        instruct = (
            "Instruct: Encode who follows this account and what that implies about audience type, "
            "credibility, coordination, or suspicious amplification."
        )
    social_hints = [
        f"count_{direction_name}: {int(total_count)}",
        f"has_{direction_name}: {_format_bool(total_count > 0)}",
        f"reciprocal_{direction_name}_count: {int(reciprocal_count)}",
    ]
    query = "\n".join(
        [
            "Query:",
            "EGO_PROFILE_BRIEF:",
            ego_profile_brief,
            f"{heading}:",
            *(neighbor_cards or ["- None"]),
            "SOCIAL_HINTS:",
            *social_hints,
            "</END>",
        ]
    )
    return f"{instruct}\n{query}"


def _graph_prompt_partitioned(
    direction_name,
    ego_profile_brief,
    support_cards,
    contrast_cards,
    total_count,
    reciprocal_count,
    candidate_count,
    selected_count,
    mean_sim_support,
    mean_sim_contrast,
    reciprocal_ratio_selected,
):
    heading_prefix = "FOLLOWING" if direction_name == "following" else "FOLLOWER"
    if direction_name == "following":
        instruct = (
            "Instruct: Encode who this account chooses to follow, separating supportive neighbors from contrasting "
            "neighbors to capture social role, coordination, fandom, promotion, or organic behavior."
        )
    else:
        instruct = (
            "Instruct: Encode who follows this account, separating supportive neighbors from contrasting neighbors "
            "to capture audience type, credibility, coordination, or suspicious amplification."
        )
    social_hints = [
        f"count_{direction_name}: {int(total_count)}",
        f"candidate_count_{direction_name}: {int(candidate_count)}",
        f"selected_count_{direction_name}: {int(selected_count)}",
        f"has_{direction_name}: {_format_bool(total_count > 0)}",
        f"reciprocal_{direction_name}_count: {int(reciprocal_count)}",
        f"mean_sim_{direction_name}_support: {float(mean_sim_support):.4f}",
        f"mean_sim_{direction_name}_contrast: {float(mean_sim_contrast):.4f}",
        f"reciprocal_ratio_{direction_name}_selected: {float(reciprocal_ratio_selected):.4f}",
    ]
    query = "\n".join(
        [
            "Query:",
            "EGO_PROFILE_BRIEF:",
            ego_profile_brief,
            f"{heading_prefix}_SUPPORT:",
            *(support_cards or ["- None"]),
            f"{heading_prefix}_CONTRAST:",
            *(contrast_cards or ["- None"]),
            "SOCIAL_HINTS:",
            *social_hints,
            "</END>",
        ]
    )
    return f"{instruct}\n{query}"


def _tweet_prompt(profile_brief, tweet_behavior_summary, tweet_samples_block):
    instruct = "Instruct: Encode the posting behavior of the account for downstream bot detection."
    query = "\n".join(
        [
            "Query:",
            "PROFILE_BRIEF:",
            profile_brief,
            "TWEET_BEHAVIOR_SUMMARY:",
            tweet_behavior_summary,
            tweet_samples_block,
            "Focus on topical consistency, conversationality, promotion intensity, repetition, and automation cues.",
            "</END>",
        ]
    )
    return f"{instruct}\n{query}"


def _conflict_prompt(profile_card, tweet_card_summary, tweet_samples_block, following_cards, follower_cards, mismatch_hints):
    instruct = "Instruct: Encode cross-view consistency and inconsistency cues for bot detection."
    query = "\n".join(
        [
            "Query:",
            "PROFILE_CARD:",
            profile_card,
            "TWEET_CARD:",
            tweet_card_summary,
            tweet_samples_block,
            "GRAPH_CARD_FOLLOWING:",
            *(following_cards or ["- None"]),
            "GRAPH_CARD_FOLLOWER:",
            *(follower_cards or ["- None"]),
            "MISMATCH_HINTS:",
            *(mismatch_hints or ["- None"]),
            "Focus on whether the profile, posting behavior, and social neighborhood support or contradict each other.",
            "</END>",
        ]
    )
    return f"{instruct}\n{query}"


def _conflict_prompt_partitioned(
    profile_card,
    tweet_card_summary,
    tweet_samples_block,
    following_support_cards,
    following_contrast_cards,
    follower_support_cards,
    follower_contrast_cards,
    mismatch_hints,
):
    instruct = "Instruct: Encode cross-view consistency and inconsistency cues for bot detection."
    query = "\n".join(
        [
            "Query:",
            "PROFILE_CARD:",
            profile_card,
            "TWEET_CARD:",
            tweet_card_summary,
            tweet_samples_block,
            "FOLLOWING_SUPPORT:",
            *(following_support_cards or ["- None"]),
            "FOLLOWING_CONTRAST:",
            *(following_contrast_cards or ["- None"]),
            "FOLLOWER_SUPPORT:",
            *(follower_support_cards or ["- None"]),
            "FOLLOWER_CONTRAST:",
            *(follower_contrast_cards or ["- None"]),
            "MISMATCH_HINTS:",
            *(mismatch_hints or ["- None"]),
            "Focus on whether the profile, posting behavior, and relation-aware support/contrast neighborhoods support or contradict each other.",
            "</END>",
        ]
    )
    return f"{instruct}\n{query}"
