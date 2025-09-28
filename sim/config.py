from __future__ import annotations

"""
Configuration types for the ICL simulator.

This module hosts small, tidy dataclasses that define all runtime knobs (Dials)
and the per-run configuration (RunConfig). Moving them here keeps
`sim.orchestrator` focused on control flow while preserving the public import
surface used by tests and callers (sim.orchestrator re-exports these).
"""

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class Dials:
    # Core behavior
    closed_book: bool = True
    anonymize: bool = True
    rich: bool = False  # rationales, etc.
    verify: bool = False
    reflection_every: int = 0  # not implemented yet
    context_position: str = "pre"  # pre|post|none
    self_consistency_n: int = 1  # >1 to enable majority-vote for MCQ (answering)
    adaptive_sc: bool = False     # enable early-stopping self-consistency
    sc_quorum: int = 0            # quorum to stop SC (0=auto majority of self_consistency_n)
    reasoning: str = "none"       # none|cot (concise CoT header)

    # Decoding params
    temp_answer: float = 0.3      # T for first-pass answers
    temp_sc: float = 0.7          # T for self-consistency sampling
    top_p: float = 1.0
    min_p: float | None = None
    grammar: str = "json"        # none|json|schema
    sc_extract_n: int = 0        # >0 overrides extraction SC for Fact-Cards; 0 means use self_consistency_n

    # Evidence & retrieval controls
    q_min: int = 1               # min required option-linked quotes per option
    max_learn_boosts: int = 0    # max LEARN escalation rounds when gates fail
    mmr_lambda: float = 0.4      # diversity for option-conditioned retrieval
    span_window: int = 240       # token-span window around hits
    citation_mode: str = "off"   # off|lenient|strict (strict enforces hard gates)

    # Evidence quality
    min_sources_chosen: int = 2
    dedup_sim: float = 0.88

    # Difficulty-adaptive SC
    sc_policy: str = "fixed"      # fixed|adaptive
    sc_k_easy: int = 3
    sc_k_medium: int = 5
    sc_k_hard: int = 7

    # Reranking / Best-of-N
    best_of_n: int = 0            # 0 = disabled; when >0, collect N candidates and rerank
    rerank: str = "none"          # none|confidence|evidence|judge

    # Reflexion self-critique (one pass)
    reflexion: bool = False

    # Compression
    compress_examples: bool = False
    compress_ratio: float = 3.0

    # Controllers
    controller: str = "basic"      # basic|ltm|tot
    controller_budget: int = 6
    tot_width: int = 2
    tot_depth: int = 2
    tot_judge: str = "self"

    # Uncertainty gating
    uncertainty_gate: bool = False
    conf_threshold: float = 0.45
    entropy_threshold: float = 0.90
    max_k_escalated: int = 12
    escalate_reasoning: bool = False

    # Instruction header (APE) for student prompts
    instruction_header: str | None = None
    accumulate_notes: bool = False  # accumulate correct info into notes across steps
    rare_emphasis: bool = False  # placeholder for rare-example emphasis
    use_tools: bool = False
    tools: List[str] = field(default_factory=lambda: ["retriever"])  # default toolset
    use_fact_cards: bool = False  # two-pass LEARN/USE with persistent cards
    fact_cards_budget: int = 10  # max cards persisted
    require_citations: bool = False  # require citations for credit
    freeze_cards: bool = False  # when using Fact-Cards, use provided cards as-is (no LEARN updates)

    # Abstention + calibration
    idk_enabled: bool = False
    target_confidence: float = 0.75

    # Card quality & evidence-weighted selection
    min_cqs: float = 0.0           # 0 disables filtering; 0.55 recommended when enabled
    min_card_len: int = 40
    max_card_len: int = 300
    per_option_top_k: int = 3
    evidence_weighted_selection: bool = False


@dataclass
class RunConfig:
    # What to generate/assess
    skill_id: Optional[str] = None
    task: str = "mcq"  # mcq|saq|code|proof|table_qa
    num_steps: int = 10
    num_options: int = 5
    difficulty: str = "medium"
    dials: Dials = field(default_factory=Dials)
    domain: str = "general"

    # Few-shot exemplars
    shots_path: Optional[str] = None
    shots_k: int = 0
    shots_selector: str = "knn"  # knn|random|as-is
    shots_order: str = "similar"  # similar|easy-hard|as-is
    shots_embed_backend: str = "lexical"  # lexical|st|openai
    shots_diverse: bool = False
    shots_mmr: float = 0.5

    # Reranker (optional)
    shots_reranker: str = "none"  # none|ce
    shots_reranker_model: str = "BAAI/bge-reranker-base"

    # Alias-swap controls
    alias_family_id: Optional[str] = None
    coverage_tau: float = 0.4

    # Bayesian guardrails (optional)
    # When provided, the orchestrator can enforce token bands and talk tuning.
    guardrails_path: Optional[str] = None  # path to guardrails.json (session_bayes_report)
    talk_slopes_path: Optional[str] = None  # path to talk_slopes_by_domain_*.csv
    # Session turns limit (fallback to guardrails mean_steps if unset)
    turns_limit: Optional[int] = None
    # Auto-nudge tokens into band when outside; if False, just log alerts
    tokens_autonudge: bool = False
    # Fraction of the band range considered the "trough vicinity" around opt for alerts (0..1)
    trough_margin: float = 0.2
    # Threshold (probability slope>0) to pick rich vs lean talk by domain
    talk_ppos_threshold: float = 0.7
