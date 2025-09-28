ICL Routing & Knobs (drop‑in)

Purpose
- Add adaptive self‑consistency (SC) with early stopping, lightweight CoT reasoning, and decoding controls without changing your learners or tasks.
- Additional reasoning modes wired: `cot`, `ltm` (Least‑to‑Most), `tot` (Tree‑of‑Thought, cheap controller), `sot` (Skeleton‑of‑Thought), `selfdisco` (Self‑Discover‑style plan), `got` (Graph‑of‑Thought, internal DAG). All run internally and still return strict JSON.
 - Program‑of‑Thought (PoT): add tool `pyexec` for safe Python execution (arithmetic/table calc). Use with `--use-tools --tools pyexec --reasoning pot`. The model first emits a tiny program (`result = ...`), which is validated via an AST allowlist and executed in a sandbox. The computed value is mapped to an option when possible.

New CLI Flags
- `--adaptive-sc` enables early‑stopping SC. Stops when quorum is reached.
- `--sc-quorum K` sets the quorum; default is simple majority of `--self-consistency`.
- `--reasoning {none,cot}` adds a concise internal CoT hint while keeping JSON‑only outputs.
- `--temperature` first‑pass T (default 0.3).
- `--temperature-sc` SC sampling T (default 0.7).
- `--top-p` nucleus sampling (default 1.0 = off). `--min-p` optional.
- Few‑shot: `--shots path.jsonl --shots-k 4 --shots-selector knn|random|as-is`. Exemplar record fields supported: `stem`, `options` (array), `answer_index` (int), optional `rationale`. Examples are passed as structured JSON to the model.
- Ordering: `--shots-order {similar,easy-hard,as-is}`. If exemplars have `difficulty` or `level` in {easy,medium,hard}, `easy-hard` sorts accordingly.
- Embedding KNN: `--shots-embed-backend {lexical,st,openai}` with optional MMR diversification `--shots-diverse --shots-mmr 0.5`. Caches embeddings under `.cache/emb/`.
- Reranking (optional): `--shots-reranker ce --shots-reranker-model BAAI/bge-reranker-base` reorders KNN picks using a cross‑encoder when available.
- APE header search: `scripts/ape_optimize.py --steps 20 [--mock]` writes the best header to `docs/ape/header.json`.
- Active Prompt: `scripts/active_prompt.py --log runs/<file>.jsonl --topk 8` prints items to label next.
- SC policy: `--sc-policy {fixed,adaptive}` with `--sc-k-easy/--sc-k-medium/--sc-k-hard`.
- Best-of-N: `--best-of N --rerank {confidence,evidence,judge}` to generate extra candidates and pick the best.
- Reflexion: `--reflexion` to add one critique+revise candidate into the pool.
- Uncertainty gating: `--uncertainty-gate --conf-threshold 0.45 --entropy-threshold 0.90 --max-k-escalated 12 [--escalate-reasoning]` escalates sampling (and optionally switches to ToT) when average confidence is low or vote entropy is high.
- Grammar: `--grammar {none,json,schema}`. `schema` passes an OpenAI‑style `json_schema` to enforce MCQ answer shape when supported.
- Compression: `--compress-examples --compress-ratio 3.0` applies a lightweight LLMLingua‑style compression to few‑shot examples.
  - Coming soon: `--compress-engine llmlingua` (optional dependency) for higher‑quality compression.

Examples
- Cheap pass only (baseline):
  `.venv/bin/python -m sim.cli --steps 5 --closed-book --self-consistency 1 --temperature 0.3`

- CoT + adaptive SC with early stop:
  `.venv/bin/python -m sim.cli --steps 10 --closed-book --reasoning cot --self-consistency 7 --adaptive-sc --temperature 0.3 --temperature-sc 0.7 --top-p 0.9`

- With Fact‑Cards and citation credit:
  `.venv/bin/python -m sim.cli --steps 10 --closed-book --fact-cards --require-citations --reasoning cot --self-consistency 5 --adaptive-sc --temperature 0.3 --temperature-sc 0.7`

- Few‑shot CoT (KNN‑picked 4 shots):
  `.venv/bin/python -m sim.cli --steps 10 --closed-book --shots data/examples.jsonl --shots-k 4 --shots-selector knn --reasoning cot --self-consistency 5 --adaptive-sc`

- Difficulty‑adaptive SC + rerank by confidence:
  `.venv/bin/python -m sim.cli --steps 10 --closed-book --sc-policy adaptive --sc-k-easy 3 --sc-k-medium 5 --sc-k-hard 7 --best-of 6 --rerank confidence`

How It Works
- First pass uses `--temperature` for a quick answer; if `--adaptive-sc` and quorum not met, additional samples are drawn at `--temperature-sc` until quorum or N votes.
- When `--reasoning cot` is set, the model is asked to think step‑by‑step internally but still return strict JSON.
- Decoding options are forwarded to provider SDKs; unsupported params are ignored by the engine.

Notes
- Defaults preserve prior behavior when `--adaptive-sc` is omitted.
- These knobs apply to MCQ answering. Other task types remain unchanged.
- Program‑of‑Thought (safe Python) for arithmetic MCQ:
  `.venv/bin/python -m sim.cli --steps 10 --closed-book --use-tools --tools pyexec --reasoning pot --self-consistency 3`

- Uncertainty gating with ToT escalation:
  `.venv/bin/python -m sim.cli --steps 10 --closed-book --self-consistency 3 --adaptive-sc --uncertainty-gate --escalate-reasoning --entropy-threshold 0.85`
