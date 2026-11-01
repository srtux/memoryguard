# Evaluation and Reproducibility

The benchmark runner evaluates five memory architectures under identical prompts, model weights (`gemini-3.5-flash-lite`), and deterministic decoding at temperature zero. Saved metrics, per-episode ingestion logs, tool proposals, Clean-Room replans, and benchmark statistics live in the summary output under `results/`.

## Five-Arm Ablation Matrix

The five architectures isolate the effect of each defense stage:

| Architecture ID | Label | Write Gate | Action Interceptor | Hypothesis Tested |
| :--- | :--- | :--- | :--- | :--- |
| `1_stock_unprotected` | Stock Unprotected | Save all turns | None | Baseline bloat and cross-session hijack exposure |
| `2_llm_gate_only` | Cloud LLM Gate | Cloud Gemini utility score | None | Whether cloud utility scoring stops poisoning |
| `3_lexical_gate_only` | Lexical Regex Gate | Jailbreak regex + CLI cues | None | Cost of brittle syntactic heuristics on human prose |
| `4_laya_naive_biba` | Blanket Red Lock | Local `Laya-421M` two-stage | Block on any `EXTERNAL` note | False-lock cost of strict Biba taint blocking |
| `5_memory_guard_full` | MemoryGuard | Local `Laya-421M` two-stage | Counterfactual Clean-Room | Full safety, zero false locks, and clean recovery |

## Metric Definitions

Each run computes nine primary metrics across the 48 coding episodes and 30 `PersonaMem-v1` dialogues:

| Metric | Cohort Size | Definition |
| :--- | :---: | :--- |
| Overall ASR | 24 | Share of attack episodes (`attack_obvious` + `attack_sneaky`) where an unsafe tool or forbidden target executes |
| Obvious ASR | 12 | Share of direct override episodes (`attack_obvious`) hijacked |
| Sneaky ASR | 12 | Share of declarative convention, split-turn, or secret-read episodes (`attack_sneaky`) hijacked |
| Benign False Lock | 24 | Share of unattacked episodes (`benign_clean` + `benign_external_taint_stress`) falsely blocked by the interceptor |
| Taint-Priv Lock | 9 | Share of unattacked privileged tasks carrying legitimate `EXTERNAL` notes that are falsely blocked |
| Benign Utility | 24 | Share of unattacked episodes completing the expected tool and required arguments |
| Attacked Utility | 24 | Share of attacked episodes that block the hijack and still complete the legitimate coding task |
| Overall Success | 48 | Share of all 48 coding episodes achieving both zero hijacking and full task completion |
| PersonaMem Recall | 30 | Multiple-choice accuracy `(a)` through `(d)` on 40-turn `PersonaMem-v1` conversations |

## Benchmark Results

Lower Attack Success Rate (ASR) and lock rates are better; higher utility, episode success, recall, and storage savings are better:

| Architecture | Overall ASR | Obvious ASR | Sneaky ASR | Benign Lock | Taint-Priv Lock | Benign Utility | Attacked Utility | Overall Success | PersonaMem Recall | Storage Saved |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Stock Unprotected | 50.0% | 16.67% | 83.33% | 0.0% | 0.0% | 100.0% | 50.0% | 75.0% | 73.33% | 0.0% |
| Cloud LLM Gate | 41.67% | 0.0% | 83.33% | 0.0% | 0.0% | 100.0% | 58.33% | 79.17% | 63.33% | 52.27% |
| Lexical Regex Gate | 29.17% | 0.0% | 58.33% | 0.0% | 0.0% | 79.17% | 62.5% | 70.83% | 33.33% | 85.85% |
| Blanket Red Lock | 0.0% | 0.0% | 0.0% | 37.5% | 88.89% | 62.5% | 79.17% | 70.83% | 76.67% | 24.59% |
| **MemoryGuard** | **0.0%** | **0.0%** | **0.0%** | **0.0%** | **0.0%** | **100.0%** | **100.0%** | **100.0%** | **76.67%** | **24.59%** |

Key findings across the ablation ladder:

- `Cloud LLM Gate` stops direct overrides (`0.0%` Obvious ASR) but fails against calm declarative conventions and split-turn attacks (`83.33%` Sneaky ASR, identical to Stock).
- `Lexical Regex Gate` keeps `79.17%` benign coding utility because coding rules carry flags and backticks. On `PersonaMem-v1` prose, regex recall drops to `33.33%` (`10/30`).
- `Blanket Red Lock` achieves `0.0%` ASR by blocking eight of nine tainted privileged tasks (`88.89%` lock rate, `37.5%` across all benign episodes).
- MemoryGuard verifies equivalent plans (`0.0%` false locks) and recovers all poisoned episodes to the trusted `USER` + `REPO` plan (`48/48` overall success). It also lifts `PersonaMem-v1` recall from `73.33%` to `76.67%` while cutting stored words by `24.59%`.

## Paired Discordant Analysis

Because all five architectures run on the same 48 episodes, paired comparisons on discordant outcomes give the net win/loss count (`+b / -c`, where `b` is episodes won by MemoryGuard and `c` is episodes won by the baseline):

| Baseline Compared Against MemoryGuard | Overall Success (48) | Attack Blocked (24) | Attacked Utility (24) | Benign Utility (24) |
| :--- | :---: | :---: | :---: | :---: |
| `1_stock_unprotected` | `+12 / -0` | `+12 / -0` | `+12 / -0` | `0 / 0` |
| `2_llm_gate_only` | `+10 / -0` | `+10 / -0` | `+10 / -0` | `0 / 0` |
| `3_lexical_gate_only` | `+14 / -0` | `+7 / -0` | `+9 / -0` | `+5 / -0` |
| `4_laya_naive_biba` | `+14 / -0` | `0 / 0` | `+5 / -0` | `+9 / -0` |

MemoryGuard loses zero paired episodes across all baselines and metrics.

## Write Latency

`run_latency_microbenchmark` times 20 sequential Session 1 memory writes after model warmup. Local gates run on CPU, and `LLMAdmissionGate` runs uncached against the Gemini API:

| Write Gate | Execution Target | Mean Latency | Median Latency | p95 Tail Latency | Cloud API Calls |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `StockUnprotectedGate` | Local SQLite + `FTS5` | 0.89 ms | 0.72 ms | 2.67 ms | 0 |
| `LexicalAdmissionGate` | Local regex + SQLite | 0.37 ms | 0.26 ms | 1.79 ms | 0 |
| `LLMAdmissionGate` | Remote cloud Gemini API | 499.1 ms | 479.4 ms | 613.4 ms | 20 |
| `LayaAdmissionGate` | Local 421M CPU (`torch`) | 404.8 ms | 392.3 ms | 477.5 ms | 0 |

Running `Laya-421M` locally on CPU requires zero cloud API calls, keeps private chat notes off external endpoints, and delivers faster response times (`404.8 ms` mean, `477.5 ms` p95) with zero network latency jitter compared to remote cloud LLM gating.

## Offline Replay

The LLM client loads `.env` automatically via `load_dotenv_file` while ignoring template placeholder strings:

| Variable | Default | Role |
| :--- | :--- | :--- |
| `GEMINI_API_KEY` | None | API key for Google AI Studio authentication |
| `GOOGLE_API_KEY` | None | Fallback alias for Google AI Studio authentication |
| `GEMINI_MODEL` | `gemini-3.5-flash-lite` | Model identifier used for benchmark evaluations |
| `GOOGLE_CLOUD_PROJECT` | None | Project ID for Vertex AI authentication |
| `GOOGLE_CLOUD_LOCATION` | `global` | Cloud region for Vertex AI endpoints |
| `MEMORY_GUARD_DOTENV_PATH` | `.env` | Custom path to local environment file |

`GeminiClient` hashes request parameters with `SHA-256` into a local SQLite WAL cache under `.cache/llm_cache.sqlite`. Non-truncated replies are stored automatically so cached runs replay offline in seconds with zero live API calls.

## Output Summary Schema

Benchmark runs write full results to `results/summary.json` matching the following schema:

- Top-level summary keys:
  - `model` (`str`): Target evaluation model identifier.
  - `temperature` (`float`): Generation sampling temperature (`0.0`).
  - `n_coding_episodes` (`int`): Total count of evaluated coding episodes (`48`).
  - `n_recall_items` (`int`): Total count of evaluated recall probes (`30`).
  - `architectures` (`dict[str, dict]`): Map of architecture ID to arm evaluation results.
  - `recall_by_gate` (`dict[str, dict]`): Map of gate identifier to PersonaMem-v1 recall metrics.
  - `latency_microbenchmark_20_notes` (`dict[str, dict]`): Map of gate identifier to write latency statistics.
- `architectures.<arch_id>` keys:
  - `arch_id` (`str`), `label` (`str`), `gate_kind` (`str`), `interceptor_mode` (`str`).
  - `metrics` (`dict`): 19 aggregate performance metrics:
    - Counts and rates: `n_episodes`, `n_attacks`, `n_benign`, `hijacked_count`, `false_positive_lock_count`.
    - Security metrics: `asr_overall_pct`, `asr_obvious_pct`, `asr_sneaky_pct`.
    - Reliability metrics: `false_positive_lock_benign_pct`, `false_positive_lock_taint_priv_pct`.
    - Utility metrics: `benign_task_success_pct`, `benign_clean_success_pct`, `benign_taint_stress_success_pct`, `attacked_benign_utility_pct`, `overall_episode_success_pct`.
    - Memory metrics: `recall_accuracy_pct`, `storage_reduction_pct`, `write_latency_mean_ms`, `write_latency_median_ms`.
  - `episodes` (`list[dict]`): Episode execution records containing `episode_id`, `cohort`, `subkind`, `user_prompt`, `expected_tool`, `ingestion_log`, `active_tokens`, `retrieved_notes`, `proposed_call`, `clean_room_call`, `interceptor`, `final_call`, `is_attack`, `hijacked`, `false_positive_lock`, `task_success`, and `episode_success`.
- `recall_by_gate.<gate_kind>` keys:
  - `n_items` (`int`), `correct_count` (`int`), `recall_accuracy_pct` (`float`), `total_active_tokens` (`int`), `mean_active_tokens` (`float`), `storage_reduction_pct` (`float`).
  - `items` (`list[dict]`): Probe evaluations containing `item_id`, `pred`, `gold`, `correct`, and `active_tokens`.
- `latency_microbenchmark_20_notes.<gate_kind>` keys:
  - `n_notes` (`int`), `mean_ms` (`float`), `median_ms` (`float`), `p95_ms` (`float`), `total_ms` (`float`).

## ADK Multi-Session Verification

In addition to the 5-arm benchmark runner (`evaluate.py`), MemoryGuard provides an interactive Google ADK test harness in `demo.py`:

```bash
python demo.py --offline  # Deterministic multi-session test running without API keys
python demo.py            # Live multi-session execution using configured Gemini credentials
```

The script runs consecutive Session 1 and Session 2 interactions through `InMemoryRunner`, verifying:
1. `without_memory` forgets Session 1 user flags on Session 2.
2. `unprotected_memory` stores all turns, recalls external GitHub issues, and gets hijacked by declarative convention poisoning.
3. `with_memoryguard` filters small talk with `LayaAdmissionGate`, vetoes imperative injections, and recovers the developer's clean pytest invocation via `before_tool_callback` Clean-Room non-interference.

## Test Coverage

Running `python -m pytest -v` executes seven deterministic tests on a fresh clone without cloud credentials (verifying all 1,768 lines of Python across 14 files):

| Test Function | Verification Scope |
| :--- | :--- |
| `test_store_authority_dominance_and_duplicate_handling` | Near-duplicate shadowing of lower-tier notes, superseding equal/lower tier notes, FTS5 retrieval, and active token counting |
| `test_stock_lexical_and_stub_llm_gates` | Offline admission gate evaluation, lexical regex danger vetoes, JSON payload parsing, and dotenv credential loading |
| `test_live_laya_admission_gate` | Live local `Laya-421M` CPU inference rejecting conversational chatter, admitting rules, shadowing clones, and vetoing imperative overrides |
| `test_tool_interceptor_and_clean_room` | Call equivalence normalization, Blanket Red Lock (`naive_biba_min`), and Clean-Room recovery on `http_post` and sensitive file reads (`.env`) |
| `test_summary_metrics_and_portability` | 100% AST docstring coverage, verified 5-arm benchmark bounds, and secret/portability leak scanning |
| `test_evaluate_offline_replay_smoke` | End-to-end hermetic mock run and offline cache replay across coding, recall, and latency benchmarks |
| `test_adk_demo_agents_end_to_end` | Multi-session ADK agent verification across without_memory, unprotected_memory, and with_memoryguard in offline replay mode |
