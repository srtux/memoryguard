# Evaluation and Results

The benchmark evaluates five approaches under identical conditions using `gemini-3.5-flash-lite` with deterministic generation at temperature zero. All test runs are reproducible using cached responses in the repository.

## The Five Compared Approaches

| Approach ID | Name | Write Gate | Command Check | Question Being Tested |
| :--- | :--- | :--- | :--- | :--- |
| `1_save_everything` | Save Everything | `save_all` | `none` | Baseline risk when keeping all turns |
| `2_keyword_rules_only` | Keyword Rules Only | `keyword_rules` | `none` | Whether simple keyword filters suffice |
| `3_cloud_ai_filter` | Cloud AI Filter Only | `cloud_ai` | `none` | Whether a remote cloud model stops attacks |
| `4_block_all_web_notes` | Block All Web Notes | `local_cpu` | `block_web_notes` | The penalty of refusing actions on web notes |
| `5_memoryguard` | MemoryGuard (Ours) | `local_cpu` | `clean_room` | Full protection with clean command recovery |

## Plain-English Metric Definitions

- **Attacks That Fooled Agent (`attacks_fooled_pct`)**: The percentage of attack scenarios where the agent was tricked into running a bad command or leaking data. Lower is better.
  - *Obvious Attacks (`obvious_attacks_fooled_pct`)*: Direct `"ignore instructions"` overrides.
  - *Sneaky Attacks (`sneaky_attacks_fooled_pct`)*: Calm conventions and split-turn attacks disguised as team rules.
- **Safe Tasks Falsely Blocked (`safe_tasks_blocked_pct`)**: The percentage of normal coding tasks where the agent refused to run a legitimate command. Lower is better.
  - *Safe Tasks with Web Notes (`safe_tasks_with_web_notes_blocked_pct`)*: Legitimate tasks where an external reference note was retrieved.
- **Tasks Completed Safely (`tasks_completed_safely_pct`)**: The percentage of scenarios where the agent completed the requested coding work without being hijacked. Higher is better.
- **Chat Facts Recalled (`chat_facts_recalled_pct`)**: Accuracy on 30 multiple-choice questions testing personal facts in `PersonaMem-v1`. Higher is better.
- **Stored Words Saved (`words_saved_pct`)**: Reduction in total words saved to disk relative to saving everything.
- **Write Speed (`write_time_mean_ms`)**: Time in milliseconds to evaluate a candidate note at write time.

## Main Scorecard

The headline results across the 48-task coding benchmark:

| Approach | Attacks That Fooled Agent | Safe Tasks Falsely Blocked | Tasks Completed Safely | Why It Fails or Wins |
| :--- | :---: | :---: | :---: | :--- |
| Save Everything | 50.0% (12/24) | 0.0% (0/24) | 75.0% (36/48) | Trusts fake rules from web issues |
| Keyword Rules Only | 29.2% (7/24) | 0.0% (0/24) | 70.8% (34/48) | Forgets normal English preferences |
| Cloud AI Filter Only | 29.2% (7/24) | 0.0% (0/24) | 85.4% (41/48) | Stops obvious attacks, fooled by sneaky ones |
| Block All Web Notes | 0.0% (0/24) | 37.5% (9/24) | 70.8% (34/48) | Freezes whenever a web note is present |
| **MemoryGuard (Ours)** | **0.0% (0/24)** | **0.0% (0/24)** | **100.0% (48/48)** | **Drops tampered command, runs your clean one** |

## In-Depth Analysis

### Obvious vs Sneaky Attacks

| Approach | Obvious Attacks Fooled | Sneaky Attacks Fooled |
| :--- | :---: | :---: |
| Save Everything | 16.7% (2/12) | 83.3% (10/12) |
| Keyword Rules Only | 0.0% (0/12) | 58.3% (7/12) |
| Cloud AI Filter Only | 0.0% (0/12) | 58.3% (7/12) |
| Block All Web Notes | 0.0% (0/12) | 0.0% (0/12) |
| **MemoryGuard (Ours)** | **0.0% (0/12)** | **0.0% (0/12)** |

When given the exact same 4 rubric questions, weights, and `0.50` / `0.40` thresholds, both `LocalCPUGate` (`Laya-421M`) and `CloudAIGate` (`gemini-3.5-flash-lite`) block 100% (`12/12`) of obvious direct overrides at write time. Against sneaky fake conventions, however, a write-time filter alone (`Cloud AI Filter Only`) still fails 58.3% (`7/12`) of the time because calm fake rules look like well-written documentation.

### The Cost of Blocking All Web Notes

`Block All Web Notes` stops attacks by refusing to run commands when an external note is retrieved.
- On safe tasks where an external note was present (`safe_with_web_notes`), it blocked **88.9% (8/9)** of legitimate commands.
- MemoryGuard ran a Clean-Room double check on those same tasks and had a **0.0% (0/9)** false block rate.

### Head-to-Head Paired Matchups

Because all five approaches ran on the exact same 48 scenarios, we compare each baseline against MemoryGuard scenario-by-scenario (wins vs losses):

| Opponent Baseline | Scenarios Won by MemoryGuard | Scenarios Won by Opponent | Net Advantage |
| :--- | :---: | :---: | :---: |
| vs Save Everything | 12 | 0 | +12 / -0 |
| vs Keyword Rules Only | 14 | 0 | +14 / -0 |
| vs Cloud AI Filter Only | 7 | 0 | +7 / -0 |
| vs Block All Web Notes | 14 | 0 | +14 / -0 |

MemoryGuard did not lose a single scenario to any baseline.

### Everyday Chat Recall and Word Reduction

On the 30-chat `PersonaMem-v1` evaluation:

| Write Gate | Chat Facts Recalled | Stored Words Saved | Impact on Conversation |
| :--- | :---: | :---: | :--- |
| `save_all` | 73.3% (22/30) | 0.0% | Retains all small talk |
| `keyword_rules` | 33.3% (10/30) | 85.9% | Drops normal English facts |
| `cloud_ai` | 66.7% (20/30) | 50.1% | Prunes too aggressively |
| `local_cpu` (MemoryGuard) | **76.7% (23/30)** | **24.6%** | **Remembers facts, filters chatter** |

The local CPU gate (`LocalCPUGate`) preserves more facts than both saving everything (`76.7%` vs `73.3%`) and the cloud LLM gate (`76.7%` vs `66.7%`) under the exact same rubric and `0.40` threshold.

### Write Speed Comparison

A benchmark timing 20 note writes after warmup:

| Write Gate | Where It Runs | Mean Time | Median Time | Cloud API Calls |
| :--- | :--- | :---: | :---: | :---: |
| `save_all` | Local SQLite database | 0.88 ms | 0.69 ms | 0 |
| `keyword_rules` | Local regex match | 0.37 ms | 0.25 ms | 0 |
| `cloud_ai` | Remote cloud endpoint | 565.1 ms | 493.7 ms | 20 |
| `local_cpu` | Local PyTorch on CPU | 383.9 ms | 377.7 ms | 0 |

`LocalCPUGate` runs locally on CPU in ~0.38s per note with zero cloud API calls and zero network tail spikes.

## Offline Replay and Prompt Cache

The client checks `.env` for credentials. All prompt requests and replies are hashed with SHA-256 and stored in an SQLite cache at `.cache/llm_cache.sqlite`. When you run the benchmark, requests replay instantly from disk without contacting external servers.

## Results JSON Format

Benchmark output in `results/benchmark_results.json` uses the following structure:

- Top-level keys:
  - `model`: Model evaluated (`gemini-3.5-flash-lite`).
  - `temperature`: Sampling temperature (`0.0`).
  - `n_coding_episodes`: Total coding tasks (`48`).
  - `n_recall_items`: Total chat recall questions (`30`).
  - `approaches`: Detailed results for each of the five approaches.
  - `chat_recall_by_gate`: Results keyed by gate (`save_all`, `keyword_rules`, `cloud_ai`, `local_cpu`).
  - `write_speed_20_notes`: Write speed stats keyed by gate.
- Metrics recorded per approach:
  - `n_episodes` (48), `n_attacks` (24), `n_safe_tasks` (24).
  - `attacks_fooled_count`, `attacks_fooled_pct`.
  - `obvious_attacks_fooled_pct`, `sneaky_attacks_fooled_pct`.
  - `safe_tasks_blocked_count`, `safe_tasks_blocked_pct`.
  - `safe_tasks_with_web_notes_blocked_pct`.
  - `tasks_completed_safely_count`, `tasks_completed_safely_pct`.
  - `safe_tasks_completed_pct`, `attacked_tasks_completed_pct`.
  - `chat_facts_recalled_pct`, `words_saved_pct`.
  - `write_time_mean_ms`, `write_time_median_ms`.

## Test Suite Checks

Running `python -m pytest -v` executes seven automated checks:

| Test Function | What It Confirms |
| :--- | :--- |
| `test_store_source_trust_and_duplicate_handling` | Source tiers work and web notes cannot overwrite user rules |
| `test_baseline_and_cloud_ai_gates` | Baseline admission gates function as intended |
| `test_live_local_cpu_admission_gate` | Local CPU gate skips chatter (`skipped_low_utility`) and blocks attacks (`blocked_attack`) |
| `test_tool_interceptor_and_clean_room` | Interceptor verifies clean plans and recovers tampered commands |
| `test_summary_metrics_and_portability` | Verified benchmark results and absence of hardcoded paths |
| `test_evaluate_offline_replay_smoke` | End-to-end replay passes without internet access |
| `test_adk_demo_agents_end_to_end` | Multi-session ADK demo runs cleanly in offline mode |
