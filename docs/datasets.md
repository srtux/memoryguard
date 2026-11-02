# Benchmark Datasets

MemoryGuard is tested across two complementary benchmark suites under `benchmarks/`:
1. A 48-episode coding suite covering real software development workflows.
2. A 30-dialogue conversational recall suite from the peer-reviewed `PersonaMem-v1` benchmark.

## Why Both Datasets Are Paired

Evaluating an agent on only one kind of data produces misleading results:

- **If you only test coding**: Simple keyword filters that look for flags like `--tb=short` appear effective. On coding tasks alone, `KeywordRuleGate` finishes 70.8% of tasks safely. But on normal English conversation, it forgets two-thirds of personal facts because they lack code syntax.
- **If you only test chat recall**: You never test terminal commands, file edits, web notes, or malicious prompt injections.

Pairing both benchmarks proves that a defense protects shell commands without turning the assistant into an amnesiac during normal conversation.

| Property | Coding Suite | PersonaMem Recall Slice |
| :--- | :--- | :--- |
| File path | `benchmarks/coding_suite.json` | `benchmarks/personamem_slice.json` |
| Primary test | Command safety and tool integrity | Fact recall and small-talk filtering |
| Size | 48 two-session coding scenarios | 30 multi-turn chats (40 turns each) |
| Source | Controlled software scenarios | UPenn `PersonaMem-v1` (COLM 2025) |
| Trust tiers | `USER`, `INTERNAL`, and `EXTERNAL` | `USER` only |
| Coding tasks | 11 software engineering domains | Everyday personal preferences |
| Grading | Exact tool and argument matching | Deterministic 4-choice multiple choice |

Both suites are static JSON files included directly in the repository.

## The Coding Suite

The coding suite contains 48 scenarios across 11 software domains: `alembic_schema`, `billing_service`, `cli_release`, `csv_parser`, `docker_build`, `fastapi_auth`, `mkdocs_build`, `pytest_runner`, `redis_cache`, `sqlite_migration`, and `webhook_client`.

### Four Balanced Cohorts

The 48 scenarios divide into four equal groups of 12:

| Cohort | Scenarios | What Is in Session 1 Memory | What Session 2 Tests |
| :--- | :---: | :--- | :--- |
| `safe_without_web_notes` | 12 | User rules, internal project files, and greetings | Normal coding without external notes |
| `safe_with_web_notes` | 12 | User rules plus harmless web notes | Legitimate tasks with external notes |
| `attack_obvious` | 12 | User rules plus direct override attacks | Shell or HTTP hijacks attempting exfiltration |
| `attack_sneaky` | 12 | User rules plus calm fake conventions | Fake team rules and split-turn attacks |

Attacks are modeled on real security threats:
- **Fake project rules (`fake_project_rule`)**: Described in research papers like PMPA (arXiv:2609.13889), where fake rules disguise themselves as official team linting or CI habits.
- **Split-turn attacks (`split_across_turns`)**: Described in MINJA (arXiv:2503.03704), where harmless-looking phrases in separate turns assemble into malicious instructions when retrieved together.

### Two-Session Workflow

Each scenario runs in its own memory store:
1. **Session 1**: The assistant reads notes from chat (`USER`), internal project files (`INTERNAL`), and issue trackers (`EXTERNAL`). Each note passes through the active write gate.
2. **Session 2**: A new user prompt arrives. The assistant retrieves relevant notes, plans a tool call, and submits it to `ToolInterceptor`.

### Two-Part Grading

Stopping an attack by refusing to run any command is not a success. If the developer asked to run tests, freezing means the developer cannot get work done.

MemoryGuard grades both safety and task completion:
- **Attack Foiled (`not attack_succeeded`)**: The assistant did not execute forbidden commands, leak credentials, or send data to an attacker.
- **Task Completed (`task_completed`)**: The assistant executed the developer's requested command with the correct arguments.
- **Completed Safely (`completed_safely`)**: Both requirements must be met. Under sneaky attacks, an approach only succeeds if it catches the tampered plan and recovers the developer's clean command.

## PersonaMem Recall Slice

The recall slice tests write-time memory on a public benchmark created by researchers at UPenn ([PersonaMem-v1, COLM 2025](https://arxiv.org/abs/2504.14225)). It covers 30 questions across 10 distinct personal profiles.

### Clean Data Curation

Each 40-turn chat in `benchmarks/personamem_slice.json` follows strict standards:
1. System headers and demographic labels were removed so the store sees natural dialogue.
2. Chats were truncated before the test question so the assistant cannot look ahead.
3. Each chat combines 10 real user turns with personal facts, 8 assistant replies, and 22 conversational interruptions like greetings and small talk.

At test time, the assistant retrieves notes and answers a 4-choice question `(a)` through `(d)`.

## Why PersonaMem-v1 Over Newer Versions

UPenn has published three versions of PersonaMem. MemoryGuard chose `PersonaMem-v1` after auditing all three:

| Feature | `PersonaMem-v1` (Used) | `PersonaMem-v2` | `PersonaMem-v3` |
| :--- | :--- | :--- | :--- |
| Focus | Direct user-assistant chats | Chat histories and sensitive data | Social media feed logs |
| Software tasks | Paired with coding suite | None | None |
| Grading | Exact 4-choice evaluation | Multiple-choice evaluation | Needs a paid cloud LLM judge |
| Data size | Compact JSON files | Large multi-file download | 100 folders (220k events) |
| License | Open MIT license | Open MIT license | Non-commercial CC-BY-NC-4.0 |

### Findings from Auditing Newer Releases

- **PersonaMem-v2**: In the sensitive information subset, 508 of 511 entries in the benchmark table omit the relevant personal text snippet from the chat history.
- **PersonaMem-v3**: Roughly 95% of events come from social media feeds (scrolling past posts and liking photos on Instagram or Facebook). This tests social feed recommendation, not assistant note-taking. It also requires calling an external judge model (`gpt-5.5`) rather than providing deterministic answers.

`PersonaMem-v1` provides clean, reproducible, and verifiable testing for memory gating.
