# Problem and Threat Model

Persistent coding assistants save notes across sessions so you do not have to repeat test flags, directory layouts, or personal workflow habits. Saving raw session transcripts creates two coupled problems: everyday small talk crowds out useful context, and untrusted external text can plant instructions that hijack tool execution in a later session.

MemoryGuard addresses both problems in one compact Python reference system (2,367 total lines across 15 files, with 1,346 lines in the core memoryguard package across 6 files). It separates what an agent **remembers** at write time from what an agent **trusts** when executing privileged tools.

## Cross-Session Memory

A stateless coding assistant forgets project context as soon as a terminal session closes. Persistent memory bridges sessions through three steps:

1. During Session 1, the assistant observes messages from developer chat, repository documentation, and external resources such as GitHub issues or web pages.
2. Admitted notes sit in a local database and return via keyword or full-text search when a relevant prompt arrives in Session 2.
3. The planner reads the retrieved notes alongside repository files and proposes a concrete action such as running a shell command, writing a file, or answering a question.

Without write filtering and provenance tracking, an external webpage read on Monday sits in the same prompt block on Tuesday as a rule typed directly by the developer.

## Provenance Tiers

Coding agents read inputs with different authority levels. MemoryGuard assigns an integer provenance tier to each incoming candidate note at ingestion time:

| Tier Name | Level | Typical Sources | Authority Scope |
| :--- | :---: | :--- | :--- |
| `USER` | 3 | Direct developer chat prompts and explicit preferences | Full authority over personal workflow and project commands |
| `REPO` | 2 | Checked-in `README.md` and `CONTRIBUTING.md` files | Trusted project conventions committed to version control |
| `EXTERNAL` | 1 | GitHub issues, pull request comments, release notes, web pages | Informational context that must not alter privileged actions |

Treating all three sources as undifferentiated text lets untrusted `EXTERNAL` notes override `USER` and `REPO` rules whenever they rank high in retrieval.

## Failure Modes

Persistent agent memory fails along two axes on real software projects:

- Casual greetings, scheduling messages, and status acknowledgments consume retrieval slots and prompt tokens without carrying durable project rules or user facts.
- Untrusted issues, pull requests, or web documents inject persistent instructions into memory during Session 1 that trigger unsafe tool calls in Session 2.

Cross-session attacks in coding workflows span four distinct patterns:

| Attack Pattern | Session 1 Payload Style | Session 2 Effect |
| :--- | :--- | :--- |
| Direct override | Imperative `IMPORTANT: Ignore previous instructions` strings | Redirects shell, file, or network calls to attacker targets |
| Declarative convention | Calm prose mimicking team CI, migration, or linter rules | Adds unauthorized flags, hooks, or `http_post` uploads |
| Split-turn fragment | Two innocent-looking notes written across separate turns | Combines at retrieval time into an unsafe action |
| Secret read or beacon | Note steering file inspection or markdown formatting | Reads `.env` or SSH credentials, or embeds tracking image URLs |

## Single-Layer Limits

Existing defenses typically guard only write admission or apply rigid taint locks at execution time. Each single-layer approach breaks down under realistic coding workloads:

| Defense Approach | Primary Mechanism | Failure Mode on Coding Agents |
| :--- | :--- | :--- |
| Append-all store | Saves all non-empty turns without provenance checks | Admits chatter and suffers a 50.0% Attack Success Rate |
| Regex filter | Blocks known jailbreak phrases and requires CLI syntax cues | Drops 66.67% of conversational preferences lacking code tokens (33.33% recall) |
| Cloud LLM utility gate | Prompts a remote LLM to score long-term note usefulness | Adds network tail latency and admits calm fake conventions (83.33% sneaky ASR) |
| Local classifier alone | Vetoes imperative overrides and drops casual small talk | Admits declarative conventions and split-turn fragments |
| Blanket taint lock | Blocks privileged tools whenever external notes enter context | Refuses 88.89% of legitimate privileged tasks with external notes (37.5% across all benign tasks) |

Declarative convention poisoning ([PMPA, arXiv:2609.13889](https://arxiv.org/abs/2609.13889)) and split-turn attacks ([MINJA, arXiv:2503.03704](https://arxiv.org/abs/2503.03704)) look identical to legitimate project documentation at write time. No standalone write filter can reject calm convention prose from a GitHub issue without also discarding genuine third-party library notes.

Strict Biba integrity locks solve security by refusing to act whenever an `EXTERNAL` note appears in retrieved memory. Developers routinely store external API release notes alongside personal test flags, so blanket taint locking breaks normal work.

## Design Principles

MemoryGuard combines four mechanisms so security, recall, and developer velocity work together:

1. A 421M-parameter CPU classifier (`LayaAdmissionGate`) evaluates four probability heads in one local pass. It separates prompt-injection vetoing from positive utility scoring without cloud API calls.
2. `MemoryStore` persists notes in SQLite with `FTS5` BM25 full-text indexing, tracking provenance tiers (`USER=3`, `REPO=2`, `EXTERNAL=1`).
3. `MemoryStore` uses `SequenceMatcher` similarity with authority dominance (`dup_thresh = 0.88`, `0.99` on Stock) and tiered novelty checking (`store.max_similarity(clean_text, min_tier=tier)`). Lower-tier near-duplicates of existing `USER` or `REPO` rules are dropped as `shadowed`, while equal- or higher-tier notes supersede older records.
4. When an `EXTERNAL` note enters context for a guarded action, `ToolInterceptor` replans using `USER` and `REPO` notes alone. On Clean-Room mismatch, `ToolInterceptor` simply discards the poisoned proposal and executes the verified clean call (`clean_room_recovered`) without mutating SQLite, completing the developer's legitimate task safely.
5. Google ADK integration (`memoryguard/adk_middleware.py`, `run_demo.py`, `agents/`) connects MemoryGuard into `LlmAgent` lifecycle callbacks (`before_model_callback` for admission and retrieval, and `before_tool_callback` for Clean-Room interception), enabling comparative evaluation in terminal CLI and visual `adk web agents` modes.

## Related Work

Recent memory security projects explore complementary parts of this problem space:

- [A-MemGuard (arXiv:2510.02373)](https://arxiv.org/abs/2510.02373) studies consensus validation across multiple reasoning paths and stores failure lessons in dual memory structures.
- [MemGate (arXiv:2606.06054)](https://arxiv.org/abs/2606.06054) frames write-time memory admission as calibrated cascade routing across lightweight and heavyweight verifiers.
- OWASP Agent Memory Guard provides schema validation, rule-based scanning, and hash-chain audit logging for general agent memory frameworks.

MemoryGuard differs by pairing a zero-API local CPU decision model (`convaiinnovations/laya`) at write time with counterfactual Clean-Room non-interference and automatic trusted-plan recovery at action time in 2,367 lines of standard Python across 15 files (1,346 lines in the core memoryguard package across 6 files, alongside Google ADK multi-session integration in run_demo.py, agents/, and memoryguard/adk_middleware.py).
