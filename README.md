# Local AI Lab · Build and run AI agents on your own computer

A hands-on course in Google Colab notebooks. Every agent runs on an **open model on the same machine** (Meta Llama via [Ollama](https://ollama.com)), with no API keys, no cloud AI, and no data leaving the box.

It follows Stanford's CS329Z *Engineering AI Agents* (Fall 2026) topic by topic, using a running example: reviewing service contracts against a law firm's playbook. **Course site:** https://shivamkalakaar.github.io/local-ai-lab-agents/ (lessons with one-click Colab buttons). Every notebook ends with worked solutions to its exercises.

## How to run
1. Click an **Open in Colab** badge below.
2. In Colab: `Runtime → Change runtime type → T4 GPU` (free).
3. Run the first code cell. It installs Ollama, downloads the model (about 5 GB) and takes 2–3 minutes. Then run the cells top to bottom.

Each notebook is self-contained, so a Colab disconnect only means rerunning the setup cell. The same notebooks also run on a laptop with Ollama installed (`ollama pull llama3.1:8b`), or against vLLM by setting `LAL_BASE_URL`.

## Notebooks
| # | Topic | What you build | |
|---|---|---|---|
| 00 | Setup | First local model call, speed, temperature | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shivamkalakaar/local-ai-lab-agents/blob/main/notebooks/00_setup_first_call.ipynb) |
| 01 | What an agent is | Agent loop from scratch vs. a fixed workflow | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shivamkalakaar/local-ai-lab-agents/blob/main/notebooks/01_agent_loop.ipynb) |
| 02 | LLMs for builders | Forced JSON, validation, majority vote, best-of-n | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shivamkalakaar/local-ai-lab-agents/blob/main/notebooks/02_llms_for_builders.ipynb) |
| 03 | Context engineering | Event log, masking, compaction, pinned rules | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shivamkalakaar/local-ai-lab-agents/blob/main/notebooks/03_context_engineering.ipynb) |
| 04 | Retrieval (RAG) | Hybrid BM25 + embeddings, RRF, recall@k, checked citations | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shivamkalakaar/local-ai-lab-agents/blob/main/notebooks/04_retrieval_rag.ipynb) |
| 05 | Tools | Registry: validate, approve, run, truncate | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shivamkalakaar/local-ai-lab-agents/blob/main/notebooks/05_tools.ipynb) |
| 06 | Design patterns | Chaining, routing, parallel, checker loops, code actions | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shivamkalakaar/local-ai-lab-agents/blob/main/notebooks/06_design_patterns.ipynb) |
| 07 | Frameworks | DSPy, LangGraph, crash-safe steps | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shivamkalakaar/local-ai-lab-agents/blob/main/notebooks/07_frameworks.ipynb) |
| 08 | Memory | Corrections log, approved lessons, versioned memory | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shivamkalakaar/local-ai-lab-agents/blob/main/notebooks/08_memory.ipynb) |
| 09 | Multi-agent | One agent + checker vs. a team of five | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shivamkalakaar/local-ai-lab-agents/blob/main/notebooks/09_multi_agent.ipynb) |
| 10 | Evaluation | Code graders, baselines, pass@k vs pass^k, careful judge | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shivamkalakaar/local-ai-lab-agents/blob/main/notebooks/10_evaluation.ipynb) |
| 11 | Optimization | GEPA prompt tuning of a 3B model, GRPO core | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shivamkalakaar/local-ai-lab-agents/blob/main/notebooks/11_optimization_gepa.ipynb) |
| 12 | Data and flywheels | Synthetic cases, traces, corrections, SFT data | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shivamkalakaar/local-ai-lab-agents/blob/main/notebooks/12_data_flywheel.ipynb) |
| 13 | Safety | Prompt injection, safe reader, injection twins | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shivamkalakaar/local-ai-lab-agents/blob/main/notebooks/13_safety.ipynb) |
| 14 | Long-running agents | Checklist, progress file, one step per session | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shivamkalakaar/local-ai-lab-agents/blob/main/notebooks/14_long_running_agents.ipynb) |
| 15 | Capstone | Full contract-review pipeline with eval | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shivamkalakaar/local-ai-lab-agents/blob/main/notebooks/15_capstone_contract_review.ipynb) |

## Projects
| | Project | |
|---|---|---|
| P1 | **Build your own agent harness**, no frameworks: parser, tool registry, loop with limits, transcript, injection resistance. Deterministic tests with a scripted model, then the local model. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shivamkalakaar/local-ai-lab-agents/blob/main/projects/P1_build_your_own_harness.ipynb) |
| P2 | **Build an eval task** (Terminal-Bench / Harbor structure, no Docker): instruction, environment, hidden verifier, oracle. Prove the task is sound, then run the agent 5× and report pass^k. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shivamkalakaar/local-ai-lab-agents/blob/main/projects/P2_build_an_eval_task.ipynb) |

These are our own homework-style projects, inspired by the skills CS329Z practices. They are not the course's assignments.

## Part 2 · Self-improving agents (Stanford CS329A)
Every assigned paper from [CS329A](https://cs329a.stanford.edu/) (Autumn 2025) is summarized on the course site. Each lecture has a runnable notebook on the local model.

| # | Topic | Papers | |
|---|---|---|---|
| A01 | Test-time compute scaling | 4 papers | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shivamkalakaar/local-ai-lab-agents/blob/main/cs329a/a01_test_time_compute.ipynb) |
| A02 | Robust verification | 4 papers | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shivamkalakaar/local-ai-lab-agents/blob/main/cs329a/a02_verification.ipynb) |
| A03 | Learning from feedback with tools and code | 3 papers | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shivamkalakaar/local-ai-lab-agents/blob/main/cs329a/a03_feedback_tools_code.ipynb) |
| A04 | Multi-step reasoning and planning | 5 papers | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shivamkalakaar/local-ai-lab-agents/blob/main/cs329a/a04_planning_search.ipynb) |
| A05 | Train-time scaling and RL | 3 papers | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shivamkalakaar/local-ai-lab-agents/blob/main/cs329a/a05_rl_training.ipynb) |
| A06 | Open-ended evolution of self-improving agents | 4 papers | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shivamkalakaar/local-ai-lab-agents/blob/main/cs329a/a06_open_ended_evolution.ipynb) |
| A07 | Self-improvement with search and deep research agents | 3 papers | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shivamkalakaar/local-ai-lab-agents/blob/main/cs329a/a07_search_deep_research.ipynb) |
| A08 | Agentic frameworks for software engineering | 3 papers | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shivamkalakaar/local-ai-lab-agents/blob/main/cs329a/a08_agentic_swe.ipynb) |
| A09 | Augmenting agents with memory | 4 papers | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shivamkalakaar/local-ai-lab-agents/blob/main/cs329a/a09_memory.ipynb) |
| A10 | Agentic evaluations and long-horizon tasks | 3 papers | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shivamkalakaar/local-ai-lab-agents/blob/main/cs329a/a10_evals_long_horizon.ipynb) |
| A11 | Guest lectures and reasoning frontiers | 11 papers | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shivamkalakaar/local-ai-lab-agents/blob/main/cs329a/a11_guest_lectures.ipynb) |

### Past student projects, rebuilt small
Small rebuilds of the CS329A Winter 2025 student projects. They credit the original authors and run on the local model.

| # | Project | Original | |
|---|---|---|---|
| AP1 | AppBench: Benchmarking AI-Generated Web Applications | [paper](https://cs329a.stanford.edu/winter2025projects/appbench.pdf) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shivamkalakaar/local-ai-lab-agents/blob/main/cs329a_projects/ap1_appbench.ipynb) |
| AP2 | PRIME: Planning with Reflective Iterative Multi-Agentic Exploration | [paper](https://cs329a.stanford.edu/winter2025projects/prime.pdf) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shivamkalakaar/local-ai-lab-agents/blob/main/cs329a_projects/ap2_prime.ipynb) |
| AP3 | Enhancing Mathematical Reasoning in LLMs through Reasoning Distillation, GRPO, and Multi-agent PRM Reranking | [paper](https://cs329a.stanford.edu/winter2025projects/enhancing.pdf) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shivamkalakaar/local-ai-lab-agents/blob/main/cs329a_projects/ap3_math_reasoning.ipynb) |
| AP4 | Towards Effective Meta-Agents: Designing Agents that Design Agents (paper: Inefficiencies of Meta Agents for Agent Design) | [paper](https://cs329a.stanford.edu/winter2025projects/batu.pdf) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shivamkalakaar/local-ai-lab-agents/blob/main/cs329a_projects/ap4_meta_agents.ipynb) |
| AP5 | ARCHON Augmented: Planning and Web-Enhanced Components | [paper](https://cs329a.stanford.edu/winter2025projects/archon.pdf) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shivamkalakaar/local-ai-lab-agents/blob/main/cs329a_projects/ap5_archon_augmented.ipynb) |

## What's in the repo
- `lal/`: small helpers (start Ollama, call the model, load data, scoring, the P1 reference harness)
- `data/`: 20 **synthetic** service agreements with planted problems and their answers, plus a sample playbook (`data/gen.py` regenerates them). No real contracts or client data.
- `scripts/src/`: notebook sources; `python scripts/build.py` rebuilds `notebooks/` and `projects/`

## Models
- `llama3.1:8b` (main) and `llama3.2:3b` (fast student for notebook 11), from Meta, served by Ollama
- `BAAI/bge-small-en-v1.5` for embeddings (notebook 04)

Licensed MIT. Built by Local AI Lab.
