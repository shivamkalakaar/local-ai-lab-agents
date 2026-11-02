# Local AI Lab · Build and run AI agents on your own computer

A hands-on course in Google Colab notebooks. Every agent runs on an **open model on the same machine** (Meta Llama via [Ollama](https://ollama.com)), with no API keys, no cloud AI, and no data leaving the box.

It follows Stanford's CS329Z *Engineering AI Agents* (Fall 2026) topic by topic, using a running example: reviewing service contracts against a law firm's playbook. Reading companion: [Agent Engineering School](https://claude.ai/artifact/41sHaxGwFWf7SjJWDdtK5w).

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

## What's in the repo
- `lal/`: small helpers (start Ollama, call the model, load data, scoring, the P1 reference harness)
- `data/`: 20 **synthetic** service agreements with planted problems and their answers, plus a sample playbook (`data/gen.py` regenerates them). No real contracts or client data.
- `scripts/src/`: notebook sources; `python scripts/build.py` rebuilds `notebooks/` and `projects/`

## Models
- `llama3.1:8b` (main) and `llama3.2:3b` (fast student for notebook 11), from Meta, served by Ollama
- `BAAI/bge-small-en-v1.5` for embeddings (notebook 04)

Licensed MIT. Built by Local AI Lab.
