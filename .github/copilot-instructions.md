# AgentBeats Project Topology & AI Context Guide

## 1. Core Architecture Setup
* **`build_what_i_mean/` (The Green Agent / Evaluator)**: This directory contains the benchmarking framework. Think of it as the 'teacher' or 'simulator'. It loads spatial game scenarios (from `/data`), runs the simulated turns (8 seeds × 40 turns = 320 evaluate instances), provides the correct/incorrect feedback, scores the results, and controls the chat workflow.
* **`purple-agent/` root and `purple_openai/` (The Purple Agent)**: This is the actual AI test-taker. It is the agent taking instructions, generating `.json` classifications (Stage 1), and producing `[BUILD]` coordinates (Stage 2) by hitting our external LLM model via an OpenAI compatible API (running on Kaggle in this case). 
* Whenever making edits to the AI's logic, ensure you are editing the `/Users/mehemudazad/Desktop/agentbeats/purple-agent/purple_openai/` logic directly, not a copy inside the `build_what_i_mean` directory.

## 2. API Statelessness & "Memory"
* **Standard APIs are Stateless**: The LLM running on Kaggle (or via OpenAI API) does *not* remember previous turns on its own. Every API call is a blank slate.
* **How the Agent "Remembers"**: The Purple Agent fakes memory via Python code tracking (`purple_openai/state.py`). 
  1. It reads the feedback from the Green Agent (evaluator).
  2. It updates a Python dictionary (`SpeakerProfile`) tracking patterns like *"Lisa prefers 4 blocks"*.
  3. On the next API call, the Python script dynamically injects this `"SPEAKER HISTORY"` into the system prompt. The model reads the "cheat sheet" and acts as if it remembers the past.

## 3. Model & Version Logging

**v1.1 - The Pragmatic Failsafe (March 9, 2026)**
* **Issue:** Qwen-32B was returning high confidence (e.g. 0.90) on Stage 1 classification even when ambiguous (missing exact colour or exact quantity). This bypassed the `[ASK]` functionality entirely and just blindly guessed coordinates.
* **Fix applied:**
    * Updated `purple_openai/prompts.py` to add "CRITICAL CONFIDENCE RULES" enforcing confidence=0.2 if exact parameters are missing.
    * Updated `purple_openai/agent.py` to add a hardcoded python `if not explicit_colors` failsafe to force the `_decide_ask()` function regardless of LLM confidence.
    * Added comprehensive file logging to `purple_agent.log` for LLM brain tracking.

**v1.2 - Targeted Question Reformulation (March 9, 2026)**
* **Issue:** The Green Agent (evaluator) was not answering quantity-based questions correctly because its fallback evaluator only checks for the word "color". Asking "How many blocks..." returned the useless default: "I can answer questions about the target structure."
* **Root Cause:** The `_fallback_answer` in `green_agent.py` only has a single `if "color" in question.lower()` branch. All other question types hit an empty `return` statement. The Green Agent is FIXED and cannot be modified.
* **Fix applied (Purple Agent only):**
    * Updated `purple_openai/agent.py` `_decide_ask()` to always ask color-based questions, even for `count` ambiguity: *"What color are all the blocks in the target structure?"*. This triggers the Green Agent's color parser and returns the full color list (e.g. "Colors in target: Blue, Blue, Green"), from which the count can be inferred.
    * Updated `purple_openai/prompts.py` `DIRECT_SYSTEM` to instruct the model to count occurrences of colors in the answer list to infer missing block quantities.

## 4. Academic Context for Ambiguity Handling
When modifying or discussing the `[ASK]` behavior, refer to these NLP/Psycholinguistics framework concepts:
* **Rational Speech Act (RSA)**: Modeling when a pragmatic listener should infer a state vs request clarification.
* **Clarification Question Generation (CQG)**: For task-oriented dialogue systems.
* **Epistemic Uncertainty Estimation**: Dealing with the model's 'overconfidence' when guessing missing attributes. 