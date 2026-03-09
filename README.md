# AgentBeats: Pragmatic Builder

This repository contains two interacting agents for a communicative 3D block-building benchmark ("Build-What-I-Mean"):
- **Green Agent**: The Instruction Giver & Benchmark Orchestrator (located in `build_what_i_mean/`)
- **Purple Agent**: The Instruction Follower & Builder (located in `purple_agent/`)

---

## How to Run the Purple Agent Evaluation

This guide covers how to run the Purple Agent (using your custom OpenAI-compatible server deployed on Kaggle) without cluttering your terminal with excessive logs.

### 1. Setup Your `.env` File
Ensure your `.env` file in the `build_what_i_mean` directory has the exact endpoints pointing to your Kaggle Ngrok tunnel.

Create or edit `build_what_i_mean/.env`:
```ini
OPENAI_API_KEY="sk-no-key-required"
OPENAI_BASE_URL="https://YOUR_NGROK_URL.ngrok-free.app/v1"
PURPLE_MODEL="Qwen/Qwen2.5-Coder-32B-Instruct"
OPENAI_TIMEOUT=120
```

### 2. Run the Evaluation
Navigate to the `pragmatic_builder` directory and start the evaluation.

```bash
cd build_what_i_mean/pragmatic_builder
AGENT_TRANSCRIPT_DIR=logs/transcripts uv run python -m agentbeats.run_scenario scenario_openai_purple.toml
```

*(Note: We removed the `--show-logs` flag and `AGENT_DEBUG=1` so the terminal only displays top-level evaluation progress.)*

### 3. View the Detailed Outputs (Log File)
The Purple Agent writes all of its detailed processing (LLM prompts, raw responses, Stage 1 classification, and Stage 2 coordinate generation) directly to a dedicated log file rather than the terminal.

To watch the purple agent's internal workings in real-time, open a **new terminal tab** and run:
```bash
tail -f purple_agent/purple_agent.log
```

### 4. Review the Full Transcript
When the run finishes, the evaluation tool will have saved a complete chat recap (the inputs/outputs the platform tracked) under:
`build_what_i_mean/pragmatic_builder/logs/transcripts/`

You can look at the latest folder inside there to see how the system scored your agent round by round.

---
For detailed information about the inner workings of the Purple Agent's two-stage architecture, pragmatic inference rules, and local development, please see [`purple_agent/README.md`](purple_agent/README.md).