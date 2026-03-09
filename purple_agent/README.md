# Purple Agent — OpenAI (rita)

Standalone LLM-powered purple agent for the **Build-What-I-Mean** benchmark.

Implements the A2A protocol and exposes an endpoint that the green agent (benchmark) talks to.  
Uses a **two-stage LLM pipeline** with speaker-aware pragmatic inference.

---

## Architecture & How it works

The agent uses a **two-stage LLM pipeline** to efficiently handle pragmatic inference (figuring out what a speaker *means* when they leave out details), parsing, and block-coordinate generation.

Instead of passing the entire dialogue repeatedly into a single LLM prompt and crossing our fingers that it learns the speaker's patterns, this architecture extracts the pattern symbolically into a state manager and injects it as an explicit rule.

### Data Flow

```text
Green agent  →  [TASK_DESCRIPTION] + [SPEAKER] + [START_STRUCTURE]  →  Purple agent
                                                                              │
                                                                  ┌───────────▼────────────┐
                                                                  │      Stage 1 (LLM)     │
                                                                  │  Classify ambiguity    │
                                                                  │ (color / count / none) │
                                                                  └───────────┬────────────┘
                                                                              │
                                                                  ┌───────────▼────────────┐
                                                                  │      Decision gate     │
                                                                  │ ├─ Resolve via Profile │
                                                                  │ └─ [ASK] if unsure     │
                                                                  └───────────┬────────────┘
                                                                              │
                                                                  ┌───────────▼────────────┐
                                                                  │      Stage 2 (LLM)     │
                                                                  │ Generate [BUILD] coords│
                                                                  └───────────┬────────────┘
                                                                              │
Purple agent  ←  [BUILD];Color,x,y,z;...  ←───────────────(Validates)─────────┘
```

### Component Breakdown

1. **State Manager (`state.py`)**: 
   - Maintains a **Speaker Profile** for every unique speaker encountered in a session. 
   - Tracks how many times they omitted a colour, what colour the target actually was, and tallies their "fill" conventions (e.g. "When *Anna* is vague, it usually means *Yellow*").
   - Maintains the short-term conversation context for error recovery.

2. **Stage 1 - Classification (`agent.py` / `prompts.py`)**:
   - The green agent's instruction is fed to the LLM alongside the mathematical block state.
   - The LLM's only job in Stage 1 is **Language Understanding**. It outputs strict JSON deciding if there is missing information (`ambiguity: "color" | "count" | "none"`) and extracts explicit modifiers.
   - **Why?** It prevents the LLM from getting confused between "deciding what to do" and "calculating 3D math". 

3. **The Decision Gate (`agent.py`)**:
   - This is pure Python logic. 
   - If Stage 1 detects a missing `color`, the Gate checks the `SpeakerProfile`. 
   - If we have seen this speaker do this before, we **skip asking** and resolve the missing colour immediately internally.
   - If we *don't* know the speaker's pattern yet, or confidence is low, the Gate aborts the build and issues an `[ASK]` command to the Green Agent. This costs points (-5), but acquires the ground-truth answer.

4. **Feedback Loop (`agent.py > _update_profile_from_feedback`)**:
   - Once the Green Agent grades a round, it sends us the correct `target_structure`.
   - The Purple Agent compares the target structure to what it explicitly parsed in Stage 1 to deduce what the missing colour/count actually was. It then logs this truth into the State Manager to learn for future turns.

5. **Stage 2 - Generation (`agent.py` / `validators.py`)**:
   - The resolved instruction (original text + the explicit fill logic injected by the Gate) is sent to the LLM. 
   - The LLM's only job here is **Spatial Reasoning**. It parses the 3D grid and returns coordinates.
   - Finally, `validators.py` intercepts the string, snaps all arbitrary floats/integers to legal grid integers, capitalizes colours, and ensures the protocol format passes the Green Agent's strict regex.

---

## Project structure

```
purple_openai/          ← you are here (project root)
├── server.py           ← A2A server entry point
.agentbeats/
    ├── README.md
    ├── build_what_i_mean/
    └── purple_agent/          ← you are here (project root)
        ├── server.py           ← A2A server entry point
        ├── agent.py            ← two-stage pipeline (PurplePipeline)
        ├── state.py            ← session / speaker-profile state management
        ├── prompts.py          ← LLM prompt templates
        ├── validators.py       ← coordinate validation + response format
        ├── pyproject.toml
        ├── Dockerfile
        ├── .env.example
        └── tests/
            └── test_purple.py  ← unit / smoke tests (no LLM calls)
```

---

## Quick start

### 1. Configure the Agent

Make a copy of the default environment variables:

```bash
cp .env.example .env
```
Edit `.env` to provide your API key and preferred model:

```ini
OPENAI_API_KEY="sk-..."
PURPLE_MODEL="gpt-4o"
```

### 2. Run the Web Server Locally
The benchmark orchestrator interacts with this agent via a local HTTP server using the A2A protocol.

```bash
# From this directory (purple_agent/)
uv run python server.py --host 127.0.0.1 --port 9022

# Or using the installed script entry point
uv run purple-agent --host 127.0.0.1 --port 9022

# Enable verbose debug logging
uv run python server.py --host 127.0.0.1 --port 9022 --debug
```

### 4. Run tests (no API key required)

```bash
uv run pytest tests/
```

---

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | *(required)* | OpenAI API key |
| `PURPLE_MODEL` | `gpt-4o` | Model to use |
| `OPENAI_BASE_URL` | *(empty)* | Override API base URL (e.g. Azure OpenAI) |
| `PURPLE_TEMPERATURE` | `0.2` | Sampling temperature |
| `PURPLE_MAX_TOKENS` | `1024` | Max tokens per LLM call |
| `OPENAI_TIMEOUT` | `60` | LLM call timeout (seconds) |
| `AGENT_DEBUG` | `0` | Set to `1` for verbose logging |

---

## Running with Docker

```bash
# Build
docker build -t purple-agent .

# Run
docker run --env-file .env -p 9022:9022 purple-agent
```

---

## Running with the green agent (full benchmark)

Point the green agent's scenario config at this server:

```toml
[[participants]]
role = "rita"
endpoint = "http://127.0.0.1:9022"
cmd = "python server.py --host 127.0.0.1 --port 9022"
```

Or use the `scenario_openai_purple.toml` inside `build_what_i_mean/pragmatic_builder/` which already has the correct endpoint configured.
