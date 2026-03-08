# Build What I Mean — Purple Agent

> This document explains the benchmark (green agent), and proposes
> a design for the purple agent before we write any code.

---

## Quick Start (existing setup)

Based on the minimal template for building [A2A (Agent-to-Agent)](https://a2a-protocol.org/latest/) green agents compatible with the [AgentBeats](https://agentbeats.dev) platform.

## Project Structure

```
pragmatic_builder/
├─ builder_agent.py   # Main server entrypoint + agent card
├─ green_agent.py     # Agent logic
├─ evaluator_proxy.py # Proxy server for evaluation flows
└─ agentbeats/        # AgentBeats integration helpers
data/                 # Scenario data files
Dockerfile            # Docker configuration
pyproject.toml        # Python dependencies
.github/
└─ workflows/
   └─ test-and-publish.yml # CI workflow
```
## How to Play

### Running Locally

```bash
# Install dependencies
uv sync

# Run the builder agent (purple agent dummy)
uv run pragmatic_builder/builder_agent.py --host 127.0.0.1 --port 9019

# Run the green agent (evaluation)
uv run pragmatic_builder/evaluator_proxy.py --host 127.0.0.1 --port 9009
```

### Running the default Scenario
```bash
cd pragmatic_builder
AGENT_TRANSCRIPT_DIR=logs/transcripts AGENT_DEBUG=1 uv run python -m agentbeats.run_scenario scenario.toml --show-logs
```

### Running a Scenario with a questionnaire
```bash
cd pragmatic_builder
AGENT_QA_MODE=dummy AGENT_TRANSCRIPT_DIR=logs/transcripts AGENT_DEBUG=1 uv run python -m agentbeats.run_scenario scenario_question_dummy.toml --show-logs
```

### Running a Scenario with OpenAI QA
```bash
cd pragmatic_builder
export OPENAI_API_KEY="your_openai_api_key_here"
AGENT_QA_MODE=openai AGENT_TRANSCRIPT_DIR=logs/transcripts AGENT_DEBUG=1 uv run python -m agentbeats.run_scenario scenario_question_dummy.toml --show-logs
```

### Running a Scenario with an OpenAI Purple Agent
```bash
cd pragmatic_builder
export OPENAI_API_KEY="your_openai_api_key_here"
export OPENAI_MODEL="gpt-4o-mini"
AGENT_TRANSCRIPT_DIR=logs/transcripts AGENT_DEBUG=1 uv run python -m agentbeats.run_scenario scenario_openai_purple.toml --show-logs
```

### Run Scenario Agents + CLI Client (writes results.json)
```bash
cd pragmatic_builder
AGENT_TRANSCRIPT_DIR=logs/transcripts \
  uv run python -m agentbeats.run_scenario scenario.toml --serve-only &
```

```bash
cd pragmatic_builder
uv run python -m agentbeats.client_cli scenario.toml results.json
```

## Running with Docker (not tested yet)

```bash
# Build the green agent image
docker build -t my-agent-green -f Dockerfile .

# Build the purple agent image
docker build -t my-agent-purple -f Dockerfile.purple .

# Run the green agent (evaluation)
docker run -p 9009:9009 my-agent-green

# Run the purple builder agent
docker run -p 9018:9018 my-agent-purple
```

## Testing

Run A2A conformance tests against your agent.

```bash
# Install test dependencies
uv sync --extra test

# Start your agent (uv or docker; see above)

# Run tests against your running agent URL
uv run pytest --agent-url http://localhost:9009
```

## Publishing

The repository includes a GitHub Actions workflow that automatically builds, tests, and publishes a Docker image of your agent to GitHub Container Registry.

If your agent needs API keys or other secrets, add them in Settings → Secrets and variables → Actions → Repository secrets. They'll be available as environment variables during CI tests.

- **Push to `main`** → publishes `latest` tag:
```
ghcr.io/<your-username>/<your-repo-name>:latest
```

- **Create a git tag** (e.g. `git tag v1.0.0 && git push origin v1.0.0`) → publishes version tags:
```
ghcr.io/<your-username>/<your-repo-name>:1.0.0
ghcr.io/<your-username>/<your-repo-name>:1
```

Once the workflow completes, find your Docker image in the Packages section (right sidebar of your repository). Configure the package visibility in package settings.

> **Note:** Organization repositories may need package write permissions enabled manually (Settings → Actions → General). Version tags must follow [semantic versioning](https://semver.org/) (e.g., `v1.0.0`).

---

## 1. Benchmark Explanation (Green Agent)

### What is it testing?

**Build What I Mean** is a **pragmatics benchmark** rooted in psycholinguistics.
It tests whether an AI agent can interpret natural-language block-building
instructions — including *deliberately ambiguous* ones — the way a human
cooperative partner would, using conversational context to fill in gaps.

The green agent is the **assessor**: it generates instructions, sends them to the
purple agent one-by-one, and scores the responses. The purple agent's job is to
receive each instruction and respond with the correct block structure.

### The 3-D Grid

| Axis | Meaning | Valid values |
|---|---|---|
| X | left–right | −400, −300, −200, −100, 0, 100, 200, 300, 400 |
| Z | front–back | −400, −300, −200, −100, 0, 100, 200, 300, 400 |
| Y | height | 50 (ground), 150, 250, 350, 450 (+100 per stacked block) |

A block is written as `Color,x,y,z` (e.g. `Red,-400,50,-400`).
The full structure is a `;`-separated list of all blocks currently on the grid.

### Prompt format the purple agent receives

Every message from the green agent looks like this:

```
[TASK_DESCRIPTION] Grid: 9x9 cells. Origin="middle square" ...scoring rules...
[SPEAKER] Anna
[START_STRUCTURE] Blue,0,50,0;Blue,0,150,0
Place a yellow block on top of the blue stack.
```

It always contains:
- `[TASK_DESCRIPTION]` — the grid rules and scoring summary (same every turn)
- `[SPEAKER]` — who is giving this instruction
- `[START_STRUCTURE]` — blocks already on the grid before this instruction
- Free text — the natural-language instruction

### What the purple agent must reply

Exactly one of:

| Reply | Meaning | Cost/Gain |
|---|---|---|
| `[BUILD];Color,x,y,z;...` | Declare the complete final structure | +10 if correct, −10 if wrong |
| `[ASK];Your question` | Ask a clarifying question (green agent answers, then waits for a `[BUILD]`) | −5 per question |

An invalid prefix (anything else) triggers a free retry (0 points, loop continues).

### Session structure

Each evaluation run iterates over **8 seeds**. Each seed is an independent game with:
- **Two speakers** (randomly named) active in sequence
- **~20 instructions per speaker** across a fixed ordering of trial types
- One continuous A2A context (conversation history is **not** reset between instructions)

---

## 2. The Core Challenge — Three Trial Types

### `fully_spec` — Fully Specified (straightforward)

The instruction completely describes what to build. No information is missing.

> *"Place a red block in each corner of the grid. Then put a green block on
> top of each red block."*

Target: `Red,-400,50,-400;Red,400,50,-400;Red,400,50,400;Red,-400,50,400;Green,-400,150,-400;...`

A capable LLM handles these well. The difficulty is precision: exact coordinates,
and every pre-existing block from `[START_STRUCTURE]` must be included in the output.

---

### `color_under` — Colour Underspecified (hard)

The number and position of blocks is clear, but the **colour of some blocks is
not stated**. There are two variants (`a` and `b`) that use the **exact same
sentence** but expect different colours:

> *"Stack five purple blocks in the middle of the grid, then stack four blocks
> in front of them."*

| Version | What the four front blocks must be |
|---|---|
| `a` | **Yellow** |
| `b` | **Purple** (same as the first stack) |

The only way to resolve this is from context established earlier in the session
with this same speaker. If the speaker has consistently used one colour convention
in prior rounds, that pattern is the signal.

---

### `number_under` — Number Underspecified (hard)

The colour is clear but the **count of some blocks is not stated**.

> *"Stack two yellow blocks on the middle square. Stack two green blocks
> directly in front of the yellow ones. Then stack red blocks directly in
> front of the green ones."*

| Version | How many red blocks |
|---|---|
| `a` | **3** |
| `b` | **2** |

Again, identical sentence — only speaker-level context resolves the count.

---

## 3. The Two-Speaker Design

Each session has two speakers, each with a fixed ordering:

| Speaker type | Trial ordering | Critical trials given |
|---|---|---|
| "Pia-type" | `PiaOrdering` | Always the `b` version |
| "Lisa-type" | `LisaOrdering` | Alternates `critical_a` and `critical_b` |

**The implication for the purple agent:**

A Pia-type speaker is *consistent* throughout a session — she always resolves
ambiguity the same way (always `b`). Once you've seen a few of her instructions,
you can reliably predict her convention.

A Lisa-type speaker alternates — she sometimes uses `a`, sometimes `b`. Asking
may be more rational with this speaker, or you can try to track subcategories.

The `[SPEAKER]` tag in every prompt is therefore a critical piece of information.
The purple agent must maintain **separate context per speaker** within a session.

---

## 4. What Makes This Hard

| Challenge | Detail |
|---|---|
| **Coordinate precision** | One wrong digit = −10 pts. Coordinates must snap to valid grid values. |
| **Full structure output** | Must output every block currently on the grid, not just the new ones. Forgetting `[START_STRUCTURE]` blocks is a common failure mode. |
| **Colour ambiguity** | Identical sentences, different targets. Cannot be solved by reading the instruction alone. |
| **Count ambiguity** | Same. Requires cross-round inference about the speaker's convention. |
| **Question cost** | Each `[ASK]` costs −5. A wrong `[BUILD]` costs −10. So asking is worthwhile when accuracy would be below ~75%, not otherwise. |
| **Speaker tracking** | Two speakers are interleaved. Must not mix up their established conventions. |
| **Long sessions** | 8 seeds × ~40 instructions each. LLM context windows and costs matter. |

---

## 5. Proposed Purple Agent Design

### 5.1 High-Level Goal

The purple agent must do three things well:

1. **Parse and apply instructions accurately** — grid mechanics, coordinate arithmetic, start-structure handling.
2. **Detect underspecification** — recognise when colour or count is missing.
3. **Resolve ambiguity from speaker context** — use intra-session history for the current speaker to fill in missing values.

### 5.2 Approach: LLM with Full Conversation History + Speaker-Aware Prompting

The simplest implementation that can handle all three goals:

- Use an OpenAI chat model with a carefully written **system prompt**.
- Maintain the **full message history** within an A2A context (one context = one seed).
  The LLM therefore sees all prior instructions and responses as conversation history.
- The system prompt explicitly instructs the model to:
  - Track each speaker's patterns (colours they use for unspecified slots, stack sizes they prefer)
  - Use those patterns when ambiguity is detected
  - Prefer `[BUILD]` over `[ASK]`, but ask if the confidence is genuinely low

```
Session history (what the LLM sees each turn):
  system:    <rules, coordinate system, scoring, strategy>
  user:      <turn 1 — speaker Anna, instruction 1>
  assistant: [BUILD];...
  user:      Feedback: Correct +10 | ...
  user:      <turn 2 — speaker Anna, instruction 2>
  assistant: [BUILD];...
  ...
  user:      A new task is starting...   ← seed boundary (soft reset)
  user:      <turn 1 — new seed, speaker Emma, instruction 1>
```

### 5.3 [ASK] vs [BUILD] Decision Policy

```
fully_spec instruction
  → [BUILD] directly

underspec + speaker has shown ≥2 prior examples of same ambiguity type
  → infer the missing value from those examples → [BUILD]

underspec + little or no speaker history yet
  → [ASK] once to get the answer → then [BUILD]

underspec + speaker is Lisa-type (alternating a/b)
  → may need to ask more, or track which version is current
```

Because every question costs −5 and a wrong build costs −10, asking once to
guarantee a +10 correct build is almost always worth it (+10 − 5 = +5 vs a
probable −10 for guessing wrong).

### 5.4 Output Validation (client-side)

The green agent checks for the exact prefixes `[BUILD]` and `[ASK]`.
The purple agent's server code should:
1. Check the LLM output starts with one of the two valid prefixes.
2. If not, either re-prompt the LLM or apply a basic fix (e.g. strip extra text).
3. Optionally: snap coordinates to the nearest valid grid values before returning.

### 5.5 Model Recommendation

| Model | Notes |
|---|---|
| `gpt-4o` / `gpt-4.1` | Best spatial reasoning and instruction-following. Recommended. |
| `gpt-4o-mini` | Cheaper; acceptable for fully_spec but struggles with precise coordinate arithmetic. |
| Temperature | 0.1 – 0.2. This is a precision task; lower is better. |

---

## 6. Open Questions to Settle Before Coding

1. **Single-stage vs two-stage LLM call**
   - Single stage: one call per turn, LLM does classify + resolve + emit simultaneously.
   - Two stage: first call classifies ambiguity + extracts knowns, second call generates the `[BUILD]` string.
   Which gives better coordinate accuracy?

2. **Speaker profile storage**
   - Rely entirely on raw chat history (implicit, no extra code)?
   - Or maintain an explicit JSON-structured speaker profile injected into every prompt (compact, reliable)?

3. **`[ASK]` policy strictness**
   - Never ask (all `[BUILD]`, risk −10 on ambiguous trials)?
   - Ask once on the first underspec encounter per speaker?
   - Ask whenever LLM signals uncertainty?

4. **Seed boundary handling**
   The green agent sends `"A new task is starting, now you will play the game again."` between seeds.
   Should the purple agent fully reset its conversation history, or keep it (so it can leverage patterns learned earlier)?
   (Note: speakers are re-randomised each seed, so cross-seed memory may not help.)

5. **Output coordinate validation**
   Should the server snap LLM-generated coordinates to valid grid positions, or return verbatim?
   Snapping reduces −10 penalties from off-by-one errors but may hide real model mistakes.
