# Purple Agent Improvements

Based on an analysis of the Green Agent (`build_what_i_mean`) and its evaluation criteria, here are the key improvements that should be made to the Purple Agent (`purple_agent`) to maximize its score and adhere to the benchmark's pragmatic goals.

## 1. Implement the Decision Gate Logic (Missing Speaker Profile Application)

**The Problem:**
The `purple_agent` currently tracks speaker profiles beautifully (recording `color_fills` and `count_fills` in `state.py` based on feedback). However, it **never actually uses them** in the Decision Gate to avoid asking questions.
In `agent.py`, any detected ambiguity results in an unconditional `[ASK]` command:
```python
        if ambiguity != "none":
            logger.info(
                "[GATE] Ambiguity present ('%s'), asking for clarification",
                ambiguity
            )
            return self._decide_ask(...)
```
This contradicts the benchmark's stated goal of inferring values from prior examples to save points (an `[ASK]` costs -5 points, while a correct `[BUILD]` grants +10 points).

**The Solution:**
Update `agent.py`'s `_handle_instruction` to check the `speaker.inferred_color()` or `speaker.inferred_count()`. If the speaker has a consistent history (e.g., they have omitted a color ≥ 2 times and we learned what it was), the agent should resolve the ambiguity internally and supply that as the `resolution` string to `_stage2_build` instead of returning `_decide_ask()`.

## 2. Utilize Stage 1 Confidence Scores

**The Problem:**
Stage 1's LLM prompt asks for a `confidence_in_build` float between 0.0 and 1.0. This score is extracted in `_handle_instruction` but is completely ignored by the Decision Gate.

**The Solution:**
The Decision Gate should use the confidence score as an additional heuristic. Even if `ambiguity == "none"`, if the `confidence_in_build` is extremely low (e.g., < 0.4) because the instruction is overly complex or contradictory, it might be mathematically safer to take a -5 point penalty to `[ASK]` rather than risk a -10 point penalty for an incorrect build.

## 3. Retain Cross-Turn Conversational Context Better

**The Problem:**
The `SessionState` stores the conversation history to pass to the direct/fallback prompt. However, it blindly truncates it if it gets too long:
```python
        if len(self.conversation) > 10:
            self.conversation = self.conversation[-6:]
```
This might truncate the `[TASK_DESCRIPTION]` or `[START_STRUCTURE]` from earlier if a conversation goes long, leaving the fallback prompt without the essential grid properties.

**The Solution:**
When truncating the conversation history, ensure the `[TASK_DESCRIPTION]` system prompt and the current turn's `[START_STRUCTURE]` are preserved so the LLM doesn't hallucinate rules when falling back.

## 4. Coordinate Snapping Masks LLM Reasoning Errors

**The Problem:**
`validators.py` contains a `snap()` function that forces any arbitrary float or integer into a valid grid value (e.g., 53 snaps to 50). While this prevents formatting errors, the Green Agent README notes this might hide real model mistakes. 

**The Solution:**
While snapping is good for resilience, you might want to log when a snap actually occurs (i.e. `if snapped_val != original_val: log.warning(...)`). This allows you to track whether the LLM actually understands the spatial mechanics or is just getting lucky due to the validator roundings.
