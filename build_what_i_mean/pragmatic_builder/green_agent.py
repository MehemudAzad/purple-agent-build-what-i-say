import asyncio
import logging
import sys
from pathlib import Path
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    InvalidParamsError,
    TaskState,
    Part,
    TextPart,
)
from a2a.utils import (
    new_agent_text_message,
)
from a2a.utils.errors import ServerError

from building_task import BuildingGameTask
from agentbeats.models import EvalRequest, EvalResult
from agentbeats.tool_provider import ToolProvider
from agentbeats.conversation_recorder import ConversationRecorder
from agentbeats.question_answerer import QuestionAnswerer

logger = logging.getLogger(__name__)


class BuildingInstructorGreenAgent:
    def __init__(self, debug: bool = False, transcript_path: str | None = None):
        self._required_roles = ["rita"]
        self._required_config_keys = ["list1_path", "list2_path"]
        self._tool_provider = ToolProvider()
        self._debug = debug
        self._recorder = ConversationRecorder(transcript_path) if transcript_path else None
        self._qa = QuestionAnswerer.from_env()

    async def _debug_pause(self, prompt: str) -> None:
        if not self._debug or not sys.stdin.isatty():
            return
            return
        await asyncio.to_thread(input, prompt)

    async def run_eval(self, req: EvalRequest, updater: TaskUpdater) -> None:
        logger.info(f"Starting structure evaluation: {req}")

        list1_path = self._resolve_path(req.config["list1_path"])
        list2_path = self._resolve_path(req.config["list2_path"])

        # Run 8 trials with different seeds
        num_seeds = 8
        all_accuracies = []
        all_avg_questions = []
        all_results = {}

        purple_agent_role = 'rita'

        async def turn(role: str, prompt: str) -> str:
            if self._recorder:
                self._recorder.record(f"GREEN -> {role}: {prompt}")
            response = await self._tool_provider.talk_to_agent(prompt, str(req.participants[role]),
                                                               new_conversation=False)
            logger.info(f"{role}: {response}")
            await updater.update_status(TaskState.working, new_agent_text_message(f"{role}: {response}"))
            if self._recorder:
                self._recorder.record(f"{role} -> GREEN: {response}")
            await self._debug_pause("Press Enter to continue...\n")
            return response

        async def send_feedback(role: str, message: str) -> None:
            if self._recorder:
                self._recorder.record(f"GREEN -> {role}: {message}")
            await self._tool_provider.talk_to_agent(
                message,
                str(req.participants[role]),
                new_conversation=False,
            )
            await updater.update_status(TaskState.working, new_agent_text_message(message))
            await self._debug_pause("Press Enter to continue...\n")

        for seed in range(num_seeds):
            logger.info(f"Starting trial {seed + 1}/{num_seeds} with seed {seed}")

            # Send transition message between tasks (except for the first one)
            if seed > 0:
                transition_message = "A new task is starting, now you will play the game again."
                await send_feedback(purple_agent_role, transition_message)

            building_task = BuildingGameTask(list1_path, list2_path, seed=seed)
            trials = building_task.run(None)
            logger.info(f"Created trials for seed {seed}: {trials}")

            results = {}
            num_correct = 0
            scored_count = 0
            questions_count = 0
            total_score = 0  # Track cumulative score

            task_description = f"[TASK_DESCRIPTION] {trials['grid_context']})"

            for speaker in [trials["instructions_A"], trials["instructions_B"]]:
                prompt_chain = []
                response_chain = []
                for instruction in speaker:
                    round_questions_count = 0
                    round_score = 0  # Track score for this round
                    # MODIFIED: Include speaker name in the prompt
                    speaker_name = instruction['speaker']
                    prompt = f"{task_description}\n[SPEAKER] {speaker_name}\n[START_STRUCTURE] {instruction['start_structure']}\n{instruction['instruction']}"
                    prompt_chain.append(prompt)
                    built = False
                    eval_result = {}
                    while built is not True:
                        instruction_response = await turn(purple_agent_role, prompt)
                        response_chain.append(instruction_response)
                        eval_result = await self.eval_message(
                            instruction_response,
                            instruction["target_structure"],
                        )
                        round_questions_count += eval_result["num_questions"]
                        round_score += eval_result.get("points", 0)  # Accumulate points
                        prompt = eval_result['message']
                        built = eval_result['built']

                    total_score += round_score  # Add round score to total

                    # Include score in feedback
                    feedback_msg = f"Feedback: {eval_result['message']} | Round score: {round_score:+d} | Total score: {total_score:+d}"
                    await send_feedback(purple_agent_role, feedback_msg)

                    if eval_result["num_correct"] is not None:
                        scored_count += 1
                        num_correct += eval_result["num_correct"]
                    questions_count += round_questions_count
                    results[instruction["round"]] = {
                        "prompts": prompt_chain,
                        "responses": response_chain,
                        "eval_feedback_message": eval_result["message"],
                        "num_correct": eval_result["num_correct"],
                        "num_questions": round_questions_count,
                        "response_feedback": None,
                        # MODIFIED: Store speaker information in results
                        "speaker": speaker_name,
                        "round_score": round_score  # Store round score
                    }

            # Calculate metrics for this seed
            accuracy = (num_correct / scored_count * 100.0) if scored_count else 0.0
            avg_questions = questions_count / len(trials["instructions_A"] + trials["instructions_B"])

            all_accuracies.append(accuracy)
            all_avg_questions.append(avg_questions)
            all_results[f"seed_{seed}"] = {
                "accuracy": accuracy,
                "avg_questions_per_instruction": avg_questions,
                "total_score": total_score,  # Add total score for this seed
                "results": results
            }

            logger.info(
                f"Seed {seed} - Accuracy: {accuracy:.2f}%, Avg Questions: {avg_questions:.2f}, Total Score: {total_score}")

        # Calculate overall averages
        overall_accuracy = sum(all_accuracies) / len(all_accuracies) if all_accuracies else 0.0
        overall_avg_questions = sum(all_avg_questions) / len(all_avg_questions) if all_avg_questions else 0.0

        # Calculate average score across all seeds
        all_scores = [all_results[f"seed_{seed}"]["total_score"] for seed in range(num_seeds)]
        overall_avg_score = sum(all_scores) / len(all_scores) if all_scores else 0.0

        logger.info(
            f"Overall - Accuracy: {overall_accuracy:.2f}%, Avg Questions: {overall_avg_questions:.2f}, Avg Score: {overall_avg_score:.2f}")

        try:
            result = EvalResult(
                accuracy=overall_accuracy,
                avg_questions_per_instruction=overall_avg_questions,
                overall_avg_score=overall_avg_score
            )
            await updater.add_artifact(
                parts=[
                    Part(root=TextPart(text=result.model_dump_json())),
                ],
                name="Result",
            )
            # Also add detailed results for each seed
            # TODO: decide how to present these results.
            # import json
            # detailed_results = {
            #     "overall_accuracy": overall_accuracy,
            #     "overall_avg_questions": overall_avg_questions,
            #     "overall_avg_score": overall_avg_score,  # Add average score
            #     "individual_seeds": all_results
            # }
            # await updater.add_artifact(
            #     parts=[
            #         Part(root=TextPart(text=json.dumps(detailed_results, indent=2))),
            #     ],
            #     name="Detailed_Results",
            # )
        finally:
            self._tool_provider.reset()
        return result

    def validate_request(self, request: EvalRequest) -> tuple[bool, str]:
        return True, "ok"

    async def eval_message(self, response: str, target_structure: str):
        # Strip whitespace
        response = response.strip()

        # Extract action type and content
        if response.startswith("[BUILD]"):
            action = "[BUILD]"
            content_str = response[7:]  # Everything after "[BUILD]"
        elif response.startswith("[ASK]"):
            action = "[ASK]"
            content_str = response[5:]  # Everything after "[ASK]"
        else:
            # Invalid response format - give Rita a chance to retry
            points = 0
            logger.warning(f"Invalid response format (not [BUILD] or [ASK]): {response[:100]}")
            return {
                "message": f"Invalid response format. Expected [BUILD] or [ASK], but got: {response[:50]}...",
                "num_correct": 0,
                "num_questions": 0,
                "built": False,  # Allow retry
                "points": points
            }

        match action:
            case "[BUILD]":
                # Remove leading semicolon if present
                if content_str.startswith(";"):
                    content_str = content_str[1:]

                # Parse coordinates
                coords = [c.strip() for c in content_str.split(";") if c.strip()]
                content = self._normalize_structure(coords)
                target_structure_set = self._normalize_structure(target_structure.split(";"))

                if content == target_structure_set:
                    points = 10
                    return {"message": f"Correct structure built! +{points} points. {target_structure}",
                            "num_correct": 1,
                            "num_questions": 0,
                            "built": True,
                            "points": points
                            }
                else:
                    points = -10
                    return {"message": f"Incorrect structure. {points} points. Expected: {target_structure}, but got: {';'.join(content)}",
                            "num_correct": 0,
                            "num_questions": 0,
                            "built": True,
                            "points": points
                            }

            case "[ASK]":
                # Take everything after [ASK] as the question (with or without semicolon)
                question = content_str.lstrip(";").strip()

                if self._qa:
                    answer = await self._qa.answer(
                        question=question,
                        target_structure=target_structure,
                    )
                else:
                    answer = self._fallback_answer(question, target_structure)
                points = -5
                return {"message": f"Answer: {answer} ({points} points for asking)",
                        "num_correct": None,
                        "num_questions": 1,
                        "built": False,
                        "points": points}

            case _:
                # Fallback case (should not reach here, but keeping for safety)
                points = -10
                return {"message": f"Invalid response format. {points} points. Expected [BUILD] or [ASK]. Moving to next instruction.",
                        "num_correct": 0,
                        "num_questions": 0,
                        "built": True,
                        "points": points
                        }


    @staticmethod
    def _fallback_answer(question: str, target_structure: str) -> str:
        colors = []
        for block in target_structure.split(";"):
            if block:
                colors.append(block.split(",", 1)[0])
        unique_colors = sorted(set(colors))
        if "color" in question.lower() and unique_colors:
            return f"Colors in target: {', '.join(unique_colors)}."
        return "I can answer questions about the target structure."


    @staticmethod
    def _normalize_structure(items) -> set[str]:
        # Normalize color casing and strip empty/invalid entries for stable comparisons.
        normalized = set()
        for item in items:
            item = item.strip()
            if not item:
                continue
            parts = item.split(",")
            if len(parts) != 4:
                continue
            color = parts[0].strip().capitalize()
            coords = [p.strip() for p in parts[1:]]
            normalized.add(",".join([color, *coords]))
        return normalized


    @staticmethod
    def _resolve_path(path_str: str) -> str:
        path = Path(path_str)
        if path.is_absolute() or path.exists():
            return str(path)
        repo_root = Path(__file__).resolve().parent.parent
        candidate = repo_root / path
        return str(candidate)