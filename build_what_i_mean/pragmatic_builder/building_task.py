from __future__ import annotations

import csv
import random
from typing import Any, Dict, List

NAMES = [
    'Anna', 'Emma', 'Lena', 'Sarah', 'Julia',
    'Marie', 'Laura', 'Nina', 'Sophie', 'Clara'
]


class BuildingGameTask:
    """Task for generating building game instructions with two speakers."""

    def __init__(self, list1_path: str, list2_path: str, seed: int | None = None) -> None:
        self.list1_path = list1_path
        self.list2_path = list2_path
        self.list1_data = self._load_csv(list1_path)
        self.list2_data = self._load_csv(list2_path)
        self.rng = random.Random(seed)

        # Fixed orderings for each speaker
        self.LisaOrdering = [
            "fully_spec", "fully_spec", "critical_a", "critical_b", "fully_spec",
            "critical_a", "critical_a", "fully_spec", "critical_b", "critical_a",
            "critical_b", "fully_spec", "critical_a", "fully_spec", "critical_a",
            "fully_spec", "critical_a", "fully_spec", "critical_b", "critical_a"
        ]
        self.PiaOrdering = [
            "fully_spec", "fully_spec", "critical", "critical", "fully_spec",
            "critical", "critical", "fully_spec", "critical", "critical",
            "fully_spec", "critical", "critical", "critical", "fully_spec",
            "critical", "critical", "fully_spec", "critical", "fully_spec"
        ]

    def _load_csv(self, path: str) -> List[Dict[str, str]]:
        """Load CSV file and return list of dictionaries."""
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            return list(reader)

    def get_ground_truth(self, list_id: int, trial_id: str) -> Dict[str, str] | None:
        """Return the raw CSV row for a given trial ID."""
        if list_id not in (1, 2):
            return None
        data = self.list1_data if list_id == 1 else self.list2_data
        return self._get_instruction_data(trial_id, data)

    def _get_instruction_data(self, trial_number: str, data: List[Dict[str, str]]) -> Dict[str, str] | None:
        """Get instruction data for a specific trial number."""
        for row in data:
            if row['trialNumber'] == trial_number:
                return row
        return None

    def _categorize_trials(self, data: List[Dict[str, str]]) -> Dict[str, List[str]]:
        """Categorize trials by type (fully_spec, color_under, number_under)."""
        categories = {
            'fully_spec': [],
            'color_under': [],
            'number_under': []
        }

        seen_base_numbers = set()

        for row in data:
            trial_num = row['trialNumber']
            # Extract base number (remove 'a' or 'b' suffix)
            if trial_num[-1] in ['a', 'b']:
                base_num = trial_num[:-1]
            else:
                base_num = trial_num

            # Only process each base number once
            if base_num in seen_base_numbers:
                continue
            seen_base_numbers.add(base_num)

            # Categorize based on trial type in the row
            trial_type = row.get('trialType', '')
            if trial_type == 'fully_spec':
                categories['fully_spec'].append(base_num)
            elif trial_type == 'color_under':
                categories['color_under'].append(base_num)
            elif trial_type == 'number_under':
                categories['number_under'].append(base_num)

        return categories

    def run(self, payload: Any) -> Dict[str, Any]:
        """Generate building instructions with fixed orderings."""
        if payload is None:
            payload = {}

        if not isinstance(payload, dict):
            raise ValueError("Building game input must be a dictionary or None")

        # Step 1: Pick two speaker names and randomly assign to first/second position
        names = self.rng.sample(NAMES, 2)
        role_a_speaker = names[0]  # uses PiaOrdering (critical = 'b' version)
        role_b_speaker = names[1]  # uses LisaOrdering (critical_a / critical_b)
        first_speaker = self.rng.choice([role_a_speaker, role_b_speaker])
        second_speaker = role_b_speaker if first_speaker == role_a_speaker else role_a_speaker

        # Step 2: Randomly choose list for fully_spec trials
        fully_spec_list = self.rng.choice([1, 2])
        fully_spec_data = self.list1_data if fully_spec_list == 1 else self.list2_data

        # Step 3: Randomly choose list for color_under and number_under trials (critical trials)
        underspec_list = self.rng.choice([1, 2])
        underspec_data = self.list1_data if underspec_list == 1 else self.list2_data

        # Categorize trials from both lists
        fully_spec_categories = self._categorize_trials(fully_spec_data)
        underspec_categories = self._categorize_trials(underspec_data)

        # Get other list for second speaker
        other_fully_spec_list = 2 if fully_spec_list == 1 else 1
        other_fully_spec_data = self.list2_data if other_fully_spec_list == 2 else self.list1_data
        other_fully_spec_categories = self._categorize_trials(other_fully_spec_data)

        other_underspec_list = 2 if underspec_list == 1 else 1
        other_underspec_data = self.list2_data if other_underspec_list == 2 else self.list1_data
        other_underspec_categories = self._categorize_trials(other_underspec_data)

        # Prepare pools of trials for each speaker
        # First speaker gets trials from the randomly selected lists
        first_fully_spec_pool = fully_spec_categories['fully_spec'][:]
        first_critical_pool = underspec_categories['color_under'][:] + underspec_categories['number_under'][:]

        # Second speaker gets trials from the other lists
        second_fully_spec_pool = other_fully_spec_categories['fully_spec'][:]
        second_critical_pool = other_underspec_categories['color_under'][:] + other_underspec_categories[
                                                                                  'number_under'][:]

        # Randomize the pools
        self.rng.shuffle(first_fully_spec_pool)
        self.rng.shuffle(first_critical_pool)
        self.rng.shuffle(second_fully_spec_pool)
        self.rng.shuffle(second_critical_pool)

        # Helper function to create instruction with specific version
        def create_instruction_with_version(trial_base: str, speaker: str, list_id: int,
                                            version: str = '') -> Dict[str, Any] | None:
            data = self.list1_data if list_id == 1 else self.list2_data

            # Get trial data
            if version:
                trial_with_version = f"{trial_base}{version}"
                trial_data = self._get_instruction_data(trial_with_version, data)
            else:
                trial_data = self._get_instruction_data(trial_base, data)

            # If versioned trial doesn't exist, try base number
            if trial_data is None and version:
                trial_data = self._get_instruction_data(trial_base, data)

            if trial_data is None:
                return None

            return {
                "speaker": speaker,
                "start_structure": trial_data['startStructure'],
                "instruction": trial_data['sentenceW'],
                "trial_id": trial_data["trialNumber"],
                "list_id": list_id,
                "target_structure": trial_data["targetStructure"],
                "trial_type": trial_data.get('trialType', '')
            }

        # Generate instructions based on ordering
        def generate_instructions_for_speaker(speaker: str, fully_spec_pool: List[str],
                                              critical_pool: List[str],
                                              fully_spec_list_id: int,
                                              critical_list_id: int) -> List[Dict[str, Any]]:
            instructions = []
            ordering = self.LisaOrdering if speaker == role_b_speaker else self.PiaOrdering

            fully_spec_idx = 0
            critical_idx = 0

            for order_item in ordering:
                if order_item == "fully_spec":
                    if fully_spec_idx < len(fully_spec_pool):
                        trial_base = fully_spec_pool[fully_spec_idx]
                        instr = create_instruction_with_version(trial_base, speaker,
                                                                fully_spec_list_id, '')
                        if instr:
                            instructions.append(instr)
                        fully_spec_idx += 1

                elif order_item == "critical":
                    # For Pia, always use 'b' version (symmetric/consistent with context)
                    if critical_idx < len(critical_pool):
                        trial_base = critical_pool[critical_idx]
                        instr = create_instruction_with_version(trial_base, speaker,
                                                                critical_list_id, 'b')
                        if instr:
                            instructions.append(instr)
                        critical_idx += 1

                elif order_item == "critical_a":
                    # For Lisa, use 'a' version
                    if critical_idx < len(critical_pool):
                        trial_base = critical_pool[critical_idx]
                        instr = create_instruction_with_version(trial_base, speaker,
                                                                critical_list_id, 'a')
                        if instr:
                            instructions.append(instr)
                        critical_idx += 1

                elif order_item == "critical_b":
                    # For Lisa, use 'b' version
                    if critical_idx < len(critical_pool):
                        trial_base = critical_pool[critical_idx]
                        instr = create_instruction_with_version(trial_base, speaker,
                                                                critical_list_id, 'b')
                        if instr:
                            instructions.append(instr)
                        critical_idx += 1

            return instructions

        # Generate instructions for both speakers
        if first_speaker == role_a_speaker:
            instructions_A = generate_instructions_for_speaker(
                role_a_speaker, first_fully_spec_pool, first_critical_pool,
                fully_spec_list, underspec_list
            )
            instructions_B = generate_instructions_for_speaker(
                role_b_speaker, second_fully_spec_pool, second_critical_pool,
                other_fully_spec_list, other_underspec_list
            )
        else:
            instructions_A = generate_instructions_for_speaker(
                role_b_speaker, first_fully_spec_pool, first_critical_pool,
                fully_spec_list, underspec_list
            )
            instructions_B = generate_instructions_for_speaker(
                role_a_speaker, second_fully_spec_pool, second_critical_pool,
                other_fully_spec_list, other_underspec_list
            )

        # Add round numbers
        for i, instr in enumerate(instructions_A):
            instr['round'] = i + 1

        for i, instr in enumerate(instructions_B):
            instr['round'] = i + len(instructions_A) + 1

        grid_context = (
            "Grid: 9x9 cells. Origin=\"middle square\": center (0,0), is highlighted. "
            "The grid is the x–z plane. In front of you is the bottom left corner "
            "(-400,0,400) and the bottom right corner (400,0,400). Top right corner "
            "is (400,0,-400), top left corner is (-400,0,-400). Valid x,z: "
            "[-400,-300,-200,-100,0,100,200,300,400]. Y(ground)=50; each extra block "
            "adds +100; valid y values are [50,150,250,350,450]. The grid may or may "
            "not contain an existing structure. The grid might be empty. Output: "
            "\"Coordinates:Color,x,y,z;Color,x,y,z;\" items separated by \";\"; no spaces; "
            "write coordinates of all blocks that are on the grid, including the initial "
            "coordinates; color should be capitalized. Only one question is allowed. "
            "Scoring: Building the correct structure earns +10 points. Asking a question "
            "costs -5 points. Building an incorrect structure costs -10 points."
        )

        return {
            "type": "building_game",
            "grid_context": grid_context,
            "chosen_list": "mixed",
            "first_speaker": first_speaker,
            "second_speaker": second_speaker,
            "instructions_A": instructions_A,
            "instructions_B": instructions_B
        }