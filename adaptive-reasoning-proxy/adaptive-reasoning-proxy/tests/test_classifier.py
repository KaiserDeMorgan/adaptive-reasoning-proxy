"""Tests for the regex task router."""

import pytest

from classifier import THRESHOLDS, TaskType, classify


class TestClassify:
    @pytest.mark.parametrize(
        "prompt,expected",
        [
            ("What is the capital of France?", TaskType.FACTUAL),
            ("Who is the president of Brazil?", TaskType.FACTUAL),
            ("Define entropy in thermodynamics", TaskType.FACTUAL),
            ("List the planets in the solar system", TaskType.FACTUAL),
            ("Why does ice float on water?", TaskType.REASONING),
            ("Explain how a transformer model works", TaskType.REASONING),
            ("Prove that the square root of 2 is irrational", TaskType.REASONING),
            ("Compare REST and GraphQL", TaskType.REASONING),
            ("Write a function to reverse a linked list", TaskType.CODE),
            ("Implement binary search in Python", TaskType.CODE),
            ("Create a class for a bank account", TaskType.CODE),
            ("Write a script that renames files", TaskType.CODE),
        ],
    )
    def test_classifies_each_task_type(self, prompt, expected):
        assert classify(prompt) == expected

    def test_unmatched_prompt_defaults_to_creative(self):
        assert classify("Once upon a time in a distant galaxy") == TaskType.CREATIVE

    def test_code_takes_priority_over_reasoning(self):
        # Contains both "explain" (reason) and "function" (code); code wins.
        assert classify("Explain and write a function for quicksort") == TaskType.CODE

    def test_classification_is_case_insensitive(self):
        assert classify("WHAT IS gravity") == TaskType.FACTUAL

    def test_empty_prompt_defaults_to_creative(self):
        assert classify("") == TaskType.CREATIVE


class TestThresholds:
    def test_every_task_type_has_a_threshold(self):
        for task in TaskType:
            assert task in THRESHOLDS

    def test_thresholds_are_in_valid_range(self):
        for value in THRESHOLDS.values():
            assert 0.0 < value < 1.0

    def test_reasoning_threshold_is_highest(self):
        # Reasoning tasks should be allowed to run longest (highest threshold).
        assert THRESHOLDS[TaskType.REASONING] == max(THRESHOLDS.values())

    def test_factual_threshold_is_lowest(self):
        # Factual tasks settle fastest, so they get the lowest threshold.
        assert THRESHOLDS[TaskType.FACTUAL] == min(THRESHOLDS.values())
