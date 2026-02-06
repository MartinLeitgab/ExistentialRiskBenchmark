"""
Unit tests for prompt generation utilities.
"""

import logging
import pytest
from utils.prompt_generator import (
    generate_prompt,
    generate_all_variants,
    generate_prompt_id,
    BASE_DIMENSIONS,
)

# Configure logger for tests
logger = logging.getLogger(__name__)


class TestPromptGeneration:
    """Test prompt generation functions."""

    def test_generate_prompt_basic(self, test_scenarios):
        """Test basic prompt generation."""
        logger.info("=" * 60)
        logger.info("TEST: Basic prompt generation")
        logger.info("=" * 60)

        scenario = test_scenarios[0]
        logger.info("Input Context: %s", scenario['context'])
        logger.info("Input Action A: %s", scenario['action_a'])
        logger.info("Input Action B: %s", scenario['action_b'])
        logger.info("Dimensions: %s", BASE_DIMENSIONS)

        result = generate_prompt(
            context=scenario["context"],
            action_a=scenario["action_a"],
            action_b=scenario["action_b"],
            dimensions=BASE_DIMENSIONS
        )

        logger.info("")
        logger.info("--- OUTPUT ---")
        logger.info("User Prompt:")
        logger.info("%s", result['user_prompt'])
        logger.info("")
        logger.info("System Prompt:")
        logger.info("%s", result['system_prompt'])
        logger.info("")
        logger.info("Metadata: %s", result['metadata'])

        assert "user_prompt" in result
        assert "system_prompt" in result
        assert "metadata" in result
        assert scenario["context"] in result["user_prompt"]
        assert scenario["action_a"] in result["user_prompt"]
        assert scenario["action_b"] in result["user_prompt"]
        logger.info("Test passed: Basic prompt generation")

    def test_generate_prompt_with_variants(self, test_scenarios, test_variants):
        """Test prompt generation with different variants."""
        logger.info("=" * 60)
        logger.info("TEST: Prompt generation with variants")
        logger.info("=" * 60)

        scenario = test_scenarios[0]
        logger.info("Total variants to test: %d", len(test_variants))

        for i, variant in enumerate(test_variants):
            logger.info("")
            logger.info("--- Variant %d/%d ---", i+1, len(test_variants))
            logger.info("Variant ID: %s", variant.get('variant_id', 'unknown'))
            logger.info("Variant Dimensions: %s", variant['dimensions'])

            result = generate_prompt(
                context=scenario["context"],
                action_a=scenario["action_a"],
                action_b=scenario["action_b"],
                dimensions=variant["dimensions"]
            )

            logger.info("Generated Prompt ID: %s", result['metadata']['prompt_id'])
            logger.info("User Prompt Preview (first 200 chars): %s...", result['user_prompt'][:200])

            assert result["metadata"]["prompt_id"] == variant["variant_id"]
            assert result["user_prompt"] is not None
            assert len(result["user_prompt"]) > 0

        logger.info("")
        logger.info("Test passed: All %d variants generated successfully", len(test_variants))

    def test_action_order_inversion(self, test_scenarios):
        """Test that action_order=inverted swaps actions."""
        logger.info("=" * 60)
        logger.info("TEST: Action order inversion")
        logger.info("=" * 60)

        scenario = test_scenarios[0]
        logger.info("Original Action A: %s", scenario['action_a'])
        logger.info("Original Action B: %s", scenario['action_b'])

        # Regular order
        regular_dims = BASE_DIMENSIONS.copy()
        regular_dims["action_order"] = "regular"
        logger.info("")
        logger.info("--- Regular Order ---")
        logger.info("Dimensions: %s", regular_dims)

        regular_result = generate_prompt(
            context=scenario["context"],
            action_a=scenario["action_a"],
            action_b=scenario["action_b"],
            dimensions=regular_dims
        )

        logger.info("Action Mapping: %s", regular_result['metadata']['action_mapping'])
        logger.info("Prompt Preview: %s...", regular_result['user_prompt'][:150])

        # Inverted order
        inverted_dims = BASE_DIMENSIONS.copy()
        inverted_dims["action_order"] = "inverted"
        logger.info("")
        logger.info("--- Inverted Order ---")
        logger.info("Dimensions: %s", inverted_dims)

        inverted_result = generate_prompt(
            context=scenario["context"],
            action_a=scenario["action_a"],
            action_b=scenario["action_b"],
            dimensions=inverted_dims
        )

        mapping = inverted_result["metadata"]["action_mapping"]
        logger.info("Action Mapping: %s", mapping)
        logger.info("Prompt Preview: %s...", inverted_result['user_prompt'][:150])

        # Check that actions are swapped
        logger.info("")
        logger.info("--- Verification ---")
        logger.info("Original A maps to: %s (expected: B)", mapping['original_A'])
        logger.info("Original B maps to: %s (expected: A)", mapping['original_B'])

        assert mapping["original_A"] == "B"
        assert mapping["original_B"] == "A"
        logger.info("Test passed: Actions correctly swapped in inverted order")

    @pytest.mark.parametrize("dimension,expected_min_variants", [
        ("framing", 3),
        ("response_format", 3),
        ("ethical_framing", 4),
    ])
    def test_generate_all_variants(self, dimension, expected_min_variants):
        """Test variant generation for each dimension."""
        logger.info("=" * 60)
        logger.info("TEST: Generate all variants for dimension '%s'", dimension)
        logger.info("=" * 60)

        variants = generate_all_variants(vary_dim=dimension)
        actual_count = len(variants)

        logger.info("Expected minimum: %d", expected_min_variants)
        logger.info("Actual generated: %d", actual_count)

        logger.info("")
        logger.info("--- Generated Variants ---")
        for i, variant in enumerate(variants):
            logger.info("")
            logger.info("Variant %d:", i+1)
            logger.info("  Prompt ID: %s", variant.get('prompt_id', 'N/A'))
            logger.info("  Vary Dimension: %s", variant.get('vary_dim', 'N/A'))
            logger.info("  Dimensions: %s", variant.get('dimensions', {}))

        assert actual_count >= expected_min_variants

        for i, variant in enumerate(variants):
            assert "prompt_id" in variant
            assert "dimensions" in variant
            assert "vary_dim" in variant
            assert variant["vary_dim"] == dimension

        logger.info("")
        logger.info("Test passed: Generated %d variants for '%s'", actual_count, dimension)

    def test_generate_prompt_id_consistency(self):
        """Test that same dimensions produce same ID."""
        logger.info("=" * 60)
        logger.info("TEST: Prompt ID consistency")
        logger.info("=" * 60)

        dims = BASE_DIMENSIONS.copy()
        logger.info("Dimensions used: %s", dims)

        id1 = generate_prompt_id(dims)
        id2 = generate_prompt_id(dims)

        logger.info("")
        logger.info("--- Results ---")
        logger.info("First ID generation:  %s", id1)
        logger.info("Second ID generation: %s", id2)
        logger.info("IDs match: %s", id1 == id2)

        assert id1 == id2
        logger.info("Test passed: Same dimensions produce identical IDs")

    def test_generate_prompt_id_uniqueness(self):
        """Test that different dimensions produce different IDs."""
        logger.info("=" * 60)
        logger.info("TEST: Prompt ID uniqueness")
        logger.info("=" * 60)

        dims1 = BASE_DIMENSIONS.copy()
        dims2 = BASE_DIMENSIONS.copy()
        dims2["framing"] = "third_person"

        logger.info("Dimensions Set 1: %s", dims1)
        logger.info("Dimensions Set 2: %s", dims2)

        id1 = generate_prompt_id(dims1)
        id2 = generate_prompt_id(dims2)

        logger.info("")
        logger.info("--- Results ---")
        logger.info("ID for regular framing:      %s", id1)
        logger.info("ID for third_person framing: %s", id2)
        logger.info("IDs differ: %s", id1 != id2)

        assert id1 != id2
        logger.info("Test passed: Different dimensions produce unique IDs")


class TestPromptContent:
    """Test prompt content generation."""

    def test_framing_first_person(self, test_scenarios):
        """Test first-person framing."""
        logger.info("=" * 60)
        logger.info("TEST: First-person framing")
        logger.info("=" * 60)

        dims = BASE_DIMENSIONS.copy()
        dims["framing"] = "first_person"
        logger.info("Dimensions: %s", dims)

        result = generate_prompt(
            context=test_scenarios[0]["context"],
            action_a=test_scenarios[0]["action_a"],
            action_b=test_scenarios[0]["action_b"],
            dimensions=dims
        )

        prompt_lower = result["user_prompt"].lower()
        has_first_person = "you" in prompt_lower

        logger.info("")
        logger.info("--- Generated Prompt ---")
        logger.info("%s", result['user_prompt'])
        logger.info("")
        logger.info("Contains 'you': %s", has_first_person)

        assert has_first_person
        logger.info("Test passed: First-person framing detected")

    def test_framing_third_person(self, test_scenarios):
        """Test third-person framing."""
        logger.info("=" * 60)
        logger.info("TEST: Third-person framing")
        logger.info("=" * 60)

        dims = BASE_DIMENSIONS.copy()
        dims["framing"] = "third_person"
        logger.info("Dimensions: %s", dims)

        result = generate_prompt(
            context=test_scenarios[0]["context"],
            action_a=test_scenarios[0]["action_a"],
            action_b=test_scenarios[0]["action_b"],
            dimensions=dims
        )

        prompt_lower = result["user_prompt"].lower()
        has_third_person = "the ai system" in prompt_lower or "ai system" in prompt_lower

        logger.info("")
        logger.info("--- Generated Prompt ---")
        logger.info("%s", result['user_prompt'])
        logger.info("")
        logger.info("Contains 'the ai system' or 'ai system': %s", has_third_person)

        assert has_third_person
        logger.info("Test passed: Third-person framing detected")

    def test_ethical_framing_present(self, test_scenarios):
        """Test that ethical framing is included when specified."""
        logger.info("=" * 60)
        logger.info("TEST: Ethical framing (consequentialist)")
        logger.info("=" * 60)

        dims = BASE_DIMENSIONS.copy()
        dims["ethical_framing"] = "consequentialist"
        logger.info("Dimensions: %s", dims)

        result = generate_prompt(
            context=test_scenarios[0]["context"],
            action_a=test_scenarios[0]["action_a"],
            action_b=test_scenarios[0]["action_b"],
            dimensions=dims
        )

        prompt_lower = result["user_prompt"].lower()
        has_consequentialist = "consequences" in prompt_lower or "outcomes" in prompt_lower

        logger.info("")
        logger.info("--- Generated Prompt ---")
        logger.info("%s", result['user_prompt'])
        logger.info("")
        logger.info("Contains 'consequences' or 'outcomes': %s", has_consequentialist)

        assert has_consequentialist
        logger.info("Test passed: Consequentialist ethical framing detected")


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_invalid_dimension_value(self):
        """Test handling of invalid dimension value."""
        logger.info("=" * 60)
        logger.info("TEST: Invalid dimension value handling")
        logger.info("=" * 60)

        from utils.prompt_generator import generate_all_variants

        invalid_dim = "invalid_dimension"
        logger.info("Attempting to generate variants with: '%s'", invalid_dim)

        try:
            with pytest.raises(ValueError) as exc_info:
                generate_all_variants(vary_dim=invalid_dim)
            logger.info("Caught expected ValueError: %s", exc_info.value)
            logger.info("Test passed: ValueError raised for invalid dimension")
        except AssertionError as e:
            logger.error("Test failed: Expected ValueError was not raised")
            raise

    def test_empty_context(self):
        """Test prompt generation with empty context."""
        logger.info("=" * 60)
        logger.info("TEST: Empty context handling")
        logger.info("=" * 60)

        logger.info("Input: context='', action_a='Action A', action_b='Action B'")

        result = generate_prompt(
            context="",
            action_a="Action A",
            action_b="Action B",
            dimensions=BASE_DIMENSIONS
        )

        logger.info("")
        logger.info("--- Generated Output ---")
        logger.info("User Prompt:")
        logger.info("%s", result['user_prompt'])
        logger.info("")
        logger.info("System Prompt:")
        logger.info("%s", result['system_prompt'])
        logger.info("")
        logger.info("Metadata: %s", result['metadata'])
        logger.info("User prompt length: %d chars", len(result['user_prompt']))

        # Should still generate valid prompt
        assert "user_prompt" in result
        assert len(result["user_prompt"]) > 0
        logger.info("Test passed: Valid prompt generated with empty context")