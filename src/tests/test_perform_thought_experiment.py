# File: tests/test_ethicist_ai.py

import unittest
from unittest.mock import patch, MagicMock
from HappyChoicesAI.perform_thought_experiment import (
    analyze_parallels,
    analyze_criteria_changes,
    analyze_percentage_changes,
    analyze_proxies_impact,
    quantify_proxies,
    summarize_thought_experiment,
)  # Adjust the import as needed
from HappyChoicesAI.ai_state import EthicistAIState, HistoricalExample


class TestEthicistAI(unittest.TestCase):

    @patch("HappyChoicesAI.perform_thought_experiment.ChatPromptTemplate.from_template")
    @patch("HappyChoicesAI.perform_thought_experiment.llm")
    def test_analyze_parallels(self, mock_llm, mock_prompt_template):
        # Arrange
        mock_output = MagicMock()
        mock_output.choices = [MagicMock(text="Mocked Criteria for Parallels")]

        # Mock the chain object and its invoke method
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_output

        # Mock the prompt_template | llm to return our mock chain
        mock_prompt_template.return_value.__or__.return_value = mock_chain

        state = EthicistAIState(
            situation="Test Situation",
            historical_examples=[
                HistoricalExample(
                    situation="Test Historical Examples",
                    action_taken="Test Action Taken",
                    reasoning="Test Reasoning",
                )
            ],
        )
        proposed_action = "Test Proposed Action"

        # Act
        result = analyze_parallels(state, proposed_action)

        # Assert
        self.assertEqual(result, "Mocked Criteria for Parallels")

    @patch("HappyChoicesAI.perform_thought_experiment.ChatPromptTemplate.from_template")
    @patch("HappyChoicesAI.perform_thought_experiment.llm")
    def test_analyze_criteria_changes(self, mock_llm, mock_prompt_template):
        # Arrange
        mock_output = MagicMock()
        mock_output.choices = [MagicMock(text="Mocked Criteria for Criteria Changes")]

        # Mock the chain object and its invoke method
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_output

        # Mock the prompt_template | llm to return our mock chain
        mock_prompt_template.return_value.__or__.return_value = mock_chain

        state = EthicistAIState(situation="Test Situation", criteria="Test Criteria")
        proposed_action = "Test Proposed Action"

        # Act
        result = analyze_criteria_changes(state, proposed_action)

        # Assert
        self.assertEqual(result, "Mocked Criteria for Criteria Changes")

    @patch("HappyChoicesAI.perform_thought_experiment.ChatPromptTemplate.from_template")
    @patch("HappyChoicesAI.perform_thought_experiment.llm")
    def test_analyze_percentage_changes(self, mock_llm, mock_prompt_template):
        # Arrange
        mock_output = MagicMock()
        mock_output.choices = [MagicMock(text="Mocked Criteria for Percentage Changes")]

        # Mock the chain object and its invoke method
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_output

        # Mock the prompt_template | llm to return our mock chain
        mock_prompt_template.return_value.__or__.return_value = mock_chain

        state = EthicistAIState(situation="Test Situation", criteria="Test Criteria")
        proposed_action = "Test Proposed Action"
        criteria_changes = "Test Criteria Changes"

        # Act
        result = analyze_percentage_changes(state, proposed_action, criteria_changes)

        # Assert
        self.assertEqual(result, "Mocked Criteria for Percentage Changes")

    @patch("HappyChoicesAI.perform_thought_experiment.ChatPromptTemplate.from_template")
    @patch("HappyChoicesAI.perform_thought_experiment.llm")
    def test_analyze_proxies_impact(self, mock_llm, mock_prompt_template):
        # Arrange
        mock_output = MagicMock()
        mock_output.choices = [MagicMock(text="Mocked Criteria for Proxies Impact..")]

        # Mock the chain object and its invoke method
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_output

        # Mock the prompt_template | llm to return our mock chain
        mock_prompt_template.return_value.__or__.return_value = mock_chain

        state = EthicistAIState(situation="Test Situation", criteria="Test Criteria")
        proposed_action = "Test Proposed Action"
        criteria_changes = "Test Criteria Changes"

        # Act
        result = analyze_proxies_impact(state, proposed_action, criteria_changes)

        # Assert
        self.assertEqual(result, "Mocked Criteria for Proxies Impact..")

    @patch("HappyChoicesAI.perform_thought_experiment.ChatPromptTemplate.from_template")
    @patch("HappyChoicesAI.perform_thought_experiment.llm")
    def test_quantify_proxies(self, mock_llm, mock_prompt_template):
        # Arrange
        mock_output = MagicMock()
        mock_output.choices = [MagicMock(text="Mocked Criteria for Quantified Proxies")]

        # Mock the chain object and its invoke method
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_output

        # Mock the prompt_template | llm to return our mock chain
        mock_prompt_template.return_value.__or__.return_value = mock_chain

        state = EthicistAIState(situation="Test Situation", criteria="Test Criteria")
        proposed_action = "Test Proposed Action"
        proxies_impact = "Test Proxies Impact"

        # Act
        result = quantify_proxies(state, proposed_action, proxies_impact)

        # Assert
        self.assertEqual(result, "Mocked Criteria for Quantified Proxies")

    @patch("HappyChoicesAI.perform_thought_experiment.ChatPromptTemplate.from_template")
    @patch("HappyChoicesAI.perform_thought_experiment.llm")
    def test_summarize_thought_experiment(self, mock_llm, mock_prompt_template):
        # Arrange
        mock_output = MagicMock()
        mock_output.choices = [MagicMock(text="Mocked Criteria for Thought Experiment")]

        # Mock the chain object and its invoke method
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_output

        # Mock the prompt_template | llm to return our mock chain
        mock_prompt_template.return_value.__or__.return_value = mock_chain

        state = EthicistAIState(situation="Test Situation", criteria="Test Criteria")
        proposed_action = "Test Proposed Action"
        parallels = "Test Parallels"
        criteria_changes = "Test Criteria Changes"
        percentage_changes = "Test Percentage Changes"
        proxies_impact = "Test Proxies Impact"
        quantified_proxies = "Test Quantified Proxies"

        # Act
        result = summarize_thought_experiment(
            proposed_action,
            parallels,
            criteria_changes,
            percentage_changes,
            proxies_impact,
            quantified_proxies,
            state,
        )

        # Assert
        self.assertEqual(result, "Mocked Criteria for Thought Experiment")


if __name__ == "__main__":
    unittest.main()
