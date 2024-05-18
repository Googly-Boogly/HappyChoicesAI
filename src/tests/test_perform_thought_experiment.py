# File: tests/test_perform_thought_experiment.py

import unittest
from unittest.mock import patch, MagicMock
from HappyChoicesAI.perform_thought_experiment import (
    perform_thought_experiments,
    propose_all_actions,
    perform_thought_experiment_chain,
    analyze_parallels,
    analyze_criteria_changes,
    analyze_percentage_changes,
    analyze_proxies_impact,
    quantify_proxies,
    summarize_thought_experiment,
)

from HappyChoicesAI.ai_state import StateManager, EthicistAIState


class TestPerformThoughtExperiments(unittest.TestCase):

    def setUp(self):
        # Initialize the state
        state = StateManager.get_instance().state
        state.situation = "Test Situation"
        state.historical_examples = ["Historical Example 1", "Historical Example 2"]
        state.criteria = "Test Criteria"
        state.thought_experiments = []
        state.best_action = None

    def tearDown(self):
        # Reset the state for other tests
        state = StateManager.get_instance().state
        state.situation = ""
        state.historical_examples = []
        state.criteria = ""
        state.thought_experiments = []
        state.best_action = None

    @patch("HappyChoicesAI.perform_thought_experiment.create_prompt_template_propose_actions")
    @patch("HappyChoicesAI.perform_thought_experiment.llm")
    def test_propose_all_actions(self, mock_llm, mock_create_prompt_template):
        # Arrange
        mock_output = MagicMock()
        mock_output.content = '{"actions": ["action1", "action2", "action3"]}'

        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_output

        mock_create_prompt_template.return_value.__or__.return_value = mock_chain

        # Act
        result = propose_all_actions()

        # Assert
        self.assertEqual(result, ["action1", "action2", "action3"])

    @patch("HappyChoicesAI.perform_thought_experiment.llm")
    @patch("HappyChoicesAI.perform_thought_experiment.create_prompt_template_propose_actions")
    def test_propose_all_actions_malformed_json(self, mock_create_prompt_template, mock_llm):
        # Arrange
        mock_output = MagicMock()
        mock_output.content = '{"actions": ["action1", "action2", "action3"'

        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_output

        mock_create_prompt_template.return_value.__or__.return_value = mock_chain

        # Act
        result = propose_all_actions()

        # Assert
        self.assertEqual(result, [])

    @patch("HappyChoicesAI.perform_thought_experiment.llm")
    @patch("HappyChoicesAI.perform_thought_experiment.create_prompt_template_propose_actions")
    def test_propose_all_actions_json_without_actions_key(self, mock_create_prompt_template, mock_llm):
        # Arrange
        mock_output = MagicMock()
        mock_output.content = '{"not_actions": ["action1", "action2", "action3"]}'

        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_output

        mock_create_prompt_template.return_value.__or__.return_value = mock_chain

        # Act
        result = propose_all_actions()

        # Assert
        self.assertEqual(result, [])

    @patch("HappyChoicesAI.perform_thought_experiment.llm")
    @patch("HappyChoicesAI.perform_thought_experiment.create_prompt_template_propose_actions")
    def test_propose_all_actions_empty_output(self, mock_create_prompt_template, mock_llm):
        # Arrange
        mock_output = MagicMock()
        mock_output.content = ''

        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_output

        mock_create_prompt_template.return_value.__or__.return_value = mock_chain

        # Act
        result = propose_all_actions()

        # Assert
        self.assertEqual(result, [])

    @patch("HappyChoicesAI.perform_thought_experiment.analyze_parallels")
    @patch("HappyChoicesAI.perform_thought_experiment.analyze_criteria_changes")
    @patch("HappyChoicesAI.perform_thought_experiment.analyze_percentage_changes")
    @patch("HappyChoicesAI.perform_thought_experiment.analyze_proxies_impact")
    @patch("HappyChoicesAI.perform_thought_experiment.quantify_proxies")
    @patch("HappyChoicesAI.perform_thought_experiment.summarize_thought_experiment")
    def test_perform_thought_experiment_chain(
            self,
            mock_summarize,
            mock_quantify_proxies,
            mock_analyze_proxies,
            mock_analyze_percentage,
            mock_analyze_criteria,
            mock_analyze_parallels
    ):
        # Arrange
        mock_summarize.return_value = "Mocked Summary"
        mock_quantify_proxies.return_value = "Mocked Quantified Proxies"
        mock_analyze_proxies.return_value = "Mocked Proxies Impact"
        mock_analyze_percentage.return_value = "Mocked Percentage Changes"
        mock_analyze_criteria.return_value = "Mocked Criteria Changes"
        mock_analyze_parallels.return_value = "Mocked Parallels"

        proposed_action = "Test Proposed Action"
        state = StateManager.get_instance().state

        # Act
        result = perform_thought_experiment_chain(proposed_action)

        # Assert
        self.assertEqual(result, "Mocked Summary")
        self.assertEqual(len(state.thought_experiments), 1)
        self.assertEqual(state.thought_experiments[0]["proposed_action"], proposed_action)
        self.assertEqual(state.thought_experiments[0]["parallels"], "Mocked Parallels")
        self.assertEqual(state.thought_experiments[0]["criteria_changes"], "Mocked Criteria Changes")
        self.assertEqual(state.thought_experiments[0]["percentage_changes"], "Mocked Percentage Changes")
        self.assertEqual(state.thought_experiments[0]["proxies_impact"], "Mocked Proxies Impact")
        self.assertEqual(state.thought_experiments[0]["quantified_proxies"], "Mocked Quantified Proxies")
        self.assertEqual(state.thought_experiments[0]["summary"], "Mocked Summary")

    @patch("HappyChoicesAI.perform_thought_experiment.get_analyze_parallels_prompt")
    @patch("HappyChoicesAI.perform_thought_experiment.llm")
    def test_analyze_parallels(self, mock_llm, mock_get_analyze_parallels_prompt):
        # Arrange
        mock_output = MagicMock()
        mock_output.content = "Mocked Parallels"

        # Mock the chain object and its invoke method
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_output

        # Mock the prompt template to return our mock chain
        mock_get_analyze_parallels_prompt.return_value.__or__.return_value = mock_chain

        state = StateManager.get_instance().state
        proposed_action = "Test Proposed Action"

        # Act
        result = analyze_parallels(state, proposed_action)

        # Assert
        self.assertEqual(result, "Mocked Parallels")

    @patch("HappyChoicesAI.perform_thought_experiment.get_analyze_criteria_changes_prompt")
    @patch("HappyChoicesAI.perform_thought_experiment.llm")
    def test_analyze_criteria_changes(self, mock_llm, mock_get_analyze_criteria_changes_prompt):
        # Arrange
        mock_output = MagicMock()
        mock_output.content = "Mocked Criteria Changes"

        # Mock the chain object and its invoke method
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_output

        # Mock the prompt template to return our mock chain
        mock_get_analyze_criteria_changes_prompt.return_value.__or__.return_value = mock_chain

        state = StateManager.get_instance().state
        proposed_action = "Test Proposed Action"

        # Act
        result = analyze_criteria_changes(state, proposed_action)

        # Assert
        self.assertEqual(result, "Mocked Criteria Changes")

    @patch("HappyChoicesAI.perform_thought_experiment.get_analyze_percentage_changes_prompt")
    @patch("HappyChoicesAI.perform_thought_experiment.llm")
    def test_analyze_percentage_changes(self, mock_llm, mock_get_analyze_percentage_changes_prompt):
        # Arrange
        mock_output = MagicMock()
        mock_output.content = "Mocked Percentage Changes"

        # Mock the chain object and its invoke method
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_output

        # Mock the prompt template to return our mock chain
        mock_get_analyze_percentage_changes_prompt.return_value.__or__.return_value = mock_chain

        state = StateManager.get_instance().state
        proposed_action = "Test Proposed Action"
        criteria_changes = "Test Criteria Changes"

        # Act
        result = analyze_percentage_changes(state, proposed_action, criteria_changes)

        # Assert
        self.assertEqual(result, "Mocked Percentage Changes")

    @patch("HappyChoicesAI.perform_thought_experiment.get_analyze_proxies_impact_prompt")
    @patch("HappyChoicesAI.perform_thought_experiment.llm")
    def test_analyze_proxies_impact(self, mock_llm, mock_get_analyze_proxies_impact_prompt):
        # Arrange
        mock_output = MagicMock()
        mock_output.content = "Mocked Proxies Impact"

        # Mock the chain object and its invoke method
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_output

        # Mock the prompt template to return our mock chain
        mock_get_analyze_proxies_impact_prompt.return_value.__or__.return_value = mock_chain

        state = StateManager.get_instance().state
        proposed_action = "Test Proposed Action"
        criteria_changes = "Test Criteria Changes"

        # Act
        result = analyze_proxies_impact(state, proposed_action, criteria_changes)

        # Assert
        self.assertEqual(result, "Mocked Proxies Impact")

    @patch("HappyChoicesAI.perform_thought_experiment.get_quantify_proxies_prompt")
    @patch("HappyChoicesAI.perform_thought_experiment.llm")
    def test_quantify_proxies(self, mock_llm, mock_get_quantify_proxies_prompt):
        # Arrange
        mock_output = MagicMock()
        mock_output.content = "Mocked Quantified Proxies"

        # Mock the chain object and its invoke method
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_output

        # Mock the prompt template to return our mock chain
        mock_get_quantify_proxies_prompt.return_value.__or__.return_value = mock_chain

        state = StateManager.get_instance().state
        proposed_action = "Test Proposed Action"
        proxies_impact = "Test Proxies Impact"

        # Act
        result = quantify_proxies(state, proposed_action, proxies_impact)

        # Assert
        self.assertEqual(result, "Mocked Quantified Proxies")

    @patch("HappyChoicesAI.perform_thought_experiment.get_summarize_thought_experiment_prompt")
    @patch("HappyChoicesAI.perform_thought_experiment.llm")
    def test_summarize_thought_experiment(self, mock_llm, mock_get_summarize_thought_experiment_prompt):
        # Arrange
        mock_output = MagicMock()
        mock_output.content = "Mocked Summary"

        # Mock the chain object and its invoke method
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_output

        # Mock the prompt template to return our mock chain
        mock_get_summarize_thought_experiment_prompt.return_value.__or__.return_value = mock_chain

        state = StateManager.get_instance().state
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
        self.assertEqual(result, "Mocked Summary")


if __name__ == "__main__":
    unittest.main()
