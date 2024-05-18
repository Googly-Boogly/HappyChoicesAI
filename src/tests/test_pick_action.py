# File: tests/test_pick_best_action.py

import unittest
from unittest.mock import patch, MagicMock
from HappyChoicesAI.pick_action import pick_best_action, argue_best_action, \
    decide_what_the_best_action_to_take_is
from HappyChoicesAI.ai_state import StateManager


class TestPickBestAction(unittest.TestCase):

    def setUp(self):
        # Initialize the state
        state = StateManager.get_instance().state
        state.thought_experiments = [
            {"summary": "Thought experiment 1"},
            {"summary": "Thought experiment 2"},
            {"summary": "Thought experiment 3"},
        ]
        state.best_action = None

    def tearDown(self):
        # Reset the state for other tests
        state = StateManager.get_instance().state
        state.thought_experiments = []
        state.best_action = None

    @patch('HappyChoicesAI.pick_action.llm')
    @patch('HappyChoicesAI.pick_action.get_argue_best_action_prompt')
    @patch('HappyChoicesAI.pick_action.get_decide_best_action_prompt')
    def test_id_assignment(self, mock_get_decide_best_action_prompt, mock_get_argue_best_action_prompt, mock_llm):
        # Ensure IDs are correctly assigned
        state = StateManager.get_instance().state
        pick_best_action()
        for idx, te in enumerate(state.thought_experiments):
            self.assertEqual(te["id"], idx + 1)

    @patch('HappyChoicesAI.pick_action.llm')
    @patch('HappyChoicesAI.pick_action.get_argue_best_action_prompt')
    @patch('HappyChoicesAI.pick_action.get_decide_best_action_prompt')
    @patch('HappyChoicesAI.pick_action.make_other_thought_experiments_pretty_text')
    def test_argument_creation(self, mock_make_other_text, mock_get_decide_best_action_prompt, mock_get_argue_best_action_prompt, mock_llm):
        # Mock the functions
        mock_make_other_text.side_effect = lambda x: "Formatted text"
        mock_argue_best_action_prompt = MagicMock()
        mock_argue_best_action_prompt.__or__.return_value.invoke.return_value = MagicMock(content='{"for": "For argument", "against": "Against argument"}')
        mock_get_argue_best_action_prompt.return_value = mock_argue_best_action_prompt

        state = StateManager.get_instance().state
        pick_best_action()

        for te in state.thought_experiments:
            self.assertIn("arguments_for", te)
            self.assertIn("arguments_against", te)
            self.assertEqual(te["arguments_for"], "For argument")
            self.assertEqual(te["arguments_against"], "Against argument")

    @patch('HappyChoicesAI.pick_action.llm')
    @patch('HappyChoicesAI.pick_action.get_decide_best_action_prompt')
    @patch('HappyChoicesAI.pick_action.get_argue_best_action_prompt')
    def test_decide_best_action(self, mock_get_argue_best_action_prompt, mock_get_decide_best_action_prompt, mock_llm):
        # Mock the LLM response
        mock_output = MagicMock()
        mock_output.content = '{"reasoning": "Best action reasoning", "id": 1}'
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_output
        mock_get_decide_best_action_prompt.return_value.__or__.return_value = mock_chain

        state = StateManager.get_instance().state
        pick_best_action()

        self.assertEqual(state.best_action, 1)

    @patch('HappyChoicesAI.pick_action.llm')
    @patch('HappyChoicesAI.pick_action.get_decide_best_action_prompt')
    @patch('HappyChoicesAI.pick_action.get_argue_best_action_prompt')
    def test_malformed_json_output(self, mock_get_argue_best_action_prompt, mock_get_decide_best_action_prompt, mock_llm):
        # Mock the LLM response with malformed JSON
        mock_output = MagicMock()
        mock_output.content = '{"reasoning": "Best action reasoning", "id": 1'
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_output
        mock_get_decide_best_action_prompt.return_value.__or__.return_value = mock_chain

        state = StateManager.get_instance().state
        for thought in state.thought_experiments:
            thought["for"] = "For argument"
            thought["against"] = "Against argument"
        pick_best_action()

        self.assertIsNone(state.best_action)

    @patch('HappyChoicesAI.pick_action.llm')
    @patch('HappyChoicesAI.pick_action.get_decide_best_action_prompt')
    @patch('HappyChoicesAI.pick_action.get_argue_best_action_prompt')
    def test_json_without_id_key(self, mock_get_argue_best_action_prompt, mock_get_decide_best_action_prompt, mock_llm):
        # Mock the LLM response without ID key
        mock_output = MagicMock()
        mock_output.content = '{"reasoning": "Best action reasoning"}'
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_output
        mock_get_decide_best_action_prompt.return_value.__or__.return_value = mock_chain

        state = StateManager.get_instance().state
        pick_best_action()

        self.assertIsNone(state.best_action)


    @patch("HappyChoicesAI.pick_action.get_argue_best_action_prompt")
    @patch("HappyChoicesAI.pick_action.llm")
    def test_argue_best_action(self, mock_llm, mock_get_argue_best_action_prompt):
        # Arrange
        mock_output = MagicMock()
        mock_output.content = '{"for": "Argument for", "against": "Argument against"}'

        # Mock the chain object and its invoke method
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_output

        # Mock the prompt template to return our mock chain
        mock_prompt_template = MagicMock()
        mock_prompt_template.__or__.return_value = mock_chain
        mock_get_argue_best_action_prompt.return_value = mock_prompt_template

        thought_experiment_to_argue = "Test Thought Experiment"
        other_thought_experiments = "Other Thought Experiments"
        state = StateManager.get_instance().state
        state.situation = "Test Dilemma"
        state.historical_examples = ["Historical Example 1"]

        # Act
        result = argue_best_action(thought_experiment_to_argue, other_thought_experiments)

        # Assert
        self.assertEqual(result, {"for": "Argument for", "against": "Argument against"})

    @patch("HappyChoicesAI.pick_action.get_decide_best_action_prompt")
    @patch("HappyChoicesAI.pick_action.llm")
    def test_decide_what_the_best_action_to_take_is(self, mock_llm, mock_get_decide_best_action_prompt):
        # Arrange
        mock_output = MagicMock()
        mock_output.content = '{"reasoning": "Best reasoning", "id": 1}'

        # Mock the chain object and its invoke method
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_output

        # Mock the prompt template to return our mock chain
        mock_prompt_template = MagicMock()
        mock_prompt_template.__or__.return_value = mock_chain
        mock_get_decide_best_action_prompt.return_value = mock_prompt_template

        state = StateManager.get_instance().state
        state.thought_experiments = [
            {
                "id": 1,
                "proposed_action": "Proposed Action 1",
                "parallels": "Parallels 1",
                "criteria_changes": "Criteria Changes 1",
                "percentage_changes": "Percentage Changes 1",
                "proxies_impact": "Proxies Impact 1",
                "quantified_proxies": "Quantified Proxies 1",
                "summary": "Summary 1",
                "arguments_for": "Arguments For 1",
                "arguments_against": "Arguments Against 1",
            }
        ]

        # Act
        result = decide_what_the_best_action_to_take_is()

        # Assert
        self.assertEqual(result, {"reasoning": "Best reasoning", "id": 1})


if __name__ == '__main__':
    unittest.main()
