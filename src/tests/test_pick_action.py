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

    def test_id_assignment(self):
        # Ensure IDs are correctly assigned
        state = StateManager.get_instance().state
        pick_best_action()
        for idx, te in enumerate(state.thought_experiments):
            self.assertEqual(te["id"], idx + 1)

    @patch('HappyChoicesAI.pick_action.make_other_thought_experiments_pretty_text')
    @patch('HappyChoicesAI.pick_action.argue_best_action')
    def test_argument_creation(self, mock_argue_best_action, mock_make_other_text):
        # Mock the functions
        mock_make_other_text.side_effect = lambda x: "Formatted text"
        mock_argue_best_action.side_effect = lambda x, y: {"for": "For argument", "against": "Against argument"}

        state = StateManager.get_instance().state
        pick_best_action()

        for te in state.thought_experiments:
            self.assertIn("arguments_for", te)
            self.assertIn("arguments_against", te)
            self.assertEqual(te["arguments_for"], "For argument")
            self.assertEqual(te["arguments_against"], "Against argument")

    @patch('HappyChoicesAI.pick_action.llm')
    @patch('HappyChoicesAI.pick_action.create_prompt_template')
    def test_decide_best_action(self, mock_create_prompt_template, mock_llm):
        # Mock the LLM response
        mock_output = MagicMock()
        mock_output.content = '{"reasoning": "Best action reasoning", "id": 1}'
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_output
        mock_create_prompt_template.return_value.__or__.return_value = mock_chain

        state = StateManager.get_instance().state
        pick_best_action()

        self.assertEqual(state.best_action, 1)

    @patch('HappyChoicesAI.pick_action.llm')
    @patch('HappyChoicesAI.pick_action.create_prompt_template')
    def test_malformed_json_output(self, mock_create_prompt_template, mock_llm):
        # Mock the LLM response with malformed JSON
        mock_output = MagicMock()
        mock_output.content = '{"reasoning": "Best action reasoning", "id": 1'
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_output
        mock_create_prompt_template.return_value.__or__.return_value = mock_chain

        state = StateManager.get_instance().state
        pick_best_action()

        self.assertIsNone(state.best_action)

    @patch('HappyChoicesAI.pick_action.llm')
    @patch('HappyChoicesAI.pick_action.create_prompt_template')
    def test_json_without_id_key(self, mock_create_prompt_template, mock_llm):
        # Mock the LLM response without ID key
        mock_output = MagicMock()
        mock_output.content = '{"reasoning": "Best action reasoning"}'
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_output
        mock_create_prompt_template.return_value.__or__.return_value = mock_chain

        state = StateManager.get_instance().state
        pick_best_action()

        self.assertIsNone(state.best_action)


if __name__ == '__main__':
    unittest.main()
