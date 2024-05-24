# File: tests/test_pick_best_action.py

import unittest
from unittest.mock import Mock, patch, MagicMock
from HappyChoicesAI.pick_action import pick_best_action, argue_best_action, \
    decide_what_the_best_action_to_take_is
from HappyChoicesAI.ai_state import StateManager, HistoricalExample


class TestPickBestAction(unittest.TestCase):

    def setUp(self):
        # Initialize the state
        self.state = Mock()
        self.state.thought_experiments = [
            {"summary": "Thought experiment 1"},
            {"summary": "Thought experiment 2"},
            {"summary": "Thought experiment 3"},
        ]
        self.state.best_action = None
        self.random_state = Mock()
        self.random_state.state.model_used = "Model used"
        self.random_state.state.thread_count = 1
        self.random_state.thread_count = 1

    def tearDown(self):
        # Reset the state for other tests
        self.state = StateManager.get_instance().state
        self.state.thought_experiments = []
        self.state.best_action = None

    @patch('HappyChoicesAI.pick_action.StateManager.get_instance')
    @patch('HappyChoicesAI.pick_action.ModelUsedAndThreadCount.get_instance')
    @patch('HappyChoicesAI.pick_action.llm')
    @patch('HappyChoicesAI.pick_action.get_argue_best_action_prompt')
    @patch('HappyChoicesAI.pick_action.get_decide_best_action_prompt')
    def test_id_assignment(self, mock_get_decide_best_action_prompt, mock_get_argue_best_action_prompt,
                           mock_llm, mock_ModelUsedAndThreadCount, real_state_mock):
        real_state_mock.return_value = Mock(state=self.state)
        # Ensure IDs are correctly assigned
        state = StateManager.get_instance().state
        mock_ModelUsedAndThreadCount.return_value = Mock(state=self.random_state)
        pick_best_action()
        for idx, te in enumerate(state.thought_experiments):
            self.assertEqual(te["id"], idx + 1)

    @patch('HappyChoicesAI.pick_action.StateManager.get_instance')
    @patch('HappyChoicesAI.pick_action.ModelUsedAndThreadCount.get_instance')
    @patch('HappyChoicesAI.pick_action.get_decide_best_action_prompt')
    @patch('HappyChoicesAI.pick_action.get_argue_best_action_prompt')
    @patch('HappyChoicesAI.pick_action.retry_fail_json_output')
    def test_argument_creation(self, mock_retry_fail_json_output, mock_get_argue_best_action_prompt,
                               mock_get_decide_best_action_prompt, mock_ModelUsedAndThreadCount,
                               real_state_mock):
        # Mock the functions
        # mock_make_other_text.side_effect = lambda x: "Formatted text"
        real_state_mock.return_value = Mock(state=self.state)
        mock_ModelUsedAndThreadCount.return_value = Mock(state=self.random_state)
        mock_retry_fail_json_output.side_effect = [
            {"for": "For argument", "against": "Against argument"},  # First call for arguments
            {"for": "For argument", "against": "Against argument"},  # Second call for arguments
            {"for": "For argument", "against": "Against argument"},  # Third call for arguments
            {"id": 1}  # Final call for deciding the best action
        ]

        pick_best_action()

        for te in self.state.thought_experiments:
            self.assertIn("arguments_for", te)
            self.assertIn("arguments_against", te)
            self.assertEqual(te["arguments_for"], "For argument")
            self.assertEqual(te["arguments_against"], "Against argument")

        self.assertEqual(self.state.best_action, 1)

    @patch('HappyChoicesAI.pick_action.StateManager.get_instance')
    @patch('HappyChoicesAI.pick_action.llm')
    @patch('HappyChoicesAI.pick_action.get_decide_best_action_prompt')
    @patch('HappyChoicesAI.pick_action.get_argue_best_action_prompt')
    @patch('HappyChoicesAI.pick_action.ModelUsedAndThreadCount.get_instance')
    def test_decide_best_action(self, mock_ModelUsedAndThreadCount, mock_get_argue_best_action_prompt,
                                mock_get_decide_best_action_prompt, mock_llm, real_state_mock):
        # Arrange
        state_mock = Mock()
        state_mock.state = Mock(thought_experiments=[
            {
                "id": 1,
                "proposed_action": "Proposed Action 1",
                "parallels": "Parallels 1",
                "criteria_changes": "Criteria Changes 1",
                "percentage_changes": "Percentage Changes 1",
                "proxies_impact": "Proxies Impact 1",
                "quantified_proxies": "Quantified Proxies 1",
                "summary": "Summary 1",
                "arguments_for": "",
                "arguments_against": ""
            }
        ])
        real_state_mock.return_value = state_mock

        mock_output = MagicMock()
        mock_output.content = {"reasoning": "Best action reasoning", "id": 1}

        # Mock the chain object and its invoke method
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_output

        # Mock the prompt template to return our mock chain
        mock_prompt_template = MagicMock()
        mock_prompt_template.__or__.return_value = mock_chain
        mock_get_decide_best_action_prompt.return_value = mock_prompt_template
        mock_get_argue_best_action_prompt.return_value = mock_prompt_template

        # Mock the ModelUsedAndThreadCount
        mock_model = Mock()
        mock_model.state = state_mock.state
        mock_ModelUsedAndThreadCount.return_value = Mock(state=self.random_state)

        # Act
        pick_best_action()

        # Assert
        self.assertEqual(state_mock.state.best_action, 1)

    @patch('HappyChoicesAI.pick_action.StateManager.get_instance')
    @patch('HappyChoicesAI.pick_action.ModelUsedAndThreadCount.get_instance')
    @patch('HappyChoicesAI.pick_action.llm')
    @patch('HappyChoicesAI.pick_action.get_decide_best_action_prompt')
    @patch('HappyChoicesAI.pick_action.get_argue_best_action_prompt')
    def test_malformed_json_output(self, mock_get_argue_best_action_prompt, mock_get_decide_best_action_prompt,
                                   mock_llm, mock_ModelUsedAndThreadCount, real_state_mock):
        real_state_mock.return_value = Mock(state=self.state)
        # Mock the LLM response with malformed JSON
        mock_output = MagicMock()
        mock_ModelUsedAndThreadCount.return_value = Mock(state=self.random_state)
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

    @patch('HappyChoicesAI.pick_action.StateManager.get_instance')
    @patch('HappyChoicesAI.pick_action.ModelUsedAndThreadCount.get_instance')
    @patch('HappyChoicesAI.pick_action.llm')
    @patch('HappyChoicesAI.pick_action.get_decide_best_action_prompt')
    @patch('HappyChoicesAI.pick_action.get_argue_best_action_prompt')
    def test_json_without_id_key(self, mock_get_argue_best_action_prompt, mock_get_decide_best_action_prompt,
                                 mock_llm, mock_ModelUsedAndThreadCount, real_state_mock):
        real_state_mock.return_value = Mock(state=self.state)
        # Mock the LLM response without ID key
        mock_output = MagicMock()
        mock_ModelUsedAndThreadCount.return_value = Mock(state=self.random_state)
        mock_output.content = '{"reasoning": "Best action reasoning"}'
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_output
        mock_get_decide_best_action_prompt.return_value.__or__.return_value = mock_chain

        state = StateManager.get_instance().state
        pick_best_action()

        self.assertIsNone(state.best_action)

    @patch('HappyChoicesAI.pick_action.StateManager.get_instance')
    @patch('HappyChoicesAI.pick_action.ModelUsedAndThreadCount.get_instance')
    @patch('HappyChoicesAI.pick_action.make_other_thought_experiments_pretty_text')
    @patch('HappyChoicesAI.pick_action.get_decide_best_action_prompt')
    @patch('HappyChoicesAI.pick_action.get_argue_best_action_prompt')
    @patch('HappyChoicesAI.pick_action.retry_fail_json_output')
    def test_argue_best_action(self, mock_retry_fail_json_output, mock_get_argue_best_action_prompt,
                               mock_get_decide_best_action_prompt, mock_make_other_text, mock_ModelUsedAndThreadCount,
                               real_state_mock):
        real_state_mock.return_value = Mock(state=self.state)
        # Arrange
        mock_output = MagicMock()
        mock_ModelUsedAndThreadCount.return_value = Mock(state=self.random_state)
        mock_output.content = {"for": "Argument for", "against": "Argument against"}

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
        state.historical_examples = [HistoricalExample(situation="Test Dilemma", action_taken="Test Action", reasoning="Test Reasoning")]

        # Act
        result = argue_best_action(thought_experiment_to_argue, other_thought_experiments)

        # Assert
        self.assertEqual(result, {"against": "Argument against", "for": "Argument for"})

    @patch('HappyChoicesAI.pick_action.StateManager.get_instance')
    @patch('HappyChoicesAI.pick_action.ModelUsedAndThreadCount.get_instance')
    @patch("HappyChoicesAI.pick_action.get_decide_best_action_prompt")
    @patch("HappyChoicesAI.pick_action.llm")
    def test_decide_what_the_best_action_to_take_is(self, mock_llm, mock_get_decide_best_action_prompt,
                                                    mock_ModelUsedAndThreadCount, mock_get_instance):
        # Arrange
        mock_output = MagicMock()
        mock_get_instance.return_value = Mock(state=self.state)
        mock_ModelUsedAndThreadCount.return_value = Mock(state=self.random_state)
        mock_output.content = {"reasoning": "Best reasoning", "id": 1}

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
        self.assertEqual(result, {"id": 1, "reasoning": "Best reasoning"})


if __name__ == '__main__':
    unittest.main()
