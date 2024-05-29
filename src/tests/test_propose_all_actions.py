# File: tests/test_propose_all_actions.py

import unittest
from unittest.mock import patch, MagicMock
from HappyChoicesAI.perform_thought_experiment import propose_all_actions


class TestProposeAllActions(unittest.TestCase):

    @patch('HappyChoicesAI.perform_thought_experiment.FileState.get_instance')
    @patch('HappyChoicesAI.perform_thought_experiment.create_prompt_template_propose_actions')
    def test_correct_json_output(self, mock_create_prompt_template, mock_get_instance):
        # Arrange
        mock_output = MagicMock()
        mock_output.content = '{"actions": ["action1", "action2", "action3"]}'

        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_output

        mock_create_prompt_template.return_value.__or__.return_value = mock_chain

        mock_file_state_instance = MagicMock()
        mock_file_state_instance.llm = mock_chain
        mock_get_instance.return_value = mock_file_state_instance

        # Act
        result = propose_all_actions()

        # Assert
        self.assertEqual(result, ["action1", "action2", "action3"])

    @patch('HappyChoicesAI.perform_thought_experiment.FileState.get_instance')
    @patch('HappyChoicesAI.perform_thought_experiment.create_prompt_template_propose_actions')
    def test_malformed_json_output(self, mock_create_prompt_template, mock_get_instance):
        # Arrange
        mock_output = MagicMock()
        mock_output.content = '{"actions": ["action1", "action2", "action3"'

        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_output

        mock_create_prompt_template.return_value.__or__.return_value = mock_chain

        mock_file_state_instance = MagicMock()
        mock_file_state_instance.llm = mock_chain
        mock_get_instance.return_value = mock_file_state_instance

        # Act
        result = propose_all_actions()

        # Assert
        self.assertEqual(result, [])

    @patch('HappyChoicesAI.perform_thought_experiment.FileState.get_instance')
    @patch('HappyChoicesAI.perform_thought_experiment.create_prompt_template_propose_actions')
    def test_json_without_actions_key(self, mock_create_prompt_template, mock_get_instance):
        # Arrange
        mock_output = MagicMock()
        mock_output.content = '{"not_actions": ["action1", "action2", "action3"]}'

        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_output

        mock_create_prompt_template.return_value.__or__.return_value = mock_chain

        mock_file_state_instance = MagicMock()
        mock_file_state_instance.llm = mock_chain
        mock_get_instance.return_value = mock_file_state_instance

        # Act
        result = propose_all_actions()

        # Assert
        self.assertEqual(result, [])

    @patch('HappyChoicesAI.perform_thought_experiment.FileState.get_instance')
    @patch('HappyChoicesAI.perform_thought_experiment.create_prompt_template_propose_actions')
    def test_empty_output(self, mock_create_prompt_template, mock_get_instance):
        # Arrange
        mock_output = MagicMock()
        mock_output.content = ''

        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_output

        mock_create_prompt_template.return_value.__or__.return_value = mock_chain

        mock_file_state_instance = MagicMock()
        mock_file_state_instance.llm = mock_chain
        mock_get_instance.return_value = mock_file_state_instance

        # Act
        result = propose_all_actions()

        # Assert
        self.assertEqual(result, [])


if __name__ == '__main__':
    unittest.main()