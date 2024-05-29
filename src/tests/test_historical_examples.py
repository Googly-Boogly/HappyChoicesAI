# File: tests/test_reason_about_dilemma.py

import unittest
from unittest.mock import MagicMock, patch

from HappyChoicesAI.ai_state import StateManager
from HappyChoicesAI.historical_examples import (
    HistoricalExample,
    reason_about_dilemma,
)


class TestReasonAboutDilemma(unittest.TestCase):

    @patch('HappyChoicesAI.historical_examples.FileState.get_instance')
    @patch("HappyChoicesAI.historical_examples.create_prompt_template")
    def test_reason_about_dilemma_yes(self, mock_create_prompt_template, mock_get_instance):
        # Arrange
        mock_output = MagicMock()
        mock_output.content = "yes"

        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_output

        mock_create_prompt_template.return_value.__or__.return_value = mock_chain

        input_dilemma = "input dilemma"
        dilemma = HistoricalExample(
            situation="Historical dilemma", action_taken="Action", reasoning="Reasoning"
        )
        state = StateManager.get_instance().state
        state.situation = input_dilemma

        mock_file_state_instance = MagicMock()
        mock_file_state_instance.llm = mock_chain
        mock_get_instance.return_value = mock_file_state_instance

        # Act
        result = reason_about_dilemma(dilemma)

        # Assert
        self.assertTrue(result)

    @patch('HappyChoicesAI.historical_examples.FileState.get_instance')
    @patch("HappyChoicesAI.historical_examples.create_prompt_template")
    def test_reason_about_dilemma_no(self, mock_create_prompt_template, mock_get_instance):
        # Arrange
        mock_output = MagicMock()
        mock_output.content = "no"

        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_output

        mock_create_prompt_template.return_value.__or__.return_value = mock_chain

        input_dilemma = "input dilemma"
        dilemma = HistoricalExample(
            situation="Historical dilemma", action_taken="Action", reasoning="Reasoning"
        )
        state = StateManager.get_instance().state
        state.situation = input_dilemma

        mock_file_state_instance = MagicMock()
        mock_file_state_instance.llm = mock_chain
        mock_get_instance.return_value = mock_file_state_instance


        # Act
        result = reason_about_dilemma(dilemma)

        # Assert
        self.assertFalse(result)

    @patch('HappyChoicesAI.historical_examples.FileState.get_instance')
    @patch("HappyChoicesAI.historical_examples.create_prompt_template")
    def test_reason_about_dilemma_invalid_response(self, mock_create_prompt_template, mock_get_instance):
        # Arrange
        mock_output = MagicMock()
        mock_output.content = "maybe"

        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_output

        mock_create_prompt_template.return_value.__or__.return_value = mock_chain

        input_dilemma = "input dilemma"
        dilemma = HistoricalExample(
            situation="Historical dilemma", action_taken="Action", reasoning="Reasoning"
        )
        state = StateManager.get_instance().state
        state.situation = input_dilemma

        mock_file_state_instance = MagicMock()
        mock_file_state_instance.llm = mock_chain
        mock_get_instance.return_value = mock_file_state_instance

        # Act
        result = reason_about_dilemma(dilemma)

        # Assert
        self.assertFalse(result)

    @patch('HappyChoicesAI.historical_examples.FileState.get_instance')
    @patch("HappyChoicesAI.historical_examples.create_prompt_template")
    def test_reason_about_dilemma_edge_case(self, mock_create_prompt_template, mock_get_instance):
        # Arrange
        mock_output = MagicMock()
        mock_output.content = "unknown"

        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_output

        mock_create_prompt_template.return_value.__or__.return_value = mock_chain

        input_dilemma = "input dilemma"
        dilemma = HistoricalExample(
            situation="Historical dilemma", action_taken="Action", reasoning="Reasoning"
        )
        state = StateManager.get_instance().state
        state.situation = input_dilemma

        mock_file_state_instance = MagicMock()
        mock_file_state_instance.llm = mock_chain
        mock_get_instance.return_value = mock_file_state_instance

        # Act
        result = reason_about_dilemma(dilemma)

        # Assert
        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()