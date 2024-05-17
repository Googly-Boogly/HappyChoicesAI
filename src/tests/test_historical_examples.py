# File: tests/test_reason_about_dilemma.py

import unittest
from unittest.mock import MagicMock, patch

from HappyChoicesAI.ai_state import EthicistAIState
from HappyChoicesAI.historical_examples import (
    HistoricalExample,
    reason_about_dilemma,
    ChatPromptTemplate,
)


class TestReasonAboutDilemma(unittest.TestCase):

    @patch("HappyChoicesAI.historical_examples.ChatPromptTemplate.from_template")
    @patch("HappyChoicesAI.historical_examples.llm")
    def test_reason_about_dilemma_yes(self, mock_llm, mock_prompt_template):
        # Arrange
        mock_output = MagicMock()
        mock_output.choices = [MagicMock(text="Yes")]

        # Mock the chain object and its invoke method
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_output

        # Mock the prompt_template | llm to return our mock chain
        mock_prompt_template.return_value.__or__.return_value = mock_chain

        input_dilemma = "input dilemma"
        dilemma = HistoricalExample(
            situation="Historical dilemma", action_taken="Action", reasoning="Reasoning"
        )
        # Act
        result = reason_about_dilemma(dilemma, input_dilemma)

        # Assert
        self.assertTrue(result)

    @patch("HappyChoicesAI.historical_examples.ChatPromptTemplate.from_template")
    @patch("HappyChoicesAI.historical_examples.llm")
    def test_reason_about_dilemma_no(self, mock_llm, mock_prompt_template):
        # Arrange
        # Arrange
        mock_output = MagicMock()
        mock_output.choices = [MagicMock(text="No")]

        # Mock the chain object and its invoke method
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_output

        # Mock the prompt_template | llm to return our mock chain
        mock_prompt_template.return_value.__or__.return_value = mock_chain

        input_dilemma = "input dilemma"
        dilemma = HistoricalExample(
            situation="Historical dilemma", action_taken="Action", reasoning="Reasoning"
        )

        # Act
        result = reason_about_dilemma(dilemma, input_dilemma)

        # Assert
        self.assertFalse(result)

    @patch("HappyChoicesAI.historical_examples.ChatPromptTemplate.from_template")
    @patch("HappyChoicesAI.historical_examples.llm")
    def test_reason_about_dilemma_invalid_response(
        self, mock_llm, mock_prompt_template
    ):
        # Arrange
        # Arrange
        mock_output = MagicMock()
        mock_output.choices = [MagicMock(text="Maybe")]

        # Mock the chain object and its invoke method
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_output

        # Mock the prompt_template | llm to return our mock chain
        mock_prompt_template.return_value.__or__.return_value = mock_chain

        input_dilemma = "input dilemma"
        dilemma = HistoricalExample(
            situation="Historical dilemma", action_taken="Action", reasoning="Reasoning"
        )

        # Act
        result = reason_about_dilemma(dilemma, input_dilemma)

        # Assert
        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()

# LMAO, I'm sorry, I just had to share this. I'm not sure if this is a good practice or not, but I'm sure it's not a good practice to have a test that tests the test.
# (Don't worry I am a professional, I just wanted to share this with you guys)
# Many look up to me for my spectacular coding abilites (No I am serious, I am a professional)
# If you need any lessons on git I am here for you, I am a professional
