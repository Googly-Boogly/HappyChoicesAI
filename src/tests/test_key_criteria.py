import unittest
from unittest.mock import patch, MagicMock
from typing import Dict, List, Tuple
from HappyChoicesAI.ai_state import EthicistAIState

# Assuming the module containing the find_key_criteria function is named `module_name`
# and the function is imported correctly.
from HappyChoicesAI.key_criteria import find_key_criteria


class TestFindKeyCriteria(unittest.TestCase):

    @patch("HappyChoicesAI.key_criteria.ChatPromptTemplate.from_template")
    @patch("HappyChoicesAI.key_criteria.llm")
    def test_find_key_criteria(self, mock_llm, mock_prompt_template):
        # Create a mock response for chain.invoke
        mock_output = MagicMock()
        mock_output.choices = [
            MagicMock(
                text="Preserving human autonomy, profits, the human managers, and the people being hired."
            )
        ]

        # Mock the chain object and its invoke method
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_output

        # Mock the prompt_template | llm to return our mock chain
        mock_prompt_template.return_value.__or__.return_value = mock_chain

        # Create a sample state
        state = EthicistAIState(
            situation="A corporation introduces an AI system designed to manage task assignments and work schedules to optimize productivity and reduce managerial costs."
        )

        # Call the function
        find_key_criteria(state)

        # Assert the criteria in the state is set correctly
        self.assertEqual(
            state.criteria,
            "Preserving human autonomy, profits, the human managers, and the people being hired.",
        )

        # Ensure the logger is called with the correct message
        # This part assumes you have a log_it_sync function that logs messages
        # Here we would mock and assert it if it was in the scope.


if __name__ == "__main__":
    unittest.main()
