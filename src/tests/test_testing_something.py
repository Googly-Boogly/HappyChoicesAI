# # File: tests/test_perform_thought_experiment.py
#
# import unittest
# from unittest.mock import patch, MagicMock
# from HappyChoicesAI.perform_thought_experiment import summarize_thought_experiment
# from HappyChoicesAI.thought_experiments_prompts import get_summarize_thought_experiment_prompt
# from HappyChoicesAI.ai_state import StateManager, EthicistAIState
#
#
# class TestPerformThoughtExperiments(unittest.TestCase):
#
#     @patch("HappyChoicesAI.perform_thought_experiment.llm")
#     @patch("HappyChoicesAI.perform_thought_experiment.get_summarize_thought_experiment_prompt")
#     def test_summarize_thought_experiment(self, mock_get_summarize_thought_experiment_prompt, mock_llm):
#         # Arrange
#         mock_output = MagicMock()
#         mock_output.content = "Mocked Summary"  # Set the content attribute directly
#
#         # Mock the chain object and its invoke method
#         mock_chain = MagicMock()
#         mock_chain.invoke.return_value = mock_output
#
#         # Mock the prompt template to return our mock chain
#         # mock_prompt_template = MagicMock()
#         # mock_prompt_template.__or__.return_value = mock_chain
#         mock_get_summarize_thought_experiment_prompt.return_value.__or__.return_value = mock_chain
#
#         state = StateManager.get_instance().state
#         proposed_action = "Test Proposed Action"
#         parallels = "Test Parallels"
#         criteria_changes = "Test Criteria Changes"
#         percentage_changes = "Test Percentage Changes"
#         proxies_impact = "Test Proxies Impact"
#         quantified_proxies = "Test Quantified Proxies"
#         state.situations = "Test Situations"
#         state.criteria = "Test Criteria"
#         # Act
#         result = summarize_thought_experiment(
#             proposed_action,
#             parallels,
#             criteria_changes,
#             percentage_changes,
#             proxies_impact,
#             quantified_proxies,
#             state,
#         )
#
#         # Assert
#         self.assertEqual(result, "Mocked Summary")
#
#
# if __name__ == "__main__":
#     unittest.main()
#
