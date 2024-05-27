import unittest
from unittest.mock import Mock, patch
from HappyChoicesAI.summarize_result import create_markdown_format_for_all, \
    historical_examples_prompt_summary_prompt, run_create_markdown_format_for_all, \
    run_historical_examples_prompt_summary_prompt
from global_code.langchain import invoke_with_retry
from HappyChoicesAI.ai_state import HistoricalExample

from HappyChoicesAI.ai_state import StateManager, StateManagerSummary


class TestRunHistorical(unittest.TestCase):

    def setUp(self):
        self.state = Mock()
        self.state.situation = "This is the dilemma."
        self.state.historical_examples = [
            HistoricalExample("This is the historical example.", "This is the relevance to the thought experiment.", "IDGA3F"),
            HistoricalExample("This is the historical example.", "This is the relevance to the thought experiment.", "IDGAF")
        ]
        self.state.thought_experiments = [
            {
                "summary": "This is a summary.",
                "parallels": "These are parallels.",
                "arguments_for": "These are arguments for.",
                "arguments_against": "These are arguments against.",
                "criteria_changes": "These are criteria changes.",
                "percentage_changes": "These are percentage changes.",
                "proxies_impact": "This is the impact of proxies.",
                "quantified_proxies": "These are quantified proxies.",
                "proposed_action": "This is the proposed action.",
                "historical_examples": "These are historical examples.",
                "id": 1,
            },
            {
                "summary": "This is a summary.",
                "parallels": "These are parallels.",
                "arguments_for": "These are arguments for.",
                "arguments_against": "These are arguments against.",
                "criteria_changes": "These are criteria changes.",
                "percentage_changes": "These are percentage changes.",
                "proxies_impact": "This is the impact of proxies.",
                "quantified_proxies": "These are quantified proxies.",
                "proposed_action": "This is the proposed action.",
                "historical_examples": "These are historical examples.",
                "id": 2,
            }
        ]
        self.state.best_action = 1
        self.state_summary = Mock()
        self.state_summary.historical_examples_summary = "These are historical examples."
        self.state_summary.themes = "These are themes."
        self.state_summary.insights = "These are insights."
        self.state_summary.conclusion = "This is the conclusion."
        self.state_summary.lessons_learned = "These are lessons learned."


    @patch('HappyChoicesAI.summarize_result.StateManager.get_instance')
    @patch('HappyChoicesAI.summarize_result.llm')
    @patch('HappyChoicesAI.summarize_result.invoke_with_retry')
    @patch('HappyChoicesAI.summarize_result.StateManagerSummary.get_instance')
    def test_run_historical_examples_prompt_summary_prompt(self, mock_get_instance_summary, mock_invoke_with_retry, mock_llm, mock_get_instance):
        mock_get_instance.return_value = Mock(state=self.state)
        mock_get_instance_summary.return_value = Mock(state=self.state_summary)
        mock_invoke_with_retry.return_value = "This is the generated summary."

        result = run_historical_examples_prompt_summary_prompt()

        self.assertEqual(result, "This is the generated summary.")
        self.assertEqual(self.state_summary.historical_examples_summary, "This is the generated summary.")

    def test_summary_of_selected_thought_experiment(self):
        template = historical_examples_prompt_summary_prompt()
        self.assertEqual(template.template, """You are a world renowned AI utilitarian ethicist. You have been tasked with summarizing the historical examples used and their relevance to the thought experiments.
        
Dilemma:
{input_dilemma}

Historical Examples:
{historical_examples}

Thought Experiment that was chosen:
{thought_experiment_summary}

Other Thought Experiments:
{other_thought_experiments}

You need to summarize the historical examples used and their relevance to the thought experiments. Describe ny parallels or connections between the historical examples and the thought experiments.
""")
