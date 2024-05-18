import unittest
from unittest.mock import Mock, patch
from HappyChoicesAI.summarize_result import create_markdown_format_for_all, \
    introduction_prompt_gen, run_create_markdown_format_for_all, run_introduction_prompt
from global_code.langchain import invoke_with_retry


# from HappyChoicesAI.ai_state import StateManager


class TestRunIntro(unittest.TestCase):

    def setUp(self):
        self.state = Mock()
        self.state.situation = "This is the dilemma."
        self.state.final_summary.historical_examples_summary = "These are historical examples."
        self.state.final_summary.themes = "These are themes."
        self.state.final_summary.insights = "These are insights."
        self.state.final_summary.conclusion = "This is the conclusion."
        self.state.final_summary.lessons_learned = "These are lessons learned."
        # For these 2 functions make_chosen_thought_experiment_pretty_text, make_other_thought_experiments_pretty_normal
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
        self.state.final_summary = Mock()

    @patch('HappyChoicesAI.summarize_result.StateManager.get_instance')
    @patch('HappyChoicesAI.summarize_result.llm')
    @patch('HappyChoicesAI.summarize_result.invoke_with_retry')
    def test_run_summary_of_selected_thought_experiment(self, mock_invoke_with_retry, mock_llm, mock_get_instance):
        mock_get_instance.return_value = Mock(state=self.state)
        mock_invoke_with_retry.return_value = "This is the generated summary."

        result = run_introduction_prompt()

        self.assertEqual(result, "This is the generated summary.")
        self.state.final_summary.chosen_best_action_summary = "This is the generated summary."

    def test_summary_of_selected_thought_experiment(self):
        template = introduction_prompt_gen()
        self.assertEqual(template.template, """You are a world renowned AI ethicist. You have been tasked with generating the purpose and scope of the thought experiments.
        
Dilemma:
{input_dilemma}

{thought_experiments}

You need to provide a brief introduction to the thought experiments, including the purpose and scope of the thought experiments.
""")
