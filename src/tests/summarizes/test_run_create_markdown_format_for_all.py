import unittest
from unittest.mock import MagicMock, Mock, patch
from HappyChoicesAI.summarize_result import create_markdown_format_for_all, \
    run_create_markdown_format_for_all
from global_code.langchain import invoke_with_retry
# from HappyChoicesAI.ai_state import StateManager


class TestRunCreateMarkdown(unittest.TestCase):

    def setUp(self):
        self.state = Mock()
        self.state_summary = Mock()
        self.state.situation = "This is the dilemma."
        self.state_summary.historical_examples_summary = "These are historical examples."
        self.state_summary.themes = "These are themes."
        self.state_summary.insights = "These are insights."
        self.state_summary.conclusion = "This is the conclusion."
        self.state_summary.lessons_learned = "These are lessons learned."
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

    @patch('HappyChoicesAI.summarize_result.StateManagerSummary.get_instance')
    @patch('HappyChoicesAI.summarize_result.StateManager.get_instance')
    @patch('HappyChoicesAI.summarize_result.FileState.get_instance')
    @patch('HappyChoicesAI.summarize_result.invoke_with_retry')
    def test_run_create_markdown_format_for_all(self, mock_invoke_with_retry, mock_get_instance2, mock_get_instance,
                                                        mock_get_instance_summary):
        mock_get_instance.return_value = Mock(state=self.state)
        mock_get_instance_summary.return_value = Mock(state=self.state_summary)
        mock_invoke_with_retry.return_value = "This is the generated summary."

        mock_file_state_instance = MagicMock()
        mock_file_state_instance.llm = Mock()
        mock_get_instance2.return_value = mock_file_state_instance

        result = run_create_markdown_format_for_all()

        self.assertEqual(result, "This is the generated summary.")
        self.assertEqual(self.state_summary.markdown, "This is the generated summary.")

    def test_create_markdown_format_for_all(self):
        template = create_markdown_format_for_all()
        self.assertEqual(template.template, """You are a world renowned markdown expert. You have been tasked with creating a markdown format for all of the details provided.

Dilemma:
{input_dilemma}  

Thought Experiment that was chosen:
{thought_experiment_summary}

Other Thought Experiments:
{other_thought_experiments}

Historical Parallels:
{historical_examples}

Common Themes:
{common_themes}

Key Insights:
{key_insights}

Lessons Learned:
{lessons_learned}

Comprehensive Conclusion:
{conclusion}

You need to create a markdown format for all of the details provided. The markdown format should follow the provided structure. Ensure to provide an introduction, summary of the thought experiments, historical examples, common themes, key insights, lessons learned, and a comprehensive conclusion.
""")
