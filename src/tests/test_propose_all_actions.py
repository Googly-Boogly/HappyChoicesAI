import unittest
from unittest.mock import patch
from HappyChoicesAI.perform_thought_experiment import propose_all_actions


class TestProposeAllActions(unittest.TestCase):
    @patch('HappyChoicesAI.perform_thought_experiment.llm')
    def test_correct_json_output(self, MockLLM):
        MockLLM.return_value.invoke.return_value = '{"actions": ["action1", "action2", "action3"]}'
        result = propose_all_actions()
        self.assertEqual(result, {"actions": ["action1", "action2", "action3"]})

    @patch('HappyChoicesAI.perform_thought_experiment.llm')
    def test_malformed_json_output(self, MockLLM):
        MockLLM.return_value.invoke.return_value = '{"actions": ["action1", "action2", "action3"'
        result = propose_all_actions()
        self.assertEqual(result, {"actions": []})

    @patch('HappyChoicesAI.perform_thought_experiment.llm')
    def test_json_without_actions_key(self, MockLLM):
        MockLLM.return_value.invoke.return_value = '{"not_actions": ["action1", "action2", "action3"]}'
        result = propose_all_actions()
        self.assertEqual(result, {"actions": []})

    @patch('HappyChoicesAI.perform_thought_experiment.llm')
    def test_empty_output(self, MockLLM):
        MockLLM.return_value.invoke.return_value = ''
        result = propose_all_actions()
        self.assertEqual(result, {"actions": []})

if __name__ == '__main__':
    unittest.main()