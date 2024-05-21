import json
import os
from typing import Dict, List, Tuple

from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from HappyChoicesAI.ai_state import EthicistAIState, StateManager
from global_code.langchain import invoke_with_retry, retry_fail_json_output
from global_code.helpful_functions import create_logger_error, log_it_sync

load_dotenv()

# Get the API key from the environment variable
api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=api_key)
logger = create_logger_error(
    file_path=os.path.abspath(__file__), name_of_log_file="pick_action"
)
"""
TODO: Implement the pick_best_action function that will determine the best action based on the outcomes of the thought experiments.
This is what the input to the decide_what_the_best_action_to_take_is function should look like:
{
  "dilemma": "Describe the ethical dilemma here",
  "thought_experiments": [
    {
      "id": 1,
      "description": "Description of thought experiment 1",
      "arguments_for": "Arguments why this is the best option",
      "arguments_against": "Arguments why this is not the best option"
    },
    {
      "id": 2,
      "description": "Description of thought experiment 2",
      "arguments_for": "Arguments why this is the best option",
      "arguments_against": "Arguments why this is not the best option"
    }
    // Add more thought experiments as needed
  ]
}
The LLM should than output it's reasoning why it chose the best action
Than actually choose the best action example output:
{
    "id": 1,
    "reasoning": "blah blAH blah"
}
The pick_best_action function is really messy right now fix it.
"""


def pick_best_action() -> None:
    """
    Determines and updates the best action based on the outcomes of the thought experiments.
    The best action is based on maximizing happiness and minimizing suffering.
    :param state: EthicistAIState object containing all relevant data
    """
    # First have the llm read all of the thought experiments
    # For each thought experiment have the LLM argue why it is the best action and why it is not
    # Than have a final LLM look through all of the thought experiments and all of the arguments and determine the best action

    # First make the thought experiments into nice looking text

    # first add id's to all of the thought experiments
    state = StateManager.get_instance().state

    for i in range(len(state.thought_experiments)):
        state.thought_experiments[i]["id"] = i + 1

    runner = 0
    while runner < len(state.thought_experiments):
        thought_experiment = state.thought_experiments[runner]["summary"]

        # get the other thought experiments into a list of strings with their summaries
        other_thought_experiments_dict = (
                state.thought_experiments[:runner]
                + state.thought_experiments[runner + 1:])
        other_thought_experiments_list = []
        for thought in other_thought_experiments_dict:
            other_thought_experiments_list.append(thought["summary"])

        other_thought_experiments_llm_ready_text: str = make_other_thought_experiments_pretty_text(
            other_thought_experiments_list)
        for_and_against: Dict[str, str] = retry_fail_json_output(argue_best_action, thought_experiment,
                                                                 other_thought_experiments_llm_ready_text)

        if for_and_against != {}:
            state.thought_experiments[runner]["arguments_for"] = for_and_against["for"]
            state.thought_experiments[runner]["arguments_against"] = for_and_against["against"]
        else:
            state.thought_experiments[runner]["arguments_for"] = ""
            state.thought_experiments[runner]["arguments_against"] = ""

        runner += 1
    best_action: Dict[str, str] = retry_fail_json_output(decide_what_the_best_action_to_take_is)

    # check if key in dict

    if "id" in best_action:
        state.best_action = best_action["id"]


def argue_best_action(
    thought_experiment_to_argue: str, other_thought_experiments: str
) -> Dict[str, str]:
    """
    The LLM will take in the thought experiment and the context that was used to get too that
    thought experiment and argue why it is the best action and why it is not
    :param thought_experiment_to_argue: The thought experiment to argue
    :param other_thought_experiments: The other thought experiments
    :return: The argument for why the thought experiment is the best action and why it is not
    """
    state = StateManager.get_instance().state
    prompt_template = get_argue_best_action_prompt()
    parser = JsonOutputParser()
    chain = prompt_template | llm | parser

    output = chain.invoke(
        {
            "thought_experiment_1": thought_experiment_to_argue,
            "other_thought_experiments": other_thought_experiments,
            "dilemma": state.situation,
            "historical_examples": make_historical_examples_used_pretty_text(),
        }
    )
    log_it_sync(logger, custom_message=f"type(chain): {type(output)}")
    if isinstance(output, dict):
        for_argument = output["for"]
        against_argument = output["against"]
        log_it_sync(
            logger,
            custom_message=f"Arguments for the thought experiment: {for_argument}",
        )
        log_it_sync(
            logger,
            custom_message=f"Arguments against the thought experiment: {against_argument}",
        )
        return {"for": for_argument, "against": against_argument}
    return {}


def decide_what_the_best_action_to_take_is() -> Dict[str, str]:
    """
    The LLM will take in the best action determined and the all of the context that was used to get too that
    action and summarize the results
    :param state: EthicistAIState object containing the chosen best action
    """
    state = StateManager.get_instance().state
    all_thought_experiments = make_all_thought_experiment_pretty_text_with_arguments()
    prompt_template = get_decide_best_action_prompt()
    parser = JsonOutputParser()
    chain = prompt_template | llm | parser
    output = chain.invoke(
        {
            "all_thought_experiments": all_thought_experiments,
        }
    )
    if isinstance(output, dict):
        return output
    return {}


def make_other_thought_experiments_pretty_text(
        other_thought_experiments: List[str],
) -> str:
    """
    Will take in the other thought experiments and make them into nice looking text that the LLM can read
    :param other_thought_experiments: The other thought experiments
    :return: The other thought experiments in nice looking text
    """
    text = ""
    for thought in other_thought_experiments:
        text += f"""
Other Thought Experiment:
{thought}
"""
        text += "\n"
    return text


def make_all_thought_experiment_pretty_text_with_arguments() -> str:
    """
    Will take in all of the thought experiments and make them into nice looking text that the LLM can read
    :return: The thought experiments in nice looking text
    """
    state = StateManager.get_instance().state
    text = ""
    for thought in state.thought_experiments:
        text += f"""
Thought Experiment:
{thought["summary"]}

Arguments For:
{thought["arguments_for"]}

Arguments Against:
{thought["arguments_against"]}

ID: {thought["id"]}
"""
        text += "\n"
    return text


def get_argue_best_action_prompt() -> PromptTemplate:
    return PromptTemplate(
        template="""
You are a world-renowned AI ethicist. You have been tasked to argue why the first thought experiment is the best course of action and why it is not.

Dilemma: {dilemma}

Significant Historical Examples: {historical_examples}

The first thought experiment is as follows: {thought_experiment_1}

{other_thought_experiments}

Make the argument why the first thought experiment is not the best course of action and why it is.
Your answer should be concise and to the point.
ALWAYS output correct JSON format don't forget any commas or colons.
Output your answer in the following format:
{{
    "for": "Your argument for why the thought experiment is the best course of action",
    "against": "Your argument for why the thought experiment is not the best course of action"
}}
""",
        input_variables=["dilemma", "thought_experiment_1", "other_thought_experiments", "historical_examples"],
    )


def get_decide_best_action_prompt() -> PromptTemplate:
    return PromptTemplate(
        template="""
You are a world-renowned AI ethicist. You have been tasked to determine the best course of action based off of all of the thought experiments and their arguments.

All Thought Experiments:
{all_thought_experiments}

Analyze all of the thought experiments and their arguments and determine the best course of action.
Output the ID of the best course of action.
Example output:
{{
    "reasoning": "Your reasoning for why this is the best course of action"
    "id": 1,
}}
""",
        input_variables=["all_thought_experiments"],
    )


def make_historical_examples_used_pretty_text() -> str:
    """
    Will take in the historical examples used and make them into nice looking text that the LLM can read
    :return: The historical examples in nice looking text
    """
    state = StateManager.get_instance().state
    text = ""
    for example in state.historical_examples:
        text += f"""
Historical Example:
{example.situation}

Action Taken:
{example.action_taken}

Reasoning:
{example.reasoning}
"""
        text += "\n"
    return text
# Compare this snippet from src/tests/test_pick_action.py: