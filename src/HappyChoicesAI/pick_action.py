import json
import os
import threading
from typing import Dict, List, Tuple
import multiprocessing

from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from HappyChoicesAI.ai_state import EthicistAIState, ModelUsedAndThreadCount, StateManager
from global_code.langchain import invoke_with_retry, retry_fail_json_output
from global_code.helpful_functions import create_logger_error, log_it_sync

load_dotenv()

# Get the API key from the environment variable
api_key = os.getenv("OPENAI_API_KEY")
random_state = ModelUsedAndThreadCount.get_instance()
thread_count = random_state.state.thread_count
model_to_use = random_state.state.model_used

llm = ChatOpenAI(model=model_to_use, temperature=0, api_key=api_key)
logger = create_logger_error(
    file_path=os.path.abspath(__file__), name_of_log_file="pick_action"
)
"""

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

    total_thought_experiments = len(state.thought_experiments)

    threads = []

    for action in range(total_thought_experiments):
        thread = threading.Thread(target=argue_best_action_all_thought_experiments, args=(action,))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    log_it_sync(logger, custom_message=f"results: {state.thought_experiments}", log_level="debug")
    best_action: Dict[str, str] = retry_fail_json_output(decide_what_the_best_action_to_take_is)

    # check if key in dict

    if "id" in best_action:
        log_it_sync(
            logger,
            custom_message=f"Picked best action: {True}",
            log_level="info"
        )
        state.best_action = best_action["id"]


def argue_best_action_all_thought_experiments(id_of_thought_experiment: int):
    state = StateManager.get_instance().state

    thought_experiment = state.thought_experiments[id_of_thought_experiment]["summary"]

    # Get the other thought experiments into a list of strings with their summaries
    other_thought_experiments_dict = (
        state.thought_experiments[:id_of_thought_experiment]
        + state.thought_experiments[id_of_thought_experiment + 1:]
    )
    other_thought_experiments_list = [thought["summary"] for thought in other_thought_experiments_dict]

    other_thought_experiments_llm_ready_text: str = make_other_thought_experiments_pretty_text(
        other_thought_experiments_list)
    for_and_against: Dict[str, str] = retry_fail_json_output(argue_best_action, thought_experiment,
                                                             other_thought_experiments_llm_ready_text)

    if for_and_against:
        state.thought_experiments[id_of_thought_experiment]["arguments_for"] = for_and_against["for"]
        state.thought_experiments[id_of_thought_experiment]["arguments_against"] = for_and_against["against"]
    else:
        state.thought_experiments[id_of_thought_experiment]["arguments_for"] = ""
        state.thought_experiments[id_of_thought_experiment]["arguments_against"] = ""


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
    log_it_sync(logger, custom_message=f"type(chain): {type(output)}", log_level="debug")
    if isinstance(output, dict):
        for_argument = output["for"]
        against_argument = output["against"]
        log_it_sync(
            logger,
            custom_message=f"Arguments for the thought experiment: {for_argument}",
            log_level="debug"
        )
        log_it_sync(
            logger,
            custom_message=f"Arguments against the thought experiment: {against_argument}",
            log_level="debug"
        )
        log_it_sync(logger, custom_message=f"Arguments against the thought experiment: passed", log_level="info")
        return {"for": for_argument, "against": against_argument}
    log_it_sync(logger, custom_message=f"Arguments against the thought experiment: failed", log_level="info")
    return {}


def decide_what_the_best_action_to_take_is() -> Dict[str, str]:
    """
    The LLM will take in the best action determined and the all of the context that was used to get too that
    action and summarize the results
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
        log_it_sync(logger, custom_message=f"decide_what_the_best_action_to_take_is: {output}",
                    log_level="debug")
        log_it_sync(logger, custom_message=f"decide_what_the_best_action_to_take_is: {True}",
                    log_level="info")
        return output
    log_it_sync(logger, custom_message=f"decide_what_the_best_action_to_take_is Output: {output}",
                log_level="info")
    log_it_sync(logger, custom_message=f"decide_what_the_best_action_to_take_is: {False}",
                log_level="info")
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
You are a world-renowned utilitarian AI ethicist. You have been tasked to argue why the first thought experiment is the best course of action and why it is not.

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
You are a world-renowned AI utilitarian ethicist. You have been tasked to determine the best course of action based off of all of the thought experiments and their arguments.

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