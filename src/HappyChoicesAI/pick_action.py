import os
from typing import Dict, List, Tuple

from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from HappyChoicesAI.ai_state import EthicistAIState

load_dotenv()

# Get the API key from the environment variable
api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=api_key)

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


def pick_best_action(state: EthicistAIState) -> None:
    """
    Determines and updates the best action based on the outcomes of the thought experiments.
    The best action is based on maximizing happiness and minimizing suffering.
    :param state: EthicistAIState object containing all relevant data
    """
    # First have the llm read all of the thought experiments
    # For each thought experiment have the LLM argue why it is the best action and why it is not
    # Than have a final LLM look through all of the thought experiments and all of the arguments and determine the best action

    # First make the thought experiments into nice looking text
    text_of_thought_experiments = []
    for thought in state.thought_experiments:
        text = """
Thought Experiment:
        """
    rename_me: List[Tuple] = []
    runner = 0
    while runner < len(text_of_thought_experiments):
        other_thought_experiments = (
            text_of_thought_experiments[:runner]
            + text_of_thought_experiments[runner + 1 :]
        )
        text_of_other_thought_experiments = make_other_thought_experiments_pretty_text(
            other_thought_experiments
        )
        best_action = argue_best_action(
            thought_experiment_to_argue=text_of_thought_experiments[runner],
            other_thought_experiments=text_of_other_thought_experiments,
        )
        rename_me.append((text_of_thought_experiments[runner], best_action))
        runner += 1

    # Now we have all of the arguments for why each thought experiment is good and bad
    # Now we have an LLM pick the best one
    winning_thought_experiment = decide_what_the_best_action_to_take_is()
    state.best_action = winning_thought_experiment


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
    # new plan the LLM should ouput a JSON with the arguments for and against the thought experiment
    prompt_template = ChatPromptTemplate.from_template(
        """
You are a world renowned AI ethicist. You have been tasked to argue why the first thought experiment is the best course of action and why it is not.

Dilemma: {dilemma}

Significant Historical Examples: {historical_examples}

The first thought experiment is as follows: {thought_experiment_1}

{other_thought_experiments}

Make the argument why the first thought experiment is not the best course of action and why it is.
Your answer should be concise and to the point.
Output your answer in the following format:
{
    "for": "Your argument for why the thought experiment is the best course of action",
    "against": "Your argument for why the thought experiment is not the best course of action"
}
"""
    )
    chain = prompt_template | llm
    output = chain.invoke(
        {
            "thought_experiment_1": thought_experiment_to_argue,
            "other_thought_experiments": other_thought_experiments,
        }
    )
    # Turn the output into a JSON
    output_parser = JsonOutputParser()
    # The str() around output may break it
    parsed_output: dict = output_parser.parse(str(output))
    return parsed_output


def decide_what_the_best_action_to_take_is(rename_me: List[Tuple]) -> str:
    """
    The LLM will take in the best action determined and the all of the context that was used to get too that
    action and summarize the results
    :param state: EthicistAIState object containing the chosen best action
    """
    thought_experiments_and_arguments = ""
    for thought_experiment, argument in rename_me:
        thought_experiments_and_arguments += f"""
Thought Experiment:
{thought_experiment}
Argument:
{argument}
        
"""
    prompt_template = ChatPromptTemplate.from_template(
        """
You are a world renowned AI ethicist. You have been tasked to figure out the most ethical action to take in the given scenario

{thought_experiments_and_arguments}

Make the argument why the first thought experiment is the best course of action and why it is not."""
    )


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
Thought Experiment:
{thought}
"""
        text += "\n"
    return text
