import json
import os
from typing import List

from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser, SimpleJsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, ValidationError

from global_code.helpful_functions import create_logger_error, log_it_sync
from HappyChoicesAI.ai_state import EthicistAIState, StateManager

load_dotenv()

# Get the API key from the environment variable
api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=api_key)
logger = create_logger_error(
    file_path=os.path.abspath(__file__), name_of_log_file="perform_thought_experiment"
)
"""
TODO: Implement the actual agent now The perform_thought_experiment_chain is done (not tested)
The agent will need to come up with X number of thought experiments
"""


def perform_thought_experiments() -> None:
    """
    This will be an agent that will take in the pain points, reasons, and actions of a dilemma and return the best action
    The agent will go through multiple thought experiments to determine the best course of action
    """

    state = StateManager.get_instance().state
    proposed_actions = ["Proposed action 1", "Proposed action 2", "Proposed action 3"]

    for proposed_action in proposed_actions:
        perform_thought_experiment_chain(proposed_action)


class Actions(BaseModel):
    actions: List[str] = Field(description="List of actions")


def propose_all_actions():
    """
    This function will use an LLM to propose all of the hypothetical actions that could be taken in a given situation
    :return:
    """
    state = StateManager.get_instance().state
    output_parser = JsonOutputParser()
    prompt_template = PromptTemplate(
        template="""
You are a world renowned AI ethicist. 
You have been tasked to propose all of the hypothetical actions that could be taken in the following situation: 

{dilemma}

Propose all of the hypothetical actions that could be taken in this situation.
Your output should be a list of actions that could be taken.
""",
        input_variables=["query"],
    )

    chain = prompt_template | llm
    output = chain.invoke({"dilemma": state.situation})
    log_it_sync(logger, custom_message=f"Output: {output}")
    try:
        parsed_output = json.loads(str(output))
        if "actions" not in parsed_output:
            raise ValueError("JSON does not contain 'actions' key")
        # Validate the parsed output against the Actions model
        return parsed_output
    except (json.JSONDecodeError, ValidationError, ValueError) as e:
        print(f"Error parsing JSON: {e}")
        return {"actions": []}


def perform_thought_experiment_chain(
        proposed_action: str
) -> str:
    state = StateManager.get_instance().state
    historical_examples = state.historical_examples
    key_criteria = state.criteria

    # Execute the chain of prompts
    parallels = analyze_parallels(state=state, proposed_action=proposed_action)
    criteria_changes = analyze_criteria_changes(
        state=state, proposed_action=proposed_action
    )
    percentage_changes = analyze_percentage_changes(
        state=state,
        proposed_action=proposed_action,
        criteria_changes=criteria_changes,
    )
    proxies_impact = analyze_proxies_impact(
        state=state,
        proposed_action=proposed_action,
        criteria_changes=criteria_changes,
    )
    quantified_proxies = quantify_proxies(
        state=state,
        proposed_action=proposed_action,
        proxies_impact=proxies_impact,
    )
    summary = summarize_thought_experiment(
        proposed_action=proposed_action,
        parallels=parallels,
        criteria_changes=criteria_changes,
        percentage_changes=percentage_changes,
        proxies_impact=proxies_impact,
        quantified_proxies=quantified_proxies,
        state=state,
    )

    # Perform the thought experiment
    state.thought_experiments.append(
        {
            "proposed_action": proposed_action,
            "parallels": parallels,
            "criteria_changes": criteria_changes,
            "percentage_changes": percentage_changes,
            "proxies_impact": proxies_impact,
            "quantified_proxies": quantified_proxies,
            "summary": summary,
        }
    )
    return summary


def analyze_parallels(state: EthicistAIState, proposed_action: str) -> str:
    prompt_template = ChatPromptTemplate.from_template(
        """
You are a world renowned AI ethicist. You have been tasked to determine if the proposed action has parallels with historical examples.
Given the input dilemma: {input_dilemma}

the following historical examples: {historical_examples}

and the proposed action: {proposed_action}

identify parallels between the proposed action and the historical examples, including how the proposed action mirrors the examples."""
    )
    chain = prompt_template | llm
    output = chain.invoke(
        {
            "input_dilemma": state.situation,
            "historical_examples": state.historical_examples,
            "proposed_action": proposed_action,
        }
    )
    criteria = output.choices[0].text.strip()
    return criteria


def analyze_criteria_changes(state: EthicistAIState, proposed_action: str) -> str:
    prompt_template = ChatPromptTemplate.from_template(
        """
You are a world renowned AI ethicist. You have been tasked to determine how the key criteria will change because of the proposed action.
Given the input dilemma: {input_dilemma}

The proposed action: {proposed_action}

The key criteria: {key_criteria}

Describe some of the ways that the key criteria will change because of the proposed action."""
    )
    chain = prompt_template | llm
    output = chain.invoke(
        {
            "input_dilemma": state.situation,
            "key_criteria": state.criteria,
            "proposed_action": proposed_action,
        }
    )
    criteria = output.choices[0].text.strip()
    return criteria


def analyze_percentage_changes(
        state: EthicistAIState, proposed_action: str, criteria_changes: str
) -> str:
    prompt_template = ChatPromptTemplate.from_template(
        """
You are a world renowned AI Ethicist, specializing in determining the percentage change based on the proposed action.

Given the input dilemma: {input_dilemma}

The proposed action: {proposed_action}

The key criteria: {key_criteria}

The criteria changes: {criteria_changes}

For all of the key criteria mentioned, output percentage changes for all of the key criteria. The percentage should be between -100% and 100%. """
    )
    chain = prompt_template | llm
    output = chain.invoke(
        {
            "input_dilemma": state.situation,
            "key_criteria": state.criteria,
            "proposed_action": proposed_action,
            "criteria_changes": criteria_changes,
        }
    )
    criteria = output.choices[0].text.strip()
    return criteria


def analyze_proxies_impact(
        state: EthicistAIState, proposed_action: str, criteria_changes: str
) -> str:
    prompt_template = ChatPromptTemplate.from_template(
        """
You are a world renowned AI ethicist. You have been tasked to determine the impact of the proposed action on proxies for suffering and happiness.

Given the input dilemma: {input_dilemma}

The proposed action: {proposed_action}

The key criteria: {key_criteria}

The criteria changes: {criteria_changes}

Describe the potential impacts on proxies for suffering and happiness, including economic changes (without specific numbers), emotional effects, and societal impacts.
"""
    )
    chain = prompt_template | llm
    output = chain.invoke(
        {
            "input_dilemma": state.situation,
            "key_criteria": state.criteria,
            "proposed_action": proposed_action,
            "criteria_changes": criteria_changes,
        }
    )
    criteria = output.choices[0].text.strip()
    return criteria


def quantify_proxies(
        state: EthicistAIState, proposed_action: str, proxies_impact: str
) -> str:
    prompt_template = ChatPromptTemplate.from_template(
        """
You are a world renowned AI ethicist. You have been tasked to quantify the impact of the proposed action on proxies for suffering and happiness.

Given the input dilemma: {input_dilemma}

The proposed action: {proposed_action}

The key criteria: {key_criteria}

The proxies' impact: {proxies_impact}

provide numerical estimates for the changes in suffering and happiness."""
    )
    chain = prompt_template | llm
    output = chain.invoke(
        {
            "input_dilemma": state.situation,
            "key_criteria": state.criteria,
            "proposed_action": proposed_action,
            "proxies_impact": proxies_impact,
        }
    )
    criteria = output.choices[0].text.strip()
    return criteria


def summarize_thought_experiment(
        proposed_action: str,
        parallels: str,
        criteria_changes: str,
        percentage_changes: str,
        proxies_impact: str,
        quantified_proxies: str,
        state: EthicistAIState,
) -> str:
    prompt_template = ChatPromptTemplate.from_template(
        """
You are a world renowned AI ethicist. You have been tasked to summarize the results of the thought experiment.

Given the input dilemma: {input_dilemma}

The proposed action: {proposed_action}

The key criteria: {key_criteria}

the criteria changes: {criteria_changes}

The parallels: {parallels}

The percentage changes: {percentage_changes}

The proxies' impact: {proxies_impact}

The quantified proxies: {quantified_proxies}

Summarize the entire thought experiment and create a comprehensive final document."""
    )
    chain = prompt_template | llm
    output = chain.invoke(
        {
            "input_dilemma": state.situation,
            "key_criteria": state.criteria,
            "proposed_action": proposed_action,
            "proxies_impact": proxies_impact,
            "parallels": parallels,
            "criteria_changes": criteria_changes,
            "percentage_changes": percentage_changes,
            "quantified_proxies": quantified_proxies,
        }
    )
    criteria = output.choices[0].text.strip()
    return criteria
