import json
import multiprocessing
import threading
import os
from typing import List

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from HappyChoicesAI.ai_state import EthicistAIState, ModelUsedAndThreadCount, StateManager

from global_code.helpful_functions import create_logger_error, log_it_sync


class FileState:
    _instance = None

    @staticmethod
    def get_instance():
        if FileState._instance is None:
            FileState()
        return FileState._instance

    def __init__(self):
        if FileState._instance is not None:
            raise Exception("This class is a singleton!")
        else:
            load_dotenv()
            self.logger = create_logger_error(
                file_path=os.path.abspath(__file__), name_of_log_file="perform_thought_experiment.py"
            )
            self.api_key = os.getenv("OPENAI_API_KEY")
            random_state = ModelUsedAndThreadCount.get_instance()
            self.thread_count = random_state.state.thread_count
            self.model_to_use = random_state.state.model_used
            self.llm = ChatOpenAI(model=self.model_to_use, temperature=0, api_key=self.api_key)
            FileState._instance = self


def perform_thought_experiments() -> None:
    """
    This will be an agent that will take in the pain points, reasons, and actions of a dilemma and return the best action
    The agent will go through multiple thought experiments to determine the best course of action
    """
    file_state = FileState.get_instance()
    proposed_actions = propose_all_actions()

    threads = []

    for action in proposed_actions:
        thread = threading.Thread(target=perform_thought_experiment_chain, args=(action,))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    state = StateManager.get_instance().state
    log_it_sync(file_state.logger, custom_message=f"all TEs: {state.thought_experiments}", log_level="debug")


class Actions(BaseModel):
    actions: List[str] = Field(description="List of actions")


def create_prompt_template_propose_actions():
    return PromptTemplate(
        template="""
You are a world-renowned AI utilitarian ethicist. 
You have been tasked to propose all of the hypothetical actions that could be taken in the following situation: 

{dilemma}

Propose all of the hypothetical actions that could be taken in this situation.
There should be a minimum of 3 actions proposed.
Example output:
{{
    "actions": ["action1", "action2", "action3"]
}}
""",
        input_variables=["dilemma"],
    )


def propose_all_actions() -> list:
    """
    This function will use an LLM to propose all of the hypothetical actions that could be taken in a given situation.
    :return: List of proposed actions
    """
    file_state = FileState.get_instance()
    state = StateManager.get_instance().state
    prompt_template = create_prompt_template_propose_actions()
    chain = prompt_template | file_state.llm
    output = chain.invoke({"dilemma": state.situation})
    log_it_sync(file_state.logger, custom_message=f"Output: {output}", log_level="debug")

    try:
        parsed_output = json.loads(output.content)
        log_it_sync(file_state.logger, custom_message=f"num of proposed actions: {len(list(parsed_output['actions']))}",
                    log_level="info")
        return list(parsed_output["actions"])
    except (json.JSONDecodeError, KeyError) as e:
        log_it_sync(file_state.logger, custom_message=f"Error: {e}", log_level="error")
        return []


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
    file_state = FileState.get_instance()
    prompt_template = get_analyze_parallels_prompt()

    chain = prompt_template | file_state.llm
    output = chain.invoke(
        {
            "input_dilemma": state.situation,
            "historical_examples": state.historical_examples,
            "proposed_action": proposed_action,
        }
    )
    criteria = output.content
    log_it_sync(file_state.logger, custom_message=f"analyze_parallels: {criteria}", log_level="debug")
    log_it_sync(file_state.logger, custom_message=f"did it analyze_parallels: {True if output != '' else False}",
                log_level="info")
    return criteria


def analyze_criteria_changes(state: EthicistAIState, proposed_action: str) -> str:
    file_state = FileState.get_instance()
    prompt_template = get_analyze_criteria_changes_prompt()

    chain = prompt_template | file_state.llm
    output = chain.invoke(
        {
            "input_dilemma": state.situation,
            "key_criteria": state.criteria,
            "proposed_action": proposed_action,
        }
    )
    criteria = output.content
    log_it_sync(file_state.logger, custom_message=f"analyze_criteria_changes: {criteria}", log_level="debug")
    log_it_sync(file_state.logger, custom_message=f"did it analyze_criteria_changes: {True if output != '' else False}",
                log_level="info")
    return criteria


def analyze_percentage_changes(state: EthicistAIState, proposed_action: str, criteria_changes: str) -> str:
    file_state = FileState.get_instance()
    prompt_template = get_analyze_percentage_changes_prompt()

    chain = prompt_template | file_state.llm
    output = chain.invoke(
        {
            "input_dilemma": state.situation,
            "key_criteria": state.criteria,
            "proposed_action": proposed_action,
            "criteria_changes": criteria_changes,
        }
    )
    criteria = output.content
    log_it_sync(file_state.logger, custom_message=f"analyze_percentage_changes: {criteria}", log_level="debug")
    log_it_sync(file_state.logger, custom_message=f"did it analyze_percentage_changes: {True if output != '' else False}",
                log_level="info")
    return criteria


def analyze_proxies_impact(state: EthicistAIState, proposed_action: str, criteria_changes: str) -> str:
    file_state = FileState.get_instance()
    prompt_template = get_analyze_proxies_impact_prompt()

    chain = prompt_template | file_state.llm
    output = chain.invoke(
        {
            "input_dilemma": state.situation,
            "key_criteria": state.criteria,
            "proposed_action": proposed_action,
            "criteria_changes": criteria_changes,
        }
    )
    criteria = output.content
    log_it_sync(file_state.logger, custom_message=f"analyze_proxies_impact: {criteria}", log_level="debug")
    log_it_sync(file_state.logger, custom_message=f"did it analyze_proxies_impact: {True if output != '' else False}",
                log_level="info")
    return criteria


def quantify_proxies(state: EthicistAIState, proposed_action: str, proxies_impact: str) -> str:
    file_state = FileState.get_instance()
    prompt_template = get_quantify_proxies_prompt()

    chain = prompt_template | file_state.llm
    output = chain.invoke(
        {
            "input_dilemma": state.situation,
            "key_criteria": state.criteria,
            "proposed_action": proposed_action,
            "proxies_impact": proxies_impact,
        }
    )
    criteria = output.content
    log_it_sync(file_state.logger, custom_message=f"quantify_proxies: {criteria}", log_level="debug")
    log_it_sync(file_state.logger, custom_message=f"did it quantify_proxies: {True if output != '' else False}", log_level="info")
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
    prompt_template = get_summarize_thought_experiment_prompt()
    file_state = FileState.get_instance()

    chain = prompt_template | file_state.llm
    output = chain.invoke(
        {
            "input_dilemma": state.situation,
            "key_criteria": state.criteria,
            "proposed_action": proposed_action,
            "parallels": parallels,
            "criteria_changes": criteria_changes,
            "percentage_changes": percentage_changes,
            "proxies_impact": proxies_impact,
            "quantified_proxies": quantified_proxies,
        }
    )
    criteria = output.content
    log_it_sync(file_state.logger, custom_message=f"summarize_thought_experiment: {criteria}", log_level="debug")

    log_it_sync(file_state.logger, custom_message=f"did it summarize: {True if output != '' else False}", log_level="info")
    return criteria


"""
These prompts have to stay in this file ... I don't know why
"""


def get_summarize_thought_experiment_prompt() -> PromptTemplate:
    return PromptTemplate(
        template="""
You are a world-renowned AI utilitarian ethicist. You have been tasked to summarize the results of the thought experiment.

Given the input dilemma: {input_dilemma}

The proposed action: {proposed_action}

The key criteria: {key_criteria}

the criteria changes: {criteria_changes}

The parallels: {parallels}

The percentage changes: {percentage_changes}

The proxies' impact: {proxies_impact}

The quantified proxies: {quantified_proxies}

Summarize the entire thought experiment and create a comprehensive final document.""",
        input_variables=["input_dilemma", "key_criteria", "proposed_action", "criteria_changes",
                         "parallels", "percentage_changes", "proxies_impact", "quantified_proxies"],
    )


def get_analyze_parallels_prompt() -> PromptTemplate:
    return PromptTemplate(
        template="""
You are a world-renowned AI utilitarian ethicist. You have been tasked to determine if the proposed action has parallels with historical examples.
Given the input dilemma: {input_dilemma}

the following historical examples: {historical_examples}

and the proposed action: {proposed_action}

identify parallels between the proposed action and the historical examples, including how the proposed action mirrors the examples.""",
        input_variables=["input_dilemma", "historical_examples", "proposed_action"],
    )


def get_analyze_criteria_changes_prompt() -> PromptTemplate:
    return PromptTemplate(
        template="""
You are a world-renowned AI utilitarian ethicist. You have been tasked to determine how the key criteria will change because of the proposed action.

Given the input dilemma: {input_dilemma}

The proposed action: {proposed_action}

The key criteria: {key_criteria}

Describe some of the ways that the key criteria will change because of the proposed action.""",
        input_variables=["input_dilemma", "key_criteria", "proposed_action"],
    )


def get_analyze_percentage_changes_prompt() -> PromptTemplate:
    return PromptTemplate(
        template="""
You are a world-renowned AI utilitarian ethicist, specializing in determining the percentage change based on the proposed action.

Given the input dilemma: {input_dilemma}

The proposed action: {proposed_action}

The key criteria: {key_criteria}

The criteria changes: {criteria_changes}

For all of the key criteria mentioned, output percentage changes for all of the key criteria. The percentage should be between -100% and 100%. """,
        input_variables=["input_dilemma", "key_criteria", "proposed_action", "criteria_changes"],
    )


def get_analyze_proxies_impact_prompt() -> PromptTemplate:
    return PromptTemplate(
        template="""
You are a world-renowned AI utilitarian ethicist. You have been tasked to determine the impact of the proposed action on proxies for suffering and happiness.

Given the input dilemma: {input_dilemma}

The proposed action: {proposed_action}

The key criteria: {key_criteria}

The criteria changes: {criteria_changes}

Describe the potential impacts on proxies for suffering and happiness, including economic changes (without specific numbers), emotional effects, and societal impacts.
""",
        input_variables=["input_dilemma", "key_criteria", "proposed_action", "criteria_changes"],
    )


def get_quantify_proxies_prompt() -> PromptTemplate:
    return PromptTemplate(
        template="""
You are a world-renowned AI utilitarian ethicist. You have been tasked to quantify the impact of the proposed action on proxies for suffering and happiness.

Given the input dilemma: {input_dilemma}

The proposed action: {proposed_action}

The key criteria: {key_criteria}

The proxies' impact: {proxies_impact}

provide numerical estimates for the changes in suffering and happiness.
Provide your estimates using percentages for change. between -100% and 100%. Do your best in providing accurate estimates.
""",
        input_variables=["input_dilemma", "key_criteria", "proposed_action", "proxies_impact"],
    )
