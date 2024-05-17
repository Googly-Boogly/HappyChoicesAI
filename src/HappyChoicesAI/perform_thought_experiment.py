from langchain.agents import Agent
from langchain_core.prompts import ChatPromptTemplate
from HappyChoicesAI.ai_state import EthicistAIState
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI


load_dotenv()

# Get the API key from the environment variable
api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=api_key)

"""
TODO: Implement the actual agent now The perform_thought_experiment_chain is done (not tested)
The agent will need to come up with X number of thought experiments
"""


def perform_thought_experiments(state: EthicistAIState) -> None:
    """
    This will be an agent that will take in the pain points, reasons, and actions of a dilemma and return the best action
    The agent will go through multiple thought experiments to determine the best course of action
    :param state: EthicistAIState object containing all relevant data
    """
    state.thought_experiments = [
        {"happiness": 80.0, "suffering": 20.0},
        {"happiness": 60.0, "suffering": 40.0},
    ]


class ThoughtExperimentAgent(Agent):
    def __init__(self, tools, context, state):
        super().__init__(tools=tools)
        self.context = context
        self.state = state

    def act(self, input_dilemma: str):
        self.state.situation = input_dilemma
        # Perform at least 3 thought experiments
        # new plan this will become an agent. The agent will have a tool called propose_action that will be used to propose an action
        # The agent will be stateful and have access to all of the proposed previous actions and their results
        # I guess the agent will need to have a state object that will contain all of the relevant data
        # I guess the agent should decide how many proposed actions for the given dilemma
        for i in range(3):
            proposed_action = self.propose_action()
            result = self.perform_thought_experiment_chain(
                input_dilemma, proposed_action
            )
            print(result)

        # Exit the agent
        return self.use_tool("exit_agent")

    def propose_action(self) -> str:
        # Placeholder for action proposal logic
        return f"Proposed Action {len(self.state.thought_experiments) + 1}"

    def perform_thought_experiment_chain(
        self, input_dilemma: str, proposed_action: str
    ) -> str:
        historical_examples = self.state.historical_examples
        key_criteria = self.state.criteria

        # Execute the chain of prompts
        parallels = analyze_parallels(state=self.state, proposed_action=proposed_action)
        criteria_changes = analyze_criteria_changes(
            state=self.state, proposed_action=proposed_action
        )
        percentage_changes = analyze_percentage_changes(
            state=self.state,
            proposed_action=proposed_action,
            criteria_changes=criteria_changes,
        )
        proxies_impact = analyze_proxies_impact(
            state=self.state,
            proposed_action=proposed_action,
            criteria_changes=criteria_changes,
        )
        quantified_proxies = quantify_proxies(
            state=self.state,
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
            state=self.state,
        )

        # Perform the thought experiment
        self.state.thought_experiments.append(
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
