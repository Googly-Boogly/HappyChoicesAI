import os

from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from HappyChoicesAI.ai_state import EthicistAIState, StateManager
from HappyChoicesAI.key_criteria import find_key_criteria
from HappyChoicesAI.historical_examples import find_historical_examples
from HappyChoicesAI.perform_thought_experiment import perform_thought_experiments
from HappyChoicesAI.pick_action import pick_best_action
from HappyChoicesAI.summarize_result import summarize_results
from global_code.langchain import process_prompt, toggle_conversation_history
from global_code.helpful_functions import benchmark_function
import time

from global_code.helpful_functions import create_logger_error, log_it_sync

"""
First thing to do tomorrow write test!
"""
load_dotenv()
logger = create_logger_error(
    file_path=os.path.abspath(__file__), name_of_log_file="historical_examples"
)
# Get the API key from the environment variable
api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0, api_key=api_key)


@benchmark_function
def main():
    time.sleep(5)
    dilemmas = [
        (
            "An autonomous vehicle must make a decision in a scenario where it can either swerve to avoid hitting a pedestrian but risk the safety of its passengers, or stay on course and potentially harm the pedestrian.",
            "A company is testing its new self-driving car technology and is concerned about how the AI will handle situations where it must choose between passenger safety and pedestrian safety.",
        ),
        (
            "An AI system in a hospital can prioritize treating patients based on their likelihood of recovery, potentially neglecting those with lower chances of survival even if they are in critical condition.",
            "A hospital is considering implementing an AI system to manage patient treatment schedules and is worried about the ethical implications of the AI prioritizing some patients over others.",
        ),
        (
            "A city uses AI-powered surveillance cameras to reduce crime, but the AI tends to disproportionately target certain neighborhoods, leading to concerns about bias and privacy.",
            "A city council is debating the deployment of AI surveillance systems to improve public safety, but there are significant concerns about privacy and potential biases in targeting specific communities.",
        ),
        (
            "An AI system in schools can tailor education plans to each student's strengths and weaknesses, but there is a risk that it might reinforce existing inequalities by providing more resources to already high-performing students.",
            "A school district is planning to introduce AI-driven personalized learning programs and is concerned about whether this could unintentionally widen the gap between high-performing and low-performing students.",
        ),
        (
            "An AI system used for hiring tends to favor candidates from certain backgrounds based on historical data, potentially leading to a lack of diversity and perpetuating existing biases.",
            "A tech company is considering using an AI-driven hiring platform to streamline their recruitment process but is wary of the possibility that the AI might introduce or perpetuate bias in their hiring decisions.",
        ),
    ]

    # prompt_file_path = "prompts/example_prompt.txt"
    # input_data = {"context": "Jake"}
    #
    # # try:
    # output = process_prompt(prompt_file_path, input_data, use_env_vars=["OPENAI_API_KEY"], log_enabled=True)
    # print("Output:", output)
    # # except ValueError as ve:
    # #     log_it_sync(logger, custom_message=f"Error: {str(ve)}", log_level="error")
    # #     print(f"Error: {str(ve)}")
    # raise NotImplemented
    output_parser = JsonOutputParser()
    prompt_template = PromptTemplate(
        template="""
You are a world renowned AI ethicist.
You have been tasked to propose all of the hypothetical actions that could be taken in the following situation:

{dilemma}

Propose all of the hypothetical actions that could be taken in this situation.
Your output should be a list of actions that could be taken.
Respond with a JSON object containing the list of actions.
EXAMPLE:
{{
    "actions": ["Action 1", "Action 2", "Action 3"]
}}
    """,
        input_variables=["query"],
    )

    # chain = prompt_template | llm | output_parser
    # output = chain.invoke({"dilemma": dilemmas[0][0]})
    #
    # log_it_sync(logger, custom_message=f"Output: {output['actions']}")

    # situation: str = input_situation()


    situation = dilemmas[0][0]
    state = StateManager.get_instance().state
    state.situation = situation
    find_historical_examples()
    find_key_criteria()

    perform_thought_experiments()

    pick_best_action()
    markdown = True
    summarize_results(markdown=markdown)

    log_it_sync(logger, custom_message=f"Final Summary: {state.final_summary.all_thought_experiments}")
    log_it_sync(logger, custom_message=f"Final Summary: {state.final_summary.conclusion}")
    log_it_sync(logger, custom_message=f"Final Summary: {state.final_summary.insights}")
    log_it_sync(logger, custom_message=f"Final Summary: {state.final_summary.historical_examples_summary}")
    log_it_sync(logger, custom_message=f"Final Summary: {state.final_summary.themes}")
    log_it_sync(logger, custom_message=f"Final Summary: {state.final_summary.chosen_best_action_summary}")
    log_it_sync(logger, custom_message=f"Final Summary: {state.final_summary.other_thought_experiments_summary}")
    log_it_sync(logger, custom_message=f"Final Summary: {state.final_summary.introduction}")
    log_it_sync(logger, custom_message=f"Final Summary: {state.final_summary.lessons_learned}")
    if markdown:
        log_it_sync(logger, custom_message=f"Final Summary: {state.final_summary.markdown}")

if __name__ == "__main__":
    main()
