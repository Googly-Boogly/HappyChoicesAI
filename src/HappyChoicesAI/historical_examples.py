import os
import threading
from typing import List

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI
from global_code.helpful_functions import CustomError, create_logger_error, log_it_sync
from HappyChoicesAI.ai_state import Database, EthicistAIState, HistoricalExample, StateManager, ModelUsedAndThreadCount
import multiprocessing


load_dotenv()
logger = create_logger_error(
    file_path=os.path.abspath(__file__), name_of_log_file="historical_examples"
)
# Get the API key from the environment variable
api_key = os.getenv("OPENAI_API_KEY")
random_state = ModelUsedAndThreadCount.get_instance()
thread_count = random_state.state.thread_count
model_to_use = random_state.state.model_used
llm = ChatOpenAI(model=model_to_use, temperature=0, api_key=api_key)

"""
The code works, need to ensure LLM outputs are good. (not tested) (always test last it is the most boring) (plus yo boi is tired)
"""


def find_historical_examples():
    """
    Will gather all of the relevant historical examples for the current situation and save them to the overall agent
    state
    :param input_dilemma: The current dilemma
    :param state: The state object
    :return: NA
    """

    historical_dilemmas = get_historical_examples()

    threads = []

    for action in historical_dilemmas:
        thread = threading.Thread(target=reason_and_add_to_state, args=(action,))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    state = StateManager.get_instance().state
    log_it_sync(logger, custom_message=f"historical examples check: {len(state.historical_examples)}", log_level="info")


def reason_and_add_to_state(dilemma: HistoricalExample):
    state = StateManager.get_instance().state
    y_or_n = reason_about_dilemma(dilemma)
    if y_or_n:
        # log_it_sync(logger, custom_message=f"saved 1 historic example: {True}",
        #             log_level="info")
        state.historical_examples.append(dilemma)


def get_historical_examples() -> List[HistoricalExample]:
    # Placeholder for actual retrieval logic

    db = Database(
        host="mysql", database="happychoices", user="root", password="password"
    )
    historical_examples: List[HistoricalExample] = db.get_all_historical_examples()
    return historical_examples


def create_prompt_template():
    return PromptTemplate(
        template="""You are a world renowned AI utilitarian ethicist. You have been tasked to determine if this historical dilemma is applicable to the current situation. 

The situation is as follows: {situation}. 

The historical dilemma is as follows: {dilemma}.

Do you think this dilemma is applicable? Answer either Yes or No""",
        input_variables=["situation", "dilemma"],
    )


def reason_about_dilemma(dilemma: HistoricalExample) -> bool:
    """
    Will use the LLM to reason about the current dilemma and the historical dilemma to determine if they are similar
    :param dilemma: The historical dilemma
    :return: Either True or False (if the dilemmas are similar)
    """
    prompt_template = create_prompt_template()
    input_dilemma = StateManager.get_instance().state.situation
    chain = prompt_template | llm
    output = chain.invoke({"situation": input_dilemma, "dilemma": dilemma.situation})
    log_it_sync(
        logger, custom_message=f"Output from LLM: {output.content}", log_level="debug"
    )
    response = output.content
    if response in ["yes", "yes.", "Yes", "Yes."]:
        return True
    return False
