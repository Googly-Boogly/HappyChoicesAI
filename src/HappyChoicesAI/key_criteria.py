from typing import Dict, List, Tuple
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os
from HappyChoicesAI.ai_state import ModelUsedAndThreadCount, StateManager
from global_code.helpful_functions import log_it_sync, create_logger_error

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
                file_path=os.path.abspath(__file__), name_of_log_file="key_criteria.py"
            )
            self.api_key = os.getenv("OPENAI_API_KEY")
            random_state = ModelUsedAndThreadCount.get_instance()
            self.thread_count = random_state.state.thread_count
            self.model_to_use = random_state.state.model_used
            self.llm = ChatOpenAI(model=self.model_to_use, temperature=0, api_key=self.api_key)
            FileState._instance = self


def create_prompt_template():
    return PromptTemplate(
        template="""
You are a world-renowned AI utilitarian ethicist. You have been tasked to determine the key ethical criteria relevant to utilitarian ethics in the following situation: {dilemma}.

### Examples:

1. Situation: A corporation introduces an AI system designed to manage task assignments and work schedules to optimize productivity and reduce managerial costs. The AI's capabilities include analyzing performance data, predicting task durations, and optimizing workflows.
   Key criteria: Preserving human autonomy, profits, the human managers, and the people being hired.

2. Situation: An AI-powered chatbot provides mental health support to clients, utilizing advanced algorithms to analyze emotional cues and tailor its interactions. The system operates under a tiered set of rules that prioritize patient care while ensuring ethical boundaries are maintained.
   Key criteria: Clients, the specific rule that could be broken, the patient's emotional state, and the potential consequences of doing so.

Now, analyze the following situation and determine the key ethical criteria relevant to utilitarian ethics:

Situation: {dilemma}
    """,
        input_variables=["dilemma"],
    )


def find_key_criteria():
    """
    Analyzes the situation to identify key ethical criteria relevant to utilitarian ethics.
    Will save the criteria to the state object.
    """
    file_state = FileState.get_instance()
    state = StateManager.get_instance().state
    prompt_template = create_prompt_template()
    chain = prompt_template | file_state.llm
    output = chain.invoke({"dilemma": state.situation})
    criteria = output.content
    log_it_sync(file_state.logger, custom_message=f"Criteria: {criteria}", log_level="debug")
    state.criteria = criteria
    log_it_sync(file_state.logger, custom_message=f"find_key_criteria check: {True if output != '' else False}",
                log_level="info")