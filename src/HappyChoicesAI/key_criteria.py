from typing import Dict, List, Tuple
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os
from HappyChoicesAI.ai_state import StateManager
from global_code.helpful_functions import log_it_sync, create_logger_error

load_dotenv()
logger = create_logger_error(
    file_path=os.path.abspath(__file__), name_of_log_file="key_criteria"
)
# Get the API key from the environment variable
api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0, api_key=api_key)

def create_prompt_template():
    return PromptTemplate(
        template="""
You are a world-renowned AI ethicist. You have been tasked to determine the key ethical criteria relevant to utilitarian ethics in the following situation: {dilemma}.

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
    state = StateManager.get_instance().state
    prompt_template = create_prompt_template()
    chain = prompt_template | llm
    output = chain.invoke({"dilemma": state.situation})
    criteria = output.content
    log_it_sync(logger, custom_message=f"Criteria: {criteria}")
    state.criteria = criteria