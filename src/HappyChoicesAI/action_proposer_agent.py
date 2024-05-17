import os

from dotenv import load_dotenv
from langchain import hub
from langchain.agents import Agent
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools import Tool
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

from global_code.helpful_functions import create_logger_error, log_it_sync
from HappyChoicesAI.perform_thought_experiment import perform_thought_experiment_chain
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=api_key)
logger = create_logger_error(
    file_path=os.path.abspath(__file__), name_of_log_file="historical_examples"
)


class ProposeActionTool(Tool):
    def __init__(self):
        super().__init__(
            name="propose_action", description="Propose an action for the given dilemma"
        )

    def _call(self, input: str) -> str:
        # Here, input is the proposed action
        perform_thought_experiment_chain(state, input, proposed_action)
        return input


# Tool to indicate no more actions to propose
class NoMoreActionsTool(Tool):
    def __init__(self):
        super().__init__(
            name="no_more_actions", description="Indicate no more actions to propose"
        )

    def _call(self) -> str:
        return "No more actions to propose"


tools = [ProposeActionTool(), NoMoreActionsTool()]
prompt = hub.pull("hwchase17/openai-tools-agent")
log_it_sync(logger, custom_message=f"Prompt: {prompt}")
agent = create_openai_tools_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
# response = agent_executor.invoke({"input": "what is LangChain?"})
# print(response)
message_history = ChatMessageHistory()

agent_with_chat_history = RunnableWithMessageHistory(
    agent_executor,
    # This is needed because in most real world scenarios, a session id is needed
    # It isn't really used here because we are using a simple in memory ChatMessageHistory
    lambda session_id: message_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

agent_with_chat_history.invoke(
    {"input": "hi! I'm bob"},
    # This is needed because in most real world scenarios, a session id is needed
    # It isn't really used here because we are using a simple in memory ChatMessageHistory
    config={"configurable": {"session_id": "<foo>"}},
)
