from HappyChoicesAI.ai_state import EthicistAIState

from langchain_core.prompts import PromptTemplate

"""
TODO: Implement the summarize_results function that will summarize the results of the ethical dilemma. (Should be supa ezpz)
"""


def summarize_results(state: EthicistAIState) -> None:
    """
    The LLM will take in the best action determined and the all of the context that was used to get too that
    action and summarize the results
    :param state: EthicistAIState object containing the chosen best action
    """
    prompt_template = PromptTemplate(
        input_variables=[
            "input_dilemma",
            "historical_examples",
            "best_action",
            "criteria",
        ],
        template="""You are a world renowned AI ethicist summarizer. 
You have been tasked to summarize the results of the following dilemma: {input_dilemma}.
The historical examples that were considered are: {historical_examples}.
The key ethical criteria relevant to utilitarian ethics are: {criteria}.
The best action determined is: {best_action}.
Your job is to craft a concise summary of the results.""",
    )
    prompt = prompt_template.render(dilemma=input_dillema)
    response = llm(prompt)
