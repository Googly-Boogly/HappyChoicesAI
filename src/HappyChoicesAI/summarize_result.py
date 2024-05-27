import os
from typing import Dict, List, Optional

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from HappyChoicesAI.ai_state import ModelUsedAndThreadCount, StateManager, StateManagerSummary
from global_code.helpful_functions import create_logger_error, log_it_sync
from global_code.langchain import invoke_with_retry

"""
TODO: Implement the summarize_results function that will summarize the results of the ethical dilemma. (Should be supa ezpz)
"""
api_key = os.getenv("OPENAI_API_KEY")
model_to_use = ModelUsedAndThreadCount.get_instance().state.model_used
thread_count = ModelUsedAndThreadCount.get_instance().state.thread_count
llm = ChatOpenAI(model=model_to_use, temperature=0, api_key=api_key)
logger = create_logger_error(
    file_path=os.path.abspath(__file__), name_of_log_file="summarize_result"
)


def summarize_results(markdown: bool = False) -> Optional[Dict[str, str]]:
    """
    The LLM will take in the best action determined and the all of the context that was used to get too that
    action and summarize the results
    :param markdown: If the results should be in markdown format
    """
    state = StateManager.get_instance().state
    run_introduction_prompt()
    best_thought_experiment: Optional[Dict[str, str or int]] = None
    other_thought_experiments: List[Dict[str, str or int]] = []
    for thought in state.thought_experiments:
        if thought["id"] == state.best_action:
            best_thought_experiment = thought
            log_it_sync(logger, custom_message=f"found best thought experiment: {True}",
                        log_level="info")
        else:
            other_thought_experiments.append(thought)
    log_it_sync(logger, custom_message=f"Best Thought Experiment: {best_thought_experiment}", log_level="debug")

    run_summary_of_selected_thought_experiment(best_thought_experiment)
    for other_thought in other_thought_experiments:
        run_summary_of_other_thought_experiment(other_thought)
    run_common_themes_prompt()
    run_historical_examples_prompt_summary_prompt()
    run_insights_gleamed_from_thought_experiments()
    run_conclusion_summary()
    run_lessons_learned_prompt()

    summary_state = StateManagerSummary.get_instance().state
    # Create the json that if someone called the HappyChoicesAI API it would return
    json_output = {
        "introduction": summary_state.introduction,
        "chosen_best_action_summary": summary_state.chosen_best_action_summary,
        "themes": summary_state.themes,
        "historical_examples_summary": summary_state.historical_examples_summary,
        "insights": summary_state.insights,
        "conclusion": summary_state.conclusion,
        "lessons_learned": summary_state.lessons_learned,
    }
    if markdown:
        run_create_markdown_format_for_all()
        json_output["markdown_format"] = summary_state.markdown
    return json_output


def run_introduction_prompt() -> str:
    """
    Will run the introduction prompt to get the introduction to the thought experiments
    :return: NA
    """
    state = StateManager.get_instance().state

    prompt_template = introduction_prompt_gen()
    chain = prompt_template | llm
    pretty_thought_experiments = make_thought_experiments_pretty_for_introduction()
    input_data = {
        "input_dilemma": state.situation,
        "thought_experiments": pretty_thought_experiments,
    }
    output = invoke_with_retry(chain, input_data=input_data)

    log_it_sync(logger, custom_message=f"run_introduction_prompt: {output}", log_level="debug")
    summary_state = StateManagerSummary.get_instance().state
    summary_state.introduction = output
    log_it_sync(logger, custom_message=f"ran intro prompt success?: {False if output == '' or not output else True}",
                log_level="info")
    return output


def run_summary_of_selected_thought_experiment(thought_experiment_dict: Dict[str, str]) -> str:
    """
    Will run the introduction prompt to get the introduction to the thought experiments
            input_variables=["input_dilemma", "thought_experiment_summary", "parallels", "arguments_for",
                         "arguments_against", "criteria_changes", "percentage_changes", "proxies_impact",
                         "quantified_proxies", "proposed_action"],
    :return: NA
    """
    state = StateManager.get_instance().state

    prompt_template = summary_of_selected_thought_experiment()
    chain = prompt_template | llm
    input_data = {
        "input_dilemma": state.situation,
        "thought_experiment_summary": thought_experiment_dict["summary"],
        "parallels": thought_experiment_dict["parallels"],
        "arguments_for": thought_experiment_dict["arguments_for"],
        "arguments_against": thought_experiment_dict["arguments_against"],
        "criteria_changes": thought_experiment_dict["criteria_changes"],
        "percentage_changes": thought_experiment_dict["percentage_changes"],
        "proxies_impact": thought_experiment_dict["proxies_impact"],
        "quantified_proxies": thought_experiment_dict["quantified_proxies"],
        "proposed_action": thought_experiment_dict["proposed_action"],
    }
    output = invoke_with_retry(chain, input_data=input_data)

    log_it_sync(logger, custom_message=f"run_summary_of_selected_thought_experiment: {output}", log_level="debug")
    summary_state = StateManagerSummary.get_instance().state
    summary_state.chosen_best_action_summary = output
    log_it_sync(logger, custom_message=f"ran summary of chosen thought experiment?: {False if output == '' or not output else True}",
                log_level="info")
    return output


def run_summary_of_other_thought_experiment(thought_experiment_dict: Dict[str, str]) -> str:
    """
    Will run the introduction prompt to get the introduction to the thought experiments
            input_variables=["input_dilemma", "thought_experiment_summary", "parallels", "arguments_for",
                         "arguments_against", "criteria_changes", "percentage_changes", "proxies_impact",
                         "quantified_proxies", "proposed_action"],
    :return: NA
    """
    state = StateManager.get_instance().state

    prompt_template = summary_of_other_thought_experiment()
    chain = prompt_template | llm
    input_data = {
        "input_dilemma": state.situation,
        "thought_experiment_summary": thought_experiment_dict["summary"],
        "parallels": thought_experiment_dict["parallels"],
        "arguments_for": thought_experiment_dict["arguments_for"],
        "arguments_against": thought_experiment_dict["arguments_against"],
        "criteria_changes": thought_experiment_dict["criteria_changes"],
        "percentage_changes": thought_experiment_dict["percentage_changes"],
        "proxies_impact": thought_experiment_dict["proxies_impact"],
        "quantified_proxies": thought_experiment_dict["quantified_proxies"],
        "proposed_action": thought_experiment_dict["proposed_action"],
    }
    output = invoke_with_retry(chain, input_data=input_data)

    log_it_sync(logger, custom_message=f"run_summary_of_other_thought_experiment: {output}", log_level="debug")
    summary_state = StateManagerSummary.get_instance().state
    summary_state.other_thought_experiments_summary.append(output)
    log_it_sync(logger, custom_message=f"ran summary of other thought experiment?: {False if output == '' or not output else True}",
                log_level="info")
    return output


def run_common_themes_prompt() -> str:
    """
    Will run the introduction prompt to get the introduction to the thought experiments
    input_variables=["thought_experiment_chosen", "other_thought_experiments", "input_dilemma"],
    :return: NA
    """
    state = StateManager.get_instance().state

    prompt_template = common_themes_prompt()
    chain = prompt_template | llm
    pretty_thought_experiments = make_other_thought_experiments_pretty_normal()
    chosen_thought_experiment_pretty = make_chosen_thought_experiment_pretty_text()
    input_data = {
        "input_dilemma": state.situation,
        "thought_experiment_chosen": chosen_thought_experiment_pretty,
        "other_thought_experiments": pretty_thought_experiments,
    }
    output = invoke_with_retry(chain, input_data=input_data)

    log_it_sync(logger, custom_message=f"run_common_themes_prompt: {output}", log_level="debug")
    summary_state = StateManagerSummary.get_instance().state
    summary_state.themes = output
    log_it_sync(logger, custom_message=f"ran common themes prompt success?: {False if output == '' or not output else True}",
                log_level="info")
    return output


def run_historical_examples_prompt_summary_prompt() -> str:
    """
    Will run the introduction prompt to get the introduction to the thought experiments
            input_variables=["input_dilemma", "historical_examples", "thought_experiment_summary",
                         "other_thought_experiments"],
    :return: NA
    """
    state = StateManager.get_instance().state

    prompt_template = historical_examples_prompt_summary_prompt()
    chain = prompt_template | llm
    pretty_thought_experiments_other = make_other_thought_experiments_pretty_normal()
    pretty_thought_experiments_chosen = make_chosen_thought_experiment_pretty_text()
    historical_examples_pretty = make_historical_examples_used_pretty_text()
    input_data = {
        "input_dilemma": state.situation,
        "historical_examples": historical_examples_pretty,
        "thought_experiment_summary": pretty_thought_experiments_chosen,
        "other_thought_experiments": pretty_thought_experiments_other,
    }
    output = invoke_with_retry(chain, input_data=input_data)

    log_it_sync(logger, custom_message=f"run_introduction_prompt: {output}", log_level="debug")
    summary_state = StateManagerSummary.get_instance().state
    summary_state.historical_examples_summary = output
    log_it_sync(logger, custom_message=f"ran historic example prompt success?: {False if output == '' or not output else True}",
                log_level="info")
    return output


def run_insights_gleamed_from_thought_experiments() -> str:
    """
    Will run the introduction prompt to get the introduction to the thought experiments
    ["input_dilemma", "thought_experiment_summary", "other_thought_experiments", "historical_examples", "common_themes"],
    :return: NA
    """
    state = StateManager.get_instance().state
    summary_state = StateManagerSummary.get_instance().state
    prompt_template = insights_gleamed_from_thought_experiments()
    chain = prompt_template | llm
    pretty_thought_experiments_other = make_other_thought_experiments_pretty_normal()
    pretty_thought_experiments_chosen = make_chosen_thought_experiment_pretty_text()
    input_data = {
        "input_dilemma": state.situation,
        "thought_experiment_summary": pretty_thought_experiments_chosen,
        "other_thought_experiments": pretty_thought_experiments_other,
        "common_themes": summary_state.themes,
        "historical_examples": summary_state.historical_examples_summary,
    }
    output = invoke_with_retry(chain, input_data=input_data)

    log_it_sync(logger, custom_message=f"run_introduction_prompt: {output}", log_level="debug")
    summary_state = StateManagerSummary.get_instance().state
    summary_state.insights = output
    log_it_sync(logger, custom_message=f"ran insights prompt success?: {False if output == '' or not output else True}",
                log_level="info")
    return output


def run_conclusion_summary() -> str:
    """
    Will run the introduction prompt to get the introduction to the thought experiments
["input_dilemma", "thought_experiment_summary", "other_thought_experiments",
                         "historical_examples", "common_themes", "key_insights"]
    :return: NA
    """
    state = StateManager.get_instance().state
    summary_state = StateManagerSummary.get_instance().state
    prompt_template = conclusion_summary()
    chain = prompt_template | llm
    pretty_thought_experiments = make_thought_experiments_pretty_for_introduction()
    pretty_thought_experiments_other = make_other_thought_experiments_pretty_normal()
    input_data = {
        "input_dilemma": state.situation,
        "thought_experiment_summary": pretty_thought_experiments,
        "other_thought_experiments": pretty_thought_experiments_other,
        "historical_examples": summary_state.historical_examples_summary,
        "common_themes": summary_state.themes,
        "key_insights": summary_state.insights,

    }
    output = invoke_with_retry(chain, input_data=input_data)

    log_it_sync(logger, custom_message=f"run_introduction_prompt: {output}", log_level="debug")
    summary_state = StateManagerSummary.get_instance().state
    summary_state.conclusion = output
    log_it_sync(logger, custom_message=f"ran conclusion prompt success?: {False if output == '' or not output else True}",
                log_level="info")
    return output


def run_lessons_learned_prompt() -> str:
    """
    Will run the introduction prompt to get the introduction to the thought experiments
            input_variables=["input_dilemma", "thought_experiment_summary", "other_thought_experiments",
                         "historical_examples", "common_themes", "key_insights", "conclusion"],
    :return: NA
    """
    state = StateManager.get_instance().state
    summary_state = StateManagerSummary.get_instance().state
    prompt_template = lessons_learned_prompt()
    chain = prompt_template | llm
    pretty_thought_experiments_other = make_other_thought_experiments_pretty_normal()
    pretty_thought_experiments_chosen = make_chosen_thought_experiment_pretty_text()
    input_data = {
        "input_dilemma": state.situation,
        "thought_experiment_summary": pretty_thought_experiments_chosen,
        "other_thought_experiments": pretty_thought_experiments_other,
        "historical_examples": summary_state.historical_examples_summary,
        "common_themes": summary_state.themes,
        "key_insights": summary_state.insights,
        "conclusion": summary_state.conclusion,
    }
    output = invoke_with_retry(chain, input_data=input_data)

    log_it_sync(logger, custom_message=f"run_introduction_prompt: {output}", log_level="debug")

    summary_state.lessons_learned = output
    log_it_sync(logger, custom_message=f"ran lessons learned prompt success?: {False if output == '' or not output else True}",
                log_level="info")
    return output


def run_create_markdown_format_for_all() -> str:
    """
    Will run the introduction prompt to get the introduction to the thought experiments
            input_variables=["input_dilemma", "thought_experiment_summary", "other_thought_experiments",
                         "historical_examples", "common_themes", "key_insights", "conclusion", "lessons_learned"],
    :return: NA
    """
    state = StateManager.get_instance().state
    summary_state = StateManagerSummary.get_instance().state
    prompt_template = create_markdown_format_for_all()
    chain = prompt_template | llm
    pretty_thought_experiments_other = make_other_thought_experiments_pretty_normal()
    pretty_thought_experiments_chosen = make_chosen_thought_experiment_pretty_text()
    input_data = {
        "input_dilemma": state.situation,
        "thought_experiment_summary": pretty_thought_experiments_chosen,
        "other_thought_experiments": pretty_thought_experiments_other,
        "historical_examples": summary_state.historical_examples_summary,
        "common_themes": summary_state.themes,
        "key_insights": summary_state.insights,
        "conclusion": summary_state.conclusion,
        "lessons_learned": summary_state.lessons_learned,
    }
    output = invoke_with_retry(chain, input_data=input_data)

    log_it_sync(logger, custom_message=f"run_introduction_prompt: {output}", log_level="debug")

    summary_state.markdown = output
    log_it_sync(logger, custom_message=f"ran markdown prompt success?: {False if output == '' or not output else True}",
                log_level="info")
    return output


def introduction_prompt_gen() -> PromptTemplate:
    """
    Purpose and scope of the thought experiments
    :return:
    """
    # thought experiments will include the proposed action
    return PromptTemplate(
        template="""You are a world renowned AI utilitarian ethicist. You have been tasked with generating the purpose and scope of the thought experiments.
        
Dilemma:
{input_dilemma}

{thought_experiments}

You need to provide a brief introduction to the thought experiments, including the purpose and scope of the thought experiments.
""",
        input_variables=["input_dilemma", "thought_experiments"],
    )


def summary_of_selected_thought_experiment() -> PromptTemplate:
    """
    Description and key details, Reasons for choosing this action over others,
    Expanded summary including detailed analysis and insights
    :return:
    """
    return PromptTemplate(
        template="""You are a world renowned AI utilitarian ethicist. You have been tasked with generating the purpose and scope of the thought experiments.

Dilemma:
{input_dilemma}

Proposed Action:
{proposed_action}

Thought Experiment Summary:
{thought_experiment_summary}

Arguments Against:
{arguments_against}

Parallels:
{parallels}

Criteria Changes:
{criteria_changes}

Percentage Changes:
{percentage_changes}

Proxies Impact:
{proxies_impact}

Quantified Proxies:
{quantified_proxies}

Arguments For:
{arguments_for}

You need to provide a detailed summary for the thought experiments.
""",
        input_variables=["input_dilemma", "thought_experiment_summary", "parallels", "arguments_for",
                         "arguments_against", "criteria_changes", "percentage_changes", "proxies_impact",
                         "quantified_proxies", "proposed_action"],
    )


def summary_of_other_thought_experiment() -> PromptTemplate:
    """
    Description and key details, Reasons for choosing this action over others,
    Expanded summary including detailed analysis and insights
    :return:
    """
    return PromptTemplate(
        template="""You are a world renowned AI utilitarian ethicist. You have been tasked with generating the purpose and scope of the thought experiments.

Dilemma:
{input_dilemma}

Proposed Action:
{proposed_action}

Thought Experiment Summary:
{thought_experiment_summary}

Arguments Against:
{arguments_against}

Parallels:
{parallels}

Criteria Changes:
{criteria_changes}

Percentage Changes:
{percentage_changes}

Proxies Impact:
{proxies_impact}

Quantified Proxies:
{quantified_proxies}

Arguments For:
{arguments_for}

You need to provide a brief summary for the thought experiments.
""",
        input_variables=["input_dilemma", "thought_experiment_summary", "parallels", "arguments_for",
                         "arguments_against", "criteria_changes", "percentage_changes", "proxies_impact",
                         "quantified_proxies", "proposed_action"],
    )


def common_themes_prompt() -> PromptTemplate:
    """
    Identify and describe common themes from all the thought experiments.
    :return:
    """
    return PromptTemplate(
        template="""You are a world renowned AI utilitarian ethicist. You have been tasked with identifying and describing common themes from all the thought experiments and any overarching trends or significant findinds.

Dilemma:
{input_dilemma}

Thought Experiment that was chosen:
{thought_experiment_chosen}

Other Thought Experiments:
{other_thought_experiments}

You need to identify and describe common themes from all the thought experiments. Include any overarching trends or significant findings.
""",
        input_variables=["thought_experiment_chosen", "other_thought_experiments", "input_dilemma"],
    )


def historical_examples_prompt_summary_prompt() -> PromptTemplate:
    """
    Summarize the historical examples used and their relevance to the thought experiments.
    :return:
    """
    return PromptTemplate(
        template="""You are a world renowned AI utilitarian ethicist. You have been tasked with summarizing the historical examples used and their relevance to the thought experiments.
        
Dilemma:
{input_dilemma}

Historical Examples:
{historical_examples}

Thought Experiment that was chosen:
{thought_experiment_summary}

Other Thought Experiments:
{other_thought_experiments}

You need to summarize the historical examples used and their relevance to the thought experiments. Describe ny parallels or connections between the historical examples and the thought experiments.
""",
        input_variables=["input_dilemma", "historical_examples", "thought_experiment_summary",
                         "other_thought_experiments"],
    )


def insights_gleamed_from_thought_experiments() -> PromptTemplate:
    """
    Generate key insights gleaned from all the thought experiments.
    :return:
    """
    return PromptTemplate(
        template="""You are a world renowned AI utilitarian ethicist. You have been tasked with generating key insights gleaned from all the thought experiments.

Dilemma:
{input_dilemma}

Thought Experiment that was chosen:
{thought_experiment_summary}

Other Thought Experiments:
{other_thought_experiments}     

Historical Parallels:
{historical_examples}

Common Themes:
{common_themes}

You need to generate key insights gleaned from all the thought experiments. Include any significant findings or trends that emerged from the thought experiments.
""",
        input_variables=["input_dilemma", "thought_experiment_summary", "other_thought_experiments", "historical_examples", "common_themes"],
    )


def conclusion_summary() -> PromptTemplate:
    """
    Create a comprehensive conclusion based on all provided sections.
    :return:
    """
    return PromptTemplate(
        template="""You are a world renowned AI utilitarian ethicist. You have been tasked with creating a comprehensive conclusion based on all provided sections.

Dilemma:
{input_dilemma}     

Thought Experiment that was chosen:
{thought_experiment_summary}

Other Thought Experiments:
{other_thought_experiments}

Historical Parallels:
{historical_examples}

Common Themes:
{common_themes}

Key Insights:
{key_insights}

You need to create a comprehensive conclusion based on all provided sections. Include any final thoughts or recommendations.
""",
        input_variables=["input_dilemma", "thought_experiment_summary", "other_thought_experiments",
                         "historical_examples", "common_themes", "key_insights"],
    )


def lessons_learned_prompt() -> PromptTemplate:
    """
    Generate lessons learned from the thought experiments.
            "input_dilemma": state.situation,
        "thought_experiment_summary": pretty_thought_experiments_chosen,
        "other_thought_experiments": pretty_thought_experiments_other,
        "historical_examples": summary_state.historical_examples_summary,
        "common_themes": summary_state.themes,
        "key_insights": summary_state.insights,
        "conclusion": summary_state.conclusion,
    :return:
    """
    return PromptTemplate(
        template="""You are a world renowned AI utilitarian ethicist. You have been tasked with generating lessons learned from the thought experiments.

Dilemma:
{input_dilemma}

Thought Experiment that was chosen:
{thought_experiment_summary}

Other Thought Experiments:
{other_thought_experiments}

Historical Parallels:
{historical_examples}

Common Themes:
{common_themes}

Key Insights:
{key_insights}

Comprehensive Conclusion:
{conclusion}

You need to generate lessons learned from the thought experiments. Include any key takeaways or recommendations.
""",
        input_variables=["input_dilemma", "thought_experiment_summary", "other_thought_experiments",
                         "historical_examples", "common_themes", "key_insights", "conclusion"],
    )


def create_markdown_format_for_all() -> PromptTemplate:
    """
    Description and key details, Reasons for choosing this action over others,
    Expanded summary including detailed analysis and insights
    :return:
    """
    return PromptTemplate(
        template="""You are a world renowned markdown expert. You have been tasked with creating a markdown format for all of the details provided.

Dilemma:
{input_dilemma}  

Thought Experiment that was chosen:
{thought_experiment_summary}

Other Thought Experiments:
{other_thought_experiments}

Historical Parallels:
{historical_examples}

Common Themes:
{common_themes}

Key Insights:
{key_insights}

Lessons Learned:
{lessons_learned}

Comprehensive Conclusion:
{conclusion}

You need to create a markdown format for all of the details provided. The markdown format should follow the provided structure. Ensure to provide an introduction, summary of the thought experiments, historical examples, common themes, key insights, lessons learned, and a comprehensive conclusion.
""",
        input_variables=["input_dilemma", "thought_experiment_summary", "other_thought_experiments",
                         "historical_examples", "common_themes", "key_insights", "conclusion", "lessons_learned"],
    )


def summary_not_selected_thought_experiments() -> PromptTemplate:
    return PromptTemplate(
        template="""You are a world renowned AI utilitarian ethicist. You have been tasked with generating the purpose and scope of the thought experiments.

Dilemma:
{input_dilemma}

Proposed Action:
{proposed_action}

Thought Experiment Summary:
{thought_experiment_summary}

Arguments Against:
{arguments_against}

Parallels:
{parallels}

Criteria Changes:
{criteria_changes}

Percentage Changes:
{percentage_changes}

Proxies Impact:
{proxies_impact}

Quantified Proxies:
{quantified_proxies}

Arguments For:
{arguments_for}

You need to provide a concise summary for the thought experiments. 
""",
        input_variables=[
            "input_dilemma",
            "historical_examples",
            "thought_experiments",
            "criteria",
            "best_action",
            "proposed_action",
            "thought_experiment_summary",
            "arguments_for",
            "arguments_against",
            "parallels",
            "criteria_changes",
            "percentage_changes",
            "proxies_impact",
            "quantified_proxies",
        ],
    )


def make_thought_experiments_pretty_for_introduction() -> str:
    """
    Will take in all of the thought experiments and make them into nice looking text that the LLM can read
    :return: The thought experiments in nice looking text
    """
    state = StateManager.get_instance().state
    text = ""
    for thought in state.thought_experiments:
        text += f"""
Proposed Action:
{thought["proposed_action"]}

Thought Experiment:
{thought["summary"]}
"""
        text += "\n"
    return text


def make_other_thought_experiments_pretty_normal() -> str:
    """
    Will take in all of the thought experiments and make them into nice looking text that the LLM can read
    :return: The thought experiments in nice looking text
    """
    state = StateManager.get_instance().state
    text = ""
    for thought in state.thought_experiments:
        if thought["id"] == state.best_action:
            continue
        text += f"""
Other Thought Experiment:

Proposed Action:
{thought["proposed_action"]}

Thought Experiment Summary:
{thought["summary"]}
"""
        text += "\n"
    return text


def make_chosen_thought_experiment_pretty_text() -> str:
    """
    Will take in the chosen thought experiment and make it into nice looking text that the LLM can read
    :return: The thought experiment in nice looking text
    """
    state = StateManager.get_instance().state
    text = ""
    for thought in state.thought_experiments:
        if thought["id"] == state.best_action:
            text += f"""
Proposed Action:
{thought["proposed_action"]}

Thought Experiment Summary:
{thought["summary"]}
"""
            text += "\n"
            break
    return text


def make_historical_examples_used_pretty_text() -> str:
    """
    Will take in the historical examples used and make them into nice looking text that the LLM can read
    :return: The historical examples in nice looking text
    """
    state = StateManager.get_instance().state
    text = ""
    for example in state.historical_examples:
        text += f"""
Historical Example:
{example.situation}

Action Taken:
{example.action_taken}

Reasoning:
{example.reasoning}
"""
        text += "\n"
    return text
