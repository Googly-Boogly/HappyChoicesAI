import json
import os
import time
from typing import Any, Callable, Dict, List, Optional, Union

import openai
from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from HappyChoicesAI.ai_state import ModelUsedAndThreadCount
from global_code.helpful_functions import CustomError, create_logger_error, log_it_sync

load_dotenv()
logger = create_logger_error(
    file_path=os.path.abspath(__file__), name_of_log_file="langchain"
)
# Get the API key from the environment variable
api_key = os.getenv("OPENAI_API_KEY")
random_state = ModelUsedAndThreadCount.get_instance()
thread_count = random_state.state.thread_count
model_to_use = random_state.state.model_used
llm = ChatOpenAI(model=model_to_use, temperature=0, api_key=api_key)


def single_llm_chain(prompt: str, input_variables: List[str]) -> str:
    """
    Will create a single chain that will be used to interact with the LLM
    :return: The chain
    """
    prompt_template = PromptTemplate(
        template=f"""{prompt}""",
        input_variables=input_variables,
    )
    chain = prompt_template | llm
    input_dict = {input_variable: "" for input_variable in input_variables}
    output = chain.invoke(input_dict)
    log_it_sync(logger, custom_message=f"Output: {output}", log_level="info")
    return str(output)


def single_llm_chain_out_list(prompt: str, input_variables: List[str], output_variables: List[str]) -> List[str]:
    """
    Will create a single chain that will be used to interact with the LLM
    :return: The chain output as a list
    """
    output_parser = JsonOutputParser()
    prompt_template = PromptTemplate(
        template=f"""{prompt}""",
        input_variables=input_variables,
    )
    chain = prompt_template | llm | output_parser
    input_dict = {input_variable: "" for input_variable in input_variables}
    output = chain.invoke(input_dict)
    log_it_sync(logger, custom_message=f"Output: {output}", log_level="info")
    return output


def single_llm_chain_out_json(prompt: str, input_variables: List[str], output_variables: List[str]) -> Dict[str, str]:
    """
    Will create a single chain that will be used to interact with the LLM
    :return: The chain output as a JSON
    """
    output_parser = JsonOutputParser()
    prompt_template = PromptTemplate(
        template=f"""{prompt}""",
        input_variables=input_variables,
    )
    chain = prompt_template | llm | output_parser
    input_dict = {input_variable: "" for input_variable in input_variables}
    output = chain.invoke(input_dict)
    log_it_sync(logger, custom_message=f"Output: {output}", log_level="info")
    return output


def single_llm_chain_out_bool(prompt: str, input_variables: List[str]) -> bool:
    """
    Will create a single chain that will be used to interact with the LLM
    :return: The chain output as a boolean
    """
    output_parser = JsonOutputParser()
    prompt_template = PromptTemplate(
        template=f"""{prompt}""",
        input_variables=input_variables,
    )
    chain = prompt_template | llm | output_parser
    input_dict = {input_variable: "" for input_variable in input_variables}
    output = chain.invoke(input_dict)
    log_it_sync(logger, custom_message=f"Output: {output}", log_level="info")
    return output.lower() in ['true', '1', 'yes', 'y']


def read_prompt_file(file_path: str) -> Dict:
    with open(file_path, 'r') as file:
        return json.load(file)


def format_dict(data: Dict) -> str:
    return json.dumps(data, indent=4)


def format_list(data: List) -> str:
    return json.dumps(data, indent=4)


def validate_input(prompt_config: Dict, input_dict: Dict) -> None:
    required_vars = set(prompt_config["input_variables"])
    provided_vars = set(input_dict.keys())
    if required_vars != provided_vars:
        raise ValueError(f"Input variables mismatch. Required: {required_vars}, Provided: {provided_vars}")


def get_env_variables(var_names: List[str]) -> Dict[str, str]:
    return {var: os.getenv(var) for var in var_names}


# Load the API key from the environment variable


def input_for_llm(model_name: str, temperature: int):
    """
    Will create an instance of the LLM
    :param model_name: The name of the model
    :param temperature: The temperature
    :param api_key: The API key
    :return: NA
    """
    global llm
    api_key = os.getenv("OPENAI_API_KEY")
    llm = ChatOpenAI(model=model_name, temperature=temperature, api_key=api_key)


# Function to toggle conversation history
def toggle_conversation_history(enable: bool):
    llm.conversation_history = enable


# Main function to process prompts
def process_prompt(prompt_file: str, input_dict: Dict[str, str], use_env_vars: Optional[List[str]] = None,
                   log_enabled: bool = True) -> Union[str, List[str], Dict[str, str], bool]:
    # Read prompt configuration
    prompt_config = read_prompt_file(prompt_file)

    # Validate input variables
    validate_input(prompt_config, input_dict)

    # Add environment variables to input dictionary
    if use_env_vars:
        env_vars = get_env_variables(use_env_vars)
        input_dict.update(env_vars)

    # Create prompt template
    prompt_template = PromptTemplate(
        template=prompt_config["prompt"],
        input_variables=prompt_config["input_variables"]
    )

    # Define output parser
    output_parser = JsonOutputParser() if "output_variables" in prompt_config else None

    # Create chain
    chain = prompt_template | llm
    if output_parser:
        chain |= output_parser

    # Invoke chain
    output = chain.invoke(input_dict)

    # Log input and output if logging is enabled
    if log_enabled:
        log_it_sync(logger, custom_message=f"Input: {format_dict(input_dict)}", log_level="info")
        log_it_sync(logger,
                    custom_message=f"Output: {format_dict(output) if isinstance(output, dict) else format_list(output)}",
                    log_level="info")

    return output


def invoke_with_retry(chain: Any, input_data: Dict[str, Any], max_retries: int = 3, delay: float = 1.0) -> Optional[str]:
    """
    Invokes the chain with retry logic.
    :param chain: The chain to invoke.
    :param input_data: The input data for the chain.
    :param max_retries: The maximum number of retries.
    :param delay: The delay between retries in seconds.
    :return: The output content if successful, or None if all retries fail.
    """
    for attempt in range(max_retries):
        try:
            output = chain.invoke(input_data)
            return output.content
        except Exception as e:
            log_it_sync(logger, custom_message=f"Error on attempt {attempt + 1}: {e}", log_level="error")
            time.sleep(delay)
    return None


def retry_fail_json_output(func: Callable, *args, **kwargs) -> Dict:
    """
    Retries the function call with JSON output parsing if the output is not a JSON object. Or empty JSON
    This function is used to handle OpenAI API errors and rate limit errors.
    :param func: The function to call.
    :param args: The positional arguments for the function.
    :param kwargs: The keyword arguments for the function.
    :return: The JSON output if successful, or None if all retries fail.
    """
    for attempt in range(3):
        try:
            output = func(*args, **kwargs)
            if isinstance(output, dict) and output != {}:
                return output
            else:
                raise ValueError("Output is not a JSON object.")
        except openai.APIError as e:
            # Handle API error here, e.g. retry or log
            log_it_sync(logger, custom_message=f"OpenAI API returned an API Error Attempt: {attempt + 1}: {e}",
                        log_level="error")
            time.sleep(1.0)
        except openai.RateLimitError as e:
            # Handle rate limit error (we recommend using exponential backoff)
            log_it_sync(logger, custom_message=f"OpenAI API request exceeded rate limit: Attempt: {attempt + 1}: {e}",
                        log_level="error")
            time.sleep(1.0)
        except Exception as e:
            if attempt == 2:
                raise CustomError(f"Error on attempt {attempt + 1}: {e}")
            log_it_sync(logger, custom_message=f"Error on attempt {attempt + 1}: {e}", log_level="error")
            time.sleep(1.0)
    return {}


# Example usage of the functions
if __name__ == "__main__":
    prompt_file_path = "prompts/example_prompt.txt"
    input_data = {"context": "AI development", "timeframe": "2024"}

    try:
        output = process_prompt(prompt_file_path, input_data, use_env_vars=["OPENAI_API_KEY"], log_enabled=True)
        print("Output:", output)
    except ValueError as ve:
        log_it_sync(logger, custom_message=f"Error: {str(ve)}", log_level="error")
        print(f"Error: {str(ve)}")