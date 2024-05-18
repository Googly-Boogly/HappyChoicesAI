import inspect
import os
import logging
import time
import traceback
from typing import Any, Callable, Optional
import asyncio
import yaml
from datetime import datetime


class CustomError(Exception):
    """
    A robust custom error class that logs errors, includes error type, error code, and context information.
    """

    def __init__(
        self,
        message: str,
        error_type: Optional[str] = None,
        error_code: Optional[int] = None,
        context_info: Optional[dict] = None,
        config: Optional[dict] = None,
    ):
        """
        Initialize the CustomError with message, error_type, error_code, and context information.

        :param message: The error message.
        :param error_type: The type of the error (optional).
        :param error_code: The error code (optional).
        :param context_info: Additional context information as a dictionary (optional).
        :param config: Configuration dictionary for logging settings (optional).
        """
        self.message = message
        self.error_type = error_type
        self.error_code = error_code
        self.timestamp = datetime.now().isoformat()
        self.context_info = context_info or {}

        if config is None:
            config = load_config()

        self.logger = create_logger_error(
            file_path=os.path.abspath(__file__),
            name_of_log_file="error_log",
            config=config,
        )

        # Log the error upon initialization
        self.log_error()

        super().__init__(self.message)

    def log_error(self):
        """
        Log the error details using the configured logger.
        """
        log_message = f"{self.timestamp} - {self.error_type if self.error_type else 'Error'}: {self.message}"
        if self.error_code:
            log_message += f" (Error Code: {self.error_code})"
        if self.context_info:
            log_message += f" | Context: {self.context_info}"

        self.logger.error(log_message)
        self.logger.error("Stack Trace: %s", traceback.format_exc())

    def __str__(self):
        """
        Return a detailed string representation of the error.
        """
        error_str = f"[{self.timestamp}] {self.error_type if self.error_type else 'Error'}: {self.message}"
        if self.error_code:
            error_str += f" (Error Code: {self.error_code})"
        if self.context_info:
            error_str += f" | Context: {self.context_info}"
        return error_str

    def __repr__(self):
        """
        Return a representation of the error suitable for debugging.
        """
        return (
            f"CustomError(message={self.message!r}, error_type={self.error_type!r}, "
            f"error_code={self.error_code!r}, timestamp={self.timestamp!r}, "
            f"context_info={self.context_info!r})"
        )


def load_config(config_file: str = "config.yaml") -> dict:
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    return config


def create_logger_error(file_path: str, name_of_log_file: str) -> logging.Logger:
    """
    Creates a logger object.
    :param config: Configuration dictionary for logging settings.
    :param file_path: Absolute path of the file, this code: os.path.abspath(__file__)
    :param name_of_log_file: Name of the log file.
    :return: The python logger object.
    """
    config = load_config()
    logger = logging.getLogger(name_of_log_file)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    if config["logging"]["enabled"]:
        # Sets the format of the console log and adds it if required
        if config["logging"]["log_to_console"]:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

        if config["logging"]["log_to_file"]:
            caller_dir = os.path.dirname(os.path.abspath(file_path))
            logs_dir = os.path.join(caller_dir, config["logging"]["log_directory"])
            os.makedirs(logs_dir, exist_ok=True)

            # Create a subfolder named after the calling file (without extension)
            calling_file_name = os.path.splitext(os.path.basename(file_path))[0]
            file_logs_dir = os.path.join(logs_dir, calling_file_name)
            os.makedirs(file_logs_dir, exist_ok=True)

            log_file = os.path.join(file_logs_dir, f"{name_of_log_file}.log")
            handler = logging.FileHandler(log_file)
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        # Set logging level
        logger.setLevel(getattr(logging, config["logging"]["level"].upper()))

    return logger


async def log_it(
    logger: logging.Logger,
    error: Optional[Exception] = None,
    custom_message: Optional[str] = None,
    log_level: Optional[str] = None,
) -> Optional[Exception]:
    """
    Takes in logger and error and logs the error in the correct way.
    Works with both synchronous and asynchronous code.
    :param log_level: Has to be one of the following: debug, info, warning, error, critical
    :param custom_message: A custom message
    :param logger: Logger object made in one of the logger maker functions
    :param error: The error that occurred, this code: except Exception as e:
    :return: The Exception that was passed in or nothing if it was None
    """
    if isinstance(error, Exception):
        log_message = f"An exception occurred on line {traceback.extract_tb(error.__traceback__)[-1].lineno}: {error}"
        if log_level is not None:
            if custom_message is not None:
                log_message_custom_message = log_message + "\n" + custom_message
                if log_level == "debug":
                    logger.debug(log_message_custom_message)
                elif log_level == "info":
                    logger.info(log_message_custom_message)
                elif log_level == "warning":
                    logger.warning(log_message_custom_message)
                elif log_level == "error":
                    logger.error(log_message_custom_message)
                elif log_level == "critical":
                    logger.critical(log_message_custom_message)
                else:
                    raise CustomError(
                        "log_level must be one of the following: debug, info, warning, error, critical"
                    )
            else:
                if log_level == "debug":
                    logger.debug(log_message)
                elif log_level == "info":
                    logger.info(log_message)
                elif log_level == "warning":
                    logger.warning(log_message)
                elif log_level == "error":
                    logger.error(log_message)
                elif log_level == "critical":
                    logger.critical(log_message)
                else:
                    raise CustomError(
                        "log_level must be one of the following: debug, info, warning, error, critical"
                    )
        else:
            logger.error(log_message)
    elif error is None:
        if custom_message is not None:
            if log_level is not None:
                if log_level == "debug":
                    logger.debug(custom_message)
                elif log_level == "info":
                    logger.info(custom_message)
                elif log_level == "warning":
                    logger.warning(custom_message)
                elif log_level == "error":
                    logger.error(custom_message)
                elif log_level == "critical":
                    logger.critical(custom_message)
                else:
                    raise CustomError(
                        "log_level must be one of the following: debug, info, warning, error, critical"
                    )
            else:
                logger.info(custom_message)
    return error


def log_it_sync(
    logger: logging.Logger,
    error: Optional[Exception] = None,
    custom_message: Optional[str] = None,
    log_level: Optional[str] = None,
) -> Optional[Exception]:
    """
    Takes in logger and error and logs the error in the correct way.
    :param logger:
    :param error:
    :param custom_message:
    :param log_level:
    :return:
    """
    output = asyncio.run(log_it(logger, error, custom_message, log_level))
    return output


logger_benchmark = create_logger_error(
    file_path=os.path.abspath(__file__), name_of_log_file="logger_benchmark"
)


def benchmark_function(func: Optional[Callable] = None, *, file_prefix: Optional[str] = None):
    """
    Benchmarks your function and logs the benchmark to a log file.
    :param func: The function to be decorated.
    :param file_prefix: if you want to add a prefix to the function name in the log.
    :return: A decorator that benchmarks the decorated function.

    Usage:
        from this_module import benchmark_function

        @benchmark_function
        def some_function(*args, **kwargs):
            # function implementation

        @benchmark_function(file_prefix="prefix_")
        def another_function(*args, **kwargs):
            # function implementation
    """

    if func is None:
        # If func is None, it means the decorator was called with arguments
        def decorator(func: Callable):
            return benchmark_function(func, file_prefix=file_prefix)

        return decorator

    def wrapper(*args, **kwargs):
        calling_frame = inspect.stack()[1]
        file_path: str = calling_frame[1]
        file_name: str = func.__name__

        if file_prefix is not None:
            file_name = file_prefix + file_name

        start_time = time.time()

        result: Any = func(*args, **kwargs)

        end_time = time.time()
        duration = end_time - start_time
        log_it_sync(
            logger_benchmark,
            custom_message=f"{file_name} took {duration} seconds to execute.",
        )
        return result

    return wrapper
