"""
The boss error called everywhere :)
"""
import sys
from BIT_ADMIT_AI.logger import logging

# Figure out the error
def error_message_detail(error):
    _, _, exc_tb = sys.exc_info()
    if exc_tb is not None:
        file_name = exc_tb.tb_frame.f_code.co_filename
        line = exc_tb.tb_lineno
    else:
        file_name = "Unknown"
        line = "Unknown"
    error_message = f"Error occurred in [{file_name}] line [{line}] message [{error}]"
    logging.error(error_message)
    return error_message


class BitAdmitAIException(Exception):
    def __init__(self, error_message, sys):
        """param: error_mesage: str format of the error from the error_message_detail()"""
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message)

    def __str__(self):
        return self.error_message
