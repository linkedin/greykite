"""
Test for logging.py
"""
from testfixtures import LogCapture

from greykite.common.constants import LOGGER_NAME
from greykite.common.logging import LoggingLevelEnum
from greykite.common.logging import log_message


def test_log_message():
    with LogCapture(LOGGER_NAME) as log_capture:
        log_message("Test log message.", LoggingLevelEnum.CRITICAL)
        log_capture.check(
            (LOGGER_NAME, "CRITICAL", "Test log message."))
