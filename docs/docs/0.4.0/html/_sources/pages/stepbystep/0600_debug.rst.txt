Debugging
=========

If you see an error, check the error message to see if it pinpoints the issue.
The most common issues are misspecified configuration and incorrect data format.
Proper configuration is described at :doc:`/pages/stepbystep/0400_configuration`.

If the error message is not clear, try increasing logging to get more info, as shown below.
You may also increase the ``verbose`` parameter in ComputationParam
(See :doc:`/pages/stepbystep/0400_configuration`) for more logging during
sklearn grid search. Then, run your code again.

.. code-block:: python

    import logging
    from greykite.common.constants import LOGGER_NAME
    from greykite.common.logging import LoggingLevelEnum
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(LoggingLevelEnum.DEBUG.value)
