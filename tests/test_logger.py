"""Test module for the the Logger."""

import os
import sys
import tempfile
import threading
import time
from typing import Any, Dict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))
from icos_fl.utils.logger import Logger

# We want to have logger as a module object accessible from all functions,
# but we only want to start actual logging after the input arguments are
# passed.
log = None

# Create secure temporary directories for testing
TEMP_DIR_GENERAL = tempfile.mkdtemp(prefix="testLoggerGeneral_")
TEMP_DIR_ERRORS = tempfile.mkdtemp(prefix="testLoggerErrors_")
TEMP_DIR_EXCEPTIONS = tempfile.mkdtemp(prefix="testLoggerExceptions_")
TEMP_DIR_THREADS = tempfile.mkdtemp(prefix="testLoggerThreads_")

# A list of configuration
LOGGER_CONF = [
    {
        "usefile": True,
        "logdir": TEMP_DIR_GENERAL,
        "logfbasename": "scripta_manent",
        "rotmaxbytes": "4k",
        "rotbakcount": 2,
    },
    {
        "level": "error",
        "logdir": TEMP_DIR_ERRORS,
        "logfbasename": "errors",
        "logext": "log",
    },
    {"level": "exception", "logdir": TEMP_DIR_EXCEPTIONS, "logfbasename": "failures"},
    {
        "usefile": True,
        "logdir": TEMP_DIR_THREADS,
        "logfbasename": "scripta_manent",
        "rotmaxbytes": "4k",
        "rotbakcount": 2,
    },
    {"usefile": False, "useconsole": True, "level": "debug", "usecolor": True},
]


def log_simple_messages() -> None:
    """Log a series of test messages at different severity levels."""
    # Create a Logger instance (test Singleton)
    log = Logger()

    # Log some messages
    log.debug("From: log_simple_messages")
    log.debug("Hello, I am a debug message.")
    log.info("Hello everybody, I am an informative message.")
    log.warning("ATTENTION: We have a situation here.")
    log.error("...Well, that did not end up the way it should.")
    log.critical("deep shit")

    # Log trace tree
    try:
        raise Exception("Intentional Exception.")
    except Exception:
        log.exception()


def reconfigure_and_log(conf: Dict[str, Any]) -> None:
    """Reconfigure the logger with new settings and log test messages."""
    # Set the configuration directly using reconfigure method
    log.reconfigure(conf)
    print(log)

    # Log the basic messages but with different logger configuration
    log.debug("From: reconfigure_and_log")
    log_simple_messages()


def worker(name: str, per: float) -> None:
    """Worker thread function that performs operations and logs its progress.

    Args:
        name: Name identifier for the worker
        per: Period between operations in seconds
    """
    start_message = f"{name}: PrefPer: {per} -- Starting main loop"
    log.debug(start_message)
    counter = 0

    while True:
        tik = time.time()

        time_seed = int(tik * 1000)
        iters = (time_seed % 100 + 1) * [1, 10, 100][time_seed % 3]

        iterations_message = f"{name}: {iters} iters"
        log.debug(iterations_message)

        for _ in range(iters):
            try:
                seed1 = hash(f"{name}:{counter}:{tik}")
                seed2 = hash(f"{name}:{counter + 1}:{tik}")

                x = (seed1 % 1000) / 1000.0
                y = (seed2 % 1000) / 1000.0

                counter += 1
                if counter == 10000:
                    counter = 0
                    y = 0
                _ = x / y
            except Exception:
                log.exception()

        dur = time.time() - tik
        duration_message = f"{name}: Work duration {dur}"
        log.info(duration_message)
        sleep_dur = max(0.0, per - dur)
        time.sleep(sleep_dur)


def test() -> None:
    """Run the logger test suite with different configurations and worker threads."""
    # Create a Logger instance (test Singleton)
    global log
    log = Logger()
    print(log)

    # Test how logger behaves when instanciated with no arguments.
    log_simple_messages()

    # Change the configuration of the logger and log some extra messages
    for conf in LOGGER_CONF:
        reconfigure_and_log(conf)

    # Create N worker threads and let them log
    workers = []
    workers.append(threading.Thread(target=worker, args=("alpha", 0.5)))
    workers.append(threading.Thread(target=worker, args=("beta", 1.0)))
    workers.append(threading.Thread(target=worker, args=("gamma", 2.0)))
    workers.append(threading.Thread(target=worker, args=("delta", 3.0)))
    workers.append(threading.Thread(target=worker, args=("epsilon", 3.0)))

    # Raise threads
    for w in workers:
        w.start()


if __name__ == "__main__":
    test()
