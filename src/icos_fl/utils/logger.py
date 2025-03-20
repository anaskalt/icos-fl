"""Contains classes responsible for logging.

The main logger class Logger is actually a wrapper of the standard logging.Logger,
configured according to our needs. It can log both to stdout and file.

Module contains an extra class FakeLogger which provides the same interface but is
nothing more than print() wrapper. FakeLogger can be used as a placeholder, since
it can be called at any point Logger can.
"""

import logging
import os
import sys
import traceback
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, Optional, Union

from icos_fl.utils.colors import BWHT, CRI_CLR, DBG_CLR, ERR_CLR, FGRY, INF_CLR, WRN_CLR, paint
from icos_fl.utils.singleton import Singleton


class Logger(metaclass=Singleton):
    """A logging.Logger wrapper class.

    Args:
        level: The level of debug messages that will be filtered.
        logdir: Directory to store the log files.
        logfbasename: The basename of the logfile.
        logext: The file extension to be used for the log file.
        useconsole: Enable output to stdout.
        usefile: Enable output to file.
        usecolor: Enable colored output.
        logformat: Output message format string.
        rotmaxbytes: Max size in bytes of the rotation buffer.
        rotbakcount: Number of rotation buffers to use.
    """

    DEFAULT_LOGDIR = os.path.join(os.path.expanduser("~"), ".icos-fl", "logs")
    DEFAULT_LOGFBASENAME = "events"
    DEFAULT_LOGEXT = "log"
    DEFAULT_LOGLEVEL = logging.DEBUG
    DEFAULT_LOGFORMAT = "[%(asctime)s] %(levelname)-8s %(message)s"
    DEFAULT_ROT_MAX_BYTES = 100000
    DEFAULT_ROT_BAK_COUNT = 4

    def __init__(
        self,
        usefile: bool = False,
        useconsole: bool = True,
        usecolor: bool = True,
        level: int = DEFAULT_LOGLEVEL,
        logformat: str = DEFAULT_LOGFORMAT,
        logdir: str = DEFAULT_LOGDIR,
        logfbasename: str = DEFAULT_LOGFBASENAME,
        logext: str = DEFAULT_LOGEXT,
        rotmaxbytes: int = DEFAULT_ROT_MAX_BYTES,
        rotbakcount: int = DEFAULT_ROT_BAK_COUNT,
    ) -> None:
        """Initialize logger with given parameters.

        Args:
            usefile: Enable output to file.
            useconsole: Enable output to stdout.
            usecolor: Enable colored output.
            level: The level of debug messages that will be filtered.
            logformat: Output message format string.
            logdir: Directory to store the log files.
            logfbasename: The basename of the logfile.
            logext: The file extension to be used for the log file.
            rotmaxbytes: Max size in bytes of the rotation buffer.
            rotbakcount: Number of rotation buffers to use.
        """
        # Unload the arguments
        self.usefile = usefile
        self.useconsole = useconsole
        self.usecolor = usecolor
        self.level = level
        self.logformat = logformat
        self.logdir = logdir
        self.logfbasename = logfbasename
        self.logext = logext
        self.rotmaxbytes = rotmaxbytes
        self.rotbakcount = rotbakcount

        # Create logger -- From official docs:
        #       Note that Loggers are never instantiated directly, but always
        #       through the module-level function logging.getLogger(__name__).
        self.__logger = logging.getLogger(__name__)
        self._setup_logger()

        # Initial Message
        if self.usefile:
            self.__logger.info("Logger started")

    def report(self) -> str:
        """Reports the current options of the Logger.

        Returns:
            A report message with the current logger setup.
        """
        msg = "\n"
        msg += paint(FGRY, "                   -= ICOS-FL Logger =-") + "\n"
        msg += (
            paint(
                FGRY,
                " ------------------------------------------------------------------------------",
            )
            + "\n"
        )
        msg += (
            paint(FGRY, "   Log level                   : ")
            + paint(BWHT, f" {self.level} ({self.strlvl(self.level)})")
            + "\n"
        )
        msg += (
            paint(FGRY, "   Use colored messages        : ")
            + paint(BWHT, f" {self.usecolor}")
            + "\n"
        )
        msg += (
            paint(FGRY, "   Message format              : ")
            + paint(BWHT, f" {self.logformat}")
            + "\n"
        )
        msg += (
            paint(FGRY, "   Export to console (stdout)  : ")
            + paint(BWHT, f" {self.useconsole}")
            + "\n"
        )
        msg += (
            paint(FGRY, "   Export to file              : ")
            + paint(BWHT, f" {self.usefile}")
            + "\n"
        )
        if self.usefile:
            msg += (
                paint(FGRY, "     log directory             : ")
                + paint(BWHT, f" {self.logdir}")
                + "\n"
            )
            msg += (
                paint(FGRY, "     log file basename         : ")
                + paint(BWHT, f" {self.logfbasename}")
                + "\n"
            )
            msg += (
                paint(FGRY, "     log file extension        : ")
                + paint(BWHT, f" {self.logext}")
                + "\n"
            )
            msg += (
                paint(FGRY, "     log file                  : ")
                + paint(BWHT, f" {self.logfile}")
                + "\n"
            )
            msg += (
                paint(FGRY, "     rotation files            : ")
                + paint(BWHT, f" {self.rotbakcount}")
                + "\n"
            )
            msg += (
                paint(FGRY, "     rotation file bytes size  : ")
                + paint(BWHT, f" {self.rotmaxbytes}")
                + "\n"
            )
        return msg

    def __str__(self) -> str:
        """Return a string representation of the Logger instance."""
        return self.report()

    def _setup_logger(self) -> None:
        """Get logging.Logger instance and configure it."""
        # Set debug level
        self.__logger.setLevel(self.level)

        # Create Formatter for the main (file) handler
        log_formatter = logging.Formatter(self.logformat)

        if self.useconsole:
            # Create handler for the stdout stream
            console_handler = logging.StreamHandler(sys.stdout)

            # Add message formatter to the handler
            console_handler.setFormatter(log_formatter)

            # Add that handler to the logger
            self.__logger.addHandler(console_handler)

        if self.usefile:
            # Make sure that the directory exists, if not create it
            if not os.path.exists(self.logdir):
                os.makedirs(self.logdir)

            # Create File Handler (Rotating One)
            rot_file_handler = RotatingFileHandler(
                self.logfile,
                mode="a",
                maxBytes=self.rotmaxbytes,
                backupCount=self.rotbakcount,
                encoding=None,
                delay=False,
            )

            # Set formatter to the main (file) handler
            rot_file_handler.setFormatter(log_formatter)

            # Add main handler (file handler) to the logger object
            self.__logger.addHandler(rot_file_handler)

    def strlvl(self, level: int) -> str:
        """Return the string representation of each logging level.

        - level < 10 : DEBUG
        - level < 20 : INFO
        - level < 30 : WARNING
        - level < 40 : ERROR
        - level < 50 : CRITICAL

        Args:
            level: The logging level

        Returns:
            The string representation of the level
        """
        # From interger to string
        if level <= logging.DEBUG:
            return "DEBUG"
        elif level <= logging.INFO:
            return "INFO"
        elif level <= logging.WARNING:
            return "WARNING"
        elif level <= logging.ERROR:
            return "ERROR"
        else:
            return "CRITICAL"

    def reconfigure(self, conf: Optional[Dict[str, Any]] = None) -> None:
        """Reconfigure the logger according to the current setup.

        Drops all the logger handlers and creates new ones going
        through the _setup_logger() all over again.

        If conf argument is set, which is expected to be
        a dictionary of logger attributes, then the logger attributes
        are reset/reassigned to the new ones and then handles are
        recreated.

        Example:
            # for example this
            logger.reconfigure(conf={'level':"WARNING", 'logext':".txt"})

            # is equivalent with this
            logger.level = "WARNING"
            logger.logext = ".txt"
            logger.reconfigure()

        Args:
            conf: A dictionary of new options, where each
                item is a logger attribute.
        """
        # If a dictionary of new attribute values is provided then
        # update those attributes before proceeding to reconfiguration
        if conf:
            for key, val in conf.items():
                self.debug(f"Change LOG.{key} to {val}")
                setattr(self, key, val)
        # Drop all handlers
        while self.__logger.hasHandlers():
            self.__logger.removeHandler(self.__logger.handlers[0])
        # Set-up logger again
        self._setup_logger()
        self.__logger.info("Logger reconfigured")

    # ###########
    # Log methods
    # ###########

    def debug(self, msg: str) -> None:
        """Logs a debug message."""
        if self.usecolor:
            msg = paint(DBG_CLR, msg)
        self.__logger.debug(msg)

    def info(self, msg: str) -> None:
        """Logs an info message."""
        if self.usecolor:
            msg = paint(INF_CLR, msg)
        self.__logger.info(msg)

    def warning(self, msg: str) -> None:
        """Logs a warning message."""
        if self.usecolor:
            msg = paint(WRN_CLR, msg)
        self.__logger.warning(msg)

    def error(self, msg: str) -> None:
        """Logs an error message."""
        if self.usecolor:
            msg = paint(ERR_CLR, msg)
        self.__logger.error(msg)

    def exception(self, msg: Optional[str] = None) -> None:
        """Logs an exception message and the exception trace."""
        pad = "\t"
        trace = pad + traceback.format_exc().replace("\n", f"\n{pad}")
        if self.usecolor:
            if msg is None:
                msg = paint(ERR_CLR, "Exception occurred:\n") + paint(ERR_CLR, trace)
            else:
                msg = paint(ERR_CLR, msg) + "\n\n"
                msg += paint(ERR_CLR, trace)
        self.__logger.error(msg)

    def critical(self, msg: str) -> None:
        """Logs a critical message."""
        if self.usecolor:
            msg = paint(CRI_CLR, msg)
        self.__logger.critical(msg)

    # ###############
    # Getters/Setters
    # ###############

    @property
    def usefile(self) -> bool:
        """Get whether logging to file is enabled."""
        return self._usefile

    @usefile.setter  # type: ignore[attr-defined, no-redef]
    def usefile(self, val: bool) -> None:
        """Set whether logging to file is enabled."""
        # Could perform some sanity check here
        self._usefile = val

    @property
    def useconsole(self) -> bool:
        """Get whether logging to stdout is enabled."""
        return self._useconsole

    @useconsole.setter  # type: ignore[attr-defined, no-redef]
    def useconsole(self, val: bool) -> None:
        """Set whether logging to stdout is enabled."""
        # Could perform some sanity check here
        self._useconsole = val

    @property
    def usecolor(self) -> bool:
        """Get whether colored output is enabled."""
        return self._usecolor

    @usecolor.setter  # type: ignore[attr-defined, no-redef]
    def usecolor(self, val: bool) -> None:
        """Set whether colored output is enabled."""
        # Could perform some sanity check here
        self._usecolor = val

    @property
    def level(self) -> int:
        """Get the current logging level."""
        return self._level

    @level.setter  # type: ignore[attr-defined, no-redef]
    def level(self, val: Union[int, str]) -> None:
        """Set the logging level."""
        # Could perform some sanity check here
        if isinstance(val, str):
            if val.lower() == "debug":
                val = logging.DEBUG
            elif val.lower() == "info":
                val = logging.INFO
            elif val.lower() == "warning":
                val = logging.WARNING
            elif val.lower() == "error":
                val = logging.ERROR
            elif val.lower() == "exception":
                val = logging.ERROR
            elif val.lower() == "critical":
                val = logging.CRITICAL
            else:
                val = logging.DEBUG
        self._level = val

    @property
    def logformat(self) -> str:
        """Get the current log message format."""
        return self._logformat

    @logformat.setter  # type: ignore[attr-defined, no-redef]
    def logformat(self, val: str) -> None:
        """Set the log message format."""
        # Could perform some sanity check here
        self._logformat = val

    @property
    def logdir(self) -> str:
        """Get the current log directory."""
        return self._logdir

    @logdir.setter  # type: ignore[attr-defined, no-redef]
    def logdir(self, val: str) -> None:
        """Set the log directory path."""
        # Get rid of trailing slash
        self._logdir = val
        if len(val) > 1:
            self._logdir = val.rstrip("/")

    @property
    def logfbasename(self) -> str:
        """Get the current log file basename."""
        return self._logfbasename

    @logfbasename.setter  # type: ignore[attr-defined, no-redef]
    def logfbasename(self, val: str) -> None:
        """Set the log file basename."""
        # Could perform some sanity check here
        self._logfbasename = val.strip("/")

    @property
    def logext(self) -> str:
        """Get the current log file extension."""
        return self._logext

    @logext.setter  # type: ignore[attr-defined, no-redef]
    def logext(self, val: str) -> None:
        """Set the log file extension."""
        # Could perform some sanity check here
        self._logext = val

    @property
    def rotmaxbytes(self) -> int:
        """Get the current max size of the rotation buffer."""
        return self._rotmaxbytes

    @rotmaxbytes.setter  # type: ignore[attr-defined, no-redef]
    def rotmaxbytes(self, val: Union[int, str]) -> None:
        """Set the max size of the rotation buffer in bytes.

        Also allow input in the form of 1k, 2M, 3.2M.
        """
        # Could perform some sanity check here
        if isinstance(val, str):
            tmp = val.strip().lower()
            if tmp.endswith("k") or tmp.endswith("kb"):
                num = tmp.split("k")[0]
                val = int(float(num) * 1024)
            if tmp.endswith("m") or tmp.endswith("mb"):
                num = tmp.split("m")[0]
                val = int(float(num) * 1024 * 1024)
            else:
                val = int(val)
        self._rotmaxbytes = val

    @property
    def rotmaxcount(self) -> int:
        """Get the current number of rotation buffers."""
        return self._rotmaxcount

    @rotmaxcount.setter  # type: ignore[attr-defined, no-redef]
    def rotmaxcount(self, val: int) -> None:
        """Set the number of rotation buffers."""
        # Could perform some sanity check here
        self._rotmaxcount = val

    @property
    def logfile(self) -> str:
        """Get the full path to the log file."""
        return f"{self.logdir}/{self.logfbasename}.{self.logext}"
