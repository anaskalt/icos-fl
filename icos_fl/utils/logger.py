"""Advanced logging system for ICOS Federated Learning.

Provides a robust logging system with file rotation, console output, and color support.
Built on top of Python's logging module with additional features like:
- Colored output for different log levels
- File rotation with size limits
- Configurable console/file output
- Singleton pattern for global access
"""

import os
import sys
import logging
import traceback
from logging.handlers import RotatingFileHandler
from typing import Optional, Dict
from icos_fl.utils.singleton import Singleton
from icos_fl.utils.colors import paint
from icos_fl.utils.colors import BWHT, FGRY
from icos_fl.utils.colors import DBG_CLR, INF_CLR, WRN_CLR, ERR_CLR, CRI_CLR

class ICOSLogger(metaclass=Singleton):
    """Advanced logging system for ICOS-FL components.

    Attributes:
        DEFAULT_LOGDIR: Default directory for log files
        DEFAULT_LOGEXT: Default log file extension
        DEFAULT_LOGLEVEL: Default logging level
        DEFAULT_LOGFORMAT: Default log message format
        DEFAULT_ROT_MAX_BYTES: Default max size for log rotation
        DEFAULT_ROT_BAK_COUNT: Default number of backup files
    """
    
    DEFAULT_LOGDIR = "/var/log/icos-fl"
    DEFAULT_LOGEXT = "log"
    DEFAULT_LOGLEVEL = logging.INFO
    DEFAULT_LOGFORMAT = "[%(asctime)s] %(levelname)-8s %(message)s"
    DEFAULT_ROT_MAX_BYTES = 10485760  # 10MB
    DEFAULT_ROT_BAK_COUNT = 3

    def __init__(self, name: str, 
                 usefile: bool = False,
                 useconsole: bool = True,
                 usecolor: bool = True,
                 level: int = DEFAULT_LOGLEVEL,
                 logformat: str = DEFAULT_LOGFORMAT,
                 logdir: str = DEFAULT_LOGDIR,
                 logfname: str = None,
                 logext: str = DEFAULT_LOGEXT,
                 rotmaxbytes: int = DEFAULT_ROT_MAX_BYTES,
                 rotbakcount: int = DEFAULT_ROT_BAK_COUNT):
        """Initialize logger with configuration.

        Args:
            name: Logger identifier/prefix
            usefile: Enable logging to file
            useconsole: Enable console output
            usecolor: Enable colored output
            level: Logging level (DEBUG, INFO, etc)
            logformat: Message format string
            logdir: Directory for log files
            logfname: Log file base name
            logext: Log file extension
            rotmaxbytes: Max bytes before rotation
            rotbakcount: Number of backup files
        """
        self.name = name
        self.usefile = usefile
        self.useconsole = useconsole
        self.usecolor = usecolor
        self.level = level
        self.logformat = logformat
        self.logdir = logdir
        self.logfname = logfname or name.lower()
        self.logext = logext
        self.rotmaxbytes = rotmaxbytes 
        self.rotbakcount = rotbakcount

        self._logger = logging.getLogger(name)
        self._setup_logger()

    def _setup_logger(self) -> None:
        """Configure logger handlers and formatters."""
        self._logger.setLevel(self.level)
        self._logger.handlers.clear()

        formatter = logging.Formatter(self.logformat)

        if self.useconsole:
            console = logging.StreamHandler(sys.stdout)
            console.setFormatter(formatter)
            self._logger.addHandler(console)

        if self.usefile:
            if not os.path.exists(self.logdir):
                os.makedirs(self.logdir)

            file_handler = RotatingFileHandler(
                filename=self.logfile,
                maxBytes=self.rotmaxbytes,
                backupCount=self.rotbakcount
            )
            file_handler.setFormatter(formatter)
            self._logger.addHandler(file_handler)

    def reconfigure(self, config: Dict) -> None:
        """Update logger configuration.

        Args:
            config: Dictionary of logger attributes to update
        """
        for key, val in config.items():
            if hasattr(self, key):
                setattr(self, key, val)
        self._setup_logger()
        self.info("Logger reconfigured")

    def debug(self, msg: str) -> None:
        """Log debug message."""
        if self.usecolor:
            msg = paint(DBG_CLR, msg)
        self._logger.debug(msg)

    def info(self, msg: str) -> None:
        """Log info message."""
        if self.usecolor:
            msg = paint(INF_CLR, msg)
        self._logger.info(msg)

    def warning(self, msg: str) -> None:
        """Log warning message."""
        if self.usecolor:
            msg = paint(WRN_CLR, msg)
        self._logger.warning(msg)

    def error(self, msg: str) -> None:
        """Log error message."""
        if self.usecolor:
            msg = paint(ERR_CLR, msg)
        self._logger.error(msg)

    def critical(self, msg: str) -> None:
        """Log critical message."""
        if self.usecolor:
            msg = paint(CRI_CLR, msg)
        self._logger.critical(msg)

    def exception(self, msg: Optional[str] = None) -> None:
        """Log exception with traceback.

        Args:
            msg: Optional context message
        """
        pad = '\t'
        trace = pad + traceback.format_exc().replace('\n', f'\n{pad}')
        if self.usecolor:
            if msg is None:
                msg = paint(ERR_CLR, "Exception occurred:\n") + paint(ERR_CLR, trace)
            else:
                msg = paint(ERR_CLR, msg) + "\n\n" + paint(ERR_CLR, trace)
        self._logger.error(msg)

    @property
    def logfile(self) -> str:
        """Full path to log file."""
        return f"{self.logdir}/{self.logfname}.{self.logext}"

    def __str__(self) -> str:
        """Return formatted logger configuration."""
        msg = "\n"
        msg += paint(FGRY, f"            -= ICOS-FL Logger ({self.name}) =-\n")
        msg += paint(FGRY, " ---------------------------------------------------\n")
        msg += paint(FGRY, " Log level             : ") + paint(BWHT, f"{self.level}\n")
        msg += paint(FGRY, " Use color             : ") + paint(BWHT, f"{self.usecolor}\n")
        msg += paint(FGRY, " Console output        : ") + paint(BWHT, f"{self.useconsole}\n")
        msg += paint(FGRY, " File output           : ") + paint(BWHT, f"{self.usefile}\n")
        if self.usefile:
            msg += paint(FGRY, " Log file             : ") + paint(BWHT, f"{self.logfile}\n")
            msg += paint(FGRY, " Rotation size        : ") + paint(BWHT, f"{self.rotmaxbytes}\n")
            msg += paint(FGRY, " Backup count         : ") + paint(BWHT, f"{self.rotbakcount}\n")
        return msg