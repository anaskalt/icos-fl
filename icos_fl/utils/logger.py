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

    def strlvl(self, level: int) -> str:
        """ Return string representation of log level

        - level <10: DEBUG
        - level <20: INFO
        - level <30: WARNING
        - level <40: ERROR
        - level <50: CRITICAL

        :param int level: Log level
        :return: String representation of log level
        :rtype: str
        """
        if level <= logging.DEBUG:
            return "DEBUG"
        if level <= logging.INFO:
            return "INFO"
        if level <= logging.WARNING:
            return "WARNING"
        if level <= logging.ERROR:
            return "ERROR"
        if level <= logging.CRITICAL:
            return "CRITICAL"
        return "UNKNOWN"

    def report(self) -> str:
        """Return formatted logger configuration."""
        msg = "\n"
        msg += paint(FGRY, f"            -= ICOS-FL Logger ({self.name}) =-\n")
        msg += paint(FGRY, " ---------------------------------------------------\n")
        msg += paint(FGRY, " Log level             : ") + paint(BWHT, f"{self.level} ({self.strlvl(self.level)})\n")
        msg += paint(FGRY, " Log level             : ") + paint(BWHT, f"{self.level}\n")
        msg += paint(FGRY, " Use color             : ") + paint(BWHT, f"{self.usecolor}\n")
        msg += paint(FGRY, " Log format            : ") + paint(BWHT, f"{self.logformat}\n")
        msg += paint(FGRY, " Console output        : ") + paint(BWHT, f"{self.useconsole}\n")
        msg += paint(FGRY, " File output           : ") + paint(BWHT, f"{self.usefile}\n")
        if self.usefile:
            msg += paint(FGRY, " Log file             : ") + paint(BWHT, f"{self.logfile}\n")
            msg += paint(FGRY, " Rotation size        : ") + paint(BWHT, f"{self.rotmaxbytes}\n")
            msg += paint(FGRY, " Backup count         : ") + paint(BWHT, f"{self.rotbakcount}\n")
        return msg

    def __str__(self):
        return self.report()

    def _setup_logger(self) -> None:
        """Configure logger handlers and formatters."""
        self._logger.setLevel(self.level)

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
                mode='a',
                maxBytes=self.rotmaxbytes,
                backupCount=self.rotbakcount,
                encoding=None,
                delay=False
            )
            file_handler.setFormatter(formatter)
            self._logger.addHandler(file_handler)

    def reconfigure(self, config: Dict) -> None:
        """Update logger configuration.

        Args:
            config: Dictionary of logger attributes to update
        """
        if config:
            for key, val in config.items():
                if hasattr(self, key):
                    self.debug(f'Change LOG.{key} to {val}')
                    setattr(self, key, val)
        while self._logger.hasHandlers():
            self._logger.removeHandler(self._logger.handlers[0])
        self._setup_logger()
        self._logger.info("Logger reconfigured")

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

    #######################
    # Setters and getters #
    #######################

    @property
    def usefile(self) -> bool:
        """Enable file logging."""
        return self._usefile
    
    @usefile.setter
    def usefile(self, value: bool) -> None:
        self._usefile = value
    
    @property
    def useconsole(self) -> bool:
        """Enable console logging."""
        return self._useconsole
    
    @useconsole.setter
    def useconsole(self, value: bool) -> None:
        self._useconsole = value
    
    @property
    def usecolor(self) -> bool:
        """Enable colored output."""
        return self._usecolor
    
    @usecolor.setter
    def usecolor(self, value: bool) -> None:
        self._usecolor = value
    
    @property
    def level(self) -> int:
        """Logging level."""
        return self._level
    
    @level.setter
    def level(self, value) -> None:
        """
        Accepts either an integer logging level or a string representing the log level.
        If a string is provided, it is converted to its corresponding numeric value using
        a predefined mapping. The accepted string values (case-insensitive) are:
        
            - "debug"     → logging.DEBUG
            - "info"      → logging.INFO
            - "warning"   → logging.WARNING
            - "error"     → logging.ERROR
            - "exception" → logging.ERROR
            - "critical"  → logging.CRITICAL

        If an unrecognized string is provided, the default level logging.DEBUG is used.
        """
        if isinstance(value, str):
            level_mapping = {
                "debug": logging.DEBUG,
                "info": logging.INFO,
                "warning": logging.WARNING,
                "error": logging.ERROR,
                "exception": logging.ERROR,
                "critical": logging.CRITICAL,
            }
            value = level_mapping.get(value.lower(), logging.DEBUG)
        self._level = value
    
    @property
    def logformat(self) -> str:
        """Log message format."""
        return self._logformat
    
    @logformat.setter
    def logformat(self, value: str) -> None:
        self._logformat = value
    
    @property
    def logdir(self) -> str:
        """Directory for log files."""
        return self._logdir
    
    @logdir.setter
    def logdir(self, value: str) -> None:
        if len(value) > 1:
            self._logdir = value.rstrip('/')
        self._logdir = value
    
    @property
    def logfname(self) -> str:
        """Base name for log files."""
        return self._logfname
    
    @logfname.setter
    def logfname(self, value: str) -> None:
        self._logfname = value.strip('/')
    
    @property
    def logext(self) -> str:
        """Log file extension."""
        return self._logext
    
    @logext.setter
    def logext(self, value: str) -> None:
        self._logext = value
    
    @property
    def rotmaxbytes(self) -> int:
        """Max size for log rotation."""
        return self._rotmaxbytes
    
    @rotmaxbytes.setter
    def rotmaxbytes(self, value) -> None:
        """Allow setting the value in format '1K', '2M', etc.
        
        Accepts an integer or a string. If a string is provided, it can contain
        a floating-point number (e.g., '1.5K') and supports both lower and uppercase
        suffixes for kilobytes (K, KB) and megabytes (M, MB).
        """
        if isinstance(value, str):
            val = value.strip().lower()
            if val.endswith("kb"):
                numeric_part = val[:-2].strip()
                value = int(float(numeric_part) * 1024)
            elif val.endswith("k"):
                numeric_part = val[:-1].strip()
                value = int(float(numeric_part) * 1024)
            elif val.endswith("mb"):
                numeric_part = val[:-2].strip()
                value = int(float(numeric_part) * 1024 * 1024)
            elif val.endswith("m"):
                numeric_part = val[:-1].strip()
                value = int(float(numeric_part) * 1024 * 1024)
            else:
                value = int(float(val))
        self._rotmaxbytes = value
    
    @property
    def rotbakcount(self) -> int:
        """Number of backup files."""
        return self._rotbakcount
    
    @rotbakcount.setter
    def rotbakcount(self, value: int) -> None:
        self._rotbakcount = value

    @property
    def logfile(self) -> str:
        """Full path to log file."""
        return f"{self.logdir}/{self.logfname}.{self.logext}"