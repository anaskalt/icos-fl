"""Color definitions and utilities for console output formatting.

This module provides ANSI color code constants and utility functions for colored console output.
Colors are defined for different emphasis levels (plain, bright/bold, faint) and include
helper functions for applying colors to text.

Colors available:
- Base colors: BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE
- Emphasis: Plain (e.g. RED), Bold/Bright (e.g. BRED), Faint (e.g. FRED)
- Special: RST (reset to default)

Example:
    >>> from icos_fl.utils.colors import RED, BBLUE, paint
    >>> print(paint(RED, "Error message"))
    >>> print(paint(BBLUE, "Bold blue text"))
"""

# Reset
RST = '\033[0m'

# Plain colors
BLK = '\033[30;22m'
RED = '\033[31;22m'
GRN = '\033[32;22m'
YEL = '\033[33;22m'
BLU = '\033[34;22m'
MAG = '\033[35;22m'
CYA = '\033[36;22m'
WHT = '\033[39;22m'
GRY = '\033[37;22m'

# Bright/Bold colors
BBLK = '\033[30;1m'
BRED = '\033[31;1m'
BGRN = '\033[32;1m'
BYEL = '\033[33;1m'
BBLU = '\033[34;1m'
BMAG = '\033[35;1m'
BCYA = '\033[36;1m'
BWHT = '\033[39;1m'
BGRY = '\033[37;1m'

# Faint colors
FBLK = '\033[30;2m'
FRED = '\033[31;2m'
FGRN = '\033[32;2m'
FYEL = '\033[33;2m'
FBLU = '\033[34;2m'
FMAG = '\033[35;2m'
FCYA = '\033[36;2m'
FWHT = '\033[39;2m'
FGRY = '\033[37;2m'

# Log level colors
DBG_CLR = FGRY   # Debug messages
INF_CLR = WHT    # Info messages
WRN_CLR = YEL    # Warning messages
ERR_CLR = RED    # Error messages
CRI_CLR = BRED   # Critical messages

BDBG_CLR = FGRY
BINF_CLR = BWHT
BWRN_CLR = BYEL
BERR_CLR = BRED
BCRI_CLR = BRED

def paint(color: str, text: str, reset: str = RST) -> str:
    """Apply ANSI color codes to text.

    Args:
        color: ANSI color code to apply
        text: Text string to color
        reset: ANSI code to apply after text (default: RST)

    Returns:
        Colored text string with reset code appended

    Example:
        >>> print(paint(RED, "Error"))
        >>> print(paint(BBLUE, "Title", FGRY))
    """
    return f"{color}{text}{reset}"

def perror(msg: str, lpadsz: int = 0, endl: str = None) -> None:
    """Print an error message in red with ERROR prefix.

    Args:
        msg: Message to print
        lpadsz: Left padding spaces (default: 0)
        endl: Custom line ending (default: None)
    """
    pad = ' ' * lpadsz
    text = f"{pad}{paint(BRED, 'ERROR: ')}{paint(ERR_CLR, msg)}"
    print(text, end=endl)

def pwarning(msg: str, lpadsz: int = 0, endl: str = None) -> None:
    """Print a warning message in yellow with WARNING prefix.

    Args:
        msg: Message to print
        lpadsz: Left padding spaces (default: 0)
        endl: Custom line ending (default: None)
    """
    pad = ' ' * lpadsz
    text = f"{pad}{paint(BWRN_CLR, 'WARNING: ')}{paint(WRN_CLR, msg)}"
    print(text, end=endl)

def pinfo(msg: str, lpadsz: int = 0, endl: str = None) -> None:
    """Print an info message in white with INFO prefix.

    Args:
        msg: Message to print
        lpadsz: Left padding spaces (default: 0)
        endl: Custom line ending (default: None)
    """
    pad = ' ' * lpadsz
    text = f"{pad}{paint(BINF_CLR, 'INFO: ')}{paint(INF_CLR, msg)}"
    print(text, end=endl)

def pdebug(msg: str, lpadsz: int = 0, endl: str = None) -> None:
    """Print a debug message in grey with DEBUG prefix.

    Args:
        msg: Message to print
        lpadsz: Left padding spaces (default: 0)
        endl: Custom line ending (default: None)
    """
    pad = ' ' * lpadsz
    text = f"{pad}{paint(BDBG_CLR, 'DEBUG: ')}{paint(DBG_CLR, msg)}"
    print(text, end=endl)