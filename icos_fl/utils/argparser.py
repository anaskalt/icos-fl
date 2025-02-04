"""Command-line argument parser for ICOS-FL components based on argparse."""

import argparse
import textwrap as _textwrap
from typing import Optional
from icos_fl.utils.colors import paint
from icos_fl.utils.colors import BCYA, FGRY, BGRY, BWHT

HELP_MESSAGE_WIDTH = 80
COLOR_FOR_DEFAULT_OPTS = FGRY
HELP_TEXT_COLOR = FGRY

class LineWrapRawTextHelpFormatter(argparse.RawTextHelpFormatter):
    """Custom formatter for colored help text with line wrapping."""
    
    def _split_lines(self, text: str, width: int) -> list:
        """Split text into colored, wrapped lines."""
        text = self._whitespace_matcher.sub(' ', text).strip()
        if 'DEFAULT:' in text:
            pre, after = text.split('DEFAULT:')
            tmp = _textwrap.wrap(pre, HELP_MESSAGE_WIDTH)
            lines = [paint(HELP_TEXT_COLOR, l) for l in tmp]
            lines.append(paint(COLOR_FOR_DEFAULT_OPTS, 'DEFAULT:' + after))
        else:
            tmp = _textwrap.wrap(text, HELP_MESSAGE_WIDTH)
            lines = [paint(HELP_TEXT_COLOR, l) for l in tmp]
        return lines

class ICOSArgParser:
    """Argument parser for ICOS-FL components."""

    def __init__(self, description: str, component: str):
        """Initialize parser with component description.
        
        Args:
            description: Program description
            component: Component identifier (client/server)
        """
        self._parser = argparse.ArgumentParser(
            prog=f"icos-fl-{component}",
            formatter_class=LineWrapRawTextHelpFormatter,
            description=paint(BCYA, f" {component} ") + paint(FGRY, description),
            epilog=paint(BGRY, "\tICOS Federated Learning\n")
        )
        self._args = None
        self._component = component
        self.filloptions()

    def __str__(self) -> str:
        """Return formatted argument values."""
        msg = paint(FGRY, "            -= Input Arguments =-") + "\n"
        msg += paint(FGRY, " ---------------------------------------------------") + "\n"
        msg += paint(FGRY, " Version              : ") + paint(BWHT, f" {self.version}") + "\n"
        msg += paint(FGRY, " Quiet mode           : ") + paint(BWHT, f" {self.quiet}") + "\n"
        msg += paint(FGRY, " Configuration file   : ") + paint(BWHT, f" {self.config}") + "\n"
        msg += paint(FGRY, " Dry run              : ") + paint(BWHT, f" {self.dry_run}") + "\n"
        return msg

    def filloptions(self) -> None:
        """Fill parser with expected options."""
        self._parser.add_argument('-v', '--version',
            action="store_true",
            default=False,
            help="print package version and exit"
        )

        self._parser.add_argument('-q', '--quiet',
            action="store_true", 
            default=False,
            help="suppress logo and verbose messages DEFAULT: no"
        )

        self._parser.add_argument('-c', '--config',
            type=str,
            default=f'config/{self._component}.yaml',
            help="path to configuration file DEFAULT: config/{component}.yaml"
        )

        self._parser.add_argument('-d', '--dry-run',
            action="store_true",
            default=False,
            help="validate configuration without execution DEFAULT: no"
        )

    @property 
    def parser(self) -> argparse.ArgumentParser:
        return self._parser

    @property
    def args(self) -> Optional[argparse.Namespace]:
        if not self._args:
            self._args = self.parser.parse_args()
        return self._args

    @property
    def version(self) -> bool:
        return self.args.version
    
    @property
    def quiet(self) -> bool:
        return self.args.quiet

    @property
    def config(self) -> str:
        return self.args.config

    @property
    def dry_run(self) -> bool:
        return self.args.dry_run