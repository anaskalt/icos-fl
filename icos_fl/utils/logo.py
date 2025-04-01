"""ASCII art graphical resources for the ICOS-FL framework.

This module contains various ASCII art definitions used for visual representation
of the ICOS-FL framework components. These graphical resources can be imported
and displayed in terminal outputs to visually enhance user interfaces or documentation.

Available ASCII art:
- LOGO1, ICOS_FL_LOGO: Main project logos
- ICOS_FL_CLIENT_LOGO: Client component logo
- ICOS_FL_SERVER_LOGO: Server component logo
- OBJECT1, OBJECT2, OBJECT3: Visual representations of system architecture
- NEURAL_NET: Representation of a neural network with federated learning
- FEDERATED_ARCH: Diagram of federated architecture
- CLOUD_SYSTEM: Cloud infrastructure visualization
- ROBOT_ICS: Robot character representation

Functions:
- version(): Returns a formatted version string
- print_banner(): Displays a customizable banner with logo and optional text
- print_start_banner(): Displays a startup banner with logo and version
- print_client_banner(): Displays the client banner
- print_server_banner(): Displays the server banner
- print_completion_banner(): Displays a completion banner for training

Example:
    >>> from icos_fl.utils.logo import ICOS_FL_LOGO
    >>> print(ICOS_FL_LOGO)
    >>> from icos_fl.utils.logo import print_start_banner
    >>> print_start_banner()
"""

import icos_fl
from icos_fl.utils.colors import BBLU, BCYA, BGRN, BMAG, BWHT, BYEL, FGRY, paint

LOGO1 = r"""
  ██╗ ██████╗ ██████╗ ███████╗    ███████╗██╗
  ██║██╔════╝██╔═══██╗██╔════╝    ██╔════╝██║
  ██║██║     ██║   ██║███████╗    █████╗  ██║
  ██║██║     ██║   ██║╚════██║    ██╔══╝  ██║
  ██║╚██████╗╚██████╔╝███████╗    ██║     ███████╗
  ╚═╝ ╚═════╝ ╚═════╝ ╚══════╝    ╚═╝     ╚══════╝
               I C O S   F L
"""

OBJECT1 = r"""
     [Node-1]        [Node-2]        [Node-3]
    ╭─────────╮     ╭─────────╮     ╭─────────╮
    │ ◢▆▅▄▃▂  │     │  ▂▃▄▅▆◣ │     │ ⌘  ∞  ⌘ │
    │ TARGET  │<═══>│  LEARN  │<═══>│ PROCESS │
    │ ▔▀▀▀▀▀▔ │     │ ▔▀▀▀▀▀▔ │     │ ▔▀▀▀▀▀▔ │
    ╰─────────╯     ╰─────────╯     ╰─────────╯
         ▲               ▲               ▲
         │               │               │
         └───────┬───────┴───────┬───────┘
                 │    ICOS-FL    │
            ╔════╧══════════════╧════╗
            ║ Federated Learning     ║
            ║   Meta-OS Platform     ║
            ╚════════════════════════╝
"""

OBJECT2 = r"""
          ┌───────────────────┐
     ┌────┤  Meta-OS Control  ├────┐
     │    └───────────────────┘    │
     ▼           ICOS-FL           ▼
  ┌──────┐                     ┌──────┐
  │ ╭──╮ │      Federated      │ ╭──╮ │
  │ │DB│ │◄────  Learning  ───►│ │DB│ │
  │ ╰──╯ │                     │ ╰──╯ │
  └──────┘                     └──────┘
     ▲                            ▲
     │                            │
   ┌─┴──┐                      ┌-─┴─┐
   │Node│                      │Node│
   └────┘                      └────┘
"""

OBJECT3 = r"""
       ╔════════════════════╗
       ║  ICOS-FL Services  ║
       ╚════════════════════╝
              │  │  │
     ┌────────┘  │  └────────┐
     ▼           ▼           ▼
  ┌─────┐    ┌─────┐     ┌─────┐
  │ ◉_◉ │    │ ⊙_⊙ │     │ ◎_◎ │
  │  ▀  │    │  ▀  │     │  ▀  │
  └──┬──┘    └──┬──┘     └──┬──┘
     │          │           │
  ┌──▼──┐    ┌──▼──┐     ┌──▼──┐
  │Agent│    │Agent│     │Agent│
  └─────┘    └─────┘     └─────┘
"""

NEURAL_NET = r"""
     ╭─○─╮       ╭─○─╮       ╭─○─╮
     │ ░ │●══════●│ ░ │●══════●│ ░ │
     ╰─○─╯       ╰─○─╯       ╰─○─╯
        \░░░░░░░░░░░░░░░░░░░░░/
         ▒▒ FEDERATED LAYER ▒▒
          ░░░░░░░░░░░░░░░░░░░
"""

FEDERATED_ARCH = r"""
   ╭──────────────╮      ╭──────────────╮
   │  EDGE NODE   │══════│  EDGE NODE   │
   ╰───────┬┬┬────╯      ╰───────┬┬┬────╯
           │││░                  │││░
    ╭──────┴┴┴──────╮    ╭───────┴┴┴─────╮
    │ CLOUD CORE    │════│ META-OS BRAIN │
    ╰───────┬┬┬─────╯    ╰───────┬┬┬─────╯
            ░░░                  ░░░
"""

ICOS_FL_LOGO = r"""
  ██╗ ██████╗ ██████╗ ███████╗    ███████╗██╗
  ██║██╔════╝██╔═══██╗██╔════╝    ██╔════╝██║
  ██║██║     ██║   ██║███████╗    █████╗  ██║
  ██║██║     ██║   ██║╚════██║    ██╔══╝  ██║
  ██║╚██████╗╚██████╔╝███████╗    ██║     ███████╗
  ╚═╝ ╚═════╝ ╚═════╝ ╚══════╝    ╚═╝     ╚══════╝
   ░▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀░
        ICOS FL :: Federated Learning Framework
"""

ICOS_FL_CLIENT_LOGO = r"""
  ██╗ ██████╗ ██████╗ ███████╗    ███████╗██╗
  ██║██╔════╝██╔═══██╗██╔════╝    ██╔════╝██║
  ██║██║     ██║   ██║███████╗    █████╗  ██║
  ██║██║     ██║   ██║╚════██║    ██╔══╝  ██║
  ██║╚██████╗╚██████╔╝███████╗    ██║     ███████╗
  ╚═╝ ╚═════╝ ╚═════╝ ╚══════╝    ╚═╝     ╚══════╝
   ░▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀░
                 ICOS FL CLIENT
"""

ICOS_FL_SERVER_LOGO = r"""
  ██╗ ██████╗ ██████╗ ███████╗    ███████╗██╗
  ██║██╔════╝██╔═══██╗██╔════╝    ██╔════╝██║
  ██║██║     ██║   ██║███████╗    █████╗  ██║
  ██║██║     ██║   ██║╚════██║    ██╔══╝  ██║
  ██║╚██████╗╚██████╔╝███████╗    ██║     ███████╗
  ╚═╝ ╚═════╝ ╚═════╝ ╚══════╝    ╚═╝     ╚══════╝
   ░▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀░
                 ICOS FL SERVER
"""

CLOUD_SYSTEM = r"""
        ╭──────────────────────────╮
        │  ███████████████████████│
        │  ██ ░░▒▒CLOUD▒▒░░ ██████│
        ╰───────────┬─┬───────────╯
             ╭──────┴─┴──────╮
        ╭────┤ ▓▓▓▓▓▓▓▓▓▓▓▓ ├───-─╮
        │    ╰──────┬─┬──────╯    │
        │  ╭────────┴─┴────────╮  │
        ╰──┤   ▒▒▒▒▒▒▒▒▒▒▒▒▒   ├──╯
           ╰───────────────────╯
"""

ROBOT_ICS = r"""
      ╭───────╮
     █│▒▒▒▒▒▒▒│█
    ╭─┤░░░░░░░├─╮
    │░│●     ●│░│
    ╰─┤  ███  ├─╯
     █╰┬─────┬╯█
       ╰┼┬┬┬┼╯
        │││││
        ╰┴┴┴╯
"""


def version(ver: str | None = None) -> str:
    """Returns the version message of ICOS-FL.

    Args:
        ver: Optional version string (uses icos_fl.__version__ if None)

    Returns:
        str: The formatted version message
    """
    version_info = ver if ver is not None else icos_fl.__version__
    vmsg = "\n" + paint(FGRY, " ICOS-FL : ") + paint(BWHT, version_info) + "\n"
    return vmsg


def print_banner(
    logo: str,
    title: str = "",
    message: str = "",
    border_color: str = BCYA,
    logo_color: str = BBLU,
    title_color: str = BWHT,
    message_color: str = BGRN,
    ver: str | None = None,
    show_version: bool = True,
) -> None:
    """Displays a customizable banner with logo and optional text.

    Args:
        logo: The ASCII art logo to display
        title: Optional title to display above the logo (default: "")
        message: Optional message to display below the logo (default: "")
        border_color: ANSI color code for the border (default: BCYA)
        logo_color: ANSI color code for the logo (default: BBLU)
        title_color: ANSI color code for the title (default: BWHT)
        message_color: ANSI color code for the message (default: BGRN)
        ver: Optional version string (uses icos_fl.version if None)
        show_version: Whether to display version information (default: True)
    """
    version_info = ver if ver is not None else icos_fl.__version__
    border = paint(border_color, "=" * 60)

    print("\n" + border)

    if title:
        print(paint(title_color, title))

    print(paint(logo_color, logo))

    if message:
        print(paint(message_color, message))

    print(border + "\n")

    if show_version:
        print(paint(FGRY, "  Version: ") + paint(BWHT, version_info) + "\n")


def print_start_banner(ver: str | None = None) -> None:
    """Displays a compact banner for application startup.

    Args:
        ver: Optional version string (uses icos_fl.version if None)
    """
    print_banner(
        logo=LOGO1,
        message="  ICOS-FL: Federated Learning Framework for Resource Monitoring",
        logo_color=BBLU,
        message_color=BGRN,
        ver=ver,
    )


def print_client_banner(ver: str | None = None) -> None:
    """Displays the client banner with version information.

    Args:
        ver: Optional version string (uses icos_fl.version if None)
    """
    print_banner(
        logo=ICOS_FL_CLIENT_LOGO,
        logo_color=BGRN,
        title="  ICOS-FL CLIENT",
        title_color=BWHT,
        ver=ver,
    )


def print_server_banner(ver: str | None = None) -> None:
    """Displays the server banner with version information.

    Args:
        ver: Optional version string (uses icos_fl.version if None)
    """
    print_banner(
        logo=ICOS_FL_SERVER_LOGO,
        logo_color=BMAG,
        title="  ICOS-FL SERVER",
        title_color=BWHT,
        ver=ver,
    )


def print_completion_banner() -> None:
    """Displays a completion banner at the end of training."""
    print_banner(
        logo=CLOUD_SYSTEM,
        title="  ICOS-FL TRAINING COMPLETED",
        logo_color=BBLU,
        title_color=BYEL,
        show_version=False,
    )
