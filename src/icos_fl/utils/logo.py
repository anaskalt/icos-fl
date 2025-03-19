"""ASCII art graphical resources for the ICOS-FL framework.

This module contains various ASCII art definitions used for visual representation
of the ICOS-FL framework components. These graphical resources can be imported
and displayed in terminal outputs to visually enhance user interfaces or documentation.

Available ASCII art:
- LOGO1, ICOS_FL_LOGO: Main project logos
- OBJECT1, OBJECT2, OBJECT3: Visual representations of system architecture
- NEURAL_NET: Representation of a neural network with federated learning
- FEDERATED_ARCH: Diagram of federated architecture
- CLOUD_SYSTEM: Cloud infrastructure visualization
- ROBOT_ICS: Robot character representation

Example:
    >>> from icos_fl.utils.logo import ICOS_FL_LOGO
    >>> print(ICOS_FL_LOGO)
"""

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
