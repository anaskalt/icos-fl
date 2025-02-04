"""Configuration management for ICOS-FL components.

Provides functionality to obtain configuration from YAML files.
Configuration follows a hierarchical structure with separate files
for client and server components.
"""

import yaml
from icos_fl.utils.logger import ICOSLogger as LOGGER
from icos_fl.utils.colors import paint, FGRY, BWHT

# Get logger instance
LOG = LOGGER("ConfigFetcher")

# Key under which configuration is found in YAML
APPLICATION_CONF_KEY = 'icos'

class ConfigData(dict):
   """Models configuration data with reporting capabilities."""

   def __init__(self, *args, **kwargs):
       """Initialize configuration data container."""
       self.update(*args, **kwargs)
   
   def longest_key(self):
       """Return length of longest config key.
       
       Returns:
           int: Length of longest key
       """
       return max([len(k) for k in self.keys()])

   def report(self):
       """Format configuration for display.
       
       Returns:
           str: Formatted configuration report
       """
       maxword = self.longest_key()
       msg = "\n"
       msg += paint(FGRY, "            -= Configuration Options =- \n")
       msg += paint(FGRY, " ---------------------------------------------------\n")
       for k,v in self.items():
           key = paint(FGRY, f'{k:{maxword}} :  ')
           val = paint(BWHT, f'{v}')
           msg += f' {key}{val}\n'
       return msg

   def __str__(self):
       """String representation via report()."""
       return self.report()

class ConfigFetchError(Exception):
   """Configuration fetching failure."""
   pass

class ConfigFetcher:
   """Fetches configuration from YAML files.
   
   Handles loading and validation of component-specific configs.
   """

   def __init__(self, cfile='./config/client.yaml'):
       """Initialize with config file path.
       
       Args:
           cfile (str): Path to YAML config file. Defaults to client config.
       """
       self.cfile = cfile

   def _fetch_from_file(self):
       """Load and parse YAML configuration file.
       
       Returns:
           ConfigData: Parsed configuration
           
       Raises:
           ConfigFetchError: If file parsing fails
       """
       LOG.debug(f"Parsing configuration file: {self.cfile}")
       try:
           with open(self._cfile, "r") as f:
               tree = yaml.safe_load(f)
           config = ConfigData(tree[APPLICATION_CONF_KEY])
           return config
       except Exception as e:
           raise ConfigFetchError(f"Failed to parse {self.cfile}: {str(e)}")

   def get_configuration(self):
       """Get parsed configuration.
       
       Returns:
           ConfigData: Configuration data
       """
       return self._fetch_from_file()

   @property 
   def cfile(self):
       """Config file path."""
       return self._cfile

   @cfile.setter
   def cfile(self, val):
       """Set config file path.
       
       Args:
           val (str): New file path
       """
       # Strip any trailing slashes
       self._cfile = str(val).rstrip('/')