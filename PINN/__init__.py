from . import plotting_fns as pl
from . import functions as tl
from . import models
from ._config import *
from pathlib import Path


# We get the directory containing this file (the 'PINN' package directory).
PACKAGE_DIR = Path(__file__).resolve().parent

# The project root is two levels up from __init__.py
PROJECT_ROOT_DIR = PACKAGE_DIR.parent

# Expose the variable for the user
main_dir = str(PROJECT_ROOT_DIR)