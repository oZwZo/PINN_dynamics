import os
from pathlib import Path
from .reader_funs import *
from .eval_funs import *

def make_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
