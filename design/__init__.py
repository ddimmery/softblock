import subprocess
import os
import numpy.f2py

try:
    from .nbpMatch import nbpwrap
except ImportError:
    subprocess.check_output(
        ["python3.6", "-m", "numpy.f2py", "-c", "-m", "design.nbpMatch", "design/nbpwrap.f", "design/fcorematch.f", "design/nbpMatch.pyf"],
    )
from .bernoulli import Bernoulli
from .complete import Complete
from .design import Design
from .optblock import OptBlock
from .quickblock import QuickBlock
from .matched_pair import MatchedPair
from .kallus_heuristic import Heuristic
from .kallus_psod import PSOD
from .soft_block import SoftBlock
from .greedy_neighbors import GreedyNeighbors
from .rerandomization import ReRandomization
