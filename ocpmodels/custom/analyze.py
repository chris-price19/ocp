#!/usr/bin/python

import numpy as np
import pandas as pd

import sys
import os
import re

import networkx as nx
import torch_geometric

import ase
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors as m2colors

from ase.visualize.plot import plot_atoms
from ase.constraints import FixAtoms

from pymatgen.analysis.adsorption import reorient_z
from pymatgen.io.ase import AseAtomsAdaptor

mcolors = dict(m2colors.BASE_COLORS, **m2colors.CSS4_COLORS)