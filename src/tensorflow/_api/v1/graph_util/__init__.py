# This file is MACHINE GENERATED! Do not edit.
# Generated by: tensorflow/python/tools/api/generator/create_python_api.py script.
"""Helpers to manipulate a tensor graph in python.

"""

from __future__ import print_function as _print_function

from tensorflow.lite.python.lite import _import_graph_def as import_graph_def
from tensorflow.python.framework.graph_util import convert_variables_to_constants
from tensorflow.python.framework.graph_util import extract_sub_graph
from tensorflow.python.framework.graph_util import must_run_on_cpu
from tensorflow.python.framework.graph_util import remove_training_nodes
from tensorflow.python.framework.graph_util import tensor_shape_from_node_def_name

del _print_function

import sys as _sys
from tensorflow.python.util import deprecation_wrapper as _deprecation_wrapper

if not isinstance(_sys.modules[__name__], _deprecation_wrapper.DeprecationWrapper):
  _sys.modules[__name__] = _deprecation_wrapper.DeprecationWrapper(
      _sys.modules[__name__], "graph_util")
