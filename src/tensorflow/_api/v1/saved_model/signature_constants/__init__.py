# This file is MACHINE GENERATED! Do not edit.
# Generated by: tensorflow/python/tools/api/generator/create_python_api.py script.
"""Signature constants for SavedModel save and restore operations.


"""

from __future__ import print_function as _print_function

from tensorflow.python.saved_model.signature_constants import CLASSIFY_INPUTS
from tensorflow.python.saved_model.signature_constants import CLASSIFY_METHOD_NAME
from tensorflow.python.saved_model.signature_constants import CLASSIFY_OUTPUT_CLASSES
from tensorflow.python.saved_model.signature_constants import CLASSIFY_OUTPUT_SCORES
from tensorflow.python.saved_model.signature_constants import DEFAULT_SERVING_SIGNATURE_DEF_KEY
from tensorflow.python.saved_model.signature_constants import PREDICT_INPUTS
from tensorflow.python.saved_model.signature_constants import PREDICT_METHOD_NAME
from tensorflow.python.saved_model.signature_constants import PREDICT_OUTPUTS
from tensorflow.python.saved_model.signature_constants import REGRESS_INPUTS
from tensorflow.python.saved_model.signature_constants import REGRESS_METHOD_NAME
from tensorflow.python.saved_model.signature_constants import REGRESS_OUTPUTS

del _print_function

import sys as _sys
from tensorflow.python.util import deprecation_wrapper as _deprecation_wrapper

if not isinstance(_sys.modules[__name__], _deprecation_wrapper.DeprecationWrapper):
  _sys.modules[__name__] = _deprecation_wrapper.DeprecationWrapper(
      _sys.modules[__name__], "saved_model.signature_constants")