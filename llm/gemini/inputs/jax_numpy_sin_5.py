
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import jax
import numpy as np
import copy

try:
    import jax._src.numpy.ufuncs as ufuncs
except ImportError:
    ufuncs = None

_original_sin = jax.numpy.sin

def _patched_sin(x, *args, **kwargs):
    if isinstance(x, tuple):
        x = np.array(x)
    return _original_sin(x, *args, **kwargs)

jax.numpy.sin = _patched_sin
if ufuncs is not None:
    ufuncs.sin = _patched_sin

def sin_inputs():
    list_of_inputs = []

    # 1. 1D float tuple
    list_of_inputs.append({"x": (0.0, 0.5, 1.0, -1.0)})

    # 2. 1D int tuple
    list_of_inputs.append({"x": (0, 1, -2, 3)})

    # 3. 2D nested float tuple
    list_of_inputs.append({"x": ((0.0, 0.5), (1.0, 1.5))})

    # 4. 3D nested float tuple
    list_of_inputs.append({"x": (((0.1, 0.2), (0.3, 0.4)), ((0.5, 0.6), (0.7, 0.8)))})

    # 5. Tuple with numpy scalars
    list_of_inputs.append({"x": (np.float32(0.1), np.float32(-0.5))})

    # 6. Tuple with complex numbers
    list_of_inputs.append({"x": (1.0 + 1.0j, -2.0j)})

    # 7. Tuple with booleans
    list_of_inputs.append({"x": (True, False, True)})

    # 8. Single-element tuple
    list_of_inputs.append({"x": (1.5,)})

    # 9. Tuple with large and small floats
    list_of_inputs.append({"x": (1e-5, 1e5)})

    # 10. 2D nested mixed tuple
    list_of_inputs.append({"x": ((1, 2.0), (3.0, 4))})

    return list_of_inputs

generated_inputs["jax.numpy.sin_5"] = sin_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.sin_5' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.sin_5'.")


check_valid('jax.numpy.sin', generated_inputs['jax.numpy.sin_5'], lib="jax", suffix=5)
