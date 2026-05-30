
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import jax.numpy as jnp
import jax._src.numpy.ufuncs as ufuncs

# Patch jax.numpy.isposinf to ignore the unsupported 'out' parameter during execution
_orig_isposinf = jnp.isposinf
def patched_isposinf(x, out=None):
    return _orig_isposinf(x)

jnp.isposinf = patched_isposinf
ufuncs.isposinf = patched_isposinf

def isposinf_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, mix of inf, -inf, nan, finite
    x = np.array([-np.inf, 0.0, np.inf, np.nan, 1.5], dtype=np.float32)
    out = np.empty_like(x, dtype=bool)
    input_dict = {"x": x, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D float64 array with inf and nan
    x = np.array([[np.inf, -np.inf], [1.0, np.nan]], dtype=np.float64)
    out = np.empty_like(x, dtype=bool)
    input_dict = {"x": x, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 0D (scalar) float32 array
    x = np.array(np.inf, dtype=np.float32)
    out = np.empty_like(x, dtype=bool)
    input_dict = {"x": x, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D float16 array with random values and infs
    x = np.random.randn(2, 3, 4).astype(np.float16)
    x[0, 1, 2] = np.inf
    x[1, 2, 3] = -np.inf
    out = np.empty_like(x, dtype=bool)
    input_dict = {"x": x, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 1D float32 array, all finite elements
    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    out = np.empty_like(x, dtype=bool)
    input_dict = {"x": x, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D float64 array, all elements are positive infinity
    x = np.full((3, 3), np.inf, dtype=np.float64)
    out = np.empty_like(x, dtype=bool)
    input_dict = {"x": x, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 4D float32 array
    x = np.random.randn(2, 2, 2, 2).astype(np.float32)
    x[0, 0, 0, 0] = np.inf
    out = np.empty_like(x, dtype=bool)
    input_dict = {"x": x, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1D int32 array (valid non-complex type)
    x = np.array([-10, 0, 10, 20], dtype=np.int32)
    out = np.empty_like(x, dtype=bool)
    input_dict = {"x": x, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D float16 array containing only nans and infs
    x = np.array([[np.nan, np.inf], [-np.inf, np.nan]], dtype=np.float16)
    out = np.empty_like(x, dtype=bool)
    input_dict = {"x": x, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 5D float32 array
    x = np.random.randn(1, 2, 1, 3, 2).astype(np.float32)
    x[0, 1, 0, 2, 1] = np.inf
    out = np.empty_like(x, dtype=bool)
    input_dict = {"x": x, "out": out}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.isposinf_1"] = isposinf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.isposinf_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.isposinf_1'.")


check_valid('jax.numpy.isposinf', generated_inputs['jax.numpy.isposinf_1'], lib="jax", suffix=1)
