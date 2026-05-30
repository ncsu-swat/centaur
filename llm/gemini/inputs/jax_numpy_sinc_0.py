
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_numpy_sinc_inputs():
    list_of_inputs = []

    # Input 1: Simple 1D float32 array with negative, zero, and positive values
    x = np.array([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 2: 2D float64 array
    x = np.random.uniform(-5.0, 5.0, size=(3, 3)).astype(np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 3: 3D float32 array
    x = np.random.uniform(-10.0, 10.0, size=(2, 3, 4)).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 4: 1D int32 array (to test promotion)
    x = np.array([-3, -2, -1, 0, 1, 2, 3], dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 5: 0D array (scalar)
    x = np.array(0.0, dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 6: 4D float32 array
    x = np.random.uniform(-1.0, 1.0, size=(2, 2, 2, 2)).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 7: 1D float32 array with very small values close to zero
    x = np.array([-1e-15, -1e-30, 0.0, 1e-30, 1e-15], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 8: 1D float32 array with larger values
    x = np.array([-100.0, -50.0, 50.0, 100.0], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 9: 2D int16 array
    x = np.array([[-10, 0], [10, 20]], dtype=np.int16)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 10: 1D float32 array with infinities and nan
    x = np.array([-np.inf, np.nan, np.inf], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    return list_of_inputs

generated_inputs["jax.numpy.sinc"] = jax_numpy_sinc_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.sinc' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.sinc'.")


check_valid('jax.numpy.sinc', generated_inputs['jax.numpy.sinc'], lib="jax", suffix=0)
