
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def isreal_inputs():
    list_of_inputs = []

    # Input 1: 1D float array (positive and negative)
    x = np.array([-1.5, 0.0, 2.3, 4.5], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 2: 1D complex array
    x = np.array([1 + 0j, 2 + 3j, -0.5j, 4 - 1j], dtype=np.complex64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 3: 2D integer array
    x = np.array([[1, -2, 3], [4, 5, -6]], dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 4: 3D float array
    x = np.random.randn(2, 3, 4).astype(np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 5: 0D (scalar) array
    x = np.array(5.5, dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 6: 1D boolean array
    x = np.array([True, False, True], dtype=bool)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 7: 4D complex array
    x = (np.random.randn(2, 2, 2, 2) + 1j * np.random.randn(2, 2, 2, 2)).astype(np.complex128)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 8: 1D array with special float values (nan, inf, -inf)
    x = np.array([np.nan, np.inf, -np.inf, 1.0], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 9: 2D complex array with zero imaginary part
    x = np.array([[1.0 + 0.0j, -2.0 + 0.0j], [3.0 + 0.0j, 4.0 + 0.0j]], dtype=np.complex64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 10: 1D int64 array with extreme values
    x = np.array([np.iinfo(np.int64).min, 0, np.iinfo(np.int64).max], dtype=np.int64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    return list_of_inputs

generated_inputs["jax.numpy.isreal_1"] = isreal_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.isreal_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.isreal_1'.")


check_valid('jax.numpy.isreal', generated_inputs['jax.numpy.isreal_1'], lib="jax", suffix=1)
