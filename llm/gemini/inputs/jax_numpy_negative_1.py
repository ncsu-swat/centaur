
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def negative_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array with negative, zero, and positive values
    x = np.array([0.0, -3.0, 7.5, -1.2], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 2: 0D scalar array (float64)
    x = np.array(-42.0, dtype=np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 3: 2D int32 array
    x = np.array([[-1, 2, -3], [4, -5, 6]], dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 4: 1D complex64 array
    x = np.array([1 - 2j, -3 + 4j, 5 - 6j], dtype=np.complex64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 5: 3D float32 array
    x = np.random.randn(2, 3, 4).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 6: 1D int32 array
    x = np.array([5, 0, 10, -15], dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 7: 4D float64 array
    x = np.random.randn(2, 2, 3, 3).astype(np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 8: 1D float32 array containing special values (inf, -inf, nan)
    x = np.array([np.inf, -np.inf, np.nan, 0.0, -0.0], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 9: 2D complex128 array
    x = np.array([[2.5 - 1.5j, -0.5 + 3.0j], [1.0 + 0.0j, -4.0 - 2.0j]], dtype=np.complex128)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 10: 1D int64 array with extreme values
    x = np.array([np.iinfo(np.int64).min + 1, -1, 0, 1, np.iinfo(np.int64).max], dtype=np.int64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 11: Empty array
    x = np.array([], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    return list_of_inputs

generated_inputs["jax.numpy.negative_1"] = negative_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.negative_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.negative_1'.")


check_valid('jax.numpy.negative', generated_inputs['jax.numpy.negative_1'], lib="jax", suffix=1)
