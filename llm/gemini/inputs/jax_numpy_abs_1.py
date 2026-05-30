
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def abs_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array with negative and positive values
    x = np.array([-1.5, 0.0, 2.3, -4.8], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 2: 2D float64 array
    x = np.array([[-1.0, -2.0], [3.0, -4.0]], dtype=np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 3: 3D int32 array
    x = np.array([[[1, -2], [-3, 4]], [[-5, 6], [7, -8]]], dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 4: 0D scalar array
    x = np.array(-5.5, dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 5: 1D complex64 array (computes magnitude)
    x = np.array([3 + 4j, -5 + 12j, 0 + 0j], dtype=np.complex64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 6: 2D complex128 array
    x = np.array([[-1 + 1j, -2 - 2j], [3 - 3j, 4 + 4j]], dtype=np.complex128)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 7: 4D int16 array
    x = (np.random.randint(-100, 100, size=(2, 2, 2, 2))).astype(np.int16)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 8: 1D float16 array with large values
    x = np.array([-1000.0, 5000.0, -32000.0], dtype=np.float16)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 9: 1D int64 array with extreme integers
    x = np.array([-9223372036854775807, 9223372036854775807, 0], dtype=np.int64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 10: Empty 2D array
    x = np.empty((0, 5), dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    return list_of_inputs

generated_inputs["jax.numpy.abs_1"] = abs_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.abs_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.abs_1'.")


check_valid('jax.numpy.abs', generated_inputs['jax.numpy.abs_1'], lib="jax", suffix=1)
