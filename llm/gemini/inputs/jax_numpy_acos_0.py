
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def acos_inputs():
    list_of_inputs = []

    # Input 1: 0D array (scalar equivalent), float32
    x = np.array(0.5, dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 2: 1D array, float32, range [-1, 1]
    x = np.linspace(-1.0, 1.0, 10, dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 3: 2D array, float64, range [-1, 1]
    x = np.random.uniform(-1.0, 1.0, (4, 4)).astype(np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 4: 3D array, float16, range [-1, 1]
    x = np.random.uniform(-1.0, 1.0, (2, 3, 4)).astype(np.float16)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 5: 1D array with negative values
    x = np.array([-0.9, -0.5, -0.1], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 6: 2D array of integers in range [-1, 1]
    x = np.array([[-1, 0], [1, 0]], dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 7: Complex 1D array
    x = np.array([0.5 + 0.5j, -0.2 - 0.1j], dtype=np.complex64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 8: 4D array, float32, range [-0.5, 0.5]
    x = np.random.uniform(-0.5, 0.5, (2, 2, 2, 2)).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 9: 1D array containing exact boundary values
    x = np.array([-1.0, 0.0, 1.0], dtype=np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 10: 5D array, float32, small size
    x = np.random.uniform(-0.8, 0.8, (2, 1, 3, 1, 2)).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    return list_of_inputs

generated_inputs["jax.numpy.acos"] = acos_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.acos' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.acos'.")


check_valid('jax.numpy.acos', generated_inputs['jax.numpy.acos'], lib="jax", suffix=0)
