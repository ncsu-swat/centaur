
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def ceil_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array with mixed positive and negative floats
    x = np.array([-1.5, -0.5, 0.0, 0.5, 1.5, 2.7], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 2: 2D float64 array with positive values
    x = np.array([[1.2, 2.3, 3.9], [4.1, 5.5, 6.8]], dtype=np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 3: 0D (scalar) float32 array
    x = np.array(-3.14, dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 4: 3D float32 array with random values
    x = np.random.uniform(-10.0, 10.0, (2, 3, 3)).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 5: 1D int32 array (integer arrays are valid inputs)
    x = np.array([-5, -1, 0, 2, 10], dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 6: 2D float16 array
    x = np.array([[0.1, -0.1], [10.9, -10.9]], dtype=np.float16)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 7: 4D float32 array with small fractional values
    x = np.random.uniform(-1.0, 1.0, (2, 2, 2, 2)).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 8: 1D float64 array containing very large and very small values
    x = np.array([-1e10, -1e-10, 1e-10, 1e10], dtype=np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 9: 2D int64 array
    x = np.array([[100, -200], [300, -400]], dtype=np.int64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 10: 1D float32 array with exact integer values represented as floats
    x = np.array([-3.0, -0.0, 0.0, 5.0], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    return list_of_inputs

generated_inputs["jax.numpy.ceil"] = ceil_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.ceil' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.ceil'.")


check_valid('jax.numpy.ceil', generated_inputs['jax.numpy.ceil'], lib="jax", suffix=0)
