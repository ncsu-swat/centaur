
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_numpy_equal_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D float32 array
    x = 0.0
    y = np.array([0.0, 1.0, -0.0, 2.0], dtype=np.float32)
    list_of_inputs.append({"x": x, "y": copy.deepcopy(y)})

    # Input 2: 2D float32 array with matching elements
    x = 1.5
    y = np.array([[1.5, 2.3], [-1.5, 1.5]], dtype=np.float32)
    list_of_inputs.append({"x": x, "y": copy.deepcopy(y)})

    # Input 3: Negative float, 3D float32 array
    x = -2.0
    y = np.ones((2, 2, 2), dtype=np.float32) * -2.0
    list_of_inputs.append({"x": x, "y": copy.deepcopy(y)})

    # Input 4: Float64 array, 1D
    x = 3.1415926535
    y = np.array([3.1415926535, 3.14, 3.0], dtype=np.float64)
    list_of_inputs.append({"x": x, "y": copy.deepcopy(y)})

    # Input 5: Float64 array, 2D with zero values (including -0.0)
    x = -0.0
    y = np.array([[0.0, -0.0], [-0.0, 0.0]], dtype=np.float64)
    list_of_inputs.append({"x": x, "y": copy.deepcopy(y)})

    # Input 6: Large 4D array
    x = 10.0
    y = np.random.uniform(9.0, 11.0, (2, 2, 2, 2)).astype(np.float32)
    list_of_inputs.append({"x": x, "y": copy.deepcopy(y)})

    # Input 7: Infinity comparison
    x = float('inf')
    y = np.array([1.0, float('inf'), float('-inf')], dtype=np.float32)
    list_of_inputs.append({"x": x, "y": copy.deepcopy(y)})

    # Input 8: Negative Infinity comparison with 2D array
    x = float('-inf')
    y = np.array([[float('-inf'), 0.0], [float('inf'), float('-inf')]], dtype=np.float32)
    list_of_inputs.append({"x": x, "y": copy.deepcopy(y)})

    # Input 9: 0D array (scalar array)
    x = 5.5
    y = np.array(5.5, dtype=np.float32)
    list_of_inputs.append({"x": x, "y": copy.deepcopy(y)})

    # Input 10: High-dimensional 5D array
    x = -1.25
    y = np.full((1, 2, 1, 2, 1), -1.25, dtype=np.float32)
    list_of_inputs.append({"x": x, "y": copy.deepcopy(y)})

    # Input 11: Comparison with NaN (though nan != nan, it is a valid float input)
    x = float('nan')
    y = np.array([float('nan'), 1.0, 2.0], dtype=np.float32)
    list_of_inputs.append({"x": x, "y": copy.deepcopy(y)})

    return list_of_inputs

generated_inputs["jax.numpy.equal_4"] = jax_numpy_equal_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.equal_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.equal_4'.")


check_valid('jax.numpy.equal', generated_inputs['jax.numpy.equal_4'], lib="jax", suffix=4)
