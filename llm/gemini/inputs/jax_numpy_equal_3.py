
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def equal_inputs():
    list_of_inputs = []

    # Input 1: 1D int array, positive integer
    x = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    y = 3
    list_of_inputs.append({"x": x, "y": y})

    # Input 2: 2D float array, zero integer
    x = np.array([[0.0, 1.0], [2.0, 0.0]], dtype=np.float32)
    y = 0
    list_of_inputs.append({"x": x, "y": y})

    # Input 3: 3D int array, negative integer
    x = np.random.randint(-10, 10, size=(2, 2, 2), dtype=np.int32)
    y = -5
    list_of_inputs.append({"x": x, "y": y})

    # Input 4: 1D float64 array, large integer
    x = np.array([1e6, 2e6, 3e6], dtype=np.float64)
    y = 1000000
    list_of_inputs.append({"x": x, "y": y})

    # Input 5: 4D int8 array, small integer
    x = np.random.randint(-5, 5, size=(2, 3, 1, 2), dtype=np.int8)
    y = 1
    list_of_inputs.append({"x": x, "y": y})

    # Input 6: 1-element array, matching integer
    x = np.array([42], dtype=np.int64)
    y = 42
    list_of_inputs.append({"x": x, "y": y})

    # Input 7: Empty array, integer y
    x = np.empty((0, 5), dtype=np.int32)
    y = 0
    list_of_inputs.append({"x": x, "y": y})

    # Input 8: 2D uint8 array, boundary integer
    x = np.array([[255, 0], [128, 64]], dtype=np.uint8)
    y = 255
    list_of_inputs.append({"x": x, "y": y})

    # Input 9: 5D int16 array, integer y
    x = np.ones((2, 2, 1, 2, 2), dtype=np.int16)
    y = 1
    list_of_inputs.append({"x": x, "y": y})

    # Input 10: 1D int64 array, large negative integer
    x = np.array([-1000, -2000, -3000], dtype=np.int64)
    y = -2000
    list_of_inputs.append({"x": x, "y": y})

    return list_of_inputs

generated_inputs["jax.numpy.equal_3"] = equal_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.equal_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.equal_3'.")


check_valid('jax.numpy.equal', generated_inputs['jax.numpy.equal_3'], lib="jax", suffix=3)
