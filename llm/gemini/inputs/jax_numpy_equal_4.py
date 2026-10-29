
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def equal_inputs():
    list_of_inputs = []

    # Input 1: 1D float array
    x = 0.0
    y = np.array([0.0, 1.0, -0.0, 2.5], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 2: 2D integer array with positive float scalar
    x = 2.0
    y = np.array([[1, 2], [2, 3]], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 3: 0D scalar array
    x = -1.5
    y = np.array(-1.5, dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 4: 3D float array with negative values
    x = -5.0
    y = np.array(
        [[[-5.0, 0.0], [1.0, -5.0]], [[-5.0, 2.0], [3.0, 4.0]]],
        dtype=np.float32,
    )
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 5: Large 1D array
    x = 100.0
    y = np.linspace(0, 200, 50, dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 6: 4D array with float16
    x = 1.0
    y = np.ones((2, 2, 3, 3), dtype=np.float16)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 7: Special values (inf, -inf, nan)
    x = float("inf")
    y = np.array([float("inf"), float("-inf"), float("nan"), 1e10], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 8: 2D array with float64
    x = 3.141592653589793
    y = np.full((3, 4), 3.141592653589793, dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 9: Empty array
    x = 0.0
    y = np.empty((0, 5), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 10: 1D boolean array with float comparison (0.0 == False, 1.0 == True)
    x = 1.0
    y = np.array([True, False, True, True], dtype=bool)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 11: 3D array with int64
    x = -42.0
    y = np.random.randint(-50, 50, size=(2, 3, 4)).astype(np.int64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    return list_of_inputs


generated_inputs["jax.numpy.equal_4"] = equal_inputs()

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
