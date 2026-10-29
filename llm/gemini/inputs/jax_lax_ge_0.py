
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def lax_ge_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 arrays
    x = np.array([-1.5, 0.0, 2.3, -5.0], dtype=np.float32)
    y = np.array([-1.5, -0.1, 2.0, 5.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 2: 2D int32 arrays
    x = np.array([[1, -2], [3, 4]], dtype=np.int32)
    y = np.array([[1, 2], [-3, 4]], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 3: 2D broadcasting with matching rank (1, 3) and (4, 3)
    x = np.array([[10, 20, 30]], dtype=np.int64)
    y = np.array(
        [[5, 25, 30], [10, 15, 35], [0, 20, 20], [15, 20, 25]], dtype=np.int64
    )
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 4: 0D scalars (float64)
    x = np.array(3.14159, dtype=np.float64)
    y = np.array(2.71828, dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 5: 0D scalar and 2D array (broadcasting scalar)
    x = np.array(0.0, dtype=np.float32)
    y = np.array([[-1.0, 2.0], [0.0, -3.0]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 6: 3D float32 arrays
    x = np.ones((2, 3, 4), dtype=np.float32) * -0.5
    y = np.zeros((2, 3, 4), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 7: boolean arrays
    x = np.array([True, False, True, False], dtype=np.bool_)
    y = np.array([False, False, True, True], dtype=np.bool_)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 8: int32 arrays with broadcasting (3, 1) and (3, 4)
    x = np.array([[50], [100], [150]], dtype=np.int32)
    y = np.array(
        [[50, 60, 40, 50], [100, 99, 101, 100], [0, 150, 200, 149]],
        dtype=np.int32,
    )
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 9: 4D int64 arrays
    x = np.random.randint(-100, 100, size=(2, 2, 3, 1), dtype=np.int64)
    y = np.random.randint(-100, 100, size=(2, 2, 3, 5), dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 10: Arrays with empty dimension
    x = np.empty((0, 5), dtype=np.float32)
    y = np.empty((0, 5), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    # Input 11: 2D float64 arrays
    x = np.array([[1.0, -2.5], [3.2, 0.0]], dtype=np.float64)
    y = np.array([[2.0, -2.5], [0.0, -1.0]], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "y": y}))

    return list_of_inputs


generated_inputs["jax.lax.ge"] = lax_ge_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.ge' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.ge'.")


check_valid('jax.lax.ge', generated_inputs['jax.lax.ge'], lib="jax", suffix=0)
