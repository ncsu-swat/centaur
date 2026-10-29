
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def power_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, positive integer power
    x1 = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    x2 = 2
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 2: 2D float64 array, negative integer power
    x1 = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float64)
    x2 = -2
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 3: 1D int32 array, zero power
    x1 = np.array([0, 1, 2, 3, -4], dtype=np.int32)
    x2 = 0
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 4: 2D int64 array, positive integer power
    x1 = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
    x2 = 3
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 5: 3D float32 array, power of 1
    x1 = np.ones((2, 3, 4), dtype=np.float32)
    x2 = 1
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 6: 0D scalar tensor, float32, positive power
    x1 = np.array(5.0, dtype=np.float32)
    x2 = 4
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 7: 4D float32 array, positive power
    x1 = np.random.uniform(0.5, 2.0, size=(2, 2, 3, 3)).astype(np.float32)
    x2 = 3
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 8: 1D int32 array, positive power
    x1 = np.array([-2, -1, 0, 1, 2], dtype=np.int32)
    x2 = 3
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 9: 2D float64 array, large positive integer power
    x1 = np.array([[1.01, 1.02], [0.99, 0.98]], dtype=np.float64)
    x2 = 10
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 10: 3D float64 array with negative numbers, odd integer power
    x1 = np.array(
        [[[-1.0, -2.0], [3.0, -4.0]], [[5.0, -6.0], [-7.0, 8.0]]],
        dtype=np.float64,
    )
    x2 = 3
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 11: 1D float32 array, negative integer power
    x1 = np.array([0.5, 2.0, 4.0, 8.0], dtype=np.float32)
    x2 = -1
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    return list_of_inputs


generated_inputs["jax.numpy.power_2"] = power_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.power_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.power_2'.")


check_valid('jax.numpy.power', generated_inputs['jax.numpy.power_2'], lib="jax", suffix=2)
