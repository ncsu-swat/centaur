
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def power_inputs():
    list_of_inputs = []

    # Input 1: 1D positive floats
    x1 = np.array([2.0, 4.0, 5.0], dtype=np.float32)
    x2 = np.array([3.0, 0.5, 2.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 2: 1D non-negative integers
    x1 = np.array([2, 3, 5, 10], dtype=np.int32)
    x2 = np.array([3, 2, 0, 1], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 3: 2D float64 matrices with broadcasting
    x1 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    x2 = np.array([[2.0], [3.0]], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 4: 0D scalar arrays
    x1 = np.array(4.0, dtype=np.float32)
    x2 = np.array(0.5, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 5: 3D float32 tensors
    x1 = np.random.uniform(1.0, 5.0, size=(2, 3, 4)).astype(np.float32)
    x2 = np.random.uniform(0.5, 2.0, size=(2, 3, 4)).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 6: Broadcasting 1D to 2D
    x1 = np.array([-2.0, 3.0, 1.0], dtype=np.float32)
    x2 = np.array([[4.0, 1.0, 6.0], [2.0, 3.0, 5.0]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 7: Complex numbers
    x1 = np.array([1.0 + 2.0j, -2.0 + 1.0j, 0.5 - 0.5j], dtype=np.complex64)
    x2 = np.array([2.0 + 0.0j, 0.5 + 0.0j, 3.0 + 1.0j], dtype=np.complex64)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 8: Negative float exponents with positive bases
    x1 = np.array([2.0, 4.0, 8.0, 16.0], dtype=np.float32)
    x2 = np.array([-1.0, -0.5, -2.0, -0.25], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 9: 4D tensors with broadcasting
    x1 = np.ones((1, 2, 1, 4), dtype=np.float32) * 3.0
    x2 = np.full((2, 1, 3, 1), 2.0, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 10: Int64 arrays with positive powers
    x1 = np.array([[-3, 2], [5, -4]], dtype=np.int64)
    x2 = np.array([[3, 4], [2, 3]], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 11: Zeros and ones
    x1 = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
    x2 = np.array([0.0, 5.0, 0.0, -2.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    return list_of_inputs


generated_inputs["jax.numpy.power_1"] = power_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.power_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.power_1'.")


check_valid('jax.numpy.power', generated_inputs['jax.numpy.power_1'], lib="jax", suffix=1)
