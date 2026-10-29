
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def true_divide_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 arrays of same shape
    x1 = np.array([10.0, -20.0, 30.5], dtype=np.float32)
    x2 = np.array([2.0, 4.0, -5.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 2: 2D integer arrays (true_divide outputs floats)
    x1 = np.array([[1, 2], [3, 4]], dtype=np.int32)
    x2 = np.array([[2, 4], [6, 8]], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 3: Broadcasting 1D to 2D
    x1 = np.random.randn(3, 4).astype(np.float64)
    x2 = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 4: 0D scalar arrays
    x1 = np.array(15.0, dtype=np.float32)
    x2 = np.array(3.0, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 5: Complex numbers
    x1 = np.array([1.0 + 2.0j, 3.0 - 4.0j], dtype=np.complex64)
    x2 = np.array([2.0 - 1.0j, 1.0 + 1.0j], dtype=np.complex64)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 6: 3D float arrays
    x1 = np.random.uniform(1.0, 10.0, size=(2, 3, 4)).astype(np.float32)
    x2 = np.random.uniform(1.0, 5.0, size=(2, 3, 4)).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 7: Broadcasting with leading unit dimension
    x1 = np.random.uniform(5.0, 20.0, size=(1, 4)).astype(np.float32)
    x2 = np.random.uniform(1.0, 5.0, size=(3, 1)).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 8: Unsigned integer types
    x1 = np.array([100, 200, 250], dtype=np.uint8)
    x2 = np.array([2, 5, 10], dtype=np.uint8)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 9: Mixed float64 and float32 via array types (x1 float64, x2 float32)
    x1 = np.array([[-3.0, 9.0], [12.0, -15.0]], dtype=np.float64)
    x2 = np.array([[1.5, 3.0], [4.0, 5.0]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 10: 4D tensor division
    x1 = np.random.uniform(1.0, 10.0, size=(2, 2, 3, 3)).astype(np.float32)
    x2 = np.random.uniform(0.5, 2.0, size=(2, 2, 3, 3)).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 11: Boolean arrays
    x1 = np.array([True, False, True], dtype=np.bool_)
    x2 = np.array([True, True, True], dtype=np.bool_)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    return list_of_inputs


generated_inputs["jax.numpy.true_divide"] = true_divide_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.true_divide' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.true_divide'.")


check_valid('jax.numpy.true_divide', generated_inputs['jax.numpy.true_divide'], lib="jax", suffix=0)
