
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def floor_divide_inputs():
    list_of_inputs = []

    # Input 1: 1D integer arrays
    x1 = np.array([10, -20, 30, -40, 50], dtype=np.int32)
    x2 = np.array([3, 3, -7, -7, 6], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 2: 1D floating point arrays
    x1 = np.array([10.5, -20.5, 30.2, -40.8], dtype=np.float32)
    x2 = np.array([2.5, 3.0, -4.2, -5.5], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 3: 2D integer arrays with same shape
    x1 = np.random.randint(-50, 50, size=(4, 5)).astype(np.int64)
    x2 = np.random.randint(1, 10, size=(4, 5)).astype(np.int64)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 4: Broadcasting 2D and 1D arrays
    x1 = np.arange(1, 13, dtype=np.float64).reshape((3, 4))
    x2 = np.array([2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 5: 3D float32 arrays
    x1 = np.random.uniform(10.0, 100.0, size=(2, 3, 4)).astype(np.float32)
    x2 = np.random.uniform(1.0, 5.0, size=(2, 3, 4)).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 6: 0D arrays (scalar tensors)
    x1 = np.array(-15, dtype=np.int32)
    x2 = np.array(4, dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 7: Broadcasting with a 0D array (divisor is 0D)
    x1 = np.array([[-10, 20], [30, -40]], dtype=np.int32)
    x2 = np.array(3, dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 8: Mixed shapes broadcasting (4D with 2D)
    x1 = np.random.randint(10, 100, size=(2, 3, 4, 5)).astype(np.int32)
    x2 = np.random.randint(1, 5, size=(4, 5)).astype(np.int32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 9: Mixed types (int32 dividend and float32 divisor)
    x1 = np.array([15, 22, -35, 48], dtype=np.int32)
    x2 = np.array([2.5, 4.0, 6.0, 5.5], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 10: 2D float64 arrays with negative and positive values
    x1 = np.array([[-12.5, 14.2], [25.0, -33.3]], dtype=np.float64)
    x2 = np.array([[3.0, -4.0], [-5.0, 6.0]], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    return list_of_inputs


generated_inputs["jax.numpy.floor_divide_1"] = floor_divide_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.floor_divide_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.floor_divide_1'.")


check_valid('jax.numpy.floor_divide', generated_inputs['jax.numpy.floor_divide_1'], lib="jax", suffix=1)
