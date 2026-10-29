
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def right_shift_inputs():
    list_of_inputs = []

    # Input 1: 1D int32 arrays
    x1 = np.array([1, 2, 4, 8, 16], dtype=np.int32)
    x2 = np.array([1, 1, 1, 1, 1], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 2: 1D int32 arrays with varying shift amounts
    x1 = np.array([16, 32, 64, 128], dtype=np.int32)
    x2 = np.array([1, 2, 3, 4], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 3: Scalar tensors (0D)
    x1 = np.array(64, dtype=np.int32)
    x2 = np.array(3, dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 4: 2D int32 arrays with same shape
    x1 = np.array([[8, 16], [32, 64]], dtype=np.int32)
    x2 = np.array([[1, 2], [3, 4]], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 5: 2D int32 broadcasted with 1D int32
    x1 = np.array([[16, 32, 64], [128, 200, 255]], dtype=np.int32)
    x2 = np.array([1, 2, 3], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 6: 3D int64 arrays
    x1 = np.arange(24, dtype=np.int64).reshape((2, 3, 4)) * 4
    x2 = np.ones((2, 3, 4), dtype=np.int64) * 2
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 7: Broadcasting (2, 1, 3) and (1, 4, 3)
    x1 = np.random.randint(0, 100, size=(2, 1, 3), dtype=np.int32)
    x2 = np.random.randint(0, 4, size=(1, 4, 3), dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 8: Signed negative values (int32)
    x1 = np.array([-16, -32, -64, -128], dtype=np.int32)
    x2 = np.array([1, 2, 1, 2], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 9: int32 arrays with 0 shift
    x1 = np.array([10, 20, 30, 40], dtype=np.int32)
    x2 = np.array([0, 0, 0, 0], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 10: 4D int32 array broadcasted with 0D scalar tensor
    x1 = np.random.randint(0, 64, size=(2, 2, 3, 2), dtype=np.int32)
    x2 = np.array(1, dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    # Input 11: int64 arrays with larger shift amounts
    x1 = np.array([1024, 2048, 4096, 8192], dtype=np.int64)
    x2 = np.array([5, 6, 7, 8], dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"x1": x1, "x2": x2}))

    return list_of_inputs


generated_inputs["jax.numpy.right_shift_1"] = right_shift_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.right_shift_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.right_shift_1'.")


check_valid('jax.numpy.right_shift', generated_inputs['jax.numpy.right_shift_1'], lib="jax", suffix=1)
