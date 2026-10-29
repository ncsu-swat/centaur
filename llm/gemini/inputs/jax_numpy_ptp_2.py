
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def ptp_inputs():
    list_of_inputs = []

    # Input 1: 1D array, single axis tuple, keepdims=False
    a = np.array([-5.0, 0.0, 3.5, 12.0, -2.5], dtype=np.float32)
    input_dict = {"a": a, "axis": (0,), "keepdims": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D array, axis=(1,), keepdims=True
    a = np.array([[1, 5, 2], [9, -3, 4]], dtype=np.int32)
    input_dict = {"a": a, "axis": (1,), "keepdims": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D array, axis=(0,), keepdims=False
    a = np.random.uniform(-10, 10, size=(4, 5)).astype(np.float64)
    input_dict = {"a": a, "axis": (0,), "keepdims": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D array, multiple axes tuple (0, 1), keepdims=True
    a = np.array([[-10, 20], [30, -40]], dtype=np.int64)
    input_dict = {"a": a, "axis": (0, 1), "keepdims": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D array, negative axis index, keepdims=False
    a = np.random.randn(3, 4, 5).astype(np.float32)
    input_dict = {"a": a, "axis": (-1,), "keepdims": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D array, two axes tuple, keepdims=True
    a = np.arange(24, dtype=np.int32).reshape(2, 3, 4)
    input_dict = {"a": a, "axis": (0, 2), "keepdims": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 4D array, axis=(1, 3), keepdims=False
    a = np.random.uniform(0, 100, size=(2, 3, 4, 5)).astype(np.float32)
    input_dict = {"a": a, "axis": (1, 3), "keepdims": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 4D array, all axes specified, keepdims=True
    a = np.random.randint(-50, 50, size=(2, 2, 2, 2), dtype=np.int32)
    input_dict = {"a": a, "axis": (0, 1, 2, 3), "keepdims": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 5D array, float16 dtype, keepdims=False
    a = np.random.randn(2, 2, 3, 2, 2).astype(np.float16)
    input_dict = {"a": a, "axis": (2, 4), "keepdims": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D unsigned integer array, axis=(0,), keepdims=True
    a = np.array([10, 250, 45, 0, 128], dtype=np.uint8)
    input_dict = {"a": a, "axis": (0,), "keepdims": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 3D array with negative axis indices, keepdims=True
    a = np.random.uniform(-1.0, 1.0, size=(4, 6, 8)).astype(np.float64)
    input_dict = {"a": a, "axis": (-3, -1), "keepdims": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.ptp_2"] = ptp_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.ptp_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.ptp_2'.")


check_valid('jax.numpy.ptp', generated_inputs['jax.numpy.ptp_2'], lib="jax", suffix=2)
