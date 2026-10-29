
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def ptp_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, single axis
    a = np.array([-5.0, 0.0, 3.5, 12.0, -2.5], dtype=np.float32)
    input_dict = {"a": a, "axis": [0], "keepdims": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D int32 array, axis 0, keepdims True
    a = np.array([[1, 5, -3], [4, -2, 8]], dtype=np.int32)
    input_dict = {"a": a, "axis": [0], "keepdims": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D float64 array, axis 1, keepdims False
    a = np.array([[-1.5, 2.5, 0.0], [10.2, -4.8, 3.3]], dtype=np.float64)
    input_dict = {"a": a, "axis": [1], "keepdims": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D int64 array, multiple axes [0, 1], keepdims False
    a = np.arange(-12, 12, dtype=np.int64).reshape(4, 6)
    input_dict = {"a": a, "axis": [0, 1], "keepdims": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D float32 array, negative axis [-1], keepdims True
    a = np.random.randn(2, 3, 4).astype(np.float32)
    input_dict = {"a": a, "axis": [-1], "keepdims": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D float64 array, multiple axes [0, 2], keepdims True
    a = np.random.uniform(-10.0, 10.0, size=(3, 4, 5)).astype(np.float64)
    input_dict = {"a": a, "axis": [0, 2], "keepdims": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D int32 array, axis [1], keepdims False
    a = np.random.randint(-50, 50, size=(2, 5, 3), dtype=np.int32)
    input_dict = {"a": a, "axis": [1], "keepdims": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 4D float32 array, axes [1, 2], keepdims False
    a = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_dict = {"a": a, "axis": [1, 2], "keepdims": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 4D float16 array, negative axes [-3, -1], keepdims True
    a = np.random.uniform(-5.0, 5.0, size=(2, 4, 3, 2)).astype(np.float16)
    input_dict = {"a": a, "axis": [-3, -1], "keepdims": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 4D int64 array, all axes [0, 1, 2, 3], keepdims True
    a = np.random.randint(0, 100, size=(2, 2, 3, 3), dtype=np.int64)
    input_dict = {"a": a, "axis": [0, 1, 2, 3], "keepdims": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 1D array with constant values, keepdims True
    a = np.ones(10, dtype=np.float32) * 7.5
    input_dict = {"a": a, "axis": [0], "keepdims": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.ptp_3"] = ptp_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.ptp_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.ptp_3'.")


check_valid('jax.numpy.ptp', generated_inputs['jax.numpy.ptp_3'], lib="jax", suffix=3)
