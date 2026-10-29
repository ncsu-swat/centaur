
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def count_nonzero_inputs():
    list_of_inputs = []

    # Input 1: 1D integer array, axis 0, keepdims False
    a = np.array([0, 1, -2, 0, 3, 0, -4], dtype=np.int32)
    input_dict = {"a": a, "axis": 0, "keepdims": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D boolean array, axis -1, keepdims True
    a = np.array([True, False, True, True, False], dtype=np.bool_)
    input_dict = {"a": a, "axis": -1, "keepdims": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D float32 array, axis 0, keepdims False
    a = np.array([[0.0, 1.5, 0.0], [-2.3, 0.0, 4.1]], dtype=np.float32)
    input_dict = {"a": a, "axis": 0, "keepdims": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D float64 array, axis 1, keepdims True
    a = np.array([[0.0, 0.0], [1.0, 2.0], [0.0, -3.5]], dtype=np.float64)
    input_dict = {"a": a, "axis": 1, "keepdims": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D int64 array, negative axis -2, keepdims False
    a = np.array([[-1, 0, 3], [0, 0, 0], [4, -5, 6]], dtype=np.int64)
    input_dict = {"a": a, "axis": -2, "keepdims": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D complex64 array, axis 2, keepdims True
    a = np.array(
        [[[0.0 + 0.0j, 1.0 + 0.0j], [0.0 + 2.0j, 0.0 + 0.0j]],
         [[3.0 - 1.0j, 0.0 + 0.0j], [0.0 + 0.0j, -1.0 + 1.0j]]],
        dtype=np.complex64,
    )
    input_dict = {"a": a, "axis": 2, "keepdims": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D uint8 array, axis 1, keepdims False
    a = np.arange(24, dtype=np.uint8).reshape(2, 3, 4)
    input_dict = {"a": a, "axis": 1, "keepdims": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 4D float32 array, axis 3, keepdims True
    a = np.zeros((2, 3, 4, 5), dtype=np.float32)
    a[0, 1, 2, 3] = 1.0
    a[1, 2, 3, 4] = -5.0
    input_dict = {"a": a, "axis": 3, "keepdims": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 4D int16 array, axis -3, keepdims False
    a = np.ones((2, 2, 3, 2), dtype=np.int16)
    a[0, 0, :, :] = 0
    input_dict = {"a": a, "axis": -3, "keepdims": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 5D float64 array, axis 0, keepdims True
    a = np.random.choice([0.0, 1.0, -1.0], size=(2, 2, 2, 2, 2)).astype(
        np.float64
    )
    input_dict = {"a": a, "axis": 0, "keepdims": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 2D array with all zeros, axis 0, keepdims True
    a = np.zeros((4, 5), dtype=np.int32)
    input_dict = {"a": a, "axis": 0, "keepdims": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: 2D array with all non-zeros, axis 1, keepdims False
    a = np.ones((3, 3), dtype=np.float32)
    input_dict = {"a": a, "axis": 1, "keepdims": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.count_nonzero_1"] = count_nonzero_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.count_nonzero_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.count_nonzero_1'.")


check_valid('jax.numpy.count_nonzero', generated_inputs['jax.numpy.count_nonzero_1'], lib="jax", suffix=1)
