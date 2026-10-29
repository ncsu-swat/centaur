
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def append_inputs():
    list_of_inputs = []

    # Input 1: 1D arrays, axis=0
    arr = np.array([1, 2, 3], dtype=np.int32)
    values = np.array([4, 5, 6], dtype=np.int32)
    axis = 0
    list_of_inputs.append(
        copy.deepcopy({"arr": arr, "values": values, "axis": axis})
    )

    # Input 2: 1D arrays, negative axis
    arr = np.array([1.5, 2.5], dtype=np.float32)
    values = np.array([-3.5, -4.5, 5.0], dtype=np.float32)
    axis = -1
    list_of_inputs.append(
        copy.deepcopy({"arr": arr, "values": values, "axis": axis})
    )

    # Input 3: 2D arrays, axis=0 (row-wise concatenation)
    arr = np.zeros((2, 3), dtype=np.float64)
    values = np.ones((4, 3), dtype=np.float64)
    axis = 0
    list_of_inputs.append(
        copy.deepcopy({"arr": arr, "values": values, "axis": axis})
    )

    # Input 4: 2D arrays, axis=1 (column-wise concatenation)
    arr = np.arange(6, dtype=np.int64).reshape((2, 3))
    values = np.arange(4, dtype=np.int64).reshape((2, 2))
    axis = 1
    list_of_inputs.append(
        copy.deepcopy({"arr": arr, "values": values, "axis": axis})
    )

    # Input 5: 2D arrays, axis=-2 (equivalent to axis=0)
    arr = np.ones((3, 4), dtype=np.complex64)
    values = np.zeros((2, 4), dtype=np.complex64)
    axis = -2
    list_of_inputs.append(
        copy.deepcopy({"arr": arr, "values": values, "axis": axis})
    )

    # Input 6: 3D arrays, axis=0
    arr = np.random.randn(2, 3, 4).astype(np.float32)
    values = np.random.randn(5, 3, 4).astype(np.float32)
    axis = 0
    list_of_inputs.append(
        copy.deepcopy({"arr": arr, "values": values, "axis": axis})
    )

    # Input 7: 3D arrays, axis=1
    arr = np.random.randint(-10, 10, size=(3, 2, 5), dtype=np.int32)
    values = np.random.randint(-10, 10, size=(3, 4, 5), dtype=np.int32)
    axis = 1
    list_of_inputs.append(
        copy.deepcopy({"arr": arr, "values": values, "axis": axis})
    )

    # Input 8: 3D arrays, axis=2
    arr = np.ones((4, 4, 2), dtype=np.bool_)
    values = np.zeros((4, 4, 6), dtype=np.bool_)
    axis = 2
    list_of_inputs.append(
        copy.deepcopy({"arr": arr, "values": values, "axis": axis})
    )

    # Input 9: 4D arrays, axis=-1
    arr = np.random.randn(2, 3, 4, 1).astype(np.float32)
    values = np.random.randn(2, 3, 4, 5).astype(np.float32)
    axis = -1
    list_of_inputs.append(
        copy.deepcopy({"arr": arr, "values": values, "axis": axis})
    )

    # Input 10: 4D arrays, axis=2
    arr = np.zeros((2, 2, 3, 2), dtype=np.uint8)
    values = np.ones((2, 2, 7, 2), dtype=np.uint8)
    axis = 2
    list_of_inputs.append(
        copy.deepcopy({"arr": arr, "values": values, "axis": axis})
    )

    # Input 11: 5D arrays, axis=0
    arr = np.ones((1, 2, 2, 2, 2), dtype=np.int16)
    values = np.zeros((3, 2, 2, 2, 2), dtype=np.int16)
    axis = 0
    list_of_inputs.append(
        copy.deepcopy({"arr": arr, "values": values, "axis": axis})
    )

    return list_of_inputs


generated_inputs["jax.numpy.append"] = append_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.append' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.append'.")


check_valid('jax.numpy.append', generated_inputs['jax.numpy.append'], lib="jax", suffix=0)
