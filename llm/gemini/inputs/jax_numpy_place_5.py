
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def place_inputs():
    list_of_inputs = []

    # Input 1: 1D int32 array
    arr = np.array([1, 2, 3, 4, 5, 6], dtype=np.int32)
    mask = np.array([True, False, True, False, False, True], dtype=bool)
    vals = np.array([10, 20], dtype=np.int32)
    list_of_inputs.append(
        copy.deepcopy(
            {"arr": arr, "mask": mask, "vals": vals, "inplace": False}
        )
    )

    # Input 2: 2D float32 array
    arr = np.zeros((3, 4), dtype=np.float32)
    mask = np.array(
        [
            [True, False, False, True],
            [False, True, True, False],
            [False, False, False, True],
        ],
        dtype=bool,
    )
    vals = np.array([-1.5], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy(
            {"arr": arr, "mask": mask, "vals": vals, "inplace": False}
        )
    )

    # Input 3: 2D int64 array with all True mask
    arr = np.ones((2, 3), dtype=np.int64)
    mask = np.ones((2, 3), dtype=bool)
    vals = np.array([1, 2, 3], dtype=np.int64)
    list_of_inputs.append(
        copy.deepcopy(
            {"arr": arr, "mask": mask, "vals": vals, "inplace": False}
        )
    )

    # Input 4: 3D float64 array
    arr = np.arange(24, dtype=np.float64).reshape((2, 3, 4))
    mask = arr % 2 == 0
    vals = np.array([0.0, -100.5, 42.0], dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy(
            {"arr": arr, "mask": mask, "vals": vals, "inplace": False}
        )
    )

    # Input 5: 1D float32 array with all False mask
    arr = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    mask = np.zeros(4, dtype=bool)
    vals = np.array([9.9], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy(
            {"arr": arr, "mask": mask, "vals": vals, "inplace": False}
        )
    )

    # Input 6: 4D int16 array
    arr = np.zeros((2, 2, 2, 2), dtype=np.int16)
    mask = np.ones((2, 2, 2, 2), dtype=bool)
    vals = np.array([-1, -2, -3, -4], dtype=np.int16)
    list_of_inputs.append(
        copy.deepcopy(
            {"arr": arr, "mask": mask, "vals": vals, "inplace": False}
        )
    )

    # Input 7: 2D bool array
    arr = np.array([[True, False], [False, True]], dtype=bool)
    mask = np.array([[False, True], [True, False]], dtype=bool)
    vals = np.array([True, False], dtype=bool)
    list_of_inputs.append(
        copy.deepcopy(
            {"arr": arr, "mask": mask, "vals": vals, "inplace": False}
        )
    )

    # Input 8: 1D complex64 array
    arr = np.array([1 + 1j, 2 + 2j, 3 + 3j], dtype=np.complex64)
    mask = np.array([True, False, True], dtype=bool)
    vals = np.array([1.0 + 2.0j, -3.0j], dtype=np.complex64)
    list_of_inputs.append(
        copy.deepcopy(
            {"arr": arr, "mask": mask, "vals": vals, "inplace": False}
        )
    )

    # Input 9: 3D int32 array with negative values
    arr = np.array([[[-5, -4], [-3, -2]], [[-1, 0], [1, 2]]], dtype=np.int32)
    mask = arr < 0
    vals = np.array([99], dtype=np.int32)
    list_of_inputs.append(
        copy.deepcopy(
            {"arr": arr, "mask": mask, "vals": vals, "inplace": False}
        )
    )

    # Input 10: 2D uint8 array
    arr = np.array([[0, 50], [100, 150], [200, 250]], dtype=np.uint8)
    mask = np.array([[True, False], [False, True], [True, False]], dtype=bool)
    vals = np.array([255, 0, 128], dtype=np.uint8)
    list_of_inputs.append(
        copy.deepcopy(
            {"arr": arr, "mask": mask, "vals": vals, "inplace": False}
        )
    )

    # Input 11: 1D float64 array with excess values in vals
    arr = np.array([0.5, 1.5, 2.5, 3.5, 4.5], dtype=np.float64)
    mask = np.array([True, False, True, False, False], dtype=bool)
    vals = np.array([-0.1, -0.2, -0.3, -0.4, -0.5], dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy(
            {"arr": arr, "mask": mask, "vals": vals, "inplace": False}
        )
    )

    return list_of_inputs


generated_inputs["jax.numpy.place_5"] = place_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.place_5' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.place_5'.")


check_valid('jax.numpy.place', generated_inputs['jax.numpy.place_5'], lib="jax", suffix=5)
