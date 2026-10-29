
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def place_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, alternating mask
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    mask = np.array([True, False, True, False, True], dtype=bool)
    vals = -1.0
    list_of_inputs.append(
        copy.deepcopy(
            {"arr": arr, "mask": mask, "vals": vals, "inplace": False}
        )
    )

    # Input 2: 2D float64 array, all True mask
    arr = np.ones((3, 3), dtype=np.float64)
    mask = np.ones((3, 3), dtype=bool)
    vals = 42.5
    list_of_inputs.append(
        copy.deepcopy(
            {"arr": arr, "mask": mask, "vals": vals, "inplace": False}
        )
    )

    # Input 3: 2D int32 array, all False mask
    arr = np.arange(6, dtype=np.int32).reshape((2, 3))
    mask = np.zeros((2, 3), dtype=bool)
    vals = 0.0
    list_of_inputs.append(
        copy.deepcopy(
            {"arr": arr, "mask": mask, "vals": vals, "inplace": False}
        )
    )

    # Input 4: 3D float32 array, sparse mask
    arr = np.zeros((2, 3, 4), dtype=np.float32)
    mask = np.zeros((2, 3, 4), dtype=bool)
    mask[0, 1, 2] = True
    mask[1, 2, 3] = True
    vals = 3.14159
    list_of_inputs.append(
        copy.deepcopy(
            {"arr": arr, "mask": mask, "vals": vals, "inplace": False}
        )
    )

    # Input 5: 1D array of single element
    arr = np.array([10.0], dtype=np.float32)
    mask = np.array([True], dtype=bool)
    vals = -999.0
    list_of_inputs.append(
        copy.deepcopy(
            {"arr": arr, "mask": mask, "vals": vals, "inplace": False}
        )
    )

    # Input 6: 4D float64 array
    arr = np.random.randn(2, 2, 2, 3).astype(np.float64)
    mask = arr > 0
    vals = -0.5
    list_of_inputs.append(
        copy.deepcopy(
            {"arr": arr, "mask": mask, "vals": vals, "inplace": False}
        )
    )

    # Input 7: 2D int64 array with negative values
    arr = np.array([[-10, 20], [30, -40]], dtype=np.int64)
    mask = arr < 0
    vals = 100.0
    list_of_inputs.append(
        copy.deepcopy(
            {"arr": arr, "mask": mask, "vals": vals, "inplace": False}
        )
    )

    # Input 8: 1D float32 array with inf values
    arr = np.array([np.inf, 1.0, -np.inf, 2.0], dtype=np.float32)
    mask = np.isinf(arr)
    vals = 0.0
    list_of_inputs.append(
        copy.deepcopy(
            {"arr": arr, "mask": mask, "vals": vals, "inplace": False}
        )
    )

    # Input 9: 2D complex64 array
    arr = np.array([[1 + 2j, 3 + 4j], [5 + 6j, 7 + 8j]], dtype=np.complex64)
    mask = np.array([[True, False], [False, True]], dtype=bool)
    vals = 9.0
    list_of_inputs.append(
        copy.deepcopy(
            {"arr": arr, "mask": mask, "vals": vals, "inplace": False}
        )
    )

    # Input 10: 3D float32 array, checkerboard-like mask
    arr = np.arange(12, dtype=np.float32).reshape((2, 2, 3))
    mask = (np.arange(12) % 2 == 1).reshape((2, 2, 3))
    vals = 2.71828
    list_of_inputs.append(
        copy.deepcopy(
            {"arr": arr, "mask": mask, "vals": vals, "inplace": False}
        )
    )

    return list_of_inputs


generated_inputs["jax.numpy.place_3"] = place_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.place_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.place_3'.")


check_valid('jax.numpy.place', generated_inputs['jax.numpy.place_3'], lib="jax", suffix=3)
