
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def place_inputs():
    list_of_inputs = []

    # Input 1: 1D int32 array
    arr = np.array([0, 1, 2, 3, 4], dtype=np.int32)
    mask = np.array([True, False, True, False, True], dtype=bool)
    vals = np.array([10, 20], dtype=np.int32)
    list_of_inputs.append(
        copy.deepcopy(
            {"arr": arr, "mask": mask, "vals": vals, "inplace": False}
        )
    )

    # Input 2: 2D float32 array
    arr = np.zeros((3, 3), dtype=np.float32)
    mask = np.array(
        [[True, False, True], [False, True, False], [True, False, True]],
        dtype=bool,
    )
    vals = np.array([-1.5, -2.5, -3.5], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy(
            {"arr": arr, "mask": mask, "vals": vals, "inplace": False}
        )
    )

    # Input 3: 1D float64 array
    arr = np.ones(6, dtype=np.float64)
    mask = np.ones(6, dtype=bool)
    vals = np.array([99.0], dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy(
            {"arr": arr, "mask": mask, "vals": vals, "inplace": False}
        )
    )

    # Input 4: 1D int64 array
    arr = np.arange(5, dtype=np.int64)
    mask = np.zeros(5, dtype=bool)
    vals = np.array([42, 43], dtype=np.int64)
    list_of_inputs.append(
        copy.deepcopy(
            {"arr": arr, "mask": mask, "vals": vals, "inplace": False}
        )
    )

    # Input 5: 3D float32 array
    arr = np.ones((2, 2, 2), dtype=np.float32)
    mask = np.array(
        [[[True, False], [False, True]], [[False, True], [True, False]]],
        dtype=bool,
    )
    vals = np.array([-10.0, -20.0, -30.0, -40.0], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy(
            {"arr": arr, "mask": mask, "vals": vals, "inplace": False}
        )
    )

    # Input 6: 2D int16 array
    arr = np.array([[-5, -10], [-15, -20]], dtype=np.int16)
    mask = np.array([[True, False], [True, True]], dtype=bool)
    vals = np.array([100, 200, 300], dtype=np.int16)
    list_of_inputs.append(
        copy.deepcopy(
            {"arr": arr, "mask": mask, "vals": vals, "inplace": False}
        )
    )

    # Input 7: 1D bool array
    arr = np.array([False, False, False, False], dtype=bool)
    mask = np.array([True, False, True, False], dtype=bool)
    vals = np.array([True], dtype=bool)
    list_of_inputs.append(
        copy.deepcopy(
            {"arr": arr, "mask": mask, "vals": vals, "inplace": False}
        )
    )

    # Input 8: 2D uint8 array
    arr = np.zeros((2, 4), dtype=np.uint8)
    mask = np.array(
        [[True, True, False, False], [False, False, True, True]], dtype=bool
    )
    vals = np.array([1, 2, 3, 4], dtype=np.uint8)
    list_of_inputs.append(
        copy.deepcopy(
            {"arr": arr, "mask": mask, "vals": vals, "inplace": False}
        )
    )

    # Input 9: 4D float32 array
    arr = np.zeros((2, 2, 2, 2), dtype=np.float32)
    mask = np.zeros((2, 2, 2, 2), dtype=bool)
    mask[0, 1, 0, 1] = True
    mask[1, 0, 1, 0] = True
    vals = np.array([3.14, 2.71], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy(
            {"arr": arr, "mask": mask, "vals": vals, "inplace": False}
        )
    )

    # Input 10: 1D complex64 array
    arr = np.array([1 + 2j, 3 + 4j, 5 + 6j], dtype=np.complex64)
    mask = np.array([True, False, True], dtype=bool)
    vals = np.array([0 + 0j, -1 - 1j], dtype=np.complex64)
    list_of_inputs.append(
        copy.deepcopy(
            {"arr": arr, "mask": mask, "vals": vals, "inplace": False}
        )
    )

    return list_of_inputs


generated_inputs["jax.numpy.place_4"] = place_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.place_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.place_4'.")


check_valid('jax.numpy.place', generated_inputs['jax.numpy.place_4'], lib="jax", suffix=4)
