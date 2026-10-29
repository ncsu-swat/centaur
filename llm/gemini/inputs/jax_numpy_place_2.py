
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def place_inputs():
    list_of_inputs = []

    # Input 1: 1D integer array, partial mask, positive integer val
    arr = np.zeros(6, dtype=np.int32)
    mask = np.array([True, False, True, False, True, False], dtype=bool)
    vals = 5
    input_dict = {"arr": arr, "mask": mask, "vals": vals, "inplace": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D float array, alternating mask, negative integer val
    arr = np.ones((3, 4), dtype=np.float32)
    mask = (np.arange(12) % 2 == 0).reshape((3, 4))
    vals = -10
    input_dict = {"arr": arr, "mask": mask, "vals": vals, "inplace": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D int array, all-true mask, zero val
    arr = np.arange(8, dtype=np.int64).reshape((2, 2, 2))
    mask = np.ones((2, 2, 2), dtype=bool)
    vals = 0
    input_dict = {"arr": arr, "mask": mask, "vals": vals, "inplace": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D array, all-false mask, positive val
    arr = np.full((4, 3), 7, dtype=np.int32)
    mask = np.zeros((4, 3), dtype=bool)
    vals = 99
    input_dict = {"arr": arr, "mask": mask, "vals": vals, "inplace": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 1D float64 array, single True element, negative val
    arr = np.linspace(0.0, 1.0, 10, dtype=np.float64)
    mask = np.zeros(10, dtype=bool)
    mask[3] = True
    vals = -42
    input_dict = {"arr": arr, "mask": mask, "vals": vals, "inplace": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 4D array, sparse mask, large integer val
    arr = np.zeros((2, 2, 3, 2), dtype=np.int32)
    mask = np.zeros((2, 2, 3, 2), dtype=bool)
    mask[0, 1, 2, 0] = True
    mask[1, 0, 1, 1] = True
    vals = 1000
    input_dict = {"arr": arr, "mask": mask, "vals": vals, "inplace": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 1D array of length 1, True mask, integer val
    arr = np.array([42], dtype=np.int64)
    mask = np.array([True], dtype=bool)
    vals = 1
    input_dict = {"arr": arr, "mask": mask, "vals": vals, "inplace": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D array with 1 row, multiple columns, mixed mask
    arr = np.zeros((1, 8), dtype=np.int16)
    mask = np.array([[True, True, False, False, True, False, True, False]])
    vals = -3
    input_dict = {"arr": arr, "mask": mask, "vals": vals, "inplace": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D float32 array, checkerboard-like mask, integer val
    arr = np.ones((3, 3, 3), dtype=np.float32)
    mask = (np.arange(27) % 3 == 0).reshape((3, 3, 3))
    vals = 7
    input_dict = {"arr": arr, "mask": mask, "vals": vals, "inplace": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D int8 array, diagonal mask, integer val
    arr = np.zeros((5, 5), dtype=np.int8)
    mask = np.eye(5, dtype=bool)
    vals = 12
    input_dict = {"arr": arr, "mask": mask, "vals": vals, "inplace": False}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.place_2"] = place_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.place_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.place_2'.")


check_valid('jax.numpy.place', generated_inputs['jax.numpy.place_2'], lib="jax", suffix=2)
