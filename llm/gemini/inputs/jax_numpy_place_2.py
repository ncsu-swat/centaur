
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def generate_jax_numpy_place_inputs():
    list_of_inputs = []

    # Input 1: 1D float32
    arr = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    mask = np.array([True, False, True, False], dtype=bool)
    vals = 9
    list_of_inputs.append({"arr": arr, "mask": mask, "vals": vals, "inplace": False})

    # Input 2: 2D int32
    arr = np.arange(6, dtype=np.int32).reshape(2, 3)
    mask = np.array([[False, True, False], [True, False, True]], dtype=bool)
    vals = -1
    list_of_inputs.append({"arr": arr, "mask": mask, "vals": vals, "inplace": False})

    # Input 3: 3D float64
    arr = np.ones((2, 2, 2), dtype=np.float64)
    mask = np.array([[[True, False], [False, True]], [[True, True], [False, False]]], dtype=bool)
    vals = 0
    list_of_inputs.append({"arr": arr, "mask": mask, "vals": vals, "inplace": False})

    # Input 4: 1D int64
    arr = np.array([-10, -20, -30], dtype=np.int64)
    mask = np.array([True, True, False], dtype=bool)
    vals = 42
    list_of_inputs.append({"arr": arr, "mask": mask, "vals": vals, "inplace": False})

    # Input 5: 4D float32
    arr = np.zeros((2, 1, 2, 2), dtype=np.float32)
    mask = np.ones((2, 1, 2, 2), dtype=bool)
    mask[0, 0, 0, 0] = False
    vals = 100
    list_of_inputs.append({"arr": arr, "mask": mask, "vals": vals, "inplace": False})

    # Input 6: 2D uint8
    arr = np.array([[10, 20], [30, 40]], dtype=np.uint8)
    mask = np.array([[True, False], [False, True]], dtype=bool)
    vals = 5
    list_of_inputs.append({"arr": arr, "mask": mask, "vals": vals, "inplace": False})

    # Input 7: 1D int8
    arr = np.array([1, 2, 3, 4, 5], dtype=np.int8)
    mask = np.array([False, False, True, True, False], dtype=bool)
    vals = -3
    list_of_inputs.append({"arr": arr, "mask": mask, "vals": vals, "inplace": False})

    # Input 8: 3D int16
    arr = np.ones((1, 3, 1), dtype=np.int16)
    mask = np.array([[[True], [False], [True]]], dtype=bool)
    vals = 99
    list_of_inputs.append({"arr": arr, "mask": mask, "vals": vals, "inplace": False})

    # Input 9: 2D float16
    arr = np.zeros((3, 3), dtype=np.float16)
    mask = np.eye(3, dtype=bool)
    vals = -7
    list_of_inputs.append({"arr": arr, "mask": mask, "vals": vals, "inplace": False})

    # Input 10: 5D int32
    arr = np.zeros((1, 1, 2, 1, 2), dtype=np.int32)
    mask = np.ones((1, 1, 2, 1, 2), dtype=bool)
    vals = 123
    list_of_inputs.append({"arr": arr, "mask": mask, "vals": vals, "inplace": False})

    return list_of_inputs

generated_inputs["jax.numpy.place_2"] = generate_jax_numpy_place_inputs()

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
