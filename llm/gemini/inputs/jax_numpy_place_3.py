
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_numpy_place_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, alternating mask
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    mask = np.array([True, False, True, False, True], dtype=bool)
    vals = 10.5
    list_of_inputs.append({"arr": arr, "mask": mask, "vals": vals, "inplace": False})

    # Input 2: 2D float32 array, random mask
    arr = np.random.randn(3, 4).astype(np.float32)
    mask = np.random.choice([True, False], size=(3, 4))
    vals = -1.23
    list_of_inputs.append({"arr": arr, "mask": mask, "vals": vals, "inplace": False})

    # Input 3: 3D float64 array, all True mask
    arr = np.ones((2, 2, 2), dtype=np.float64)
    mask = np.ones((2, 2, 2), dtype=bool)
    vals = 0.0
    list_of_inputs.append({"arr": arr, "mask": mask, "vals": vals, "inplace": False})

    # Input 4: 1D float32 array, all False mask
    arr = np.zeros(10, dtype=np.float32)
    mask = np.zeros(10, dtype=bool)
    vals = 3.14159
    list_of_inputs.append({"arr": arr, "mask": mask, "vals": vals, "inplace": False})

    # Input 5: 2D float64 array, identity-like mask
    arr = np.eye(5, dtype=np.float64)
    mask = (arr > 0)
    vals = 9.9
    list_of_inputs.append({"arr": arr, "mask": mask, "vals": vals, "inplace": False})

    # Input 6: 3D float32 array, random mask with negative float value
    arr = np.random.randn(2, 3, 2).astype(np.float32)
    mask = np.random.rand(2, 3, 2) > 0.5
    vals = -99.9
    list_of_inputs.append({"arr": arr, "mask": mask, "vals": vals, "inplace": False})

    # Input 7: 1D float64 array, half True mask
    arr = np.arange(8, dtype=np.float64)
    mask = np.array([True, True, True, True, False, False, False, False], dtype=bool)
    vals = 100.0
    list_of_inputs.append({"arr": arr, "mask": mask, "vals": vals, "inplace": False})

    # Input 8: 4D float32 array, small dimensions
    arr = np.random.randn(1, 2, 2, 1).astype(np.float32)
    mask = np.random.choice([True, False], size=(1, 2, 2, 1))
    vals = 0.5
    list_of_inputs.append({"arr": arr, "mask": mask, "vals": vals, "inplace": False})

    # Input 9: 2D float32 array, checkerboard-like mask
    arr = np.full((3, 3), 5.0, dtype=np.float32)
    mask = np.array([[True, False, True], [False, True, False], [True, False, True]], dtype=bool)
    vals = -5.0
    list_of_inputs.append({"arr": arr, "mask": mask, "vals": vals, "inplace": False})

    # Input 10: 1D float32 array, single element
    arr = np.array([1.1], dtype=np.float32)
    mask = np.array([True], dtype=bool)
    vals = 2.2
    list_of_inputs.append({"arr": arr, "mask": mask, "vals": vals, "inplace": False})

    return list_of_inputs

generated_inputs["jax.numpy.place_3"] = jax_numpy_place_inputs()

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
