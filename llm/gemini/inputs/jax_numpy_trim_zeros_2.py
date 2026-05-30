
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def trim_zeros_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, both trimmed, axis (0,)
    filt = np.array([0.0, 0.0, 1.0, 2.0, 3.0, 0.0], dtype=np.float32)
    input_dict = {
        "filt": filt,
        "trim": "fb",
        "axis": (0,)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D int32 array, leading trimmed, axis (0,)
    filt = np.array([0, 0, -1, 5, 0, 10, 0], dtype=np.int32)
    input_dict = {
        "filt": filt,
        "trim": "f",
        "axis": (0,)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D float64 array, trailing trimmed, axis (0,)
    filt = np.array([1.5, 0.0, 2.3, 0.0, 0.0], dtype=np.float64)
    input_dict = {
        "filt": filt,
        "trim": "b",
        "axis": (0,)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D float32 array, trim along axis 0
    filt = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, 0.0, 0.0]
    ], dtype=np.float32)
    input_dict = {
        "filt": filt,
        "trim": "fb",
        "axis": (0,)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D float32 array, trim along axis 1
    filt = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, 0.0, 0.0]
    ], dtype=np.float32)
    input_dict = {
        "filt": filt,
        "trim": "fb",
        "axis": (1,)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D int32 array, trim 'f' along axes (0, 1)
    filt = np.array([
        [0, 0, 0],
        [0, 3, 4],
        [0, 5, 0]
    ], dtype=np.int32)
    input_dict = {
        "filt": filt,
        "trim": "f",
        "axis": (0, 1)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D float32 array, trim 'b' along axis (2,)
    filt = np.zeros((3, 3, 3), dtype=np.float32)
    filt[1, 1, 0] = 1.0
    filt[1, 1, 1] = 2.0
    input_dict = {
        "filt": filt,
        "trim": "b",
        "axis": (2,)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D int16 array, trim 'fb' along axis (1, 2)
    filt = np.zeros((2, 4, 4), dtype=np.int16)
    filt[0, 1:3, 1:3] = 5
    filt[1, 1:3, 1:3] = 6
    input_dict = {
        "filt": filt,
        "trim": "fb",
        "axis": (1, 2)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 1D array with no non-zeros (all zeros), trim 'fb', axis (0,)
    filt = np.zeros((5,), dtype=np.float32)
    input_dict = {
        "filt": filt,
        "trim": "fb",
        "axis": (0,)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 4D float32 array, trim 'fb' along axis (1, 2, 3)
    filt = np.zeros((2, 3, 3, 3), dtype=np.float32)
    filt[0, 1, 1, 1] = 9.0
    filt[1, 1, 1, 1] = 9.0
    input_dict = {
        "filt": filt,
        "trim": "fb",
        "axis": (1, 2, 3)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 2D complex64 array, trim 'f' along axis (0,)
    filt = np.array([
        [0+0j, 0+0j],
        [1+1j, 2+2j],
        [0+0j, 0+0j]
    ], dtype=np.complex64)
    input_dict = {
        "filt": filt,
        "trim": "f",
        "axis": (0,)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.trim_zeros_2"] = trim_zeros_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.trim_zeros_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.trim_zeros_2'.")


check_valid('jax.numpy.trim_zeros', generated_inputs['jax.numpy.trim_zeros_2'], lib="jax", suffix=2)
