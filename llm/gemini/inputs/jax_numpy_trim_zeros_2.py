
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def trim_zeros_inputs():
    list_of_inputs = []

    # Input 1: 1D array, trim 'fb', axis=(0,)
    filt = np.array([0, 0, 1, 2, 3, 0, 0], dtype=np.int32)
    list_of_inputs.append({"filt": filt, "trim": "fb", "axis": (0,)})

    # Input 2: 1D array with negative values, trim 'f', axis=(0,)
    filt = np.array([0, -5, 2, 0, -1, 0], dtype=np.int64)
    list_of_inputs.append({"filt": filt, "trim": "f", "axis": (0,)})

    # Input 3: 1D array float32, trim 'b', axis=(-1,)
    filt = np.array([0.0, 1.5, 0.0, -2.5, 0.0, 0.0], dtype=np.float32)
    list_of_inputs.append({"filt": filt, "trim": "b", "axis": (-1,)})

    # Input 4: 2D array float64, trim 'fb', axis=(0,)
    filt = np.array(
        [[0.0, 0.0], [1.0, 2.0], [-1.0, 0.5], [0.0, 0.0]], dtype=np.float64
    )
    list_of_inputs.append({"filt": filt, "trim": "fb", "axis": (0,)})

    # Input 5: 2D array float32, trim 'f', axis=(1,)
    filt = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 2.0]], dtype=np.float32)
    list_of_inputs.append({"filt": filt, "trim": "f", "axis": (1,)})

    # Input 6: 2D array int16, trim 'b', axis=(0, 1)
    filt = np.zeros((5, 5), dtype=np.int16)
    filt[1:4, 1:4] = np.array([[1, -2, 3], [-4, 5, -6], [7, 8, 9]])
    list_of_inputs.append({"filt": filt, "trim": "b", "axis": (0, 1)})

    # Input 7: 3D array uint8, trim 'fb', axis=(0, 2)
    filt = np.zeros((4, 3, 4), dtype=np.uint8)
    filt[1:3, :, 1:3] = 5
    list_of_inputs.append({"filt": filt, "trim": "fb", "axis": (0, 2)})

    # Input 8: 3D array complex64, trim 'f', axis=(1,)
    filt = np.array(
        [[[0 + 0j], [1 + 2j], [0 + 0j]], [[0 + 0j], [-1 - 1j], [3 + 0j]]],
        dtype=np.complex64,
    )
    list_of_inputs.append({"filt": filt, "trim": "f", "axis": (1,)})

    # Input 9: 2D boolean array, trim 'fb', axis=(0, 1)
    filt = np.array(
        [[False, False, False], [False, True, False], [False, False, False]],
        dtype=np.bool_,
    )
    list_of_inputs.append({"filt": filt, "trim": "fb", "axis": (0, 1)})

    # Input 10: 4D array float32, trim 'fb', axis=(0, 1, 2, 3)
    filt = np.zeros((3, 3, 3, 3), dtype=np.float32)
    filt[1, 1, 1, 1] = -10.5
    list_of_inputs.append({"filt": filt, "trim": "fb", "axis": (0, 1, 2, 3)})

    # Input 11: 3D array int32, trim 'b', axis=(-2, -1)
    filt = np.zeros((2, 4, 4), dtype=np.int32)
    filt[:, :2, :2] = np.array([[[-1, 2], [3, -4]], [[5, -6], [7, 8]]])
    list_of_inputs.append({"filt": filt, "trim": "b", "axis": (-2, -1)})

    return [copy.deepcopy(x) for x in list_of_inputs]


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
