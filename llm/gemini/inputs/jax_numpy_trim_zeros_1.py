
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def trim_zeros_inputs():
    list_of_inputs = []

    # Input 1: 1D int array, trim='fb', axis=0
    filt = np.array([0, 0, 1, 2, 0, 3, 0], dtype=np.int32)
    trim = "fb"
    axis = 0
    list_of_inputs.append(
        copy.deepcopy({"filt": filt, "trim": trim, "axis": axis})
    )

    # Input 2: 1D float array with negative numbers, trim='f', axis=0
    filt = np.array([0.0, 0.0, -1.5, 2.0, 0.0], dtype=np.float32)
    trim = "f"
    axis = 0
    list_of_inputs.append(
        copy.deepcopy({"filt": filt, "trim": trim, "axis": axis})
    )

    # Input 3: 1D float array, trim='b', axis=0
    filt = np.array([0.0, 3.14, 0.0, 0.0], dtype=np.float64)
    trim = "b"
    axis = 0
    list_of_inputs.append(
        copy.deepcopy({"filt": filt, "trim": trim, "axis": axis})
    )

    # Input 4: 2D int array, trim='fb', axis=0
    filt = np.array([[0, 0, 0], [0, 1, 0], [0, 2, 0], [0, 0, 0]], dtype=np.int64)
    trim = "fb"
    axis = 0
    list_of_inputs.append(
        copy.deepcopy({"filt": filt, "trim": trim, "axis": axis})
    )

    # Input 5: 2D float array, trim='f', axis=1
    filt = np.array(
        [[0.0, 0.0, 1.0, 2.0], [0.0, 0.0, 3.0, 0.0]], dtype=np.float32
    )
    trim = "f"
    axis = 1
    list_of_inputs.append(
        copy.deepcopy({"filt": filt, "trim": trim, "axis": axis})
    )

    # Input 6: 2D int array, trim='b', axis=-1 (negative axis)
    filt = np.array([[1, 2, 0, 0], [3, 4, 5, 0]], dtype=np.int16)
    trim = "b"
    axis = -1
    list_of_inputs.append(
        copy.deepcopy({"filt": filt, "trim": trim, "axis": axis})
    )

    # Input 7: 3D int array, trim='fb', axis=1
    filt = np.zeros((3, 4, 2), dtype=np.int32)
    filt[:, 1:3, :] = 1
    trim = "fb"
    axis = 1
    list_of_inputs.append(
        copy.deepcopy({"filt": filt, "trim": trim, "axis": axis})
    )

    # Input 8: 3D float array, trim='b', axis=2
    filt = np.zeros((2, 2, 5), dtype=np.float64)
    filt[:, :, :3] = -2.5
    trim = "b"
    axis = 2
    list_of_inputs.append(
        copy.deepcopy({"filt": filt, "trim": trim, "axis": axis})
    )

    # Input 9: 3D bool array, trim='f', axis=0
    filt = np.zeros((5, 2, 2), dtype=bool)
    filt[2:, :, :] = True
    trim = "f"
    axis = 0
    list_of_inputs.append(
        copy.deepcopy({"filt": filt, "trim": trim, "axis": axis})
    )

    # Input 10: 4D complex array, trim='fb', axis=-2
    filt = np.zeros((2, 2, 4, 3), dtype=np.complex64)
    filt[:, :, 1:3, :] = 1.0 + 2.0j
    trim = "fb"
    axis = -2
    list_of_inputs.append(
        copy.deepcopy({"filt": filt, "trim": trim, "axis": axis})
    )

    return list_of_inputs


generated_inputs["jax.numpy.trim_zeros_1"] = trim_zeros_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.trim_zeros_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.trim_zeros_1'.")


check_valid('jax.numpy.trim_zeros', generated_inputs['jax.numpy.trim_zeros_1'], lib="jax", suffix=1)
