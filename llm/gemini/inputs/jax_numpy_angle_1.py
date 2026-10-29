
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def angle_inputs():
    list_of_inputs = []

    # Input 1: 1D complex64, radians
    z = np.array([1.0 + 2.0j, -3.0 + 4.0j, -1.0 - 1.0j], dtype=np.complex64)
    list_of_inputs.append({"z": copy.deepcopy(z), "deg": False})

    # Input 2: 1D complex128, degrees
    z = np.array([1.0 + 0.0j, 0.0 + 1.0j, -1.0 + 0.0j, 0.0 - 1.0j], dtype=np.complex128)
    list_of_inputs.append({"z": copy.deepcopy(z), "deg": True})

    # Input 3: 2D complex64 array
    z = np.array(
        [[1.0 + 3.0j, 2.0 - 5.0j], [4.0 - 3.0j, 3.0 + 2.0j]],
        dtype=np.complex64,
    )
    list_of_inputs.append({"z": copy.deepcopy(z), "deg": False})

    # Input 4: 2D complex128 array with deg=True
    z = np.array(
        [[-2.5 + 1.5j, 0.0 + 0.0j], [3.5 - 4.5j, -1.0 - 2.0j]],
        dtype=np.complex128,
    )
    list_of_inputs.append({"z": copy.deepcopy(z), "deg": True})

    # Input 5: 0D scalar complex array
    z = np.array(2.0 + 3.0j, dtype=np.complex64)
    list_of_inputs.append({"z": copy.deepcopy(z), "deg": False})

    # Input 6: 3D complex64 array
    z = np.ones((2, 3, 4), dtype=np.complex64) * (1.0 - 1.0j)
    list_of_inputs.append({"z": copy.deepcopy(z), "deg": True})

    # Input 7: Real float32 array (positive and negative values)
    z = np.array([-5.0, -1.0, 0.0, 1.0, 5.0], dtype=np.float32)
    list_of_inputs.append({"z": copy.deepcopy(z), "deg": False})

    # Input 8: Real float64 array with deg=True
    z = np.array([[-10.0, 20.0], [0.0, -30.0]], dtype=np.float64)
    list_of_inputs.append({"z": copy.deepcopy(z), "deg": True})

    # Input 9: Integer int32 array
    z = np.array([-2, -1, 0, 1, 2], dtype=np.int32)
    list_of_inputs.append({"z": copy.deepcopy(z), "deg": False})

    # Input 10: 4D complex128 array
    z = np.zeros((2, 2, 2, 2), dtype=np.complex128)
    z[0, 1, 0, 1] = -1.0 + 1.0j
    list_of_inputs.append({"z": copy.deepcopy(z), "deg": True})

    # Input 11: 1D array with special values like inf and nan
    z = np.array([np.nan + 1.0j, np.inf - 1.0j, -np.inf + 0.0j], dtype=np.complex64)
    list_of_inputs.append({"z": copy.deepcopy(z), "deg": False})

    return list_of_inputs


generated_inputs["jax.numpy.angle_1"] = angle_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.angle_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.angle_1'.")


check_valid('jax.numpy.angle', generated_inputs['jax.numpy.angle_1'], lib="jax", suffix=1)
