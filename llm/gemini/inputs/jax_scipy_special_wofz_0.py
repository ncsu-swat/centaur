
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def wofz_inputs():
    list_of_inputs = []

    # Input 1: 0D scalar float32
    z = np.array(0.0, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"z": z}))

    # Input 2: 0D scalar complex64
    z = np.array(1.0 + 2.0j, dtype=np.complex64)
    list_of_inputs.append(copy.deepcopy({"z": z}))

    # Input 3: 1D array of floats (float32)
    z = np.array([-3.0, -1.5, 0.0, 1.5, 3.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"z": z}))

    # Input 4: 1D array of float64
    z = np.array([-10.0, -1.0, 1.0, 10.0], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"z": z}))

    # Input 5: 1D array of complex128
    z = np.array([1.0 + 1.0j, -2.0 + 0.5j, 0.0 - 1.5j], dtype=np.complex128)
    list_of_inputs.append(copy.deepcopy({"z": z}))

    # Input 6: 2D array of float32
    z = np.array([[-1.0, 2.0], [3.0, -4.0]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"z": z}))

    # Input 7: 2D array of complex64
    z = np.array(
        [[0.5 + 0.5j, -0.5 - 0.5j], [1.5 - 1.5j, -1.5 + 1.5j]],
        dtype=np.complex64,
    )
    list_of_inputs.append(copy.deepcopy({"z": z}))

    # Input 8: 3D array of float64
    z = np.linspace(-2.0, 2.0, 12, dtype=np.float64).reshape((2, 2, 3))
    list_of_inputs.append(copy.deepcopy({"z": z}))

    # Input 9: 3D array of complex128
    real = np.zeros((2, 3, 2), dtype=np.float64)
    imag = np.ones((2, 3, 2), dtype=np.float64)
    z = (real + 1j * imag).astype(np.complex128)
    list_of_inputs.append(copy.deepcopy({"z": z}))

    # Input 10: 4D array of float32
    z = np.ones((1, 2, 3, 4), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"z": z}))

    return list_of_inputs


generated_inputs["jax.scipy.special.wofz"] = wofz_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.special.wofz' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.special.wofz'.")


check_valid('jax.scipy.special.wofz', generated_inputs['jax.scipy.special.wofz'], lib="jax", suffix=0)
