
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def poch_inputs():
    list_of_inputs = []

    # Input 1: 0D scalar arrays (float32)
    z = np.array(3.5, dtype=np.float32)
    m = np.array(2.0, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"z": z, "m": m}))

    # Input 2: 1D arrays of same shape (float32)
    z = np.array([1.0, 2.5, 3.0, 4.2], dtype=np.float32)
    m = np.array([0.5, 1.0, 2.0, 1.5], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"z": z, "m": m}))

    # Input 3: 2D arrays (float64)
    z = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float64)
    m = np.array([[0.5, 1.5], [2.5, 3.5]], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"z": z, "m": m}))

    # Input 4: 3D arrays (float32)
    z = np.ones((2, 3, 4), dtype=np.float32) * 2.0
    m = np.ones((2, 3, 4), dtype=np.float32) * 1.5
    list_of_inputs.append(copy.deepcopy({"z": z, "m": m}))

    # Input 5: Broadcasting 1D and 2D arrays
    z = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
    m = np.array([0.5, 1.0, 1.5, 2.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"z": z, "m": m}))

    # Input 6: Negative non-integer values
    z = np.array([-1.5, -2.3, -0.7], dtype=np.float32)
    m = np.array([0.5, 1.2, 0.3], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"z": z, "m": m}))

    # Input 7: m = 0 (Pochhammer symbol is identically 1)
    z = np.array([0.5, 1.5, 2.5, 3.5], dtype=np.float64)
    m = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"z": z, "m": m}))

    # Input 8: Fractional positive values
    z = np.array([0.1, 0.25, 0.5, 0.75], dtype=np.float32)
    m = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"z": z, "m": m}))

    # Input 9: 4D arrays
    z = np.full((2, 2, 2, 2), 3.0, dtype=np.float32)
    m = np.full((2, 2, 2, 2), 2.0, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"z": z, "m": m}))

    # Input 10: Mixed positive and negative values with float64
    z = np.array([5.5, -3.5, 2.2, -0.8], dtype=np.float64)
    m = np.array([1.5, 2.0, -0.5, 0.4], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"z": z, "m": m}))

    # Input 11: Broadcasting scalar-like 1D with 2D array
    z = np.array([4.0], dtype=np.float32)
    m = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"z": z, "m": m}))

    return list_of_inputs


generated_inputs["jax.scipy.special.poch"] = poch_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.special.poch' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.special.poch'.")


check_valid('jax.scipy.special.poch', generated_inputs['jax.scipy.special.poch'], lib="jax", suffix=0)
