
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def spence_inputs():
    list_of_inputs = []

    # Input 1: 0D scalar float32 (positive value)
    x = np.array(1.0, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 2: 0D scalar float64 (positive value)
    x = np.array(2.5, dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 3: 1D array float32 with values around 1
    x = np.array([0.1, 0.5, 1.0, 1.5, 2.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 4: 1D array float64 including negative values (returns NaN per doc)
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 5: 2D array float32
    x = np.random.uniform(0.1, 10.0, size=(3, 4)).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 6: 2D array float64
    x = np.random.uniform(0.01, 1.0, size=(2, 5)).astype(np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 7: 3D array float32
    x = np.random.uniform(1.0, 100.0, size=(2, 3, 4)).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 8: 3D array float64 with special values (0, 1, inf)
    x = np.array(
        [[[0.0, 1.0], [2.0, np.inf]], [[0.5, 1.5], [np.nan, 3.0]]],
        dtype=np.float64,
    )
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 9: 4D array float32
    x = np.random.uniform(0.5, 5.0, size=(2, 2, 2, 3)).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 10: 1D empty array float32
    x = np.array([], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    # Input 11: 1D array float64 with small values close to 0
    x = np.array([1e-6, 1e-4, 1e-2, 0.9999], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x}))

    return list_of_inputs


generated_inputs["jax.scipy.special.spence"] = spence_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.special.spence' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.special.spence'.")


check_valid('jax.scipy.special.spence', generated_inputs['jax.scipy.special.spence'], lib="jax", suffix=0)
