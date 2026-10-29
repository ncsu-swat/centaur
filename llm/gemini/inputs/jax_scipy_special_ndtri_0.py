
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def ndtri_inputs():
    list_of_inputs = []

    # Input 1: 0D scalar array, float32
    p = np.array(0.5, dtype=np.float32)
    list_of_inputs.append({"p": copy.deepcopy(p)})

    # Input 2: 0D scalar array, float64
    p = np.array(0.8413447, dtype=np.float64)
    list_of_inputs.append({"p": copy.deepcopy(p)})

    # Input 3: 1D array with standard quantiles, float32
    p = np.array([0.025, 0.05, 0.25, 0.5, 0.75, 0.95, 0.975], dtype=np.float32)
    list_of_inputs.append({"p": copy.deepcopy(p)})

    # Input 4: 1D array with boundary values (0 and 1), float64
    p = np.array([0.0, 0.5, 1.0], dtype=np.float64)
    list_of_inputs.append({"p": copy.deepcopy(p)})

    # Input 5: 2D array, float32
    p = np.linspace(0.1, 0.9, 12, dtype=np.float32).reshape(3, 4)
    list_of_inputs.append({"p": copy.deepcopy(p)})

    # Input 6: 2D array, float64
    rng = np.random.default_rng(42)
    p = rng.uniform(0.01, 0.99, size=(4, 5)).astype(np.float64)
    list_of_inputs.append({"p": copy.deepcopy(p)})

    # Input 7: 3D array, float32
    p = rng.uniform(1e-4, 1.0 - 1e-4, size=(2, 3, 4)).astype(np.float32)
    list_of_inputs.append({"p": copy.deepcopy(p)})

    # Input 8: 4D array, float64
    p = rng.uniform(0.1, 0.9, size=(2, 2, 3, 2)).astype(np.float64)
    list_of_inputs.append({"p": copy.deepcopy(p)})

    # Input 9: Extreme values near 0 and 1, float64
    p = np.array([1e-12, 1e-6, 0.5, 1.0 - 1e-6, 1.0 - 1e-12], dtype=np.float64)
    list_of_inputs.append({"p": copy.deepcopy(p)})

    # Input 10: Empty 1D array, float32
    p = np.array([], dtype=np.float32)
    list_of_inputs.append({"p": copy.deepcopy(p)})

    # Input 11: 1-element 2D array, float64
    p = np.array([[0.3]], dtype=np.float64)
    list_of_inputs.append({"p": copy.deepcopy(p)})

    return list_of_inputs


generated_inputs["jax.scipy.special.ndtri"] = ndtri_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.special.ndtri' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.special.ndtri'.")


check_valid('jax.scipy.special.ndtri', generated_inputs['jax.scipy.special.ndtri'], lib="jax", suffix=0)
