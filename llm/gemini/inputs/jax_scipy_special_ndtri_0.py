
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def ndtri_inputs():
    list_of_inputs = []

    # Input 1: 1D array, float32, standard values in (0, 1)
    p = np.array([0.1, 0.3, 0.5, 0.7, 0.9], dtype=np.float32)
    list_of_inputs.append({"p": p})

    # Input 2: 0D array (scalar), float32
    p = np.array(0.5, dtype=np.float32)
    list_of_inputs.append({"p": p})

    # Input 3: 2D array, float64, random values in (0, 1)
    p = np.random.uniform(1e-5, 1-1e-5, size=(3, 4)).astype(np.float64)
    list_of_inputs.append({"p": p})

    # Input 4: 3D array, float32, random values in (0, 1)
    p = np.random.uniform(0.01, 0.99, size=(2, 2, 3)).astype(np.float32)
    list_of_inputs.append({"p": p})

    # Input 5: 1D array, float64, values very close to boundaries 0 and 1
    p = np.array([1e-7, 1e-4, 0.5, 1-1e-4, 1-1e-7], dtype=np.float64)
    list_of_inputs.append({"p": p})

    # Input 6: 4D array, float32
    p = np.random.uniform(0.1, 0.9, size=(1, 2, 2, 3)).astype(np.float32)
    list_of_inputs.append({"p": p})

    # Input 7: 1D array, float32, uniform distribution
    p = np.linspace(0.01, 0.99, 10, dtype=np.float32)
    list_of_inputs.append({"p": p})

    # Input 8: 2D array, float64
    p = np.linspace(0.1, 0.9, 9, dtype=np.float64).reshape((3, 3))
    list_of_inputs.append({"p": p})

    # Input 9: 1D array, float64, single element
    p = np.array([0.25], dtype=np.float64)
    list_of_inputs.append({"p": p})

    # Input 10: 3D array, float32, shape (4, 1, 2)
    p = np.random.uniform(0.2, 0.8, size=(4, 1, 2)).astype(np.float32)
    list_of_inputs.append({"p": p})

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
