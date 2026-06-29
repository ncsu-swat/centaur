
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def gumbel_r_ppf_inputs():
    list_of_inputs = []

    # Input 1: 1D arrays, float32, standard values
    p = np.array([0.1, 0.5, 0.9], dtype=np.float32)
    loc = np.array([0.0, 1.0, -1.0], dtype=np.float32)
    scale = np.array([1.0, 2.0, 0.5], dtype=np.float32)
    list_of_inputs.append({"p": p, "loc": loc, "scale": scale})

    # Input 2: Scalar values represented as 0D arrays
    p = np.array(0.25, dtype=np.float32)
    loc = np.array(2.0, dtype=np.float32)
    scale = np.array(1.5, dtype=np.float32)
    list_of_inputs.append({"p": p, "loc": loc, "scale": scale})

    # Input 3: 2D arrays, float64
    p = np.random.uniform(0.05, 0.95, size=(3, 3)).astype(np.float64)
    loc = np.random.uniform(-5.0, 5.0, size=(3, 3)).astype(np.float64)
    scale = np.random.uniform(0.1, 5.0, size=(3, 3)).astype(np.float64)
    list_of_inputs.append({"p": p, "loc": loc, "scale": scale})

    # Input 4: Broadcasting shapes (5, 1) and (1, 4)
    p = np.random.uniform(0.1, 0.9, size=(5, 1)).astype(np.float32)
    loc = np.random.uniform(-2.0, 2.0, size=(1, 4)).astype(np.float32)
    scale = np.random.uniform(0.5, 2.5, size=(5, 4)).astype(np.float32)
    list_of_inputs.append({"p": p, "loc": loc, "scale": scale})

    # Input 5: High dimensional arrays (3D)
    p = np.random.uniform(0.1, 0.9, size=(2, 3, 4)).astype(np.float32)
    loc = np.random.uniform(-10.0, 10.0, size=(2, 3, 4)).astype(np.float32)
    scale = np.random.uniform(1.0, 10.0, size=(2, 3, 4)).astype(np.float32)
    list_of_inputs.append({"p": p, "loc": loc, "scale": scale})

    # Input 6: Very small probabilities
    p = np.array([1e-5, 1e-4, 1e-3], dtype=np.float32)
    loc = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    scale = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    list_of_inputs.append({"p": p, "loc": loc, "scale": scale})

    # Input 7: Very large probabilities (close to 1)
    p = np.array([0.99, 0.999, 0.9999], dtype=np.float32)
    loc = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    scale = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    list_of_inputs.append({"p": p, "loc": loc, "scale": scale})

    # Input 8: Small scale values (narrow distribution)
    p = np.random.uniform(0.1, 0.9, size=(5,)).astype(np.float32)
    loc = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    scale = np.array([1e-3, 1e-2, 1e-1, 0.2, 0.3], dtype=np.float32)
    list_of_inputs.append({"p": p, "loc": loc, "scale": scale})

    # Input 9: Large negative loc values
    p = np.random.uniform(0.1, 0.9, size=(4,)).astype(np.float32)
    loc = np.array([-100.0, -500.0, -1000.0, -5000.0], dtype=np.float32)
    scale = np.array([1.0, 5.0, 10.0, 20.0], dtype=np.float32)
    list_of_inputs.append({"p": p, "loc": loc, "scale": scale})

    # Input 10: Larger 1D tensors to test performance scale
    p = np.random.uniform(0.01, 0.99, size=(100,)).astype(np.float32)
    loc = np.random.uniform(-10.0, 10.0, size=(100,)).astype(np.float32)
    scale = np.random.uniform(0.1, 20.0, size=(100,)).astype(np.float32)
    list_of_inputs.append({"p": p, "loc": loc, "scale": scale})

    return list_of_inputs

generated_inputs["jax.scipy.stats.gumbel_r.ppf"] = gumbel_r_ppf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.gumbel_r.ppf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.gumbel_r.ppf'.")


check_valid('jax.scipy.stats.gumbel_r.ppf', generated_inputs['jax.scipy.stats.gumbel_r.ppf'], lib="jax", suffix=0)
