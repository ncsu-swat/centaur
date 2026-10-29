
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def gumbel_r_ppf_inputs():
    list_of_inputs = []

    # Input 1: 0D scalar arrays
    p = np.array(0.5, dtype=np.float32)
    loc = np.array(0.0, dtype=np.float32)
    scale = np.array(1.0, dtype=np.float32)
    list_of_inputs.append({"p": p, "loc": loc, "scale": scale})

    # Input 2: 1D arrays with standard values
    p = np.array([0.1, 0.25, 0.5, 0.75, 0.9], dtype=np.float32)
    loc = np.array([0.0, 1.0, -1.0, 2.0, -2.0], dtype=np.float32)
    scale = np.array([1.0, 0.5, 2.0, 1.5, 0.8], dtype=np.float32)
    list_of_inputs.append({"p": p, "loc": loc, "scale": scale})

    # Input 3: 2D arrays, float64 precision
    p = np.array([[0.1, 0.3], [0.7, 0.9]], dtype=np.float64)
    loc = np.array([[0.0, -2.5], [1.5, 3.0]], dtype=np.float64)
    scale = np.array([[1.0, 2.0], [0.5, 1.5]], dtype=np.float64)
    list_of_inputs.append({"p": p, "loc": loc, "scale": scale})

    # Input 4: 3D arrays
    p = np.full((2, 3, 4), 0.6, dtype=np.float32)
    loc = np.zeros((2, 3, 4), dtype=np.float32)
    scale = np.ones((2, 3, 4), dtype=np.float32)
    list_of_inputs.append({"p": p, "loc": loc, "scale": scale})

    # Input 5: Broadcasting shapes: p is (4, 1), loc is (1, 3), scale is (4, 3)
    p = np.array([[0.1], [0.4], [0.6], [0.9]], dtype=np.float32)
    loc = np.array([[-1.0, 0.0, 1.0]], dtype=np.float32)
    scale = np.full((4, 3), 2.0, dtype=np.float32)
    list_of_inputs.append({"p": p, "loc": loc, "scale": scale})

    # Input 6: Large scale and negative location values
    p = np.array([0.05, 0.5, 0.95], dtype=np.float32)
    loc = np.array([-100.0, -50.0, -10.0], dtype=np.float32)
    scale = np.array([10.0, 50.0, 100.0], dtype=np.float32)
    list_of_inputs.append({"p": p, "loc": loc, "scale": scale})

    # Input 7: Broadcasting with 1D scale and 2D p and loc
    p = np.linspace(0.01, 0.99, 12, dtype=np.float64).reshape(3, 4)
    loc = np.array([[0.0, 1.0, 2.0, 3.0]], dtype=np.float64)
    scale = np.array([0.5, 1.0, 1.5, 2.0], dtype=np.float64)
    list_of_inputs.append({"p": p, "loc": loc, "scale": scale})

    # Input 8: Extreme valid probabilities close to bounds
    p = np.array([1e-6, 1e-4, 0.5, 1.0 - 1e-4, 1.0 - 1e-6], dtype=np.float64)
    loc = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    scale = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)
    list_of_inputs.append({"p": p, "loc": loc, "scale": scale})

    # Input 9: 4D tensor inputs
    p = np.random.uniform(0.1, 0.9, size=(2, 2, 2, 2)).astype(np.float32)
    loc = np.random.randn(2, 2, 2, 2).astype(np.float32)
    scale = np.random.uniform(0.5, 3.0, size=(2, 2, 2, 2)).astype(np.float32)
    list_of_inputs.append({"p": p, "loc": loc, "scale": scale})

    # Input 10: Single element 1D arrays
    p = np.array([0.33], dtype=np.float32)
    loc = np.array([5.5], dtype=np.float32)
    scale = np.array([0.25], dtype=np.float32)
    list_of_inputs.append({"p": p, "loc": loc, "scale": scale})

    return [copy.deepcopy(inp) for inp in list_of_inputs]


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
