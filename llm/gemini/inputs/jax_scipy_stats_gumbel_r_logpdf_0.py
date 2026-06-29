
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def gumbel_r_logpdf_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D arrays with float32
    x = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    loc = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    scale = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 0D arrays (scalars)
    x = np.array(1.5, dtype=np.float32)
    loc = np.array(0.5, dtype=np.float32)
    scale = np.array(2.0, dtype=np.float32)
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative x and loc values, varying scales
    x = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    loc = np.array([-1.5, -1.5, -1.5], dtype=np.float32)
    scale = np.array([0.5, 1.5, 2.5], dtype=np.float32)
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D arrays with float64
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    loc = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float64)
    scale = np.array([[1.0, 1.5], [2.0, 2.5]], dtype=np.float64)
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Broadcast-compatible shapes (2, 1) and (1, 2)
    x = np.array([[1.0], [2.0]], dtype=np.float32)
    loc = np.array([[0.0, 1.0]], dtype=np.float32)
    scale = np.array([[1.0, 1.0]], dtype=np.float32)
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Randomly generated values (float32)
    x = np.random.randn(5).astype(np.float32)
    loc = np.random.randn(5).astype(np.float32)
    scale = np.abs(np.random.randn(5).astype(np.float32)) + 0.1
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D arrays (float32)
    x = np.random.randn(2, 3, 4).astype(np.float32)
    loc = np.random.randn(2, 3, 4).astype(np.float32)
    scale = np.random.uniform(0.1, 5.0, size=(2, 3, 4)).astype(np.float32)
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Small scale values (float64)
    x = np.array([10.0, 20.0], dtype=np.float64)
    loc = np.array([5.0, 5.0], dtype=np.float64)
    scale = np.array([1e-5, 1e-4], dtype=np.float64)
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large x and loc values
    x = np.array([100.0, -100.0], dtype=np.float64)
    loc = np.array([0.0, 0.0], dtype=np.float64)
    scale = np.array([10.0, 20.0], dtype=np.float64)
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Single element 1D arrays
    x = np.array([0.5], dtype=np.float32)
    loc = np.array([-0.5], dtype=np.float32)
    scale = np.array([2.5], dtype=np.float32)
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.scipy.stats.gumbel_r.logpdf"] = gumbel_r_logpdf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.gumbel_r.logpdf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.gumbel_r.logpdf'.")


check_valid('jax.scipy.stats.gumbel_r.logpdf', generated_inputs['jax.scipy.stats.gumbel_r.logpdf'], lib="jax", suffix=0)
