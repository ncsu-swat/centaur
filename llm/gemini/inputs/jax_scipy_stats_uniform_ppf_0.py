
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_scipy_stats_uniform_ppf_inputs():
    list_of_inputs = []

    # Input 1: Simple 1D float32
    q = np.array([0.1, 0.3, 0.5, 0.7, 0.9], dtype=np.float32)
    loc = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    scale = np.array([1.0, 2.0, 1.5, 3.0, 0.5], dtype=np.float32)
    input_dict = {"q": q, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 0D scalars (tensors), float32
    q = np.array(0.5, dtype=np.float32)
    loc = np.array(-1.0, dtype=np.float32)
    scale = np.array(2.0, dtype=np.float32)
    input_dict = {"q": q, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D arrays, float32
    q = np.array([[0.2, 0.4], [0.6, 0.8]], dtype=np.float32)
    loc = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    scale = np.array([[1.0, 2.0], [2.0, 1.0]], dtype=np.float32)
    input_dict = {"q": q, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D arrays, float64
    q = np.array([0.05, 0.95], dtype=np.float64)
    loc = np.array([-10.0, 10.0], dtype=np.float64)
    scale = np.array([5.0, 5.0], dtype=np.float64)
    input_dict = {"q": q, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Broad-castable shapes: q (2D), loc (1D), scale (0D)
    q = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    loc = np.array([1.0, 2.0], dtype=np.float32)
    scale = np.array(5.0, dtype=np.float32)
    input_dict = {"q": q, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Broad-castable shapes: q (1D), loc (2D), scale (2D)
    q = np.array([0.5, 0.5], dtype=np.float32)
    loc = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
    scale = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    input_dict = {"q": q, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Negative loc values, 1D float32
    q = np.array([0.25, 0.75], dtype=np.float32)
    loc = np.array([-5.0, -10.0], dtype=np.float32)
    scale = np.array([10.0, 20.0], dtype=np.float32)
    input_dict = {"q": q, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D arrays, float32
    q = np.ones((2, 2, 2), dtype=np.float32) * 0.5
    loc = np.zeros((2, 2, 2), dtype=np.float32)
    scale = np.ones((2, 2, 2), dtype=np.float32) * 3.0
    input_dict = {"q": q, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Extreme probability values (0 and 1)
    q = np.array([0.0, 1.0, 0.5], dtype=np.float32)
    loc = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    scale = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    input_dict = {"q": q, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D float64 with a large range
    q = np.random.uniform(0.0, 1.0, size=(3, 3)).astype(np.float64)
    loc = np.array([[-100.0, -50.0, 0.0], [50.0, 100.0, 150.0], [200.0, 250.0, 300.0]], dtype=np.float64)
    scale = np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0], [70.0, 80.0, 90.0]], dtype=np.float64)
    input_dict = {"q": q, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.scipy.stats.uniform.ppf"] = jax_scipy_stats_uniform_ppf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.uniform.ppf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.uniform.ppf'.")


check_valid('jax.scipy.stats.uniform.ppf', generated_inputs['jax.scipy.stats.uniform.ppf'], lib="jax", suffix=0)
