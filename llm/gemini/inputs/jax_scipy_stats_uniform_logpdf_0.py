
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def generate_uniform_logpdf_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D arrays, float32, standard range
    x = np.array([0.1, 0.5, 0.9], dtype=np.float32)
    loc = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    scale = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 2: 2D arrays, float64, scale > 1
    x = np.array([[1.5, 2.0], [2.5, 3.0]], dtype=np.float64)
    loc = np.array([[1.0, 1.0], [2.0, 2.0]], dtype=np.float64)
    scale = np.array([[2.0, 2.0], [2.0, 2.0]], dtype=np.float64)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 3: 0D arrays (scalars represented as tensors)
    x = np.array(0.5, dtype=np.float32)
    loc = np.array(-1.0, dtype=np.float32)
    scale = np.array(3.0, dtype=np.float32)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 4: 3D arrays with negative values for x and loc
    x = np.random.uniform(-5.0, 5.0, (2, 3, 4)).astype(np.float32)
    loc = np.full((2, 3, 4), -2.0, dtype=np.float32)
    scale = np.full((2, 3, 4), 5.0, dtype=np.float32)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 5: Broadcasting - x is 2D, loc is 1D, scale is scalar (0D)
    x = np.array([[0.2, 0.4], [0.6, 0.8]], dtype=np.float32)
    loc = np.array([0.1, 0.3], dtype=np.float32)
    scale = np.array(2.0, dtype=np.float32)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 6: Broadcasting - x is 1D, loc is 2D, scale is 2D
    x = np.array([0.0, 1.0], dtype=np.float64)
    loc = np.array([[-1.0, -1.0], [0.0, 0.0]], dtype=np.float64)
    scale = np.array([[2.0, 3.0], [1.0, 5.0]], dtype=np.float64)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 7: High dimensional arrays (4D)
    x = np.random.uniform(0, 1, (2, 2, 2, 2)).astype(np.float32)
    loc = np.zeros((2, 2, 2, 2), dtype=np.float32)
    scale = np.ones((2, 2, 2, 2), dtype=np.float32)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 8: Very small scale values
    x = np.array([0.001, 0.002], dtype=np.float32)
    loc = np.array([0.0, 0.0], dtype=np.float32)
    scale = np.array([0.005, 0.005], dtype=np.float32)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 9: Large scale values and large loc
    x = np.array([1000.0, 2000.0], dtype=np.float64)
    loc = np.array([500.0, 500.0], dtype=np.float64)
    scale = np.array([2000.0, 2000.0], dtype=np.float64)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 10: Mixed positive and negative coordinates where x is outside distribution (yielding -inf logpdf)
    x = np.array([-1.0, 2.0, 0.5], dtype=np.float32)
    loc = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    scale = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    return [copy.deepcopy(inp) for inp in list_of_inputs]

generated_inputs["jax.scipy.stats.uniform.logpdf"] = generate_uniform_logpdf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.uniform.logpdf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.uniform.logpdf'.")


check_valid('jax.scipy.stats.uniform.logpdf', generated_inputs['jax.scipy.stats.uniform.logpdf'], lib="jax", suffix=0)
