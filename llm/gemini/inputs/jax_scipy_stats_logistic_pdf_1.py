
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_scipy_stats_logistic_pdf_inputs():
    list_of_inputs = []

    # Input 1: 0D scalars (represented as 0D arrays)
    x = np.array(0.5, dtype=np.float32)
    loc = np.array(0.0, dtype=np.float32)
    scale = np.array(1.0, dtype=np.float32)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 2: 1D arrays, standard float32
    x = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    loc = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    scale = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 3: 2D arrays, float64, negative loc
    x = np.random.randn(2, 3).astype(np.float64)
    loc = np.random.randn(2, 3).astype(np.float64) - 1.0
    scale = np.abs(np.random.randn(2, 3)).astype(np.float64) + 0.1
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 4: 3D arrays, float32, large scale values
    x = np.random.randn(2, 2, 2).astype(np.float32)
    loc = np.zeros((2, 2, 2), dtype=np.float32)
    scale = np.ones((2, 2, 2), dtype=np.float32) * 5.0
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 5: Broadcastable shapes (2D and 1D)
    x = np.random.randn(3, 4).astype(np.float32)
    loc = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    scale = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 6: Broadcastable shapes (2D and 2D with singleton dimensions)
    x = np.array([[1.0], [2.0]], dtype=np.float64)
    loc = np.array([0.5, 1.5], dtype=np.float64)
    scale = np.array([[0.5], [1.5]], dtype=np.float64)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 7: High dimensions (4D arrays)
    x = np.random.randn(1, 3, 1, 3).astype(np.float32)
    loc = np.random.randn(1, 3, 1, 3).astype(np.float32)
    scale = np.exp(np.random.randn(1, 3, 1, 3)).astype(np.float32)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 8: Small scale values (steep PDF) and large x
    x = np.array([-10.0, 10.0], dtype=np.float32)
    loc = np.array([0.0, 0.0], dtype=np.float32)
    scale = np.array([0.1, 0.2], dtype=np.float32)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 9: Integer arrays (castable to float during operations)
    x = np.array([1, 2, 3], dtype=np.int32)
    loc = np.array([0, -1, 1], dtype=np.int32)
    scale = np.array([1, 2, 3], dtype=np.int32)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    # Input 10: Mixed positive and negative values with 1D broadcasting of loc/scale
    x = np.random.randn(5).astype(np.float64)
    loc = np.array([0.0], dtype=np.float64)
    scale = np.array([1.0], dtype=np.float64)
    list_of_inputs.append({"x": x, "loc": loc, "scale": scale})

    return list_of_inputs

generated_inputs["jax.scipy.stats.logistic.pdf_1"] = jax_scipy_stats_logistic_pdf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.logistic.pdf_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.logistic.pdf_1'.")


check_valid('jax.scipy.stats.logistic.pdf', generated_inputs['jax.scipy.stats.logistic.pdf_1'], lib="jax", suffix=1)
