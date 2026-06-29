
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def generate_cauchy_ppf_inputs():
    list_of_inputs = []

    # Case 1: 1D arrays, float32, standard range
    q = np.array([0.1, 0.5, 0.9], dtype=np.float32)
    loc = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    scale = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    list_of_inputs.append({"q": q, "loc": loc, "scale": scale})

    # Case 2: 2D arrays, float32
    q = np.array([[0.2, 0.4], [0.6, 0.8]], dtype=np.float32)
    loc = np.array([[1.0, -1.0], [2.0, -2.0]], dtype=np.float32)
    scale = np.array([[0.5, 1.5], [2.0, 0.1]], dtype=np.float32)
    list_of_inputs.append({"q": q, "loc": loc, "scale": scale})

    # Case 3: 3D arrays, float64
    q = np.random.uniform(0.01, 0.99, size=(2, 2, 2)).astype(np.float64)
    loc = np.random.normal(0, 1, size=(2, 2, 2)).astype(np.float64)
    scale = np.random.uniform(0.1, 5.0, size=(2, 2, 2)).astype(np.float64)
    list_of_inputs.append({"q": q, "loc": loc, "scale": scale})

    # Case 4: 0D arrays (scalars as tensors)
    q = np.array(0.25, dtype=np.float32)
    loc = np.array(-5.0, dtype=np.float32)
    scale = np.array(2.5, dtype=np.float32)
    list_of_inputs.append({"q": q, "loc": loc, "scale": scale})

    # Case 5: Broadcasting, q is 1D, loc is 2D, scale is 1D
    q = np.array([0.1, 0.9], dtype=np.float32)
    loc = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
    scale = np.array([1.0, 2.0], dtype=np.float32)
    list_of_inputs.append({"q": q, "loc": loc, "scale": scale})

    # Case 6: Edge values for q (very close to 0 and 1)
    q = np.array([1e-5, 0.5, 1.0 - 1e-5], dtype=np.float64)
    loc = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    scale = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    list_of_inputs.append({"q": q, "loc": loc, "scale": scale})

    # Case 7: Larger scale values
    q = np.linspace(0.1, 0.9, 5).astype(np.float32)
    loc = np.array([100.0, 200.0, 300.0, 400.0, 500.0], dtype=np.float32)
    scale = np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float32)
    list_of_inputs.append({"q": q, "loc": loc, "scale": scale})

    # Case 8: Small scale values (close to 0)
    q = np.array([0.5, 0.5], dtype=np.float32)
    loc = np.array([0.0, 0.0], dtype=np.float32)
    scale = np.array([1e-3, 1e-4], dtype=np.float32)
    list_of_inputs.append({"q": q, "loc": loc, "scale": scale})

    # Case 9: Multi-dimensional, float64
    q = np.random.uniform(0.1, 0.9, size=(3, 1, 4)).astype(np.float64)
    loc = np.zeros((3, 1, 4), dtype=np.float64)
    scale = np.ones((3, 1, 4), dtype=np.float64)
    list_of_inputs.append({"q": q, "loc": loc, "scale": scale})

    # Case 10: Negative loc and various scale shapes (broadcasting)
    q = np.array([0.3, 0.7], dtype=np.float32)
    loc = np.array([-10.0], dtype=np.float32)
    scale = np.array([5.0], dtype=np.float32)
    list_of_inputs.append({"q": q, "loc": loc, "scale": scale})

    return list_of_inputs

generated_inputs["jax.scipy.stats.cauchy.ppf_1"] = generate_cauchy_ppf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.cauchy.ppf_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.cauchy.ppf_1'.")


check_valid('jax.scipy.stats.cauchy.ppf', generated_inputs['jax.scipy.stats.cauchy.ppf_1'], lib="jax", suffix=1)
