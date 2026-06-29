
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def expon_ppf_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D arrays, float32, default-like range
    q = np.array([0.1, 0.3, 0.5, 0.7, 0.9], dtype=np.float32)
    loc = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    scale = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    input_dict = {"q": q, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D arrays, float32, random values
    q = np.random.uniform(0.01, 0.99, (3, 3)).astype(np.float32)
    loc = np.random.uniform(-1.0, 1.0, (3, 3)).astype(np.float32)
    scale = np.random.uniform(0.1, 5.0, (3, 3)).astype(np.float32)
    input_dict = {"q": q, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Scalar-like tensors (0D arrays), float64
    q = np.array(0.5, dtype=np.float64)
    loc = np.array(-2.0, dtype=np.float64)
    scale = np.array(3.0, dtype=np.float64)
    input_dict = {"q": q, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D arrays, float32
    q = np.random.uniform(0.01, 0.99, (2, 2, 2)).astype(np.float32)
    loc = np.zeros((2, 2, 2), dtype=np.float32)
    scale = np.ones((2, 2, 2), dtype=np.float32) * 2.5
    input_dict = {"q": q, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Broadcasted shapes (q is 2D, loc is 1D, scale is 0D)
    q = np.random.uniform(0.1, 0.9, (4, 4)).astype(np.float32)
    loc = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    scale = np.array(1.5, dtype=np.float32)
    input_dict = {"q": q, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Edge cases of q close to 0 and 1
    q = np.array([0.0, 0.0001, 0.999, 0.9999], dtype=np.float32)
    loc = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    scale = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    input_dict = {"q": q, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Large scale and shift values, float64
    q = np.random.uniform(0.1, 0.9, (5,)).astype(np.float64)
    loc = np.array([-10.0, -5.0, 0.0, 5.0, 10.0], dtype=np.float64)
    scale = np.array([100.0, 200.0, 300.0, 400.0, 500.0], dtype=np.float64)
    input_dict = {"q": q, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Very small scale values
    q = np.random.uniform(0.1, 0.9, (3,)).astype(np.float32)
    loc = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    scale = np.array([1e-5, 1e-4, 1e-3], dtype=np.float32)
    input_dict = {"q": q, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: High dimensionality (4D arrays)
    q = np.random.uniform(0.2, 0.8, (2, 2, 2, 2)).astype(np.float32)
    loc = np.random.uniform(-0.5, 0.5, (2, 2, 2, 2)).astype(np.float32)
    scale = np.random.uniform(0.5, 1.5, (2, 2, 2, 2)).astype(np.float32)
    input_dict = {"q": q, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Mixed broadcasting shapes
    q = np.random.uniform(0.1, 0.9, (1, 5)).astype(np.float32)
    loc = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)  # shape (3, 1)
    scale = np.array([2.0], dtype=np.float32)  # shape (1,)
    input_dict = {"q": q, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.scipy.stats.expon.ppf"] = expon_ppf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.expon.ppf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.expon.ppf'.")


check_valid('jax.scipy.stats.expon.ppf', generated_inputs['jax.scipy.stats.expon.ppf'], lib="jax", suffix=0)
