
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_scipy_stats_cauchy_pdf_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D float32
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    loc = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    scale = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D float32 with negative values for x and loc
    x = np.array([-1.5, 0.0, 1.5], dtype=np.float32)
    loc = np.array([-0.5, 0.5, 1.0], dtype=np.float32)
    scale = np.array([0.5, 1.5, 2.0], dtype=np.float32)
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 0D arrays (scalars as tensors) float64
    x = np.array(0.5, dtype=np.float64)
    loc = np.array(-1.0, dtype=np.float64)
    scale = np.array(2.0, dtype=np.float64)
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D arrays float32
    x = np.random.randn(2, 3).astype(np.float32)
    loc = np.zeros((2, 3), dtype=np.float32)
    scale = np.ones((2, 3), dtype=np.float32)
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D arrays float64
    x = np.random.randn(2, 2, 2).astype(np.float64)
    loc = np.random.randn(2, 2, 2).astype(np.float64)
    scale = (np.abs(np.random.randn(2, 2, 2)) + 0.1).astype(np.float64)
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Broadcasting shapes (2, 3) and (2, 1) and (1, 3)
    x = np.random.randn(2, 3).astype(np.float32)
    loc = np.random.randn(2, 1).astype(np.float32)
    scale = (np.abs(np.random.randn(1, 3)) + 0.5).astype(np.float32)
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 4D arrays
    x = np.random.randn(2, 2, 2, 2).astype(np.float32)
    loc = np.random.randn(2, 2, 2, 2).astype(np.float32)
    scale = (np.abs(np.random.randn(2, 2, 2, 2)) + 0.1).astype(np.float32)
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1D array with large scale values
    x = np.array([100.0, 200.0], dtype=np.float64)
    loc = np.array([0.0, 10.0], dtype=np.float64)
    scale = np.array([1000.0, 2000.0], dtype=np.float64)
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 1D array with small scale values
    x = np.array([0.01, 0.02], dtype=np.float32)
    loc = np.array([0.0, 0.0], dtype=np.float32)
    scale = np.array([0.001, 0.002], dtype=np.float32)
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D array x and loc, with a 1D scale of size 1
    x = np.random.randn(5, 5).astype(np.float32)
    loc = np.random.randn(5, 5).astype(np.float32)
    scale = np.array([1.5], dtype=np.float32)
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.scipy.stats.cauchy.pdf"] = jax_scipy_stats_cauchy_pdf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.cauchy.pdf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.cauchy.pdf'.")


check_valid('jax.scipy.stats.cauchy.pdf', generated_inputs['jax.scipy.stats.cauchy.pdf'], lib="jax", suffix=0)
