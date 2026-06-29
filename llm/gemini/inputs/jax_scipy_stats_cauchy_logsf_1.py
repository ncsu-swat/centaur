
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_scipy_stats_cauchy_logsf_inputs():
    list_of_inputs = []

    # Input 1: Basic float32 1D arrays
    x = np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    loc = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    scale = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float64 arrays with negative loc
    x = np.array([-2.5, -1.0, 0.5, 3.0], dtype=np.float64)
    loc = np.array([-1.0, -1.0, -1.0, -1.0], dtype=np.float64)
    scale = np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float64)
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D arrays with random values
    x = np.random.randn(3, 4).astype(np.float32)
    loc = np.zeros((3, 4), dtype=np.float32)
    scale = np.ones((3, 4), dtype=np.float32)
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Scalar arrays (0D tensors)
    x = np.array(1.5, dtype=np.float32)
    loc = np.array(-0.5, dtype=np.float32)
    scale = np.array(0.8, dtype=np.float32)
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: High dimensional (3D) arrays
    x = np.random.randn(2, 3, 2).astype(np.float32)
    loc = np.random.randn(2, 3, 2).astype(np.float32)
    scale = np.abs(np.random.randn(2, 3, 2)).astype(np.float32) + 0.1
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Large values for x (testing tails)
    x = np.array([-100.0, -50.0, 0.0, 50.0, 100.0], dtype=np.float32)
    loc = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    scale = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Broadcasting case with differing shapes
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    loc = np.array([[0.0], [1.0]], dtype=np.float32)
    scale = np.array([1.5], dtype=np.float32)
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Broadcasting scale as 0D tensor
    x = np.random.randn(4, 2).astype(np.float32)
    loc = np.random.randn(4, 2).astype(np.float32)
    scale = np.array(0.5, dtype=np.float32)
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Small positive scales (float64)
    x = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    loc = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    scale = np.array([1e-3, 1e-2, 1e-1], dtype=np.float64)
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Larger array size
    x = np.linspace(-10, 10, 100).astype(np.float32)
    loc = np.zeros(100, dtype=np.float32)
    scale = np.ones(100, dtype=np.float32) * 5.0
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.scipy.stats.cauchy.logsf_1"] = jax_scipy_stats_cauchy_logsf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.cauchy.logsf_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.cauchy.logsf_1'.")


check_valid('jax.scipy.stats.cauchy.logsf', generated_inputs['jax.scipy.stats.cauchy.logsf_1'], lib="jax", suffix=1)
