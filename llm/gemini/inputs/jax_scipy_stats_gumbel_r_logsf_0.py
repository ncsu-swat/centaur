
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def gumbel_r_logsf_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D float32 arrays
    x = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    loc = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    scale = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D float64 arrays with varying scales and locations
    x = np.array([-1.5, 0.5, 3.2], dtype=np.float64)
    loc = np.array([-1.0, 0.0, 1.0], dtype=np.float64)
    scale = np.array([0.5, 1.5, 2.5], dtype=np.float64)
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D float32 arrays (random values, scale guaranteed positive)
    x = np.random.randn(2, 3).astype(np.float32)
    loc = np.random.randn(2, 3).astype(np.float32)
    scale = np.abs(np.random.randn(2, 3).astype(np.float32)) + 0.1
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Scalar tensors (0D arrays)
    x = np.array(2.5, dtype=np.float32)
    loc = np.array(-1.0, dtype=np.float32)
    scale = np.array(0.5, dtype=np.float32)
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Broadcasting - 2D x, 1D loc and scale
    x = np.random.randn(3, 4).astype(np.float32)
    loc = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    scale = np.array([1.0, 1.1, 1.2, 1.3], dtype=np.float32)
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Broadcasting - 1D x, 2D loc, 0D scale
    x = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    loc = np.random.randn(2, 3).astype(np.float32)
    scale = np.array(1.5, dtype=np.float32)
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Large scale values
    x = np.array([10.0, 50.0, 100.0], dtype=np.float32)
    loc = np.array([0.0, 10.0, 20.0], dtype=np.float32)
    scale = np.array([100.0, 200.0, 500.0], dtype=np.float32)
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Small scale values (close to 0)
    x = np.array([0.1, -0.2, 0.0], dtype=np.float32)
    loc = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    scale = np.array([1e-4, 1e-4, 1e-4], dtype=np.float32)
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D float32 arrays
    x = np.random.randn(2, 2, 2).astype(np.float32)
    loc = np.random.randn(2, 2, 2).astype(np.float32)
    scale = np.ones((2, 2, 2), dtype=np.float32) * 3.5
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Diverse 1D float64 values with extreme negatives and positives
    x = np.array([-10.0, 0.0, 10.0, 100.0], dtype=np.float64)
    loc = np.array([-5.0, 0.0, 5.0, 50.0], dtype=np.float64)
    scale = np.array([0.1, 1.0, 10.0, 100.0], dtype=np.float64)
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.scipy.stats.gumbel_r.logsf"] = gumbel_r_logsf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.gumbel_r.logsf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.gumbel_r.logsf'.")


check_valid('jax.scipy.stats.gumbel_r.logsf', generated_inputs['jax.scipy.stats.gumbel_r.logsf'], lib="jax", suffix=0)
