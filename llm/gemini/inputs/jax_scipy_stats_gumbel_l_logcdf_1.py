
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def gumbel_l_logcdf_inputs():
    list_of_inputs = []

    # Input 1: 0D tensors (scalars) with float32
    x = np.array(1.5, dtype=np.float32)
    loc = np.array(0.0, dtype=np.float32)
    scale = np.array(2.0, dtype=np.float32)
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D tensors with negative values
    x = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    loc = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    scale = np.array([1.0, 1.5, 2.0], dtype=np.float32)
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D tensors with float64
    x = np.array([[-1.0, 2.0], [0.5, -0.5]], dtype=np.float64)
    loc = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float64)
    scale = np.array([[1.0, 2.0], [0.5, 1.5]], dtype=np.float64)
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Broadcasting (x is 2D, loc is 1D, scale is 0D)
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    loc = np.array([0.5, 1.5], dtype=np.float32)
    scale = np.array(2.0, dtype=np.float32)
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Large negative loc and small scale
    x = np.array([0.0], dtype=np.float32)
    loc = np.array([-10.0], dtype=np.float32)
    scale = np.array([0.1], dtype=np.float32)
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Large values for high-precision float64
    x = np.array([100.0, -100.0], dtype=np.float64)
    loc = np.array([50.0, -50.0], dtype=np.float64)
    scale = np.array([10.0, 10.0], dtype=np.float64)
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D tensors
    x = np.random.randn(2, 2, 2).astype(np.float32)
    loc = np.random.randn(2, 2, 2).astype(np.float32)
    scale = np.abs(np.random.randn(2, 2, 2).astype(np.float32)) + 0.1
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Uniform scale and zeros for loc
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    loc = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    scale = np.array([5.0, 5.0, 5.0], dtype=np.float32)
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Scale is large
    x = np.array([10.0], dtype=np.float32)
    loc = np.array([1.0], dtype=np.float32)
    scale = np.array([100.0], dtype=np.float32)
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Multi-dimensional broadcasting (1, 3) and (2, 1) and (2, 3)
    x = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    loc = np.array([[0.0], [1.0]], dtype=np.float32)
    scale = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.scipy.stats.gumbel_l.logcdf_1"] = gumbel_l_logcdf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.gumbel_l.logcdf_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.gumbel_l.logcdf_1'.")


check_valid('jax.scipy.stats.gumbel_l.logcdf', generated_inputs['jax.scipy.stats.gumbel_l.logcdf_1'], lib="jax", suffix=1)
