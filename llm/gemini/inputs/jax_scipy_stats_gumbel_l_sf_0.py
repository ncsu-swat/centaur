
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def gumbel_l_sf_inputs():
    list_of_inputs = []

    # Input 1: 1D arrays, float32, standard values
    x = np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    loc = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    scale = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D arrays, float64
    x = np.random.randn(2, 3).astype(np.float64)
    loc = np.zeros((2, 3), dtype=np.float64)
    scale = np.ones((2, 3), dtype=np.float64) * 2.5
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D arrays, varying loc and scale
    x = np.linspace(-5, 5, 24).reshape(2, 3, 4).astype(np.float32)
    loc = np.ones((2, 3, 4), dtype=np.float32) * -1.0
    scale = np.abs(np.random.randn(2, 3, 4).astype(np.float32)) + 0.1
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 0D arrays (scalar-like tensors)
    x = np.array(0.5, dtype=np.float32)
    loc = np.array(-0.5, dtype=np.float32)
    scale = np.array(1.5, dtype=np.float32)
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Broadcasting inputs
    x = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)  # (3, 1)
    loc = np.array([0.0, 1.0], dtype=np.float32)            # (2,)
    scale = np.array([[0.5, 1.5]], dtype=np.float32)        # (1, 2)
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: float16 type
    x = np.array([-2.0, 2.0], dtype=np.float16)
    loc = np.array([0.0, 0.0], dtype=np.float16)
    scale = np.array([0.5, 2.0], dtype=np.float16)
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Large scale values
    x = np.random.randn(5, 5).astype(np.float32) * 10
    loc = np.random.randn(5, 5).astype(np.float32)
    scale = np.ones((5, 5), dtype=np.float32) * 100.0
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Small positive scale values
    x = np.array([0.1, -0.1], dtype=np.float32)
    loc = np.array([0.0, 0.0], dtype=np.float32)
    scale = np.array([0.01, 0.001], dtype=np.float32)
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: High-dimensional tensors (4D)
    x = np.random.randn(2, 2, 2, 2).astype(np.float32)
    loc = np.random.randn(2, 2, 2, 2).astype(np.float32)
    scale = np.random.uniform(0.1, 5.0, size=(2, 2, 2, 2)).astype(np.float32)
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Mixing positive/negative x, specific loc and scale
    x = np.array([-10.0, -1.0, 0.0, 1.0, 10.0], dtype=np.float64)
    loc = np.array([2.0, 2.0, 2.0, 2.0, 2.0], dtype=np.float64)
    scale = np.array([0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float64)
    input_dict = {"x": x, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.scipy.stats.gumbel_l.sf"] = gumbel_l_sf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.gumbel_l.sf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.gumbel_l.sf'.")


check_valid('jax.scipy.stats.gumbel_l.sf', generated_inputs['jax.scipy.stats.gumbel_l.sf'], lib="jax", suffix=0)
