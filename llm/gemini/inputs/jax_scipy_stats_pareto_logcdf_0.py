
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_scipy_stats_pareto_logcdf_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D float32, standard domain
    x = np.array([2.0, 3.0, 4.0], dtype=np.float32)
    b = np.array([1.5, 2.0, 2.5], dtype=np.float32)
    loc = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    scale = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    input_dict = {"x": x, "b": b, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D float64, with non-zero loc and scale > 1
    x = np.array([5.0, 10.0, 15.0], dtype=np.float64)
    b = np.array([3.0, 3.0, 3.0], dtype=np.float64)
    loc = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    scale = np.array([2.0, 2.0, 2.0], dtype=np.float64)
    input_dict = {"x": x, "b": b, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Scalars (0D arrays)
    x = np.array(3.5, dtype=np.float32)
    b = np.array(1.2, dtype=np.float32)
    loc = np.array(0.5, dtype=np.float32)
    scale = np.array(1.5, dtype=np.float32)
    input_dict = {"x": x, "b": b, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D arrays, float32
    x = np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float32)
    b = np.array([[1.0, 1.5], [2.0, 2.5]], dtype=np.float32)
    loc = np.array([[0.0, 0.5], [1.0, 1.5]], dtype=np.float32)
    scale = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    input_dict = {"x": x, "b": b, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Broadcasting, scale and loc as 1D, x and b as 2D
    x = np.random.uniform(5.0, 10.0, size=(3, 3)).astype(np.float32)
    b = np.random.uniform(1.0, 3.0, size=(3, 3)).astype(np.float32)
    loc = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    scale = np.array([1.0, 2.0, 1.5], dtype=np.float32)
    input_dict = {"x": x, "b": b, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: x below support (should yield -inf or limit values)
    x = np.array([0.5, 0.8], dtype=np.float32)
    b = np.array([2.0, 2.0], dtype=np.float32)
    loc = np.array([0.0, 0.0], dtype=np.float32)
    scale = np.array([1.0, 1.0], dtype=np.float32)
    input_dict = {"x": x, "b": b, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Large dimensions (3D arrays)
    x = np.random.uniform(10.0, 20.0, size=(2, 2, 2)).astype(np.float64)
    b = np.random.uniform(0.5, 2.5, size=(2, 2, 2)).astype(np.float64)
    loc = np.zeros((2, 2, 2), dtype=np.float64)
    scale = np.ones((2, 2, 2), dtype=np.float64)
    input_dict = {"x": x, "b": b, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Large scale and loc parameters
    x = np.array([1050.0, 2100.0], dtype=np.float32)
    b = np.array([2.5, 3.5], dtype=np.float32)
    loc = np.array([1000.0, 2000.0], dtype=np.float32)
    scale = np.array([10.0, 50.0], dtype=np.float32)
    input_dict = {"x": x, "b": b, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Small shapes but float64
    x = np.array([[1.5], [2.5]], dtype=np.float64)
    b = np.array([[0.8], [1.2]], dtype=np.float64)
    loc = np.array([[0.1], [0.2]], dtype=np.float64)
    scale = np.array([[0.9], [1.1]], dtype=np.float64)
    input_dict = {"x": x, "b": b, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Mixed positive values with negative loc
    x = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    b = np.array([1.0, 1.5, 2.0], dtype=np.float32)
    loc = np.array([-2.0, -1.0, 0.0], dtype=np.float32)
    scale = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    input_dict = {"x": x, "b": b, "loc": loc, "scale": scale}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.scipy.stats.pareto.logcdf"] = jax_scipy_stats_pareto_logcdf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.pareto.logcdf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.pareto.logcdf'.")


check_valid('jax.scipy.stats.pareto.logcdf', generated_inputs['jax.scipy.stats.pareto.logcdf'], lib="jax", suffix=0)
