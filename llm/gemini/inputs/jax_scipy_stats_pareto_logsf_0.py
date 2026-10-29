
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def pareto_logsf_inputs():
    list_of_inputs = []

    # Input 1: 1D arrays, standard positive values
    x = np.array([1.5, 2.0, 3.0, 5.0], dtype=np.float32)
    b = np.array([1.0, 2.0, 1.5, 3.0], dtype=np.float32)
    loc = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    scale = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "b": b, "loc": loc, "scale": scale})
    )

    # Input 2: 0D scalars (tensors)
    x = np.array(2.5, dtype=np.float64)
    b = np.array(3.0, dtype=np.float64)
    loc = np.array(0.5, dtype=np.float64)
    scale = np.array(1.2, dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "b": b, "loc": loc, "scale": scale})
    )

    # Input 3: 2D arrays with broadcasting loc and scale
    x = np.array([[1.5, 2.0], [3.0, 4.0]], dtype=np.float32)
    b = np.array([[2.0, 2.5], [1.0, 1.5]], dtype=np.float32)
    loc = np.array([[0.0], [0.0]], dtype=np.float32)
    scale = np.array([[1.0, 2.0]], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "b": b, "loc": loc, "scale": scale})
    )

    # Input 4: Values of x less than loc + scale (support boundary test)
    x = np.array([0.1, 0.5, 0.9], dtype=np.float32)
    b = np.array([2.0, 2.0, 2.0], dtype=np.float32)
    loc = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    scale = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "b": b, "loc": loc, "scale": scale})
    )

    # Input 5: Negative loc values
    x = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    b = np.array([1.5, 2.5, 3.5], dtype=np.float64)
    loc = np.array([-2.0, -1.0, -0.5], dtype=np.float64)
    scale = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "b": b, "loc": loc, "scale": scale})
    )

    # Input 6: 3D tensors with broadcasting across dimensions
    x = np.ones((2, 3, 4), dtype=np.float32) * 5.0
    b = np.ones((1, 3, 1), dtype=np.float32) * 2.0
    loc = np.zeros((2, 1, 1), dtype=np.float32)
    scale = np.ones((1, 1, 4), dtype=np.float32) * 2.0
    list_of_inputs.append(
        copy.deepcopy({"x": x, "b": b, "loc": loc, "scale": scale})
    )

    # Input 7: Large b (shape parameter) and large scale
    x = np.array([100.0, 200.0, 300.0], dtype=np.float64)
    b = np.array([10.0, 20.0, 30.0], dtype=np.float64)
    loc = np.array([10.0, 20.0, 30.0], dtype=np.float64)
    scale = np.array([50.0, 50.0, 50.0], dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "b": b, "loc": loc, "scale": scale})
    )

    # Input 8: Fractional shape b < 1
    x = np.array([2.0, 4.0, 8.0], dtype=np.float32)
    b = np.array([0.1, 0.5, 0.8], dtype=np.float32)
    loc = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    scale = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "b": b, "loc": loc, "scale": scale})
    )

    # Input 9: Negative x with sufficiently negative loc
    x = np.array([-5.0, -4.0, -3.0], dtype=np.float32)
    b = np.array([2.0, 3.0, 4.0], dtype=np.float32)
    loc = np.array([-10.0, -10.0, -10.0], dtype=np.float32)
    scale = np.array([2.0, 2.0, 2.0], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "b": b, "loc": loc, "scale": scale})
    )

    # Input 10: 4D tensor inputs
    x = np.full((2, 2, 2, 2), 3.0, dtype=np.float64)
    b = np.full((2, 2, 2, 2), 1.5, dtype=np.float64)
    loc = np.full((2, 2, 2, 2), 0.5, dtype=np.float64)
    scale = np.full((2, 2, 2, 2), 1.0, dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "b": b, "loc": loc, "scale": scale})
    )

    # Input 11: Integer dtypes cast/used as tensors
    x = np.array([5, 6, 7], dtype=np.int32)
    b = np.array([2, 2, 2], dtype=np.int32)
    loc = np.array([0, 0, 0], dtype=np.int32)
    scale = np.array([1, 1, 1], dtype=np.int32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "b": b, "loc": loc, "scale": scale})
    )

    return list_of_inputs


generated_inputs["jax.scipy.stats.pareto.logsf"] = pareto_logsf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.pareto.logsf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.pareto.logsf'.")


check_valid('jax.scipy.stats.pareto.logsf', generated_inputs['jax.scipy.stats.pareto.logsf'], lib="jax", suffix=0)
