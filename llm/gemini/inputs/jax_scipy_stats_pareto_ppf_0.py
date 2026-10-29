
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def pareto_ppf_inputs():
    list_of_inputs = []

    # Input 1: 0-D scalar arrays, standard float32
    q = np.array(0.5, dtype=np.float32)
    b = np.array(2.0, dtype=np.float32)
    loc = np.array(0.0, dtype=np.float32)
    scale = np.array(1.0, dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"q": q, "b": b, "loc": loc, "scale": scale})
    )

    # Input 2: 1-D arrays, float64
    q = np.array([0.1, 0.25, 0.5, 0.75, 0.9], dtype=np.float64)
    b = np.array([1.5, 2.0, 3.0, 2.5, 1.0], dtype=np.float64)
    loc = np.array([0.0, 1.0, -1.0, 2.0, 0.5], dtype=np.float64)
    scale = np.array([1.0, 2.0, 0.5, 1.5, 3.0], dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy({"q": q, "b": b, "loc": loc, "scale": scale})
    )

    # Input 3: 2-D arrays, matching shapes
    q = np.array([[0.1, 0.3], [0.6, 0.8]], dtype=np.float32)
    b = np.array([[2.5, 1.5], [3.0, 0.5]], dtype=np.float32)
    loc = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    scale = np.array([[1.0, 1.0], [2.0, 2.0]], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"q": q, "b": b, "loc": loc, "scale": scale})
    )

    # Input 4: Broadcasting 1D to 2D
    q = np.linspace(0.01, 0.99, 5, dtype=np.float32).reshape(5, 1)
    b = np.array([1.0, 2.0, 3.0], dtype=np.float32).reshape(1, 3)
    loc = np.array([0.0], dtype=np.float32)
    scale = np.array([1.0], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"q": q, "b": b, "loc": loc, "scale": scale})
    )

    # Input 5: Boundary value q=0
    q = np.array([0.0, 0.0], dtype=np.float64)
    b = np.array([2.0, 5.0], dtype=np.float64)
    loc = np.array([10.0, -5.0], dtype=np.float64)
    scale = np.array([2.0, 1.0], dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy({"q": q, "b": b, "loc": loc, "scale": scale})
    )

    # Input 6: 3-D tensors with non-trivial shapes
    q = np.random.uniform(0.05, 0.95, size=(2, 3, 4)).astype(np.float32)
    b = np.random.uniform(0.5, 5.0, size=(2, 3, 4)).astype(np.float32)
    loc = np.zeros((2, 3, 4), dtype=np.float32)
    scale = np.ones((2, 3, 4), dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"q": q, "b": b, "loc": loc, "scale": scale})
    )

    # Input 7: Negative loc and various scales
    q = np.array([0.2, 0.4, 0.6], dtype=np.float32)
    b = np.array([3.0, 3.0, 3.0], dtype=np.float32)
    loc = np.array([-10.0, -20.0, -30.0], dtype=np.float32)
    scale = np.array([0.1, 0.5, 1.0], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"q": q, "b": b, "loc": loc, "scale": scale})
    )

    # Input 8: q very close to 1
    q = np.array([0.99, 0.999, 0.9999], dtype=np.float64)
    b = np.array([4.0, 4.0, 4.0], dtype=np.float64)
    loc = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    scale = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy({"q": q, "b": b, "loc": loc, "scale": scale})
    )

    # Input 9: Small positive b parameter
    q = np.array([[0.1, 0.5]], dtype=np.float32)
    b = np.array([[0.05, 0.1]], dtype=np.float32)
    loc = np.array([[1.0, 2.0]], dtype=np.float32)
    scale = np.array([[0.5, 0.5]], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"q": q, "b": b, "loc": loc, "scale": scale})
    )

    # Input 10: Broadcasting 3D with scalar shapes
    q = np.full((2, 2, 2), 0.5, dtype=np.float32)
    b = np.array(2.0, dtype=np.float32)
    loc = np.array(0.0, dtype=np.float32)
    scale = np.array(1.0, dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"q": q, "b": b, "loc": loc, "scale": scale})
    )

    return list_of_inputs


generated_inputs["jax.scipy.stats.pareto.ppf"] = pareto_ppf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.pareto.ppf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.pareto.ppf'.")


check_valid('jax.scipy.stats.pareto.ppf', generated_inputs['jax.scipy.stats.pareto.ppf'], lib="jax", suffix=0)
