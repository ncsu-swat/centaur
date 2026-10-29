
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def pareto_pdf_inputs():
    list_of_inputs = []

    # Input 1: 0D scalar tensors, float32, x >= loc + scale
    x = np.array(2.5, dtype=np.float32)
    b = np.array(1.5, dtype=np.float32)
    loc = np.array(0.0, dtype=np.float32)
    scale = np.array(1.0, dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "b": b, "loc": loc, "scale": scale})
    )

    # Input 2: 1D tensors, standard float64 values
    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    b = np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float64)
    loc = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    scale = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "b": b, "loc": loc, "scale": scale})
    )

    # Input 3: Negative loc, x < loc + scale (out-of-support case, pdf evaluates to 0)
    x = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    b = np.array([3.0, 3.0, 3.0], dtype=np.float32)
    loc = np.array([-2.0, -2.0, -2.0], dtype=np.float32)
    scale = np.array([2.0, 2.0, 2.0], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "b": b, "loc": loc, "scale": scale})
    )

    # Input 4: 2D tensors, matching shapes
    x = np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float32)
    b = np.array([[1.0, 2.0], [0.5, 3.5]], dtype=np.float32)
    loc = np.array([[0.0, 0.5], [1.0, -1.0]], dtype=np.float32)
    scale = np.array([[1.0, 1.5], [2.0, 0.5]], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "b": b, "loc": loc, "scale": scale})
    )

    # Input 5: Broadcasting - 1D x with scalar-like 1D parameters
    x = np.linspace(1.5, 10.0, 10, dtype=np.float64)
    b = np.array([2.5], dtype=np.float64)
    loc = np.array([0.5], dtype=np.float64)
    scale = np.array([1.0], dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "b": b, "loc": loc, "scale": scale})
    )

    # Input 6: 3D tensors
    x = np.ones((2, 3, 4), dtype=np.float32) * 5.0
    b = np.ones((2, 3, 4), dtype=np.float32) * 2.0
    loc = np.zeros((2, 3, 4), dtype=np.float32)
    scale = np.ones((2, 3, 4), dtype=np.float32) * 2.0
    list_of_inputs.append(
        copy.deepcopy({"x": x, "b": b, "loc": loc, "scale": scale})
    )

    # Input 7: Broadcasting - (3, 1) with (1, 4)
    x = np.array([[2.0], [3.0], [4.0]], dtype=np.float32)
    b = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    loc = np.array([[0.0]], dtype=np.float32)
    scale = np.array([[1.0]], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "b": b, "loc": loc, "scale": scale})
    )

    # Input 8: Large shape parameter b and small scale
    x = np.array([1.001, 1.01, 1.1], dtype=np.float64)
    b = np.array([50.0, 20.0, 10.0], dtype=np.float64)
    loc = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    scale = np.array([0.1, 0.5, 0.8], dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "b": b, "loc": loc, "scale": scale})
    )

    # Input 9: Non-zero positive loc with shifted support
    x = np.array([11.0, 15.0, 20.0], dtype=np.float32)
    b = np.array([1.2, 2.4, 3.6], dtype=np.float32)
    loc = np.array([10.0, 10.0, 10.0], dtype=np.float32)
    scale = np.array([0.5, 1.0, 2.0], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "b": b, "loc": loc, "scale": scale})
    )

    # Input 10: 4D tensors with broadcasting
    x = np.random.uniform(5.0, 10.0, size=(2, 1, 3, 4)).astype(np.float32)
    b = np.random.uniform(0.5, 3.0, size=(1, 2, 3, 1)).astype(np.float32)
    loc = np.zeros((1, 1, 1, 1), dtype=np.float32)
    scale = np.ones((1, 1, 1, 1), dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "b": b, "loc": loc, "scale": scale})
    )

    return list_of_inputs


generated_inputs["jax.scipy.stats.pareto.pdf"] = pareto_pdf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.pareto.pdf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.pareto.pdf'.")


check_valid('jax.scipy.stats.pareto.pdf', generated_inputs['jax.scipy.stats.pareto.pdf'], lib="jax", suffix=0)
