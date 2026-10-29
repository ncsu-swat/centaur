
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def cauchy_pdf_inputs():
    list_of_inputs = []

    # Input 1: 0D scalars as numpy arrays (float32)
    x = np.array(0.5, dtype=np.float32)
    loc = np.array(0.0, dtype=np.float32)
    scale = np.array(1.0, dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 2: 1D arrays of matching shape
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    loc = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    scale = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 3: 2D arrays, float64
    x = np.array([[-1.5, 0.5], [2.0, -0.5]], dtype=np.float64)
    loc = np.array([[1.0, -1.0], [0.0, 0.5]], dtype=np.float64)
    scale = np.array([[0.5, 1.5], [2.0, 1.0]], dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 4: Broadcasting shapes: x (2, 3), loc (1, 3), scale (2, 1)
    x = np.arange(6, dtype=np.float32).reshape(2, 3) - 3.0
    loc = np.array([[0.0, 1.0, 2.0]], dtype=np.float32)
    scale = np.array([[1.0], [2.0]], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 5: 3D arrays
    x = np.random.randn(2, 3, 4).astype(np.float32)
    loc = np.zeros((2, 3, 4), dtype=np.float32)
    scale = np.ones((2, 3, 4), dtype=np.float32) * 2.5
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 6: Integer arrays
    x = np.array([-5, 0, 5], dtype=np.int32)
    loc = np.array([0, 1, -1], dtype=np.int32)
    scale = np.array([1, 2, 3], dtype=np.int32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 7: Broadcasting with 0D scalar loc and scale
    x = np.linspace(-10.0, 10.0, 50, dtype=np.float64)
    loc = np.array(-2.5, dtype=np.float64)
    scale = np.array(0.75, dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 8: 4D arrays with small scales
    x = np.zeros((1, 2, 2, 3), dtype=np.float32)
    loc = np.ones((1, 2, 2, 3), dtype=np.float32)
    scale = np.full((1, 2, 2, 3), 0.1, dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 9: Negative loc, large values for x
    x = np.array([1e3, -1e3, 1e4], dtype=np.float64)
    loc = np.array([-50.0, -100.0, -200.0], dtype=np.float64)
    scale = np.array([10.0, 5.0, 20.0], dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 10: Broadcasting 1D to 2D
    x = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
    loc = np.array([0.0, 1.0], dtype=np.float32)
    scale = np.array([[0.5, 1.5]], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    return list_of_inputs


generated_inputs["jax.scipy.stats.cauchy.pdf"] = cauchy_pdf_inputs()

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
