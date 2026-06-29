
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def gamma_pdf_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D float32 arrays
    x = np.array([0.5, 1.5, 2.5], dtype=np.float32)
    a = np.array([2.0, 2.0, 2.0], dtype=np.float32)
    loc = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    scale = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    list_of_inputs.append({"x": x, "a": a, "loc": loc, "scale": scale})

    # Input 2: 0D arrays (scalars)
    x = np.array(1.0, dtype=np.float32)
    a = np.array(3.5, dtype=np.float32)
    loc = np.array(-0.5, dtype=np.float32)
    scale = np.array(2.0, dtype=np.float32)
    list_of_inputs.append({"x": x, "a": a, "loc": loc, "scale": scale})

    # Input 3: 2D arrays with float64
    x = np.array([[0.5, 1.0], [1.5, 2.0]], dtype=np.float64)
    a = np.array([[1.0, 1.5], [2.0, 2.5]], dtype=np.float64)
    loc = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float64)
    scale = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float64)
    list_of_inputs.append({"x": x, "a": a, "loc": loc, "scale": scale})

    # Input 4: Broadcasting shapes
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    a = np.array([[1.5], [2.5]], dtype=np.float32)
    loc = np.array([0.0], dtype=np.float32)
    scale = np.array([1.0], dtype=np.float32)
    list_of_inputs.append({"x": x, "a": a, "loc": loc, "scale": scale})

    # Input 5: Large values and large scales
    x = np.array([10.0, 50.0], dtype=np.float32)
    a = np.array([5.0, 10.0], dtype=np.float32)
    loc = np.array([2.0, 5.0], dtype=np.float32)
    scale = np.array([10.0, 20.0], dtype=np.float32)
    list_of_inputs.append({"x": x, "a": a, "loc": loc, "scale": scale})

    # Input 6: 3D arrays
    x = np.ones((2, 2, 2), dtype=np.float32) * 1.5
    a = np.ones((2, 2, 2), dtype=np.float32) * 3.0
    loc = np.ones((2, 2, 2), dtype=np.float32) * 0.5
    scale = np.ones((2, 2, 2), dtype=np.float32) * 1.2
    list_of_inputs.append({"x": x, "a": a, "loc": loc, "scale": scale})

    # Input 7: Negative loc, x is positive and greater than loc
    x = np.array([0.1, 0.5], dtype=np.float32)
    a = np.array([1.2, 1.8], dtype=np.float32)
    loc = np.array([-1.0, -2.0], dtype=np.float32)
    scale = np.array([0.5, 1.5], dtype=np.float32)
    list_of_inputs.append({"x": x, "a": a, "loc": loc, "scale": scale})

    # Input 8: Some x values less than loc (results in 0 PDF but is valid)
    x = np.array([-1.0, 2.0], dtype=np.float32)
    a = np.array([2.0, 2.0], dtype=np.float32)
    loc = np.array([0.0, 0.0], dtype=np.float32)
    scale = np.array([1.0, 1.0], dtype=np.float32)
    list_of_inputs.append({"x": x, "a": a, "loc": loc, "scale": scale})

    # Input 9: Small values of shape parameter a
    x = np.array([0.01, 0.1], dtype=np.float32)
    a = np.array([0.1, 0.5], dtype=np.float32)
    loc = np.array([0.0, 0.0], dtype=np.float32)
    scale = np.array([1.0, 1.0], dtype=np.float32)
    list_of_inputs.append({"x": x, "a": a, "loc": loc, "scale": scale})

    # Input 10: 1D arrays with float64, mixed values
    x = np.array([0.0, 5.0, 10.0], dtype=np.float64)
    a = np.array([4.0, 4.0, 4.0], dtype=np.float64)
    loc = np.array([-1.0, 0.0, 1.0], dtype=np.float64)
    scale = np.array([2.0, 2.0, 2.0], dtype=np.float64)
    list_of_inputs.append({"x": x, "a": a, "loc": loc, "scale": scale})

    return [copy.deepcopy(x) for x in list_of_inputs]

generated_inputs["jax.scipy.stats.gamma.pdf"] = gamma_pdf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.gamma.pdf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.gamma.pdf'.")


check_valid('jax.scipy.stats.gamma.pdf', generated_inputs['jax.scipy.stats.gamma.pdf'], lib="jax", suffix=0)
