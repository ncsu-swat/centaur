
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy


def gumbel_l_pdf_inputs():
    list_of_inputs = []

    # Input 1: 0D scalars
    x = np.array(0.5, dtype=np.float32)
    loc = np.array(0.0, dtype=np.float32)
    scale = np.array(1.0, dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 2: 1D arrays with float32
    x = np.array([-2.0, -0.5, 0.0, 1.5, 3.0], dtype=np.float32)
    loc = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    scale = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 3: 1D arrays with float64
    x = np.array([-1.5, 0.2, 2.5], dtype=np.float64)
    loc = np.array([1.0, -1.0, 0.5], dtype=np.float64)
    scale = np.array([0.5, 2.0, 1.5], dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 4: 2D arrays of matching shape
    x = np.array([[-1.0, 2.0], [0.5, -2.5]], dtype=np.float32)
    loc = np.array([[0.0, 1.0], [-1.0, 0.5]], dtype=np.float32)
    scale = np.array([[1.0, 0.5], [2.0, 1.5]], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 5: Broadcasting - 2D x, 1D loc, 0D scale
    x = np.random.randn(3, 4).astype(np.float32)
    loc = np.array([0.1, -0.2, 0.5, 0.0], dtype=np.float32)
    scale = np.array(1.2, dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 6: Broadcasting - 3D arrays with broadcastable dimensions
    x = np.random.randn(2, 1, 4).astype(np.float64)
    loc = np.random.randn(1, 3, 4).astype(np.float64)
    scale = np.abs(np.random.randn(2, 3, 1)).astype(np.float64) + 0.1
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 7: Large positive and negative values
    x = np.array([-50.0, -10.0, 0.0, 10.0, 50.0], dtype=np.float32)
    loc = np.array([-5.0, 0.0, 5.0, 10.0, 15.0], dtype=np.float32)
    scale = np.array([5.0, 2.0, 1.0, 3.0, 10.0], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 8: 3D tensors with float32
    x = np.zeros((2, 2, 2), dtype=np.float32)
    loc = np.ones((2, 2, 2), dtype=np.float32) * -1.0
    scale = np.ones((2, 2, 2), dtype=np.float32) * 2.5
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 9: 4D tensors
    x = np.random.randn(2, 3, 2, 2).astype(np.float32)
    loc = np.random.randn(2, 3, 2, 2).astype(np.float32)
    scale = np.exp(np.random.randn(2, 3, 2, 2)).astype(np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 10: Column vector loc and row vector scale broadcasting with 2D x
    x = np.linspace(-3.0, 3.0, 12, dtype=np.float64).reshape(4, 3)
    loc = np.array([[0.0], [1.0], [-1.0], [2.0]], dtype=np.float64)
    scale = np.array([[0.5, 1.0, 2.0]], dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 11: Small scale values
    x = np.array([0.01, 0.02, 0.03], dtype=np.float64)
    loc = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    scale = np.array([0.01, 0.05, 0.1], dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    return list_of_inputs


generated_inputs["jax.scipy.stats.gumbel_l.pdf"] = gumbel_l_pdf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.gumbel_l.pdf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.gumbel_l.pdf'.")


check_valid('jax.scipy.stats.gumbel_l.pdf', generated_inputs['jax.scipy.stats.gumbel_l.pdf'], lib="jax", suffix=0)
