
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def gumbel_r_pdf_inputs():
    list_of_inputs = []

    # Input 1: 0D scalars (represented as 0D numpy arrays)
    x = np.array(0.5, dtype=np.float32)
    loc = np.array(0.0, dtype=np.float32)
    scale = np.array(1.0, dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 2: 1D arrays with default-like parameters
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    loc = np.array([0.0], dtype=np.float32)
    scale = np.array([1.0], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 3: 1D arrays with float64 and negative loc
    x = np.linspace(-5.0, 5.0, 10, dtype=np.float64)
    loc = np.array([-1.5], dtype=np.float64)
    scale = np.array([2.0], dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 4: 2D arrays, matching shapes
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    loc = np.array([[0.5, 0.5], [1.0, 1.0]], dtype=np.float32)
    scale = np.array([[1.5, 2.0], [0.5, 1.0]], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 5: 2D x with broadcasted loc and scale (1D loc and 2D scale)
    x = np.random.randn(3, 4).astype(np.float32)
    loc = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    scale = np.array([[0.5], [1.0], [2.0]], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 6: 3D arrays
    x = np.random.randn(2, 3, 4).astype(np.float64)
    loc = np.zeros((2, 3, 4), dtype=np.float64)
    scale = np.ones((2, 3, 4), dtype=np.float64) * 1.5
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 7: Broadcasting with 3D and 1D
    x = np.random.randn(4, 5, 2).astype(np.float32)
    loc = np.array([0.0], dtype=np.float32)
    scale = np.array([0.5, 1.0], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 8: High dimensional 4D arrays
    x = np.random.uniform(-3, 3, size=(2, 2, 3, 3)).astype(np.float32)
    loc = np.random.uniform(-1, 1, size=(2, 2, 3, 3)).astype(np.float32)
    scale = np.random.uniform(0.1, 2.0, size=(2, 2, 3, 3)).astype(np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 9: Large values in x (testing tail behaviour)
    x = np.array([10.0, 20.0, 50.0], dtype=np.float64)
    loc = np.array([5.0], dtype=np.float64)
    scale = np.array([3.0], dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 10: Negative extremes in x
    x = np.array([-50.0, -20.0, -10.0], dtype=np.float32)
    loc = np.array([0.0], dtype=np.float32)
    scale = np.array([1.0], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    # Input 11: Small fractional scale values
    x = np.array([0.0, 0.1, 0.2, 0.3], dtype=np.float32)
    loc = np.array([0.1, 0.1, 0.1, 0.1], dtype=np.float32)
    scale = np.array([0.01, 0.05, 0.1, 0.2], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "loc": loc, "scale": scale})
    )

    return list_of_inputs


generated_inputs["jax.scipy.stats.gumbel_r.pdf"] = gumbel_r_pdf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.gumbel_r.pdf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.gumbel_r.pdf'.")


check_valid('jax.scipy.stats.gumbel_r.pdf', generated_inputs['jax.scipy.stats.gumbel_r.pdf'], lib="jax", suffix=0)
