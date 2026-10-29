
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def cauchy_isf_inputs():
    list_of_inputs = []

    # Input 1: 0D scalars, standard Cauchy (loc=0, scale=1)
    q = np.array(0.25, dtype=np.float32)
    loc = np.array(0.0, dtype=np.float32)
    scale = np.array(1.0, dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"q": q, "loc": loc, "scale": scale})
    )

    # Input 2: 1D array of probabilities, float32
    q = np.array([0.1, 0.5, 0.9], dtype=np.float32)
    loc = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    scale = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"q": q, "loc": loc, "scale": scale})
    )

    # Input 3: 1D array with non-zero loc and non-unit scale, float64
    q = np.array([0.05, 0.5, 0.95], dtype=np.float64)
    loc = np.array([-2.0, 0.0, 3.5], dtype=np.float64)
    scale = np.array([0.5, 1.5, 2.0], dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy({"q": q, "loc": loc, "scale": scale})
    )

    # Input 4: 2D array, float32
    q = np.array([[0.2, 0.4], [0.6, 0.8]], dtype=np.float32)
    loc = np.array([[1.0, -1.0], [2.0, -2.0]], dtype=np.float32)
    scale = np.array([[0.5, 1.0], [1.5, 2.0]], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"q": q, "loc": loc, "scale": scale})
    )

    # Input 5: Broadcasting - 2D q with 1D loc and 0D scale
    q = np.linspace(0.1, 0.9, 6, dtype=np.float32).reshape(2, 3)
    loc = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    scale = np.array(0.5, dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"q": q, "loc": loc, "scale": scale})
    )

    # Input 6: Broadcasting - 1D q with 2D loc and 2D scale
    q = np.array([0.3, 0.7], dtype=np.float64)
    loc = np.array([[1.0, 2.0], [-1.0, -2.0]], dtype=np.float64)
    scale = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy({"q": q, "loc": loc, "scale": scale})
    )

    # Input 7: 3D tensors, float32
    q = np.full((2, 2, 2), 0.5, dtype=np.float32)
    loc = np.zeros((2, 2, 2), dtype=np.float32)
    scale = np.ones((2, 2, 2), dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"q": q, "loc": loc, "scale": scale})
    )

    # Input 8: Extreme tail probabilities
    q = np.array([1e-4, 1e-2, 0.5, 1.0 - 1e-2, 1.0 - 1e-4], dtype=np.float64)
    loc = np.array([0.0], dtype=np.float64)
    scale = np.array([1.0], dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy({"q": q, "loc": loc, "scale": scale})
    )

    # Input 9: Large negative loc and varied scale values
    q = np.array([[0.1], [0.5], [0.9]], dtype=np.float32)
    loc = np.array([[-100.0], [-50.0], [0.0]], dtype=np.float32)
    scale = np.array([[10.0], [5.0], [1.0]], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"q": q, "loc": loc, "scale": scale})
    )

    # Input 10: 4D tensor with singleton dimensions to test broadcasting
    q = np.array([[[[0.15]]]], dtype=np.float32)
    loc = np.ones((2, 1, 3, 1), dtype=np.float32)
    scale = np.full((1, 4, 1, 1), 2.0, dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"q": q, "loc": loc, "scale": scale})
    )

    return list_of_inputs


generated_inputs["jax.scipy.stats.cauchy.isf"] = cauchy_isf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.cauchy.isf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.cauchy.isf'.")


check_valid('jax.scipy.stats.cauchy.isf', generated_inputs['jax.scipy.stats.cauchy.isf'], lib="jax", suffix=0)
