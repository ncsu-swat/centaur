
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def gumbel_l_ppf_inputs():
    list_of_inputs = []

    # Input 1: 0D scalar arrays, standard values
    p = np.array(0.5, dtype=np.float32)
    loc = np.array(0.0, dtype=np.float32)
    scale = np.array(1.0, dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"p": p, "loc": loc, "scale": scale})
    )

    # Input 2: 1D arrays of size 5, float32
    p = np.array([0.1, 0.25, 0.5, 0.75, 0.9], dtype=np.float32)
    loc = np.array([0.0, -1.0, 2.0, 0.5, -0.5], dtype=np.float32)
    scale = np.array([1.0, 0.5, 2.0, 1.5, 0.8], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"p": p, "loc": loc, "scale": scale})
    )

    # Input 3: 2D arrays, float64
    p = np.array([[0.1, 0.3], [0.6, 0.8]], dtype=np.float64)
    loc = np.array([[-2.0, 0.0], [1.5, 3.0]], dtype=np.float64)
    scale = np.array([[0.5, 1.0], [1.2, 2.0]], dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy({"p": p, "loc": loc, "scale": scale})
    )

    # Input 4: 3D arrays, float32
    p = np.random.uniform(0.05, 0.95, size=(2, 3, 4)).astype(np.float32)
    loc = np.random.uniform(-5.0, 5.0, size=(2, 3, 4)).astype(np.float32)
    scale = np.random.uniform(0.1, 3.0, size=(2, 3, 4)).astype(np.float32)
    list_of_inputs.append(
        copy.deepcopy({"p": p, "loc": loc, "scale": scale})
    )

    # Input 5: Broadcasting - 2D p, 1D loc, scalar scale
    p = np.array([[0.2, 0.4, 0.6], [0.3, 0.5, 0.7]], dtype=np.float32)
    loc = np.array([1.0, -1.0, 0.5], dtype=np.float32)
    scale = np.array(2.0, dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"p": p, "loc": loc, "scale": scale})
    )

    # Input 6: Broadcasting - (4, 1) and (1, 3)
    p = np.array([[0.1], [0.3], [0.5], [0.7]], dtype=np.float64)
    loc = np.array([[0.0, 1.0, 2.0]], dtype=np.float64)
    scale = np.array([[0.5, 1.0, 1.5]], dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy({"p": p, "loc": loc, "scale": scale})
    )

    # Input 7: Values close to boundaries (0 and 1)
    p = np.array([1e-5, 0.01, 0.5, 0.99, 1.0 - 1e-5], dtype=np.float64)
    loc = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    scale = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy({"p": p, "loc": loc, "scale": scale})
    )

    # Input 8: 4D tensor with small scale
    p = np.random.uniform(0.1, 0.9, size=(2, 2, 2, 2)).astype(np.float32)
    loc = np.zeros((2, 2, 2, 2), dtype=np.float32)
    scale = np.full((2, 2, 2, 2), 0.01, dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"p": p, "loc": loc, "scale": scale})
    )

    # Input 9: Large loc and large scale values
    p = np.array([0.15, 0.55, 0.85], dtype=np.float32)
    loc = np.array([1000.0, -500.0, 2500.0], dtype=np.float32)
    scale = np.array([50.0, 100.0, 200.0], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"p": p, "loc": loc, "scale": scale})
    )

    # Input 10: 1D array of single element
    p = np.array([0.63212056], dtype=np.float64)
    loc = np.array([-2.5], dtype=np.float64)
    scale = np.array([3.14], dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy({"p": p, "loc": loc, "scale": scale})
    )

    return list_of_inputs


generated_inputs["jax.scipy.stats.gumbel_l.ppf"] = gumbel_l_ppf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.gumbel_l.ppf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.gumbel_l.ppf'.")


check_valid('jax.scipy.stats.gumbel_l.ppf', generated_inputs['jax.scipy.stats.gumbel_l.ppf'], lib="jax", suffix=0)
