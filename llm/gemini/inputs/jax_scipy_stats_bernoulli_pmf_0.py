
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def bernoulli_pmf_inputs():
    list_of_inputs = []

    # Input 1: 0D scalars
    k = np.array(1, dtype=np.int32)
    p = np.array(0.5, dtype=np.float32)
    loc = np.array(0, dtype=np.int32)
    list_of_inputs.append({"k": k, "p": p, "loc": loc})

    # Input 2: 1D arrays, standard binary values
    k = np.array([0, 1, 1, 0], dtype=np.int32)
    p = np.array([0.2, 0.4, 0.6, 0.8], dtype=np.float32)
    loc = np.array([0, 0, 0, 0], dtype=np.int32)
    list_of_inputs.append({"k": k, "p": p, "loc": loc})

    # Input 3: Broadcasted p and loc with 1D k
    k = np.array([0, 1, 2], dtype=np.float32)
    p = np.array(0.7, dtype=np.float32)
    loc = np.array(0.0, dtype=np.float32)
    list_of_inputs.append({"k": k, "p": p, "loc": loc})

    # Input 4: Non-zero loc, shifting support to {loc, loc + 1}
    k = np.array([2, 3, 4], dtype=np.int32)
    p = np.array(0.3, dtype=np.float64)
    loc = np.array(2, dtype=np.int32)
    list_of_inputs.append({"k": k, "p": p, "loc": loc})

    # Input 5: Negative loc
    k = np.array([-1, 0, 1], dtype=np.int32)
    p = np.array(0.6, dtype=np.float32)
    loc = np.array(-1, dtype=np.int32)
    list_of_inputs.append({"k": k, "p": p, "loc": loc})

    # Input 6: 2D arrays with identical shapes
    k = np.array([[0, 1], [1, 0]], dtype=np.int32)
    p = np.array([[0.1, 0.9], [0.5, 0.5]], dtype=np.float32)
    loc = np.zeros((2, 2), dtype=np.int32)
    list_of_inputs.append({"k": k, "p": p, "loc": loc})

    # Input 7: 2D broadcasting across dimensions
    k = np.array([[0], [1], [2]], dtype=np.int32)
    p = np.array([[0.25, 0.75]], dtype=np.float32)
    loc = np.array([[0]], dtype=np.int32)
    list_of_inputs.append({"k": k, "p": p, "loc": loc})

    # Input 8: Boundary probabilities p=0.0 and p=1.0
    k = np.array([0, 1], dtype=np.float32)
    p = np.array([0.0, 1.0], dtype=np.float32)
    loc = np.array([0.0, 0.0], dtype=np.float32)
    list_of_inputs.append({"k": k, "p": p, "loc": loc})

    # Input 9: 3D tensor inputs
    k = np.random.choice([0, 1], size=(2, 3, 4)).astype(np.int32)
    p = np.full((2, 3, 4), 0.4, dtype=np.float32)
    loc = np.zeros((2, 3, 4), dtype=np.int32)
    list_of_inputs.append({"k": k, "p": p, "loc": loc})

    # Input 10: Float64 precision with fractional k (evaluated outside support)
    k = np.array([0.5, 1.5, -0.5], dtype=np.float64)
    p = np.array([0.3, 0.3, 0.3], dtype=np.float64)
    loc = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    list_of_inputs.append({"k": k, "p": p, "loc": loc})

    # Input 11: 4D tensor with broadcasting
    k = np.ones((1, 2, 1, 4), dtype=np.int32)
    p = np.array([[[[0.8]]]], dtype=np.float32)
    loc = np.zeros((1, 1, 1, 1), dtype=np.int32)
    list_of_inputs.append({"k": k, "p": p, "loc": loc})

    return list_of_inputs


generated_inputs["jax.scipy.stats.bernoulli.pmf"] = bernoulli_pmf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.bernoulli.pmf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.bernoulli.pmf'.")


check_valid('jax.scipy.stats.bernoulli.pmf', generated_inputs['jax.scipy.stats.bernoulli.pmf'], lib="jax", suffix=0)
