
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def bernoulli_cdf_inputs():
    list_of_inputs = []

    # Input 1: 0D scalars (represented as 0D numpy arrays)
    k = np.array(0, dtype=np.int32)
    p = np.array(0.5, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"k": k, "p": p}))

    # Input 2: 1D arrays with standard binary values
    k = np.array([0, 1, 0, 1], dtype=np.int32)
    p = np.array([0.2, 0.4, 0.6, 0.8], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"k": k, "p": p}))

    # Input 3: Negative and greater than 1 values for k
    k = np.array([-2, -1, 0, 1, 2, 3], dtype=np.int64)
    p = np.array([0.3, 0.3, 0.3, 0.3, 0.3, 0.3], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"k": k, "p": p}))

    # Input 4: Floating point k values
    k = np.array([-0.5, 0.0, 0.5, 1.0, 1.5], dtype=np.float32)
    p = np.array([0.7, 0.7, 0.7, 0.7, 0.7], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"k": k, "p": p}))

    # Input 5: 2D arrays with identical shape
    k = np.array([[0, 1], [1, 2]], dtype=np.int32)
    p = np.array([[0.1, 0.5], [0.9, 0.2]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"k": k, "p": p}))

    # Input 6: Broadcasting 1D k with 2D p
    k = np.array([0, 1, 2], dtype=np.int32)
    p = np.full((4, 3), 0.25, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"k": k, "p": p}))

    # Input 7: Broadcasting 2D k with 1D p
    k = np.ones((3, 4), dtype=np.float64)
    p = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"k": k, "p": p}))

    # Input 8: 3D arrays
    k = np.random.randint(-1, 3, size=(2, 3, 4)).astype(np.int32)
    p = np.random.uniform(0.0, 1.0, size=(2, 3, 4)).astype(np.float32)
    list_of_inputs.append(copy.deepcopy({"k": k, "p": p}))

    # Input 9: Boundary values for probability p (0 and 1)
    k = np.array([-1, 0, 1, 2], dtype=np.int32)
    p = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"k": k, "p": p}))

    # Input 10: Higher dimensional broadcasting with float64
    k = np.array([[[0]], [[1]]], dtype=np.float64)
    p = np.array([0.3, 0.7], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"k": k, "p": p}))

    return list_of_inputs


generated_inputs["jax.scipy.stats.bernoulli.cdf"] = bernoulli_cdf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.bernoulli.cdf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.bernoulli.cdf'.")


check_valid('jax.scipy.stats.bernoulli.cdf', generated_inputs['jax.scipy.stats.bernoulli.cdf'], lib="jax", suffix=0)
