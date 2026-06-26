
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def multigammaln_inputs():
    list_of_inputs = []

    # Input 1, valid — 1D array, d=1
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    d = 1
    input_dict = {"a": a, "d": d}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2, valid — 1D array, d=2
    a = np.array([1.5, 2.5, 3.5], dtype=np.float32)
    d = 2
    input_dict = {"a": a, "d": d}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3, valid — 2D array, d=3 (float64)
    a = np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float64)
    d = 3
    input_dict = {"a": a, "d": d}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4, valid — 3D array, d=4
    a = (np.ones((2, 2, 2), dtype=np.float32) * 5.0)
    d = 4
    input_dict = {"a": a, "d": d}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5, valid — 0D array (scalar), d=2
    a = np.array(3.0, dtype=np.float32)
    d = 2
    input_dict = {"a": a, "d": d}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6, valid — 1D array, d=5
    a = np.array([2.5, 10.0, 100.0], dtype=np.float64)
    d = 5
    input_dict = {"a": a, "d": d}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7, valid — 2D random array, d=1
    a = np.random.uniform(0.5, 10.0, size=(5, 5)).astype(np.float32)
    d = 1
    input_dict = {"a": a, "d": d}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8, valid — float64 precision, d=3
    a = np.array([1.1, 1.2, 1.3], dtype=np.float64)
    d = 3
    input_dict = {"a": a, "d": d}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9, valid — High dimension d=10, 3D array
    a = np.random.uniform(5.0, 20.0, size=(3, 3, 3)).astype(np.float32)
    d = 10
    input_dict = {"a": a, "d": d}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10, valid — Large values, d=2
    a = np.array([1000.0, 2000.0], dtype=np.float32)
    d = 2
    input_dict = {"a": a, "d": d}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.scipy.special.multigammaln_1"] = multigammaln_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.special.multigammaln_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.special.multigammaln_1'.")


check_valid('jax.scipy.special.multigammaln', generated_inputs['jax.scipy.special.multigammaln_1'], lib="jax", suffix=1)
