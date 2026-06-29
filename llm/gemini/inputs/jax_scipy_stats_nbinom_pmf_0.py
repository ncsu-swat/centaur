
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def nbinom_pmf_inputs():
    list_of_inputs = []

    # Input 1: Scalar inputs (0D arrays)
    k = np.array(5.0, dtype=np.float32)
    n = np.array(10.0, dtype=np.float32)
    p = np.array(0.5, dtype=np.float32)
    loc = np.array(0.0, dtype=np.float32)
    input_dict = {"k": k, "n": n, "p": p, "loc": loc}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D arrays of floats
    k = np.array([0, 1, 2, 3, 4], dtype=np.float32)
    n = np.array([5.0, 5.0, 5.0, 5.0, 5.0], dtype=np.float32)
    p = np.array([0.3, 0.3, 0.3, 0.3, 0.3], dtype=np.float32)
    loc = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    input_dict = {"k": k, "n": n, "p": p, "loc": loc}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D arrays (float64)
    k = np.array([[1, 2], [3, 4]], dtype=np.float64)
    n = np.array([[2.5, 3.5], [4.5, 5.5]], dtype=np.float64)
    p = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64)
    loc = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float64)
    input_dict = {"k": k, "n": n, "p": p, "loc": loc}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Broadcasting inputs of different shapes
    k = np.array([2, 4, 6], dtype=np.float32)
    n = np.array([[5], [10]], dtype=np.float32)
    p = np.array([0.5], dtype=np.float32)
    loc = np.array([0], dtype=np.float32)
    input_dict = {"k": k, "n": n, "p": p, "loc": loc}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: High dimension (3D arrays)
    k = np.ones((2, 2, 2), dtype=np.float32) * 3
    n = np.ones((2, 2, 2), dtype=np.float32) * 5
    p = np.ones((2, 2, 2), dtype=np.float32) * 0.7
    loc = np.zeros((2, 2, 2), dtype=np.float32)
    input_dict = {"k": k, "n": n, "p": p, "loc": loc}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Edge case with p close to 0 and 1
    k = np.array([1, 2], dtype=np.float32)
    n = np.array([2.0, 2.0], dtype=np.float32)
    p = np.array([0.01, 0.99], dtype=np.float32)
    loc = np.array([0.0, 0.0], dtype=np.float32)
    input_dict = {"k": k, "n": n, "p": p, "loc": loc}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Non-zero loc array offset
    k = np.array([10.0, 12.0], dtype=np.float32)
    n = np.array([5.0, 5.0], dtype=np.float32)
    p = np.array([0.5, 0.5], dtype=np.float32)
    loc = np.array([5.0, 5.0], dtype=np.float32)
    input_dict = {"k": k, "n": n, "p": p, "loc": loc}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Int32 types for k and loc with float32 parameters
    k = np.array([2, 5, 8], dtype=np.int32)
    n = np.array([3.0, 4.0, 5.0], dtype=np.float32)
    p = np.array([0.2, 0.5, 0.8], dtype=np.float32)
    loc = np.array([1, 2, 3], dtype=np.int32)
    input_dict = {"k": k, "n": n, "p": p, "loc": loc}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large 1D array size of 100 elements
    k = np.arange(100, dtype=np.float32)
    n = np.full((100,), 10.0, dtype=np.float32)
    p = np.full((100,), 0.25, dtype=np.float32)
    loc = np.zeros((100,), dtype=np.float32)
    input_dict = {"k": k, "n": n, "p": p, "loc": loc}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: High value parameters
    k = np.array([1000], dtype=np.float32)
    n = np.array([500.0], dtype=np.float32)
    p = np.array([0.6], dtype=np.float32)
    loc = np.array([100.0], dtype=np.float32)
    input_dict = {"k": k, "n": n, "p": p, "loc": loc}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.scipy.stats.nbinom.pmf"] = nbinom_pmf_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.stats.nbinom.pmf' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.stats.nbinom.pmf'.")


check_valid('jax.scipy.stats.nbinom.pmf', generated_inputs['jax.scipy.stats.nbinom.pmf'], lib="jax", suffix=0)
