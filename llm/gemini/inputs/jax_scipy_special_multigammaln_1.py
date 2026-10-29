
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def multigammaln_inputs():
    list_of_inputs = []

    # Input 1: 0D scalar array, d=1
    a = np.array(2.5, dtype=np.float32)
    d = 1
    list_of_inputs.append(copy.deepcopy({"a": a, "d": d}))

    # Input 2: 1D array, d=2 (requires a > 0.5)
    a = np.array([1.0, 1.5, 2.0, 3.5], dtype=np.float32)
    d = 2
    list_of_inputs.append(copy.deepcopy({"a": a, "d": d}))

    # Input 3: 2D array, float64, d=3 (requires a > 1.0)
    a = np.array([[1.5, 2.0], [3.0, 4.5]], dtype=np.float64)
    d = 3
    list_of_inputs.append(copy.deepcopy({"a": a, "d": d}))

    # Input 4: 1D array, d=1 (multigammaln reduces to gammaln)
    a = np.linspace(0.5, 10.0, 10, dtype=np.float64)
    d = 1
    list_of_inputs.append(copy.deepcopy({"a": a, "d": d}))

    # Input 5: 3D array, d=4 (requires a > 1.5)
    a = np.full((2, 3, 2), 5.0, dtype=np.float32)
    d = 4
    list_of_inputs.append(copy.deepcopy({"a": a, "d": d}))

    # Input 6: 2D array, d=5 (requires a > 2.0)
    a = np.linspace(2.5, 8.0, 12, dtype=np.float64).reshape(3, 4)
    d = 5
    list_of_inputs.append(copy.deepcopy({"a": a, "d": d}))

    # Input 7: 1D array, float32, d=2
    a = np.array([0.6, 1.2, 5.0, 15.0], dtype=np.float32)
    d = 2
    list_of_inputs.append(copy.deepcopy({"a": a, "d": d}))

    # Input 8: 4D array, d=1
    a = np.ones((2, 1, 3, 2), dtype=np.float32) * 2.5
    d = 1
    list_of_inputs.append(copy.deepcopy({"a": a, "d": d}))

    # Input 9: 1D array, float64, d=6 (requires a > 2.5)
    a = np.array([3.0, 4.0, 5.5, 7.2], dtype=np.float64)
    d = 6
    list_of_inputs.append(copy.deepcopy({"a": a, "d": d}))

    # Input 10: 2D array with large values, d=3
    a = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
    d = 3
    list_of_inputs.append(copy.deepcopy({"a": a, "d": d}))

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
