
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def multigammaln_inputs():
    list_of_inputs = []

    # Input 1: 0D float32 scalar tensor, d=1
    a = np.array(2.0, dtype=np.float32)
    d = np.array(1, dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"a": a, "d": d}))

    # Input 2: 1D float32 tensor, d=2
    a = np.array([1.5, 2.0, 3.5], dtype=np.float32)
    d = np.array(2, dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"a": a, "d": d}))

    # Input 3: 2D float64 tensor, d=3
    a = np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float64)
    d = np.array(3, dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"a": a, "d": d}))

    # Input 4: 1D float64 tensor, d=1
    a = np.array([0.5, 1.0, 5.0, 10.0], dtype=np.float64)
    d = np.array(1, dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"a": a, "d": d}))

    # Input 5: 3D float32 tensor, d=2
    a = np.array([[[2.0, 2.5], [3.0, 3.5]]], dtype=np.float32)
    d = np.array(2, dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"a": a, "d": d}))

    # Input 6: 0D float64 scalar tensor, d=4
    a = np.array(6.0, dtype=np.float64)
    d = np.array(4, dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"a": a, "d": d}))

    # Input 7: 2D float32 tensor, d=4
    a = np.array([[3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    d = np.array(4, dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"a": a, "d": d}))

    # Input 8: 3D float32 tensor, d=3
    a = np.full((2, 3, 4), 4.5, dtype=np.float32)
    d = np.array(3, dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"a": a, "d": d}))

    # Input 9: 1D float32 tensor with larger dimension d=5
    a = np.array([3.0, 4.0, 7.0, 12.0], dtype=np.float32)
    d = np.array(5, dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"a": a, "d": d}))

    # Input 10: 4D float64 tensor, d=2
    a = np.ones((2, 1, 3, 2), dtype=np.float64) * 3.5
    d = np.array(2, dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"a": a, "d": d}))

    # Input 11: 1D float64 tensor with large values, d=6
    a = np.linspace(5.0, 20.0, 8, dtype=np.float64)
    d = np.array(6, dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"a": a, "d": d}))

    return list_of_inputs


generated_inputs["jax.scipy.special.multigammaln_2"] = multigammaln_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.special.multigammaln_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.special.multigammaln_2'.")


check_valid('jax.scipy.special.multigammaln', generated_inputs['jax.scipy.special.multigammaln_2'], lib="jax", suffix=2)
