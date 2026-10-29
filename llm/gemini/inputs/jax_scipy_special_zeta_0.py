
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def zeta_inputs():
    list_of_inputs = []

    # Input 1: 0D scalars, float32
    x = np.array(2.0, dtype=np.float32)
    q = np.array(1.0, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "q": q}))

    # Input 2: 0D scalars, float64
    x = np.array(4.0, dtype=np.float64)
    q = np.array(0.5, dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "q": q}))

    # Input 3: 1D arrays, same shape
    x = np.array([2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    q = np.array([1.0, 2.0, 1.5, 0.5], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "q": q}))

    # Input 4: 1D arrays with float64
    x = np.array([1.5, 2.5, 3.5], dtype=np.float64)
    q = np.array([0.25, 0.75, 1.25], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "q": q}))

    # Input 5: 2D arrays, same shape
    x = np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float32)
    q = np.array([[1.0, 1.5], [2.0, 2.5]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "q": q}))

    # Input 6: Broadcasting (3, 1) and (1, 4)
    x = np.array([[2.0], [3.0], [4.0]], dtype=np.float32)
    q = np.array([[0.5, 1.0, 1.5, 2.0]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "q": q}))

    # Input 7: Broadcasting scalar x with 2D q
    x = np.array(3.0, dtype=np.float64)
    q = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "q": q}))

    # Input 8: Broadcasting 1D x with 2D q
    x = np.array([2.0, 3.0, 4.0], dtype=np.float32)
    q = np.ones((2, 3), dtype=np.float32) * 1.5
    list_of_inputs.append(copy.deepcopy({"x": x, "q": q}))

    # Input 9: 3D arrays
    x = np.full((2, 2, 3), 2.5, dtype=np.float64)
    q = np.full((2, 2, 3), 1.2, dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "q": q}))

    # Input 10: High exponent values and large q
    x = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    q = np.array([5.0, 10.0, 15.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "q": q}))

    # Input 11: Non-integer larger dimension inputs
    x = np.linspace(2.0, 5.0, 8, dtype=np.float64).reshape(2, 4)
    q = np.linspace(0.1, 2.0, 8, dtype=np.float64).reshape(2, 4)
    list_of_inputs.append(copy.deepcopy({"x": x, "q": q}))

    return list_of_inputs


generated_inputs["jax.scipy.special.zeta"] = zeta_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.special.zeta' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.special.zeta'.")


check_valid('jax.scipy.special.zeta', generated_inputs['jax.scipy.special.zeta'], lib="jax", suffix=0)
