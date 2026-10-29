
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def zeta_inputs():
    list_of_inputs = []

    # Input 1: 0D scalar float32
    x = np.array(2.0, dtype=np.float32)
    q = np.array(1.0, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "q": q}))

    # Input 2: 0D scalar float64
    x = np.array(3.5, dtype=np.float64)
    q = np.array(0.5, dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "q": q}))

    # Input 3: 1D array float32
    x = np.array([2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    q = np.array([1.0, 2.0, 0.5, 1.5], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "q": q}))

    # Input 4: 1D array float64
    x = np.array([1.5, 2.5, 3.5, 4.5], dtype=np.float64)
    q = np.array([0.1, 0.5, 2.0, 1.0], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "q": q}))

    # Input 5: 2D array float32
    x = np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float32)
    q = np.array([[1.0, 1.0], [2.0, 2.0]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "q": q}))

    # Input 6: 2D array float64
    x = np.array([[2.5, 3.5, 4.5], [5.5, 6.5, 7.5]], dtype=np.float64)
    q = np.array([[0.5, 1.0, 1.5], [2.0, 2.5, 3.0]], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"x": x, "q": q}))

    # Input 7: 3D array float32
    x = np.full((2, 3, 4), 3.0, dtype=np.float32)
    q = np.full((2, 3, 4), 1.25, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "q": q}))

    # Input 8: 3D array float64
    x = np.linspace(2.0, 8.0, 24, dtype=np.float64).reshape((2, 3, 4))
    q = np.linspace(0.5, 3.0, 24, dtype=np.float64).reshape((2, 3, 4))
    list_of_inputs.append(copy.deepcopy({"x": x, "q": q}))

    # Input 9: 4D array float32
    x = np.linspace(2.0, 10.0, 16, dtype=np.float32).reshape((2, 2, 2, 2))
    q = np.linspace(0.1, 5.0, 16, dtype=np.float32).reshape((2, 2, 2, 2))
    list_of_inputs.append(copy.deepcopy({"x": x, "q": q}))

    # Input 10: 1D array float32 with larger values
    x = np.array([5.0, 6.0, 7.0, 8.0, 9.0], dtype=np.float32)
    q = np.array([2.0, 2.5, 3.0, 3.5, 4.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"x": x, "q": q}))

    return list_of_inputs


generated_inputs["jax.lax.zeta_1"] = zeta_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.zeta_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.zeta_1'.")


check_valid('jax.lax.zeta', generated_inputs['jax.lax.zeta_1'], lib="jax", suffix=1)
