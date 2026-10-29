
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def polygamma_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, order 0 (digamma)
    n = 0
    x = np.array([0.5, 1.0, 1.5, 2.0, 5.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"n": n, "x": x}))

    # Input 2: 2D float32 array, order 1 (trigamma)
    n = 1
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"n": n, "x": x}))

    # Input 3: 1D float64 array, order 2
    n = 2
    x = np.array([0.1, 0.5, 2.5, 10.0], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"n": n, "x": x}))

    # Input 4: 0D scalar tensor, order 0
    n = 0
    x = np.array(2.5, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"n": n, "x": x}))

    # Input 5: 3D float32 array, order 3
    n = 3
    x = np.ones((2, 3, 4), dtype=np.float32) * 1.5
    list_of_inputs.append(copy.deepcopy({"n": n, "x": x}))

    # Input 6: 2D float64 array, order 4
    n = 4
    x = np.linspace(0.5, 5.0, 6, dtype=np.float64).reshape((2, 3))
    list_of_inputs.append(copy.deepcopy({"n": n, "x": x}))

    # Input 7: 1D float32 array with negative non-integer values, order 1
    n = 1
    x = np.array([-0.5, -1.5, -2.5, -3.5], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"n": n, "x": x}))

    # Input 8: 4D float32 array, order 0
    n = 0
    x = np.full((2, 2, 2, 2), 3.0, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"n": n, "x": x}))

    # Input 9: 1D float64 array with larger values, order 5
    n = 5
    x = np.array([10.0, 20.0, 50.0, 100.0], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"n": n, "x": x}))

    # Input 10: 2D float32 array with small positive values, order 2
    n = 2
    x = np.array([[0.01, 0.05], [0.1, 0.2]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"n": n, "x": x}))

    # Input 11: 3D float64 array, order 1
    n = 1
    x = np.arange(1, 9, dtype=np.float64).reshape((2, 2, 2))
    list_of_inputs.append(copy.deepcopy({"n": n, "x": x}))

    return list_of_inputs


generated_inputs["jax.scipy.special.polygamma_1"] = polygamma_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.special.polygamma_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.special.polygamma_1'.")


check_valid('jax.scipy.special.polygamma', generated_inputs['jax.scipy.special.polygamma_1'], lib="jax", suffix=1)
