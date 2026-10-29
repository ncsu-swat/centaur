
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def boxcox_inputs():
    list_of_inputs = []

    # Input 1: 1D array, lmbda = 0
    x = np.array([0.1, 0.5, 1.0, 5.0], dtype=np.float32)
    lmbda = 0
    list_of_inputs.append(copy.deepcopy({"x": x, "lmbda": lmbda}))

    # Input 2: 1D array, lmbda = 1
    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    lmbda = 1
    list_of_inputs.append(copy.deepcopy({"x": x, "lmbda": lmbda}))

    # Input 3: 2D array, lmbda = 2
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    lmbda = 2
    list_of_inputs.append(copy.deepcopy({"x": x, "lmbda": lmbda}))

    # Input 4: 2D array, lmbda = -1
    x = np.array([[0.5, 1.0], [2.0, 4.0]], dtype=np.float32)
    lmbda = -1
    list_of_inputs.append(copy.deepcopy({"x": x, "lmbda": lmbda}))

    # Input 5: 3D array, lmbda = 3
    x = np.ones((2, 2, 3), dtype=np.float32) * 2.0
    lmbda = 3
    list_of_inputs.append(copy.deepcopy({"x": x, "lmbda": lmbda}))

    # Input 6: 0D array (scalar), lmbda = 0
    x = np.array(2.5, dtype=np.float32)
    lmbda = 0
    list_of_inputs.append(copy.deepcopy({"x": x, "lmbda": lmbda}))

    # Input 7: 4D array, lmbda = -2
    x = np.ones((2, 2, 2, 2), dtype=np.float64) * 1.2
    lmbda = -2
    list_of_inputs.append(copy.deepcopy({"x": x, "lmbda": lmbda}))

    # Input 8: 1D array with small values, lmbda = 2
    x = np.array([1e-3, 1e-2, 1e-1], dtype=np.float32)
    lmbda = 2
    list_of_inputs.append(copy.deepcopy({"x": x, "lmbda": lmbda}))

    # Input 9: Large 2D array, lmbda = 1
    x = np.ones((10, 10), dtype=np.float32) * 3.0
    lmbda = 1
    list_of_inputs.append(copy.deepcopy({"x": x, "lmbda": lmbda}))

    # Input 10: 1D array with float64, lmbda = 5
    x = np.linspace(1.0, 4.0, num=10, dtype=np.float64)
    lmbda = 5
    list_of_inputs.append(copy.deepcopy({"x": x, "lmbda": lmbda}))

    return list_of_inputs


generated_inputs["jax.scipy.special.boxcox_3"] = boxcox_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.special.boxcox_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.special.boxcox_3'.")


check_valid('jax.scipy.special.boxcox', generated_inputs['jax.scipy.special.boxcox_3'], lib="jax", suffix=3)
