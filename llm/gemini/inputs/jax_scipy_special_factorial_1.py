
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def factorial_inputs():
    list_of_inputs = []

    # Input 1: 0D scalar int32
    n = np.array(5, dtype=np.int32)
    exact = False
    list_of_inputs.append(copy.deepcopy({"n": n, "exact": exact}))

    # Input 2: 1D array int32
    n = np.array([0, 1, 2, 3, 4, 5], dtype=np.int32)
    exact = False
    list_of_inputs.append(copy.deepcopy({"n": n, "exact": exact}))

    # Input 3: 1D array int64
    n = np.array([5, 10, 15, 20], dtype=np.int64)
    exact = False
    list_of_inputs.append(copy.deepcopy({"n": n, "exact": exact}))

    # Input 4: 1D array float32
    n = np.array([0.5, 1.5, 2.5, 3.5], dtype=np.float32)
    exact = False
    list_of_inputs.append(copy.deepcopy({"n": n, "exact": exact}))

    # Input 5: 2D array float64
    n = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    exact = False
    list_of_inputs.append(copy.deepcopy({"n": n, "exact": exact}))

    # Input 6: 3D array int32
    n = np.arange(8, dtype=np.int32).reshape(2, 2, 2)
    exact = False
    list_of_inputs.append(copy.deepcopy({"n": n, "exact": exact}))

    # Input 7: 1D array with negative non-integers
    n = np.array([-0.5, 0.5, 1.2], dtype=np.float32)
    exact = False
    list_of_inputs.append(copy.deepcopy({"n": n, "exact": exact}))

    # Input 8: 2D array of zeros
    n = np.zeros((3, 3), dtype=np.float32)
    exact = False
    list_of_inputs.append(copy.deepcopy({"n": n, "exact": exact}))

    # Input 9: 4D array float32
    n = np.random.uniform(0.1, 5.0, size=(2, 1, 3, 2)).astype(np.float32)
    exact = False
    list_of_inputs.append(copy.deepcopy({"n": n, "exact": exact}))

    # Input 10: Empty 1D array
    n = np.array([], dtype=np.float32)
    exact = False
    list_of_inputs.append(copy.deepcopy({"n": n, "exact": exact}))

    # Input 11: 1D array float64 with larger values
    n = np.array([10.0, 20.0, 30.0], dtype=np.float64)
    exact = False
    list_of_inputs.append(copy.deepcopy({"n": n, "exact": exact}))

    return list_of_inputs


generated_inputs["jax.scipy.special.factorial_1"] = factorial_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.special.factorial_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.special.factorial_1'.")


check_valid('jax.scipy.special.factorial', generated_inputs['jax.scipy.special.factorial_1'], lib="jax", suffix=1)
