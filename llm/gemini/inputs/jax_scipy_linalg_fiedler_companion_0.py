
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import importlib
import numpy as np
import scipy.linalg

_mod = importlib.import_module("jax.scipy.linalg")
if not hasattr(_mod, "fiedler_companion"):
    setattr(_mod, "fiedler_companion", scipy.linalg.fiedler_companion)


def fiedler_companion_inputs():
    list_of_inputs = []

    # Input 1: Float32 1D array of length 2
    a = np.array([1.0, -2.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"a": a}))

    # Input 2: Float32 1D array of length 3
    a = np.array([2.0, 3.0, -5.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"a": a}))

    # Input 3: Float64 1D array of length 4
    a = np.array([1.5, -0.5, 2.0, -1.0], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"a": a}))

    # Input 4: Float64 1D array of length 5
    a = np.array([-3.0, 1.0, 0.0, -2.0, 4.0], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"a": a}))

    # Input 5: Float32 1D array with positive and negative floats
    a = np.array([10.0, -20.0, 30.0, -40.0, 50.0, -60.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"a": a}))

    # Input 6: Float64 1D array of length 7
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"a": a}))

    # Input 7: Complex64 1D array of length 3
    a = np.array([1.0 + 1.0j, 2.0 - 1.0j, -3.0 + 2.0j], dtype=np.complex64)
    list_of_inputs.append(copy.deepcopy({"a": a}))

    # Input 8: Complex128 1D array of length 4
    a = np.array(
        [2.0 - 3.0j, -1.0 + 0.5j, 0.5 - 2.0j, 1.0 + 0.0j], dtype=np.complex128
    )
    list_of_inputs.append(copy.deepcopy({"a": a}))

    # Input 9: Float64 1D array of length 8
    a = np.array(
        [0.5, -1.5, 2.5, -3.5, 4.5, -5.5, 6.5, -7.5], dtype=np.float64
    )
    list_of_inputs.append(copy.deepcopy({"a": a}))

    # Input 10: Float32 1D array of length 10
    a = np.linspace(1.0, 5.0, 10, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"a": a}))

    return list_of_inputs


generated_inputs["jax.scipy.linalg.fiedler_companion"] = (
    fiedler_companion_inputs()
)

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.linalg.fiedler_companion' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.linalg.fiedler_companion'.")


check_valid('jax.scipy.linalg.fiedler_companion', generated_inputs['jax.scipy.linalg.fiedler_companion'], lib="jax", suffix=0)
