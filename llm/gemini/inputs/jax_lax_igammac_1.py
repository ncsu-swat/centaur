
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def igammac_inputs():
    list_of_inputs = []

    # Input 1: 1D arrays, float32, standard positive values
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    x = np.array([0.5, 1.5, 2.5], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"a": a, "x": x}))

    # Input 2: 2D arrays, float64
    a = np.array([[1.0, 2.5], [3.0, 4.5]], dtype=np.float64)
    x = np.array([[0.5, 1.0], [2.0, 3.5]], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"a": a, "x": x}))

    # Input 3: Scalar-like 0D arrays, float32
    a = np.array(2.0, dtype=np.float32)
    x = np.array(1.0, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"a": a, "x": x}))

    # Input 4: Scalar-like 0D arrays, float64
    a = np.array(5.5, dtype=np.float64)
    x = np.array(3.2, dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"a": a, "x": x}))

    # Input 5: Boundary value x=0
    a = np.array([0.5, 1.0, 5.0], dtype=np.float32)
    x = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"a": a, "x": x}))

    # Input 6: 3D arrays, float32
    a = np.full((2, 3, 4), 2.5, dtype=np.float32)
    x = np.full((2, 3, 4), 1.2, dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"a": a, "x": x}))

    # Input 7: 3D arrays, float64
    a = np.full((3, 2, 2), 4.0, dtype=np.float64)
    x = np.full((3, 2, 2), 2.0, dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"a": a, "x": x}))

    # Input 8: Large parameter values
    a = np.array([50.0, 100.0, 200.0], dtype=np.float64)
    x = np.array([45.0, 105.0, 190.0], dtype=np.float64)
    list_of_inputs.append(copy.deepcopy({"a": a, "x": x}))

    # Input 9: Small positive parameter values
    a = np.array([0.01, 0.05, 0.1], dtype=np.float32)
    x = np.array([0.001, 0.01, 0.05], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"a": a, "x": x}))

    # Input 10: 4D tensors, float32
    a = np.ones((2, 3, 4, 5), dtype=np.float32) * 3.0
    x = np.ones((2, 3, 4, 5), dtype=np.float32) * 2.0
    list_of_inputs.append(copy.deepcopy({"a": a, "x": x}))

    return list_of_inputs


generated_inputs["jax.lax.igammac_1"] = igammac_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.igammac_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.igammac_1'.")


check_valid('jax.lax.igammac', generated_inputs['jax.lax.igammac_1'], lib="jax", suffix=1)
