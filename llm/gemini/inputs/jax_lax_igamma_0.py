
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def igamma_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 arrays
    a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    x = np.array([0.5, 1.5, 2.5, 3.5], dtype=np.float32)
    list_of_inputs.append({"a": copy.deepcopy(a), "x": copy.deepcopy(x)})

    # Input 2: 0D scalar float32 arrays
    a = np.array(1.5, dtype=np.float32)
    x = np.array(2.5, dtype=np.float32)
    list_of_inputs.append({"a": copy.deepcopy(a), "x": copy.deepcopy(x)})

    # Input 3: 2D float32 arrays of same shape
    a = np.full((3, 3), 2.0, dtype=np.float32)
    x = np.array([[0.1, 0.5, 1.0], [1.5, 2.0, 2.5], [3.0, 3.5, 4.0]], dtype=np.float32)
    list_of_inputs.append({"a": copy.deepcopy(a), "x": copy.deepcopy(x)})

    # Input 4: 2D float64 arrays
    a = np.array([[0.5, 1.0], [1.5, 2.0]], dtype=np.float64)
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    list_of_inputs.append({"a": copy.deepcopy(a), "x": copy.deepcopy(x)})

    # Input 5: 2D float32 arrays
    a = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], dtype=np.float32)
    x = np.array([[0.5, 1.0, 1.5], [2.0, 2.5, 3.0]], dtype=np.float32)
    list_of_inputs.append({"a": copy.deepcopy(a), "x": copy.deepcopy(x)})

    # Input 6: Boundary case with x = 0
    a = np.array([0.5, 1.0, 2.0, 5.0], dtype=np.float32)
    x = np.zeros(4, dtype=np.float32)
    list_of_inputs.append({"a": copy.deepcopy(a), "x": copy.deepcopy(x)})

    # Input 7: 3D float32 arrays
    a = np.ones((2, 3, 4), dtype=np.float32) * 3.0
    x = np.linspace(0.1, 5.0, 24, dtype=np.float32).reshape(2, 3, 4)
    list_of_inputs.append({"a": copy.deepcopy(a), "x": copy.deepcopy(x)})

    # Input 8: 2D float32 with small values
    a = np.full((3, 4), 0.5, dtype=np.float32)
    x = np.linspace(0.01, 1.0, 12, dtype=np.float32).reshape(3, 4)
    list_of_inputs.append({"a": copy.deepcopy(a), "x": copy.deepcopy(x)})

    # Input 9: Large values float64
    a = np.array([10.0, 50.0, 100.0], dtype=np.float64)
    x = np.array([15.0, 45.0, 110.0], dtype=np.float64)
    list_of_inputs.append({"a": copy.deepcopy(a), "x": copy.deepcopy(x)})

    # Input 10: 4D float32 arrays
    a = np.random.uniform(0.5, 5.0, size=(2, 2, 2, 2)).astype(np.float32)
    x = np.random.uniform(0.1, 10.0, size=(2, 2, 2, 2)).astype(np.float32)
    list_of_inputs.append({"a": copy.deepcopy(a), "x": copy.deepcopy(x)})

    return list_of_inputs


generated_inputs["jax.lax.igamma"] = igamma_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.igamma' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.igamma'.")


check_valid('jax.lax.igamma', generated_inputs['jax.lax.igamma'], lib="jax", suffix=0)
