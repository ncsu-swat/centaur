
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def polygamma_inputs():
    list_of_inputs = []

    # Input 1: Scalar-like tensors, shape ()
    m = np.array(1.0, dtype=np.float32)
    x = np.array(1.5, dtype=np.float32)
    list_of_inputs.append({"m": m, "x": x})

    # Input 2: 1-D arrays, shape (3,)
    m = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    list_of_inputs.append({"m": m, "x": x})

    # Input 3: 2-D arrays, shape (2, 2)
    m = np.array([[1.0, 1.0], [2.0, 2.0]], dtype=np.float32)
    x = np.array([[0.5, 1.5], [2.5, 3.5]], dtype=np.float32)
    list_of_inputs.append({"m": m, "x": x})

    # Input 4: 2-D arrays, shape (2, 2)
    m = np.array([[2.0, 2.0], [2.0, 2.0]], dtype=np.float32)
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    list_of_inputs.append({"m": m, "x": x})

    # Input 5: float64 precision, shape (2,)
    m = np.array([0.0, 1.0], dtype=np.float64)
    x = np.array([0.1, 0.2], dtype=np.float64)
    list_of_inputs.append({"m": m, "x": x})

    # Input 6: Negative non-integer values for x, shape (3,)
    m = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    x = np.array([-0.5, -1.5, -2.5], dtype=np.float32)
    list_of_inputs.append({"m": m, "x": x})

    # Input 7: 3-D arrays, shape (2, 3, 4)
    m = np.ones((2, 3, 4), dtype=np.float32)
    x = np.random.uniform(0.5, 5.0, (2, 3, 4)).astype(np.float32)
    list_of_inputs.append({"m": m, "x": x})

    # Input 8: m=0.0 with larger x values, shape (3,)
    m = np.zeros((3,), dtype=np.float32)
    x = np.array([1.0, 10.0, 100.0], dtype=np.float32)
    list_of_inputs.append({"m": m, "x": x})

    # Input 9: Varying m orders, shape (5,)
    m = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    x = np.array([1.5, 2.5, 3.5, 4.5, 5.5], dtype=np.float32)
    list_of_inputs.append({"m": m, "x": x})

    # Input 10: 3-D array with float64, shape (2, 2, 2)
    m = np.ones((2, 2, 2), dtype=np.float64)
    x = np.random.uniform(1.0, 2.0, (2, 2, 2)).astype(np.float64)
    list_of_inputs.append({"m": m, "x": x})

    return list_of_inputs

generated_inputs["jax.lax.polygamma_2"] = polygamma_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.polygamma_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.polygamma_2'.")


check_valid('jax.lax.polygamma', generated_inputs['jax.lax.polygamma_2'], lib="jax", suffix=2)
