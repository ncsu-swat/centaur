
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def convolve_inputs():
    list_of_inputs = []

    # Input 1: Basic float32 arrays, full mode, default precision
    a = np.array([1.0, 2.0, 3.0, 2.0, 1.0], dtype=np.float32)
    v = np.array([4.0, 1.0, 2.0], dtype=np.float32)
    list_of_inputs.append(
        {
            "a": a,
            "v": v,
            "mode": "full",
            "precision": "default",
            "preferred_element_type": np.float32,
        }
    )

    # Input 2: same mode, high precision, float32
    a = np.array([1.5, -2.0, 3.5, 0.0, -1.2], dtype=np.float32)
    v = np.array([-0.5, 1.0, 2.5], dtype=np.float32)
    list_of_inputs.append(
        {
            "a": a,
            "v": v,
            "mode": "same",
            "precision": "high",
            "preferred_element_type": np.float32,
        }
    )

    # Input 3: valid mode, highest precision
    a = np.arange(10, dtype=np.float32)
    v = np.array([0.2, 0.5, 0.3], dtype=np.float32)
    list_of_inputs.append(
        {
            "a": a,
            "v": v,
            "mode": "valid",
            "precision": "highest",
            "preferred_element_type": np.float32,
        }
    )

    # Input 4: Complex numbers, full mode
    a = np.array([1.0 + 2.0j, -1.0 + 0.5j, 3.0 - 1.0j], dtype=np.complex64)
    v = np.array([0.5 - 0.5j, 2.0 + 1.0j], dtype=np.complex64)
    list_of_inputs.append(
        {
            "a": a,
            "v": v,
            "mode": "full",
            "precision": "default",
            "preferred_element_type": np.complex64,
        }
    )

    # Input 5: Complex64 arrays, same mode
    a = np.array([1.0j, 2.0 - 1.0j, -3.0j, 4.0], dtype=np.complex64)
    v = np.array([0.5 + 0.5j, -0.5 - 0.5j], dtype=np.complex64)
    list_of_inputs.append(
        {
            "a": a,
            "v": v,
            "mode": "same",
            "precision": "high",
            "preferred_element_type": np.complex64,
        }
    )

    # Input 6: Integer inputs with preferred_element_type float32
    a = np.array([1, 4, -2, 7, 0, 3], dtype=np.int32)
    v = np.array([1, -1, 2], dtype=np.int32)
    list_of_inputs.append(
        {
            "a": a,
            "v": v,
            "mode": "full",
            "precision": "default",
            "preferred_element_type": np.float32,
        }
    )

    # Input 7: Single element v
    a = np.array([-5.0, 2.0, 10.0, -1.0, 0.0], dtype=np.float32)
    v = np.array([3.0], dtype=np.float32)
    list_of_inputs.append(
        {
            "a": a,
            "v": v,
            "mode": "valid",
            "precision": "default",
            "preferred_element_type": np.float32,
        }
    )

    # Input 8: Larger 1D arrays
    a = np.linspace(-1.0, 1.0, 50, dtype=np.float32)
    v = np.ones(10, dtype=np.float32) / 10.0
    list_of_inputs.append(
        {
            "a": a,
            "v": v,
            "mode": "same",
            "precision": "highest",
            "preferred_element_type": np.float32,
        }
    )

    # Input 9: Equal length arrays, valid mode
    a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    v = np.array([4.0, 3.0, 2.0, 1.0], dtype=np.float32)
    list_of_inputs.append(
        {
            "a": a,
            "v": v,
            "mode": "valid",
            "precision": "high",
            "preferred_element_type": np.float32,
        }
    )

    # Input 10: v longer than a, full mode
    a = np.array([0.5, -0.5], dtype=np.float32)
    v = np.array([1.0, -2.0, 3.0, -4.0, 5.0], dtype=np.float32)
    list_of_inputs.append(
        {
            "a": a,
            "v": v,
            "mode": "full",
            "precision": "default",
            "preferred_element_type": np.float32,
        }
    )

    # Input 11: Negative and zero elements, same mode
    a = np.array([-10.0, 0.0, 10.0, -20.0, 0.0, 30.0], dtype=np.float32)
    v = np.array([0.0, -1.0, 0.0], dtype=np.float32)
    list_of_inputs.append(
        {
            "a": a,
            "v": v,
            "mode": "same",
            "precision": "default",
            "preferred_element_type": np.float32,
        }
    )

    return [copy.deepcopy(inp) for inp in list_of_inputs]


generated_inputs["jax.numpy.convolve_1"] = convolve_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.convolve_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.convolve_1'.")


check_valid('jax.numpy.convolve', generated_inputs['jax.numpy.convolve_1'], lib="jax", suffix=1)
