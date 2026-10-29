
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def correlate_inputs():
    list_of_inputs = []

    # Input 1: Basic float32 1D arrays with mode 'valid'
    a = np.array([1.0, 2.0, 3.0, 2.0, 1.0], dtype=np.float32)
    v = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    input_dict = {
        "a": a,
        "v": v,
        "mode": "valid",
        "precision": "default",
        "preferred_element_type": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Float32 with mode 'full' and high precision
    a = np.array([-1.5, 0.5, 2.0, -3.0], dtype=np.float32)
    v = np.array([1.0, -1.0, 2.0], dtype=np.float32)
    input_dict = {
        "a": a,
        "v": v,
        "mode": "full",
        "precision": "high",
        "preferred_element_type": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Float64 with mode 'same' and highest precision
    a = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float64)
    v = np.array([-0.5, 0.5], dtype=np.float64)
    input_dict = {
        "a": a,
        "v": v,
        "mode": "same",
        "precision": "highest",
        "preferred_element_type": np.dtype(np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Int32 inputs accumulated to float32
    a = np.array([1, -2, 3, -4, 5], dtype=np.int32)
    v = np.array([2, 0, -1], dtype=np.int32)
    input_dict = {
        "a": a,
        "v": v,
        "mode": "valid",
        "precision": "default",
        "preferred_element_type": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Complex64 inputs with mode 'full'
    a = np.array([1.0 + 2.0j, -3.0 + 0.5j, 2.0 - 1.0j], dtype=np.complex64)
    v = np.array([0.5 - 1.0j, 2.0 + 0.0j], dtype=np.complex64)
    input_dict = {
        "a": a,
        "v": v,
        "mode": "full",
        "precision": "high",
        "preferred_element_type": np.dtype(np.complex64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Complex128 inputs with mode 'same'
    a = np.array(
        [1.0 - 1.0j, 2.0 + 3.0j, -0.5 + 0.5j, 4.0 - 2.0j], dtype=np.complex128
    )
    v = np.array([1.0 + 0.0j, 0.0 - 1.0j], dtype=np.complex128)
    input_dict = {
        "a": a,
        "v": v,
        "mode": "same",
        "precision": "highest",
        "preferred_element_type": np.dtype(np.complex128),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Single element vector v
    a = np.array([10.0, -5.0, 2.5, 0.0, -1.0], dtype=np.float32)
    v = np.array([2.0], dtype=np.float32)
    input_dict = {
        "a": a,
        "v": v,
        "mode": "valid",
        "precision": "default",
        "preferred_element_type": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Equal length 1D arrays with mode 'full'
    a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    v = np.array([4.0, 3.0, 2.0, 1.0], dtype=np.float64)
    input_dict = {
        "a": a,
        "v": v,
        "mode": "full",
        "precision": "default",
        "preferred_element_type": np.dtype(np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large 1D arrays with mode 'valid'
    a = np.linspace(-10.0, 10.0, 100, dtype=np.float32)
    v = np.ones(10, dtype=np.float32)
    input_dict = {
        "a": a,
        "v": v,
        "mode": "valid",
        "precision": "high",
        "preferred_element_type": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Larger v than a with mode 'full'
    a = np.array([1.0, -1.0], dtype=np.float32)
    v = np.array([0.5, 1.5, -2.0, 3.0, 0.0], dtype=np.float32)
    input_dict = {
        "a": a,
        "v": v,
        "mode": "full",
        "precision": "highest",
        "preferred_element_type": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.correlate_1"] = correlate_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.correlate_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.correlate_1'.")


check_valid('jax.numpy.correlate', generated_inputs['jax.numpy.correlate_1'], lib="jax", suffix=1)
