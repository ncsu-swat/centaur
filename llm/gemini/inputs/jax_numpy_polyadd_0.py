
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def polyadd_inputs():
    list_of_inputs = []

    # Input 1: 1D float32, same length
    a1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    a2 = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    list_of_inputs.append({"a1": copy.deepcopy(a1), "a2": copy.deepcopy(a2)})

    # Input 2: 1D int32, different lengths
    a1 = np.array([1, 2], dtype=np.int32)
    a2 = np.array([3, 4, 5, 6], dtype=np.int32)
    list_of_inputs.append({"a1": copy.deepcopy(a1), "a2": copy.deepcopy(a2)})

    # Input 3: 1D float64, negative values
    a1 = np.array([-1.5, 2.5, -3.5], dtype=np.float64)
    a2 = np.array([0.5, -0.5], dtype=np.float64)
    list_of_inputs.append({"a1": copy.deepcopy(a1), "a2": copy.deepcopy(a2)})

    # Input 4: 1D int64, one scalar-like
    a1 = np.array([10], dtype=np.int64)
    a2 = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    list_of_inputs.append({"a1": copy.deepcopy(a1), "a2": copy.deepcopy(a2)})

    # Input 5: 2D float32, same shape
    a1 = np.random.randn(3, 4).astype(np.float32)
    a2 = np.random.randn(3, 4).astype(np.float32)
    list_of_inputs.append({"a1": copy.deepcopy(a1), "a2": copy.deepcopy(a2)})

    # Input 6: 2D int32, different first dimension
    a1 = np.array([[1, 2, 3]], dtype=np.int32)
    a2 = np.array([[4, 5, 6], [7, 8, 9]], dtype=np.int32)
    list_of_inputs.append({"a1": copy.deepcopy(a1), "a2": copy.deepcopy(a2)})

    # Input 7: 3D float32, same shape
    a1 = np.random.randn(2, 2, 3).astype(np.float32)
    a2 = np.random.randn(2, 2, 3).astype(np.float32)
    list_of_inputs.append({"a1": copy.deepcopy(a1), "a2": copy.deepcopy(a2)})

    # Input 8: 3D int32, different first dimension
    a1 = np.ones((1, 2, 2), dtype=np.int32)
    a2 = np.ones((3, 2, 2), dtype=np.int32) * 2
    list_of_inputs.append({"a1": copy.deepcopy(a1), "a2": copy.deepcopy(a2)})

    # Input 9: 1D complex64
    a1 = np.array([1+2j, 3+4j], dtype=np.complex64)
    a2 = np.array([5+6j], dtype=np.complex64)
    list_of_inputs.append({"a1": copy.deepcopy(a1), "a2": copy.deepcopy(a2)})

    # Input 10: 1D float16
    a1 = np.array([0.1, 0.2, 0.3], dtype=np.float16)
    a2 = np.array([0.4, 0.5], dtype=np.float16)
    list_of_inputs.append({"a1": copy.deepcopy(a1), "a2": copy.deepcopy(a2)})

    # Input 11: 1D int16, zeros
    a1 = np.zeros(5, dtype=np.int16)
    a2 = np.ones(5, dtype=np.int16)
    list_of_inputs.append({"a1": copy.deepcopy(a1), "a2": copy.deepcopy(a2)})

    return list_of_inputs

generated_inputs["jax.numpy.polyadd"] = polyadd_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.polyadd' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.polyadd'.")


check_valid('jax.numpy.polyadd', generated_inputs['jax.numpy.polyadd'], lib="jax", suffix=0)
