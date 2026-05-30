
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def kron_inputs():
    list_of_inputs = []

    # 1. 1D arrays (int32)
    a = np.array([1, 2, -3], dtype=np.int32)
    b = np.array([-1, 0, 4], dtype=np.int32)
    list_of_inputs.append({"a": a, "b": b})

    # 2. 2D arrays (float32)
    a = np.random.randn(2, 3).astype(np.float32)
    b = np.random.randn(3, 2).astype(np.float32)
    list_of_inputs.append({"a": a, "b": b})

    # 3. 2D arrays (float64)
    a = np.random.uniform(-5.0, 5.0, (4, 4)).astype(np.float64)
    b = np.random.uniform(-5.0, 5.0, (2, 2)).astype(np.float64)
    list_of_inputs.append({"a": a, "b": b})

    # 4. 3D arrays (float32)
    a = np.random.randn(2, 2, 2).astype(np.float32)
    b = np.random.randn(2, 1, 3).astype(np.float32)
    list_of_inputs.append({"a": a, "b": b})

    # 5. Complex64 inputs
    a = (np.random.randn(2, 2) + 1j * np.random.randn(2, 2)).astype(np.complex64)
    b = (np.random.randn(2, 2) + 1j * np.random.randn(2, 2)).astype(np.complex64)
    list_of_inputs.append({"a": a, "b": b})

    # 6. uint8 arrays
    a = np.array([[1, 0], [0, 1]], dtype=np.uint8)
    b = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    list_of_inputs.append({"a": a, "b": b})

    # 7. 0D arrays (scalars)
    a = np.array(5, dtype=np.int32)
    b = np.array(-3, dtype=np.int32)
    list_of_inputs.append({"a": a, "b": b})

    # 8. 2D arrays (int16)
    a = np.random.randint(-10, 10, (3, 3)).astype(np.int16)
    b = np.random.randint(-10, 10, (2, 2)).astype(np.int16)
    list_of_inputs.append({"a": a, "b": b})

    # 9. Mixed dimensions: 1D and 2D
    a = np.array([1, -2, 3], dtype=np.float32)
    b = np.random.randn(2, 2).astype(np.float32)
    list_of_inputs.append({"a": a, "b": b})

    # 10. Mixed dimensions: 3D and 1D
    a = np.random.randn(2, 3, 2).astype(np.float32)
    b = np.array([0.5, -1.5], dtype=np.float32)
    list_of_inputs.append({"a": a, "b": b})

    return list_of_inputs

generated_inputs["jax.numpy.kron"] = kron_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.kron' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.kron'.")


check_valid('jax.numpy.kron', generated_inputs['jax.numpy.kron'], lib="jax", suffix=0)
