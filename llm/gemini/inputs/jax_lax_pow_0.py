
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def pow_inputs():
    list_of_inputs = []

    # Input 1: Simple 1D float32 arrays, identical shapes
    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    y = np.array([2.0, 3.0, 0.5, -1.0], dtype=np.float32)
    list_of_inputs.append({"x": x, "y": y})

    # Input 2: 2D float64 arrays, identical shapes
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    y = np.array([[2.0, -1.0], [0.0, 0.5]], dtype=np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 3: Negative base with integer exponents (int32), identical shapes
    x = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    y = np.array([2, 3, 4], dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 4: Broadcasting with same ndim (ndim=2), same float32 dtype
    x = np.array([[1.0], [2.0]], dtype=np.float32)
    y = np.array([[2.0, 3.0], [1.0, 0.5]], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 5: Broadcasting with same ndim (ndim=2), same float32 dtype
    x = np.array([[2.0, 3.0]], dtype=np.float32)
    y = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 6: Complex64 base and Complex64 exponent, identical shapes
    x = np.array([1.0 + 2.0j, -2.0 + 1.0j], dtype=np.complex64)
    y = np.array([2.0 + 0.0j, 3.0 + 0.0j], dtype=np.complex64)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 7: Complex128 base and Complex128 exponent, identical shapes
    x = np.array([1.0 - 1.0j, 2.0 + 3.0j], dtype=np.complex128)
    y = np.array([0.5 + 0.5j, 1.0 - 2.0j], dtype=np.complex128)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 8: Scalar (0D) float32 base and 2D float32 exponent
    x = np.array(5.0, dtype=np.float32)
    y = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 9: High-dimensional 3D float32 arrays, identical shapes
    x = np.random.uniform(0.1, 5.0, size=(2, 3, 4)).astype(np.float32)
    y = np.random.uniform(-1.0, 2.0, size=(2, 3, 4)).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    # Input 10: Float64 2D base and 0D int64 exponent (scalar)
    x = np.array([[10.0], [100.0], [1000.0]], dtype=np.float64)
    y = np.array(-1, dtype=np.int64)
    list_of_inputs.append({"x": copy.deepcopy(x), "y": copy.deepcopy(y)})

    return list_of_inputs

generated_inputs["jax.lax.pow"] = pow_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.pow' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.pow'.")


check_valid('jax.lax.pow', generated_inputs['jax.lax.pow'], lib="jax", suffix=0)
