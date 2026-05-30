
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_nn_logsumexp_inputs():
    list_of_inputs = []

    # Input 1: Simple 1D float32 array
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    axis = [0]
    b = np.ones_like(a)
    keepdims = False
    return_sign = False
    where = np.array([True, True, True, True, True])
    list_of_inputs.append({
        "a": a,
        "axis": axis,
        "b": b,
        "keepdims": keepdims,
        "return_sign": return_sign,
        "where": where
    })

    # Input 2: 2D array, reducing along axis 1, with keepdims=True
    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    axis = [1]
    b = np.array([[0.5, 0.5, 1.0], [1.0, 2.0, 1.0]], dtype=np.float32)
    keepdims = True
    return_sign = False
    where = np.array([[True, True, False], [True, True, True]])
    list_of_inputs.append({
        "a": a,
        "axis": axis,
        "b": b,
        "keepdims": keepdims,
        "return_sign": return_sign,
        "where": where
    })

    # Input 3: Negative inputs, return_sign=True, negative scaling factors in b
    a = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    axis = [0]
    b = np.array([-1.0, 2.0, -3.0], dtype=np.float32)
    keepdims = False
    return_sign = True
    where = np.array([True, True, True])
    list_of_inputs.append({
        "a": a,
        "axis": axis,
        "b": b,
        "keepdims": keepdims,
        "return_sign": return_sign,
        "where": where
    })

    # Input 4: float64 array, 3D tensor, multiple axes reduction
    a = np.random.randn(2, 3, 4).astype(np.float64)
    axis = [0, 2]
    b = np.abs(np.random.randn(2, 3, 4)).astype(np.float64)
    keepdims = False
    return_sign = False
    where = np.ones((2, 3, 4), dtype=bool)
    list_of_inputs.append({
        "a": a,
        "axis": axis,
        "b": b,
        "keepdims": keepdims,
        "return_sign": return_sign,
        "where": where
    })

    # Input 5: Broadcasted b and where tensors
    a = np.random.randn(3, 4, 5).astype(np.float32)
    axis = [1]
    b = np.ones((1, 4, 1), dtype=np.float32)
    keepdims = True
    return_sign = False
    where = np.ones((3, 1, 5), dtype=bool)
    list_of_inputs.append({
        "a": a,
        "axis": axis,
        "b": b,
        "keepdims": keepdims,
        "return_sign": return_sign,
        "where": where
    })

    # Input 6: Large values (stability test)
    a = np.array([1000.0, 1001.0, 1002.0], dtype=np.float32)
    axis = [0]
    b = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    keepdims = False
    return_sign = False
    where = np.array([True, True, True])
    list_of_inputs.append({
        "a": a,
        "axis": axis,
        "b": b,
        "keepdims": keepdims,
        "return_sign": return_sign,
        "where": where
    })

    # Input 7: Small values (stability test) with float64
    a = np.array([-1000.0, -1001.0, -1002.0], dtype=np.float64)
    axis = [0]
    b = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    keepdims = True
    return_sign = False
    where = np.array([True, True, True])
    list_of_inputs.append({
        "a": a,
        "axis": axis,
        "b": b,
        "keepdims": keepdims,
        "return_sign": return_sign,
        "where": where
    })

    # Input 8: High dimensional 4D array
    a = np.random.randn(2, 2, 2, 2).astype(np.float32)
    axis = [1, 2, 3]
    b = np.ones((2, 2, 2, 2), dtype=np.float32)
    keepdims = False
    return_sign = False
    where = np.random.choice([True, False], size=(2, 2, 2, 2))
    list_of_inputs.append({
        "a": a,
        "axis": axis,
        "b": b,
        "keepdims": keepdims,
        "return_sign": return_sign,
        "where": where
    })

    # Input 9: Negative scaling factors and return_sign=True, 2D array
    a = np.random.randn(4, 4).astype(np.float32)
    axis = [0]
    b = np.random.randn(4, 4).astype(np.float32)
    keepdims = False
    return_sign = True
    where = np.ones((4, 4), dtype=bool)
    list_of_inputs.append({
        "a": a,
        "axis": axis,
        "b": b,
        "keepdims": keepdims,
        "return_sign": return_sign,
        "where": where
    })

    # Input 10: 1D array, partially masked reduction (where contains Falses)
    a = np.array([5.0, 10.0, 15.0, 20.0], dtype=np.float32)
    axis = [0]
    b = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    keepdims = False
    return_sign = False
    where = np.array([True, False, True, False])
    list_of_inputs.append({
        "a": a,
        "axis": axis,
        "b": b,
        "keepdims": keepdims,
        "return_sign": return_sign,
        "where": where
    })

    return list_of_inputs

generated_inputs["jax.nn.logsumexp_3"] = jax_nn_logsumexp_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.nn.logsumexp_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.nn.logsumexp_3'.")


check_valid('jax.nn.logsumexp', generated_inputs['jax.nn.logsumexp_3'], lib="jax", suffix=3)
