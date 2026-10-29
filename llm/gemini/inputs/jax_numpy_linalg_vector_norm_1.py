
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def vector_norm_inputs():
    list_of_inputs = []

    # Input 1: 1D array, default L2 norm
    x = np.array([1.0, -2.0, 3.0, -4.0], dtype=np.float32)
    axis = 0
    keepdims = False
    ord_val = 2
    list_of_inputs.append(
        copy.deepcopy({"x": x, "axis": axis, "keepdims": keepdims, "ord": ord_val})
    )

    # Input 2: 2D array, L1 norm along axis 0
    x = np.array([[1.5, -2.5], [3.0, 4.0]], dtype=np.float64)
    axis = 0
    keepdims = True
    ord_val = 1
    list_of_inputs.append(
        copy.deepcopy({"x": x, "axis": axis, "keepdims": keepdims, "ord": ord_val})
    )

    # Input 3: 2D array, L2 norm along axis 1
    x = np.random.randn(5, 4).astype(np.float32)
    axis = 1
    keepdims = False
    ord_val = 2
    list_of_inputs.append(
        copy.deepcopy({"x": x, "axis": axis, "keepdims": keepdims, "ord": ord_val})
    )

    # Input 4: 2D array, negative axis, ord=0 (count non-zeros)
    x = np.array([[0.0, 1.0, -2.0], [3.0, 0.0, 0.0]], dtype=np.float32)
    axis = -1
    keepdims = False
    ord_val = 0
    list_of_inputs.append(
        copy.deepcopy({"x": x, "axis": axis, "keepdims": keepdims, "ord": ord_val})
    )

    # Input 5: 3D array, ord=1 along axis 2, keepdims=True
    x = np.random.uniform(-10.0, 10.0, size=(2, 3, 4)).astype(np.float64)
    axis = 2
    keepdims = True
    ord_val = 1
    list_of_inputs.append(
        copy.deepcopy({"x": x, "axis": axis, "keepdims": keepdims, "ord": ord_val})
    )

    # Input 6: 3D array, negative ord (-1 norm)
    x = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    axis = 1
    keepdims = False
    ord_val = -1
    list_of_inputs.append(
        copy.deepcopy({"x": x, "axis": axis, "keepdims": keepdims, "ord": ord_val})
    )

    # Input 7: 4D array, ord=3 (p-norm)
    x = np.random.randn(2, 3, 2, 4).astype(np.float32)
    axis = 3
    keepdims = True
    ord_val = 3
    list_of_inputs.append(
        copy.deepcopy({"x": x, "axis": axis, "keepdims": keepdims, "ord": ord_val})
    )

    # Input 8: Complex 2D array, ord=2
    x = (np.random.randn(3, 3) + 1j * np.random.randn(3, 3)).astype(np.complex64)
    axis = 0
    keepdims = False
    ord_val = 2
    list_of_inputs.append(
        copy.deepcopy({"x": x, "axis": axis, "keepdims": keepdims, "ord": ord_val})
    )

    # Input 9: Negative ord (-2 norm), 1D array
    x = np.array([2.0, 4.0, 8.0], dtype=np.float64)
    axis = -1
    keepdims = True
    ord_val = -2
    list_of_inputs.append(
        copy.deepcopy({"x": x, "axis": axis, "keepdims": keepdims, "ord": ord_val})
    )

    # Input 10: 4D array, ord=2 along axis 1, keepdims=False
    x = np.random.randn(4, 5, 2, 3).astype(np.float32)
    axis = 1
    keepdims = False
    ord_val = 2
    list_of_inputs.append(
        copy.deepcopy({"x": x, "axis": axis, "keepdims": keepdims, "ord": ord_val})
    )

    # Input 11: 1D array, ord=4
    x = np.array([-1.0, -2.0, -3.0, 4.0, 5.0], dtype=np.float32)
    axis = 0
    keepdims = False
    ord_val = 4
    list_of_inputs.append(
        copy.deepcopy({"x": x, "axis": axis, "keepdims": keepdims, "ord": ord_val})
    )

    return list_of_inputs


generated_inputs["jax.numpy.linalg.vector_norm_1"] = vector_norm_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.linalg.vector_norm_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.linalg.vector_norm_1'.")


check_valid('jax.numpy.linalg.vector_norm', generated_inputs['jax.numpy.linalg.vector_norm_1'], lib="jax", suffix=1)
