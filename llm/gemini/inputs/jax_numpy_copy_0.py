
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def copy_inputs():
    list_of_inputs = []

    # Input 1: 1D int32 array with order 'K'
    a = np.array([1, -2, 3, -4, 5], dtype=np.int32)
    order = "K"
    list_of_inputs.append(copy.deepcopy({"a": a, "order": order}))

    # Input 2: 2D float32 array with order 'K'
    a = np.array([[-1.5, 2.3], [0.0, -4.7]], dtype=np.float32)
    order = "K"
    list_of_inputs.append(copy.deepcopy({"a": a, "order": order}))

    # Input 3: 3D float64 array with order 'K'
    a = np.random.randn(2, 3, 4).astype(np.float64)
    order = "K"
    list_of_inputs.append(copy.deepcopy({"a": a, "order": order}))

    # Input 4: 0D scalar float32 array with order 'K'
    a = np.array(-3.14, dtype=np.float32)
    order = "K"
    list_of_inputs.append(copy.deepcopy({"a": a, "order": order}))

    # Input 5: 2D int64 array with order 'K'
    a = np.arange(-6, 6, dtype=np.int64).reshape((3, 4))
    order = "K"
    list_of_inputs.append(copy.deepcopy({"a": a, "order": order}))

    # Input 6: 4D float32 array with order 'K'
    a = np.random.uniform(-10.0, 10.0, size=(2, 2, 3, 3)).astype(np.float32)
    order = "K"
    list_of_inputs.append(copy.deepcopy({"a": a, "order": order}))

    # Input 7: 1D boolean array with order 'K'
    a = np.array([True, False, True, True, False], dtype=np.bool_)
    order = "K"
    list_of_inputs.append(copy.deepcopy({"a": a, "order": order}))

    # Input 8: 2D complex64 array with order 'K'
    a = np.array([[1.0 + 2.0j, -3.0 + 0.5j], [-0.5 - 1.0j, 4.0 - 2.0j]], dtype=np.complex64)
    order = "K"
    list_of_inputs.append(copy.deepcopy({"a": a, "order": order}))

    # Input 9: 3D uint8 array with order 'K'
    a = np.arange(24, dtype=np.uint8).reshape((2, 3, 4))
    order = "K"
    list_of_inputs.append(copy.deepcopy({"a": a, "order": order}))

    # Input 10: 1D float32 array containing inf and nan values with order 'K'
    a = np.array([np.nan, -np.inf, 0.0, np.inf], dtype=np.float32)
    order = "K"
    list_of_inputs.append(copy.deepcopy({"a": a, "order": order}))

    # Input 11: 2D empty float32 array with order 'K'
    a = np.empty((0, 5), dtype=np.float32)
    order = "K"
    list_of_inputs.append(copy.deepcopy({"a": a, "order": order}))

    # Input 12: 5D int16 array with order 'K'
    a = np.ones((2, 1, 2, 1, 3), dtype=np.int16)
    order = "K"
    list_of_inputs.append(copy.deepcopy({"a": a, "order": order}))

    return list_of_inputs


generated_inputs["jax.numpy.copy"] = copy_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.copy' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.copy'.")


check_valid('jax.numpy.copy', generated_inputs['jax.numpy.copy'], lib="jax", suffix=0)
