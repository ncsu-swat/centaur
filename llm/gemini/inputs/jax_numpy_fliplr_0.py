
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def fliplr_inputs():
    list_of_inputs = []

    # Input 1: 2D float32 array
    m = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"m": m}))

    # Input 2: 2D int32 array with negative values
    m = np.array([[-1, -2], [3, -4], [5, 6]], dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"m": m}))

    # Input 3: 3D float64 array
    m = np.random.randn(2, 3, 4).astype(np.float64)
    list_of_inputs.append(copy.deepcopy({"m": m}))

    # Input 4: 4D uint8 array
    m = np.arange(24, dtype=np.uint8).reshape(2, 3, 2, 2)
    list_of_inputs.append(copy.deepcopy({"m": m}))

    # Input 5: 2D boolean array
    m = np.array([[True, False, True], [False, False, True]], dtype=np.bool_)
    list_of_inputs.append(copy.deepcopy({"m": m}))

    # Input 6: 2D complex numbers
    m = np.array([[1 + 2j, 3 - 4j], [-5j, 6 + 0j]], dtype=np.complex64)
    list_of_inputs.append(copy.deepcopy({"m": m}))

    # Input 7: 5D int64 array
    m = np.ones((1, 4, 1, 2, 3), dtype=np.int64)
    list_of_inputs.append(copy.deepcopy({"m": m}))

    # Input 8: 2D array with single column (axis 1 has size 1)
    m = np.array([[10], [20], [30]], dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"m": m}))

    # Input 9: 2D empty array (shape with zero dimension)
    m = np.empty((3, 0), dtype=np.float32)
    list_of_inputs.append(copy.deepcopy({"m": m}))

    # Input 10: 3D empty array along axis 0
    m = np.empty((0, 4, 2), dtype=np.int32)
    list_of_inputs.append(copy.deepcopy({"m": m}))

    # Input 11: 2D large random array
    m = np.random.uniform(-100.0, 100.0, size=(50, 50)).astype(np.float64)
    list_of_inputs.append(copy.deepcopy({"m": m}))

    return list_of_inputs


generated_inputs["jax.numpy.fliplr"] = fliplr_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.fliplr' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.fliplr'.")


check_valid('jax.numpy.fliplr', generated_inputs['jax.numpy.fliplr'], lib="jax", suffix=0)
