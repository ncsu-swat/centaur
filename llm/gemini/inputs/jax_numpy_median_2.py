
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def median_inputs():
    list_of_inputs = []

    # Input 1: 1D array, single axis in tuple
    a = np.array([1.0, -3.0, 5.0, 2.0, 4.0], dtype=np.float32)
    input_dict = {
        "a": a,
        "axis": (0,),
        "overwrite_input": False,
        "keepdims": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D array, reduce axis 0 with keepdims True
    a = np.array([[-1.5, 2.3, 0.0], [4.1, -5.2, 6.7]], dtype=np.float32)
    input_dict = {
        "a": a,
        "axis": (0,),
        "overwrite_input": False,
        "keepdims": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D array, reduce axis 1 with keepdims False
    a = np.arange(12, dtype=np.int32).reshape(3, 4) - 5
    input_dict = {
        "a": a,
        "axis": (1,),
        "overwrite_input": False,
        "keepdims": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D array, reduce multiple axes (0, 1)
    a = np.array([[10, 20], [30, 40], [50, 60]], dtype=np.int64)
    input_dict = {
        "a": a,
        "axis": (0, 1),
        "overwrite_input": False,
        "keepdims": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D array with negative axis specification
    a = np.random.randn(2, 3, 4).astype(np.float64)
    input_dict = {
        "a": a,
        "axis": (-1,),
        "overwrite_input": False,
        "keepdims": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D array, reduce across two axes (-2, -1)
    a = np.random.uniform(-10.0, 10.0, size=(4, 5, 6)).astype(np.float32)
    input_dict = {
        "a": a,
        "axis": (-2, -1),
        "overwrite_input": False,
        "keepdims": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 4D array, float32, keepdims False
    a = np.linspace(-5, 5, 24, dtype=np.float32).reshape(2, 3, 2, 2)
    input_dict = {
        "a": a,
        "axis": (1, 3),
        "overwrite_input": False,
        "keepdims": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1D integer array with even number of elements
    a = np.array([10, -20, 30, -40, 50, -60], dtype=np.int32)
    input_dict = {
        "a": a,
        "axis": (0,),
        "overwrite_input": False,
        "keepdims": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D array, axis (0, 2)
    a = np.random.randn(3, 4, 5).astype(np.float32)
    input_dict = {
        "a": a,
        "axis": (0, 2),
        "overwrite_input": False,
        "keepdims": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 5D array, reducing a single internal axis
    a = np.ones((2, 2, 3, 2, 2), dtype=np.float64) * -2.5
    input_dict = {
        "a": a,
        "axis": (2,),
        "overwrite_input": False,
        "keepdims": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.median_2"] = median_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.median_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.median_2'.")


check_valid('jax.numpy.median', generated_inputs['jax.numpy.median_2'], lib="jax", suffix=2)
