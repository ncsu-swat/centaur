
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

class PrecisionTuple(tuple):
    def min(self, axis=None, out=None, **kwargs):
        return self[0]
    def max(self, axis=None, out=None, **kwargs):
        return self[0]

def multi_dot_inputs():
    list_of_inputs = []

    # Input 1: 3D array of shape (3, 5, 5), float32, precision: ("default", "default")
    arrays = np.random.randn(3, 5, 5).astype(np.float32)
    list_of_inputs.append({
        "arrays": copy.deepcopy(arrays),
        "precision": PrecisionTuple(("default", "default"))
    })

    # Input 2: 3D array of shape (4, 2, 2), float32, precision: ("high", "high")
    arrays = np.random.randn(4, 2, 2).astype(np.float32)
    list_of_inputs.append({
        "arrays": copy.deepcopy(arrays),
        "precision": PrecisionTuple(("high", "high"))
    })

    # Input 3: 3D array of shape (3, 10, 10), float64, precision: ("highest", "highest")
    arrays = np.random.randn(3, 10, 10).astype(np.float64)
    list_of_inputs.append({
        "arrays": copy.deepcopy(arrays),
        "precision": PrecisionTuple(("highest", "highest"))
    })

    # Input 4: 3D array of shape (5, 3, 3), float32, precision: ("default", "default")
    arrays = np.random.randn(5, 3, 3).astype(np.float32)
    list_of_inputs.append({
        "arrays": copy.deepcopy(arrays),
        "precision": PrecisionTuple(("default", "default"))
    })

    # Input 5: 3D array of shape (2, 8, 8), float64, precision: ("high", "high")
    arrays = np.random.randn(2, 8, 8).astype(np.float64)
    list_of_inputs.append({
        "arrays": copy.deepcopy(arrays),
        "precision": PrecisionTuple(("high", "high"))
    })

    # Input 6: 3D array of shape (3, 6, 6), float32, precision: ("highest", "highest")
    arrays = np.random.randn(3, 6, 6).astype(np.float32)
    list_of_inputs.append({
        "arrays": copy.deepcopy(arrays),
        "precision": PrecisionTuple(("highest", "highest"))
    })

    # Input 7: 3D array of shape (4, 4, 4), float32, precision: ("default", "default")
    arrays = np.random.randn(4, 4, 4).astype(np.float32)
    list_of_inputs.append({
        "arrays": copy.deepcopy(arrays),
        "precision": PrecisionTuple(("default", "default"))
    })

    # Input 8: 3D array with negative/positive uniform values, precision: ("high", "high")
    arrays = np.random.uniform(-2.0, 2.0, (3, 7, 7)).astype(np.float32)
    list_of_inputs.append({
        "arrays": copy.deepcopy(arrays),
        "precision": PrecisionTuple(("high", "high"))
    })

    # Input 9: 3D array of shape (2, 12, 12), float32, precision: ("highest", "highest")
    arrays = np.random.randn(2, 12, 12).astype(np.float32)
    list_of_inputs.append({
        "arrays": copy.deepcopy(arrays),
        "precision": PrecisionTuple(("highest", "highest"))
    })

    # Input 10: 3D array of shape (3, 9, 9), float32, precision: ("default", "default")
    arrays = np.random.randn(3, 9, 9).astype(np.float32)
    list_of_inputs.append({
        "arrays": copy.deepcopy(arrays),
        "precision": PrecisionTuple(("default", "default"))
    })

    return list_of_inputs

generated_inputs["jax.numpy.linalg.multi_dot_2"] = multi_dot_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.linalg.multi_dot_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.linalg.multi_dot_2'.")


check_valid('jax.numpy.linalg.multi_dot', generated_inputs['jax.numpy.linalg.multi_dot_2'], lib="jax", suffix=2)
