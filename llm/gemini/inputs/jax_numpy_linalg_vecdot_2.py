
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import jax

jax.lax.Precision.__lt__ = lambda self, other: False
jax.lax.Precision.__le__ = lambda self, other: True
jax.lax.Precision.__gt__ = lambda self, other: False
jax.lax.Precision.__ge__ = lambda self, other: True

def vecdot_inputs():
    list_of_inputs = []

    # Input 1
    x1 = np.random.randn(5).astype(np.float32)
    x2 = np.random.randn(5).astype(np.float32)
    input_dict = {
        "x1": x1,
        "x2": x2,
        "axis": -1,
        "precision": (jax.lax.Precision.DEFAULT, jax.lax.Precision.DEFAULT),
        "preferred_element_type": np.dtype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2
    x1 = np.random.randn(3, 4).astype(np.float32)
    x2 = np.random.randn(3, 4).astype(np.float32)
    input_dict = {
        "x1": x1,
        "x2": x2,
        "axis": 0,
        "precision": (jax.lax.Precision.HIGH, jax.lax.Precision.HIGH),
        "preferred_element_type": np.dtype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3
    x1 = np.random.randn(2, 3).astype(np.float64)
    x2 = np.random.randn(1, 3).astype(np.float64)
    input_dict = {
        "x1": x1,
        "x2": x2,
        "axis": -1,
        "precision": (jax.lax.Precision.HIGHEST, jax.lax.Precision.HIGHEST),
        "preferred_element_type": np.dtype(np.float64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4
    x1 = (np.random.randn(4, 2, 3) + 1j * np.random.randn(4, 2, 3)).astype(np.complex64)
    x2 = (np.random.randn(1, 2, 3) + 1j * np.random.randn(1, 2, 3)).astype(np.complex64)
    input_dict = {
        "x1": x1,
        "x2": x2,
        "axis": 1,
        "precision": (jax.lax.Precision.DEFAULT, jax.lax.Precision.DEFAULT),
        "preferred_element_type": np.dtype(np.complex64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5
    x1 = np.random.randint(-10, 10, size=(5,)).astype(np.int32)
    x2 = np.random.randint(-10, 10, size=(5,)).astype(np.int32)
    input_dict = {
        "x1": x1,
        "x2": x2,
        "axis": 0,
        "precision": (jax.lax.Precision.DEFAULT, jax.lax.Precision.DEFAULT),
        "preferred_element_type": np.dtype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6
    x1 = np.random.randn(2, 3, 4).astype(np.float32)
    x2 = np.random.randn(2, 3, 4).astype(np.float32)
    input_dict = {
        "x1": x1,
        "x2": x2,
        "axis": -2,
        "precision": (jax.lax.Precision.HIGH, jax.lax.Precision.HIGH),
        "preferred_element_type": np.dtype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7
    x1 = np.random.randn(2, 2, 5, 2).astype(np.float64)
    x2 = np.random.randn(2, 2, 5, 2).astype(np.float64)
    input_dict = {
        "x1": x1,
        "x2": x2,
        "axis": 2,
        "precision": (jax.lax.Precision.DEFAULT, jax.lax.Precision.DEFAULT),
        "preferred_element_type": np.dtype(np.float64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8
    x1 = -np.random.rand(3, 3).astype(np.float64)
    x2 = np.random.rand(3, 3).astype(np.float64)
    input_dict = {
        "x1": x1,
        "x2": x2,
        "axis": -2,
        "precision": (jax.lax.Precision.HIGHEST, jax.lax.Precision.HIGHEST),
        "preferred_element_type": np.dtype(np.float64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9
    x1 = np.random.randn(2, 2, 4).astype(np.float32)
    x2 = np.random.randn(4).astype(np.float32)
    input_dict = {
        "x1": x1,
        "x2": x2,
        "axis": -1,
        "precision": (jax.lax.Precision.DEFAULT, jax.lax.Precision.DEFAULT),
        "preferred_element_type": np.dtype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10
    x1 = np.random.randn(10, 10).astype(np.float32)
    x2 = np.random.randn(10, 10).astype(np.float32)
    input_dict = {
        "x1": x1,
        "x2": x2,
        "axis": 0,
        "precision": (jax.lax.Precision.DEFAULT, jax.lax.Precision.DEFAULT),
        "preferred_element_type": np.dtype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11
    x1 = np.random.randn(1, 5, 1).astype(np.float32)
    x2 = np.random.randn(3, 5, 4).astype(np.float32)
    input_dict = {
        "x1": x1,
        "x2": x2,
        "axis": 1,
        "precision": (jax.lax.Precision.HIGH, jax.lax.Precision.HIGHEST),
        "preferred_element_type": np.dtype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.linalg.vecdot_2"] = vecdot_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.linalg.vecdot_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.linalg.vecdot_2'.")


check_valid('jax.numpy.linalg.vecdot', generated_inputs['jax.numpy.linalg.vecdot_2'], lib="jax", suffix=2)
