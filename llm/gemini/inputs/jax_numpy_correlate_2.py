
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import jax

jax.lax.Precision.__lt__ = lambda self, other: False
jax.lax.Precision.__gt__ = lambda self, other: False
jax.lax.Precision.__le__ = lambda self, other: True
jax.lax.Precision.__ge__ = lambda self, other: True

def correlate_inputs():
    list_of_inputs = []

    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    v = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    input_dict = {
        "a": a,
        "v": v,
        "mode": "valid",
        "precision": (jax.lax.Precision.DEFAULT, jax.lax.Precision.DEFAULT),
        "preferred_element_type": np.dtype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.array([1.0 + 2.0j, 3.0 - 1.0j, 2.0 + 0.0j], dtype=np.complex64)
    v = np.array([2.0 + 1.0j, 1.0 - 1.0j], dtype=np.complex64)
    input_dict = {
        "a": a,
        "v": v,
        "mode": "full",
        "precision": (jax.lax.Precision.HIGH, jax.lax.Precision.HIGH),
        "preferred_element_type": np.dtype(np.complex64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.array([-1.5, 2.5, -3.5, 4.5, -5.5], dtype=np.float64)
    v = np.array([-0.5, 1.5], dtype=np.float64)
    input_dict = {
        "a": a,
        "v": v,
        "mode": "same",
        "precision": (jax.lax.Precision.HIGHEST, jax.lax.Precision.HIGHEST),
        "preferred_element_type": np.dtype(np.float64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.random.randn(100).astype(np.float32)
    v = np.random.randn(10).astype(np.float32)
    input_dict = {
        "a": a,
        "v": v,
        "mode": "valid",
        "precision": (jax.lax.Precision.DEFAULT, jax.lax.Precision.DEFAULT),
        "preferred_element_type": np.dtype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.array([2.0, 3.0], dtype=np.float32)
    v = np.array([1.0, 4.0, 5.0, 6.0], dtype=np.float32)
    input_dict = {
        "a": a,
        "v": v,
        "mode": "full",
        "precision": (jax.lax.Precision.DEFAULT, jax.lax.Precision.DEFAULT),
        "preferred_element_type": np.dtype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.array([1, -2, 3, -4], dtype=np.int32)
    v = np.array([2, -1], dtype=np.int32)
    input_dict = {
        "a": a,
        "v": v,
        "mode": "same",
        "precision": (jax.lax.Precision.HIGH, jax.lax.Precision.HIGH),
        "preferred_element_type": np.dtype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.array([0.5, -1.5, 2.0, -2.5], dtype=np.float16)
    v = np.array([1.0, -1.0], dtype=np.float16)
    input_dict = {
        "a": a,
        "v": v,
        "mode": "valid",
        "precision": (jax.lax.Precision.DEFAULT, jax.lax.Precision.DEFAULT),
        "preferred_element_type": np.dtype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.array([1.0 - 1.0j, 2.0 + 2.0j, -3.0 + 3.0j], dtype=np.complex128)
    v = np.array([1.0 + 0.0j, 0.0 - 1.0j], dtype=np.complex128)
    input_dict = {
        "a": a,
        "v": v,
        "mode": "full",
        "precision": (jax.lax.Precision.HIGHEST, jax.lax.Precision.HIGHEST),
        "preferred_element_type": np.dtype(np.complex128)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
    v = np.array([2.5], dtype=np.float32)
    input_dict = {
        "a": a,
        "v": v,
        "mode": "same",
        "precision": (jax.lax.Precision.DEFAULT, jax.lax.Precision.DEFAULT),
        "preferred_element_type": np.dtype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    a = np.zeros(10, dtype=np.float32)
    v = np.zeros(5, dtype=np.float32)
    input_dict = {
        "a": a,
        "v": v,
        "mode": "valid",
        "precision": (jax.lax.Precision.DEFAULT, jax.lax.Precision.DEFAULT),
        "preferred_element_type": np.dtype(np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.correlate_2"] = correlate_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.correlate_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.correlate_2'.")


check_valid('jax.numpy.correlate', generated_inputs['jax.numpy.correlate_2'], lib="jax", suffix=2)
