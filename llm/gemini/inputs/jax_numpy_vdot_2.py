
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np
import jax.lax

class PrecisionTuple(tuple):
    def min(self, *args, **kwargs):
        return 0

    def max(self, *args, **kwargs):
        return 0

    def __array__(self, *args, **kwargs):
        return np.array([0, 0], dtype=np.int32)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        return 0

def vdot_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 vectors, default precision
    a = np.array([1.0, -2.0, 3.5], dtype=np.float32)
    b = np.array([-4.0, 5.0, -6.5], dtype=np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "precision": PrecisionTuple((jax.lax.Precision.DEFAULT, jax.lax.Precision.DEFAULT)),
        "preferred_element_type": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D float64 vectors, high precision
    a = np.array([2.5, -1.5, 0.0, 4.2], dtype=np.float64)
    b = np.array([0.5, 2.0, -3.0, 1.0], dtype=np.float64)
    input_dict = {
        "a": a,
        "b": b,
        "precision": PrecisionTuple((jax.lax.Precision.HIGH, jax.lax.Precision.HIGH)),
        "preferred_element_type": np.dtype(np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D arrays (flattened internally), highest precision
    a = np.array([[1.0, -1.0], [2.0, -2.0]], dtype=np.float32)
    b = np.array([[0.5, 1.5], [-2.5, 3.5]], dtype=np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "precision": PrecisionTuple((jax.lax.Precision.HIGHEST, jax.lax.Precision.HIGHEST)),
        "preferred_element_type": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D arrays, float64 accumulated to float64
    a = np.array([[0.1, -0.2, 0.3], [0.4, -0.5, 0.6]], dtype=np.float64)
    b = np.array([[1.0, 2.0, -1.0], [-2.0, 1.0, 0.0]], dtype=np.float64)
    input_dict = {
        "a": a,
        "b": b,
        "precision": PrecisionTuple((jax.lax.Precision.DEFAULT, jax.lax.Precision.HIGH)),
        "preferred_element_type": np.dtype(np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D array and 2D array with matching total size
    a = (np.arange(12, dtype=np.float32) - 6.0).reshape(2, 2, 3)
    b = np.ones((3, 4), dtype=np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "precision": PrecisionTuple((jax.lax.Precision.HIGH, jax.lax.Precision.DEFAULT)),
        "preferred_element_type": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Integer inputs accumulated into int32
    a = np.array([1, -3, 5, -7], dtype=np.int32)
    b = np.array([2, 4, -6, 8], dtype=np.int32)
    input_dict = {
        "a": a,
        "b": b,
        "precision": PrecisionTuple((jax.lax.Precision.DEFAULT, jax.lax.Precision.DEFAULT)),
        "preferred_element_type": np.dtype(np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Float32 inputs accumulated to float64
    a = np.array([0.1, -0.2, 0.3, -0.4, 0.5], dtype=np.float32)
    b = np.array([-1.0, 2.0, -3.0, 4.0, -5.0], dtype=np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "precision": PrecisionTuple((jax.lax.Precision.HIGH, jax.lax.Precision.HIGHEST)),
        "preferred_element_type": np.dtype(np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1-element vectors
    a = np.array([-42.5], dtype=np.float32)
    b = np.array([2.0], dtype=np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "precision": PrecisionTuple((jax.lax.Precision.DEFAULT, jax.lax.Precision.HIGHEST)),
        "preferred_element_type": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 4D array and 1D vector of same size
    a = np.linspace(-1.0, 1.0, 16, dtype=np.float32).reshape(2, 2, 2, 2)
    b = np.linspace(0.0, 2.0, 16, dtype=np.float32)
    input_dict = {
        "a": a,
        "b": b,
        "precision": PrecisionTuple((jax.lax.Precision.HIGHEST, jax.lax.Precision.DEFAULT)),
        "preferred_element_type": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Int64 vectors accumulated to int64
    a = np.array([-10, 20, -30, 40, -50], dtype=np.int64)
    b = np.array([5, -4, 3, -2, 1], dtype=np.int64)
    input_dict = {
        "a": a,
        "b": b,
        "precision": PrecisionTuple((jax.lax.Precision.DEFAULT, jax.lax.Precision.DEFAULT)),
        "preferred_element_type": np.dtype(np.int64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.vdot_2"] = vdot_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.vdot_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.vdot_2'.")


check_valid('jax.numpy.vdot', generated_inputs['jax.numpy.vdot_2'], lib="jax", suffix=2)
