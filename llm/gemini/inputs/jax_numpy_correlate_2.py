
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


class PrecisionTuple(tuple):

    def _as_numpy_ndarray(self):
        return np.array([0, 0])

    def __array__(self, dtype=None, copy=None):
        return np.array([0, 0])


def correlate_inputs():
    list_of_inputs = []

    # Input 1: Basic float32, mode='valid'
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    v = np.array([1.0, -1.0, 2.0], dtype=np.float32)
    input_dict = {
        "a": a,
        "v": v,
        "mode": "valid",
        "precision": PrecisionTuple(("default", "default")),
        "preferred_element_type": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float64, mode='full'
    a = np.array([-2.5, 0.0, 1.5, 3.2], dtype=np.float64)
    v = np.array([0.5, -1.2], dtype=np.float64)
    input_dict = {
        "a": a,
        "v": v,
        "mode": "full",
        "precision": PrecisionTuple(("high", "high")),
        "preferred_element_type": np.dtype(np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: int32 inputs, mode='same', preferred_element_type float32
    a = np.array([10, -5, 2, 7, -3, 8], dtype=np.int32)
    v = np.array([1, 2, 3], dtype=np.int32)
    input_dict = {
        "a": a,
        "v": v,
        "mode": "same",
        "precision": PrecisionTuple(("default", "default")),
        "preferred_element_type": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Complex numbers, mode='full'
    a = np.array([1.0 + 2.0j, -3.0 + 0.5j, 0.0 - 1.0j], dtype=np.complex64)
    v = np.array([2.0 - 1.0j, 1.0 + 1.0j], dtype=np.complex64)
    input_dict = {
        "a": a,
        "v": v,
        "mode": "full",
        "precision": PrecisionTuple(("highest", "highest")),
        "preferred_element_type": np.dtype(np.complex64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Complex128 inputs, mode='valid'
    a = np.array(
        [-1.5 + 2.5j, 3.0 - 4.0j, 0.5 + 0.5j, -2.0 - 1.0j], dtype=np.complex128
    )
    v = np.array([1.0 + 0.0j, -0.5 + 1.0j], dtype=np.complex128)
    input_dict = {
        "a": a,
        "v": v,
        "mode": "valid",
        "precision": PrecisionTuple(("high", "default")),
        "preferred_element_type": np.dtype(np.complex128),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Equal lengths, mode='full'
    a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    v = np.array([4.0, 3.0, 2.0, 1.0], dtype=np.float32)
    input_dict = {
        "a": a,
        "v": v,
        "mode": "full",
        "precision": PrecisionTuple(("default", "highest")),
        "preferred_element_type": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Single element v, mode='same'
    a = np.array([5.0, -2.0, 3.5, 8.1, -1.2], dtype=np.float32)
    v = np.array([2.0], dtype=np.float32)
    input_dict = {
        "a": a,
        "v": v,
        "mode": "same",
        "precision": PrecisionTuple(("default", "default")),
        "preferred_element_type": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Larger 1D arrays, mode='valid'
    a = np.linspace(-10, 10, 100, dtype=np.float64)
    v = np.ones(10, dtype=np.float64)
    input_dict = {
        "a": a,
        "v": v,
        "mode": "valid",
        "precision": PrecisionTuple(("highest", "highest")),
        "preferred_element_type": np.dtype(np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: float16 arrays, mode='same'
    a = np.array([0.1, -0.5, 1.2, -2.0, 3.0, -1.5], dtype=np.float16)
    v = np.array([1.0, 0.5, -0.5], dtype=np.float16)
    input_dict = {
        "a": a,
        "v": v,
        "mode": "same",
        "precision": PrecisionTuple(("default", "default")),
        "preferred_element_type": np.dtype(np.float16),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Array 'a' shorter than 'v', mode='full'
    a = np.array([1.0, -1.0], dtype=np.float32)
    v = np.array([2.0, 0.0, -2.0, 1.0, 3.0], dtype=np.float32)
    input_dict = {
        "a": a,
        "v": v,
        "mode": "full",
        "precision": PrecisionTuple(("high", "high")),
        "preferred_element_type": np.dtype(np.float32),
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
