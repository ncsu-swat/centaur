
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


class PrecisionTuple(tuple):
    def __array__(self, *args, **kwargs):
        return np.array(list(self), dtype=object)


def vecdot_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 vectors, axis=-1
    x1 = np.array([1.0, -2.0, 3.5, -4.2], dtype=np.float32)
    x2 = np.array([-0.5, 2.0, 1.2, 3.0], dtype=np.float32)
    list_of_inputs.append(
        {
            "x1": x1,
            "x2": x2,
            "axis": -1,
            "precision": PrecisionTuple(("default", "default")),
            "preferred_element_type": np.dtype(np.float32),
        }
    )

    # Input 2: 2D float64 matrices along axis=0
    x1 = np.random.randn(5, 3).astype(np.float64)
    x2 = np.random.randn(5, 3).astype(np.float64)
    list_of_inputs.append(
        {
            "x1": x1,
            "x2": x2,
            "axis": 0,
            "precision": PrecisionTuple(("high", "high")),
            "preferred_element_type": np.dtype(np.float64),
        }
    )

    # Input 3: 2D float32 matrices along axis=1
    x1 = np.random.randn(4, 6).astype(np.float32)
    x2 = np.random.randn(4, 6).astype(np.float32)
    list_of_inputs.append(
        {
            "x1": x1,
            "x2": x2,
            "axis": 1,
            "precision": PrecisionTuple(("highest", "highest")),
            "preferred_element_type": np.dtype(np.float32),
        }
    )

    # Input 4: Broadcasting in non-contracting dimensions (1, 4) with (3, 4)
    x1 = np.array([[-1.0, 2.0, -3.0, 4.0]], dtype=np.float32)
    x2 = np.array(
        [[1.0, 1.0, 1.0, 1.0], [0.5, -0.5, 0.5, -0.5], [2.0, 3.0, -1.0, 0.0]],
        dtype=np.float32,
    )
    list_of_inputs.append(
        {
            "x1": x1,
            "x2": x2,
            "axis": -1,
            "precision": PrecisionTuple(("default", "high")),
            "preferred_element_type": np.dtype(np.float32),
        }
    )

    # Input 5: Complex128 1D vectors
    x1 = np.array([1.0 + 2.0j, -3.0 + 0.5j, 4.0 - 1.0j], dtype=np.complex128)
    x2 = np.array([-2.0 + 1.0j, 1.5 - 2.5j, 0.0 + 3.0j], dtype=np.complex128)
    list_of_inputs.append(
        {
            "x1": x1,
            "x2": x2,
            "axis": 0,
            "precision": PrecisionTuple(("highest", "highest")),
            "preferred_element_type": np.dtype(np.complex128),
        }
    )

    # Input 6: Complex64 2D arrays along axis=-1
    x1 = (
        np.random.randn(3, 4).astype(np.float32)
        + 1j * np.random.randn(3, 4).astype(np.float32)
    ).astype(np.complex64)
    x2 = (
        np.random.randn(3, 4).astype(np.float32)
        + 1j * np.random.randn(3, 4).astype(np.float32)
    ).astype(np.complex64)
    list_of_inputs.append(
        {
            "x1": x1,
            "x2": x2,
            "axis": -1,
            "precision": PrecisionTuple(("high", "default")),
            "preferred_element_type": np.dtype(np.complex64),
        }
    )

    # Input 7: 3D float32 arrays along middle axis=1
    x1 = np.random.randn(2, 5, 4).astype(np.float32)
    x2 = np.random.randn(2, 5, 4).astype(np.float32)
    list_of_inputs.append(
        {
            "x1": x1,
            "x2": x2,
            "axis": 1,
            "precision": PrecisionTuple(("default", "default")),
            "preferred_element_type": np.dtype(np.float32),
        }
    )

    # Input 8: 3D arrays with multi-axis broadcasting
    x1 = np.random.randn(2, 1, 6).astype(np.float64)
    x2 = np.random.randn(1, 3, 6).astype(np.float64)
    list_of_inputs.append(
        {
            "x1": x1,
            "x2": x2,
            "axis": -1,
            "precision": PrecisionTuple(("high", "high")),
            "preferred_element_type": np.dtype(np.float64),
        }
    )

    # Input 9: 4D float32 arrays along negative axis=-2
    x1 = np.random.randn(2, 3, 4, 2).astype(np.float32)
    x2 = np.random.randn(2, 3, 4, 2).astype(np.float32)
    list_of_inputs.append(
        {
            "x1": x1,
            "x2": x2,
            "axis": -2,
            "precision": PrecisionTuple(("default", "default")),
            "preferred_element_type": np.dtype(np.float32),
        }
    )

    # Input 10: 1D integer vectors with int32 preferred_element_type
    x1 = np.array([3, -7, 12, -5, 8], dtype=np.int32)
    x2 = np.array([-2, 4, 1, 6, -3], dtype=np.int32)
    list_of_inputs.append(
        {
            "x1": x1,
            "x2": x2,
            "axis": 0,
            "precision": PrecisionTuple(("default", "default")),
            "preferred_element_type": np.dtype(np.int32),
        }
    )

    # Input 11: 4D float32 arrays along first axis=0
    x1 = np.random.randn(7, 2, 2, 3).astype(np.float32)
    x2 = np.random.randn(7, 2, 2, 3).astype(np.float32)
    list_of_inputs.append(
        {
            "x1": x1,
            "x2": x2,
            "axis": 0,
            "precision": PrecisionTuple(("highest", "highest")),
            "preferred_element_type": np.dtype(np.float32),
        }
    )

    return list_of_inputs


generated_inputs["jax.numpy.vecdot_2"] = vecdot_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.vecdot_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.vecdot_2'.")


check_valid('jax.numpy.vecdot', generated_inputs['jax.numpy.vecdot_2'], lib="jax", suffix=2)
