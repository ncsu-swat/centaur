
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


class PrecisionTuple(tuple):

    def __array__(self, dtype=None, copy=None):
        arr = np.empty(len(self), dtype=object)
        for i, v in enumerate(self):
            arr[i] = v
        return arr

    def __deepcopy__(self, memo):
        return PrecisionTuple(self)


def vecdot_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 arrays
    x1 = np.array([1.0, -2.0, 3.0, -4.0], dtype=np.float32)
    x2 = np.array([0.5, 1.5, -2.0, 2.5], dtype=np.float32)
    axis = 0
    precision = PrecisionTuple(("default", "default"))
    preferred_element_type = np.dtype(np.float32)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "x1": x1,
                "x2": x2,
                "axis": axis,
                "precision": precision,
                "preferred_element_type": preferred_element_type,
            }
        )
    )

    # Input 2: 2D float64 arrays along last axis
    x1 = np.random.randn(3, 5).astype(np.float64)
    x2 = np.random.randn(3, 5).astype(np.float64)
    axis = -1
    precision = PrecisionTuple(("high", "high"))
    preferred_element_type = np.dtype(np.float64)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "x1": x1,
                "x2": x2,
                "axis": axis,
                "precision": precision,
                "preferred_element_type": preferred_element_type,
            }
        )
    )

    # Input 3: 2D arrays along axis 0
    x1 = np.random.randn(4, 6).astype(np.float32)
    x2 = np.random.randn(4, 6).astype(np.float32)
    axis = 0
    precision = PrecisionTuple(("highest", "highest"))
    preferred_element_type = np.dtype(np.float32)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "x1": x1,
                "x2": x2,
                "axis": axis,
                "precision": precision,
                "preferred_element_type": preferred_element_type,
            }
        )
    )

    # Input 4: Broadcasting along batch dimensions
    x1 = np.random.randn(5, 1, 8).astype(np.float32)
    x2 = np.random.randn(1, 4, 8).astype(np.float32)
    axis = -1
    precision = PrecisionTuple(("default", "high"))
    preferred_element_type = np.dtype(np.float64)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "x1": x1,
                "x2": x2,
                "axis": axis,
                "precision": precision,
                "preferred_element_type": preferred_element_type,
            }
        )
    )

    # Input 5: Complex64 arrays
    x1 = (np.random.randn(2, 3) + 1j * np.random.randn(2, 3)).astype(
        np.complex64
    )
    x2 = (np.random.randn(2, 3) + 1j * np.random.randn(2, 3)).astype(
        np.complex64
    )
    axis = -1
    precision = PrecisionTuple(("default", "default"))
    preferred_element_type = np.dtype(np.complex64)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "x1": x1,
                "x2": x2,
                "axis": axis,
                "precision": precision,
                "preferred_element_type": preferred_element_type,
            }
        )
    )

    # Input 6: Complex128 arrays along middle axis
    x1 = (np.random.randn(4, 7, 3) + 1j * np.random.randn(4, 7, 3)).astype(
        np.complex128
    )
    x2 = (np.random.randn(4, 7, 3) + 1j * np.random.randn(4, 7, 3)).astype(
        np.complex128
    )
    axis = 1
    precision = PrecisionTuple(("high", "default"))
    preferred_element_type = np.dtype(np.complex128)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "x1": x1,
                "x2": x2,
                "axis": axis,
                "precision": precision,
                "preferred_element_type": preferred_element_type,
            }
        )
    )

    # Input 7: 4D tensors contracting along axis 2
    x1 = np.random.randn(2, 3, 6, 2).astype(np.float32)
    x2 = np.random.randn(2, 3, 6, 2).astype(np.float32)
    axis = 2
    precision = PrecisionTuple(("highest", "default"))
    preferred_element_type = np.dtype(np.float32)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "x1": x1,
                "x2": x2,
                "axis": axis,
                "precision": precision,
                "preferred_element_type": preferred_element_type,
            }
        )
    )

    # Input 8: Integer arrays accumulated into float32
    x1 = np.random.randint(-10, 10, size=(5, 3), dtype=np.int32)
    x2 = np.random.randint(-10, 10, size=(5, 3), dtype=np.int32)
    axis = 1
    precision = PrecisionTuple(("default", "default"))
    preferred_element_type = np.dtype(np.float32)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "x1": x1,
                "x2": x2,
                "axis": axis,
                "precision": precision,
                "preferred_element_type": preferred_element_type,
            }
        )
    )

    # Input 9: Negative axis on 3D array with broadcast
    x1 = np.random.randn(3, 5, 2).astype(np.float32)
    x2 = np.random.randn(1, 5, 2).astype(np.float32)
    axis = -2
    precision = PrecisionTuple(("high", "highest"))
    preferred_element_type = np.dtype(np.float32)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "x1": x1,
                "x2": x2,
                "axis": axis,
                "precision": precision,
                "preferred_element_type": preferred_element_type,
            }
        )
    )

    # Input 10: 1D arrays with float16 inputs
    x1 = np.random.randn(16).astype(np.float16)
    x2 = np.random.randn(16).astype(np.float16)
    axis = -1
    precision = PrecisionTuple(("default", "default"))
    preferred_element_type = np.dtype(np.float32)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "x1": x1,
                "x2": x2,
                "axis": axis,
                "precision": precision,
                "preferred_element_type": preferred_element_type,
            }
        )
    )

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
