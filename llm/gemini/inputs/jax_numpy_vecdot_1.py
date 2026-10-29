
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def vecdot_inputs():
    list_of_inputs = []

    # Input 1: 1D real float32 arrays
    x1 = np.array([1.0, -2.0, 3.5], dtype=np.float32)
    x2 = np.array([4.0, 5.0, -1.0], dtype=np.float32)
    list_of_inputs.append(
        {
            "x1": x1,
            "x2": x2,
            "axis": -1,
            "precision": "default",
            "preferred_element_type": np.dtype(np.float32),
        }
    )

    # Input 2: 1D complex arrays along axis 0
    x1 = np.array([1.0 + 2.0j, -3.0 + 0.5j, 4.0 - 1.0j], dtype=np.complex64)
    x2 = np.array([2.0 - 1.0j, 1.0 + 1.0j, -2.0 + 3.0j], dtype=np.complex64)
    list_of_inputs.append(
        {
            "x1": x1,
            "x2": x2,
            "axis": 0,
            "precision": "high",
            "preferred_element_type": np.dtype(np.complex64),
        }
    )

    # Input 3: 2D arrays, axis=-1, float64
    x1 = np.arange(12, dtype=np.float64).reshape(3, 4) - 5.0
    x2 = np.ones((3, 4), dtype=np.float64) * 2.0
    list_of_inputs.append(
        {
            "x1": x1,
            "x2": x2,
            "axis": -1,
            "precision": "highest",
            "preferred_element_type": np.dtype(np.float64),
        }
    )

    # Input 4: 2D arrays, axis=0
    x1 = np.array([[1.0, -2.0], [3.0, 4.0], [-5.0, 6.0]], dtype=np.float32)
    x2 = np.array([[0.5, 1.5], [-1.0, 2.0], [3.0, -0.5]], dtype=np.float32)
    list_of_inputs.append(
        {
            "x1": x1,
            "x2": x2,
            "axis": 0,
            "precision": "default",
            "preferred_element_type": np.dtype(np.float32),
        }
    )

    # Input 5: 3D arrays with broadcastable dimensions
    x1 = np.ones((2, 1, 5), dtype=np.float32) * -1.5
    x2 = np.ones((2, 3, 5), dtype=np.float32) * 2.5
    list_of_inputs.append(
        {
            "x1": x1,
            "x2": x2,
            "axis": -1,
            "precision": "high",
            "preferred_element_type": np.dtype(np.float32),
        }
    )

    # Input 6: 3D arrays, axis=1
    x1 = np.arange(24, dtype=np.float32).reshape(2, 4, 3) - 10.0
    x2 = np.arange(24, dtype=np.float32).reshape(2, 4, 3) * 0.1
    list_of_inputs.append(
        {
            "x1": x1,
            "x2": x2,
            "axis": 1,
            "precision": "highest",
            "preferred_element_type": np.dtype(np.float64),
        }
    )

    # Input 7: 4D arrays, axis=-2
    x1 = np.ones((2, 3, 4, 2), dtype=np.float32)
    x2 = np.full((2, 3, 4, 2), -0.5, dtype=np.float32)
    list_of_inputs.append(
        {
            "x1": x1,
            "x2": x2,
            "axis": -2,
            "precision": "fastest",
            "preferred_element_type": np.dtype(np.float32),
        }
    )

    # Input 8: Complex 2D arrays, precision to complex128 accumulation
    x1 = np.array(
        [[1.0 - 1.0j, 2.0 + 3.0j], [-4.0 + 0.0j, 5.0 - 2.0j]],
        dtype=np.complex128,
    )
    x2 = np.array(
        [[0.0 + 2.0j, -1.0 - 1.0j], [3.0 + 1.0j, -2.0 - 2.0j]],
        dtype=np.complex128,
    )
    list_of_inputs.append(
        {
            "x1": x1,
            "x2": x2,
            "axis": -1,
            "precision": "high",
            "preferred_element_type": np.dtype(np.complex128),
        }
    )

    # Input 9: 3D broadcasting against 1D vector along axis=-1
    x1 = np.arange(18, dtype=np.float32).reshape(2, 3, 3)
    x2 = np.array([1.0, 0.0, -1.0], dtype=np.float32)
    list_of_inputs.append(
        {
            "x1": x1,
            "x2": x2,
            "axis": -1,
            "precision": "default",
            "preferred_element_type": np.dtype(np.float32),
        }
    )

    # Input 10: 3D arrays along axis=-3
    x1 = np.ones((4, 2, 2), dtype=np.float32) * 3.0
    x2 = np.ones((4, 2, 2), dtype=np.float32) * -2.0
    list_of_inputs.append(
        {
            "x1": x1,
            "x2": x2,
            "axis": -3,
            "precision": "highest",
            "preferred_element_type": np.dtype(np.float32),
        }
    )

    return list_of_inputs


generated_inputs["jax.numpy.vecdot_1"] = vecdot_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.vecdot_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.vecdot_1'.")


check_valid('jax.numpy.vecdot', generated_inputs['jax.numpy.vecdot_1'], lib="jax", suffix=1)
