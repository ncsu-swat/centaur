
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np

def tensordot_inputs():
    list_of_inputs = []

    # Input 1: 3D and 3D arrays, contracting 2 axes, precision='default'
    a = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    b = np.ones((3, 4, 5), dtype=np.float32)
    axes = ((1, 2), (0, 1))
    precision = "default"
    preferred_element_type = np.dtype(np.float32)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "b": b,
                "axes": axes,
                "precision": precision,
                "preferred_element_type": preferred_element_type,
            }
        )
    )

    # Input 2: 2D and 2D arrays, matrix multiplication style, precision='high'
    a = np.array([[1.5, -2.0], [3.0, 4.5]], dtype=np.float32)
    b = np.array([[-1.0, 2.0, 0.5], [4.0, -5.0, 6.0]], dtype=np.float32)
    axes = ((1,), (0,))
    precision = "high"
    preferred_element_type = np.dtype(np.float32)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "b": b,
                "axes": axes,
                "precision": precision,
                "preferred_element_type": preferred_element_type,
            }
        )
    )

    # Input 3: 2D and 2D arrays, contracting axis 0 of both, precision='highest'
    a = np.array([[1.0, -2.0, 3.0], [4.0, 5.0, -1.0]], dtype=np.float64)
    b = np.array([[2.0, 0.0], [-1.0, 3.0]], dtype=np.float64)
    axes = ((0,), (0,))
    precision = "highest"
    preferred_element_type = np.dtype(np.float64)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "b": b,
                "axes": axes,
                "precision": precision,
                "preferred_element_type": preferred_element_type,
            }
        )
    )

    # Input 4: 4D and 3D arrays, contracting 2 axes with negative indices
    a = np.random.randn(2, 3, 4, 5).astype(np.float32)
    b = np.random.randn(4, 5, 6).astype(np.float32)
    axes = ((-2, -1), (0, 1))
    precision = "fastest"
    preferred_element_type = np.dtype(np.float32)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "b": b,
                "axes": axes,
                "precision": precision,
                "preferred_element_type": preferred_element_type,
            }
        )
    )

    # Input 5: 3D and 2D arrays, 1 axis contraction
    a = np.random.randn(3, 4, 5).astype(np.float32)
    b = np.random.randn(5, 6).astype(np.float32)
    axes = ((2,), (0,))
    precision = "high"
    preferred_element_type = np.dtype(np.float32)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "b": b,
                "axes": axes,
                "precision": precision,
                "preferred_element_type": preferred_element_type,
            }
        )
    )

    # Input 6: Full reduction across all axes
    a = np.arange(12, dtype=np.float32).reshape(3, 4)
    b = np.ones((3, 4), dtype=np.float32)
    axes = ((0, 1), (0, 1))
    precision = "highest"
    preferred_element_type = np.dtype(np.float32)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "b": b,
                "axes": axes,
                "precision": precision,
                "preferred_element_type": preferred_element_type,
            }
        )
    )

    # Input 7: Float32 inputs with float64 accumulation type
    a = np.random.randn(5, 6, 7).astype(np.float32)
    b = np.random.randn(7, 6, 2).astype(np.float32)
    axes = ((1, 2), (1, 0))
    precision = "default"
    preferred_element_type = np.dtype(np.float64)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "b": b,
                "axes": axes,
                "precision": precision,
                "preferred_element_type": preferred_element_type,
            }
        )
    )

    # Input 8: 1D dot product
    a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    b = np.array([-1.0, 0.5, 2.0, -3.0], dtype=np.float32)
    axes = ((0,), (0,))
    precision = "float32"
    preferred_element_type = np.dtype(np.float32)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "b": b,
                "axes": axes,
                "precision": precision,
                "preferred_element_type": preferred_element_type,
            }
        )
    )

    # Input 9: 5D and 4D arrays contracting 3 axes
    a = np.random.randn(2, 3, 2, 4, 5).astype(np.float32)
    b = np.random.randn(5, 2, 3, 6).astype(np.float32)
    axes = ((1, 2, 4), (2, 1, 0))
    precision = "tensorfloat32"
    preferred_element_type = np.dtype(np.float32)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "b": b,
                "axes": axes,
                "precision": precision,
                "preferred_element_type": preferred_element_type,
            }
        )
    )

    # Input 10: Non-contiguous axes contraction
    a = np.random.randn(4, 2, 5, 3).astype(np.float64)
    b = np.random.randn(3, 7, 4).astype(np.float64)
    axes = ((0, 3), (2, 0))
    precision = "highest"
    preferred_element_type = np.dtype(np.float64)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "a": a,
                "b": b,
                "axes": axes,
                "precision": precision,
                "preferred_element_type": preferred_element_type,
            }
        )
    )

    return list_of_inputs

generated_inputs["jax.numpy.tensordot_3"] = tensordot_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.tensordot_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.tensordot_3'.")


check_valid('jax.numpy.tensordot', generated_inputs['jax.numpy.tensordot_3'], lib="jax", suffix=3)
