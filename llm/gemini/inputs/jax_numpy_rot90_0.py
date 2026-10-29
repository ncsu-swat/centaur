
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def rot90_inputs():
    list_of_inputs = []

    # Input 1: 2D float32 array, default-like rotation
    m = np.random.randn(3, 4).astype(np.float32)
    k = 1
    axes = (0, 1)
    list_of_inputs.append(
        copy.deepcopy({"m": m, "k": int(k), "axes": tuple(axes)})
    )

    # Input 2: 2D int32 array, inverted axes, k=2
    m = np.arange(12, dtype=np.int32).reshape(3, 4)
    k = 2
    axes = (1, 0)
    list_of_inputs.append(
        copy.deepcopy({"m": m, "k": int(k), "axes": tuple(axes)})
    )

    # Input 3: 2D boolean array, negative k (clockwise)
    m = np.array([[True, False], [False, True]], dtype=bool)
    k = -1
    axes = (0, 1)
    list_of_inputs.append(
        copy.deepcopy({"m": m, "k": int(k), "axes": tuple(axes)})
    )

    # Input 4: 3D float64 array, k=0 (identity rotation)
    m = np.random.randn(2, 3, 4).astype(np.float64)
    k = 0
    axes = (0, 2)
    list_of_inputs.append(
        copy.deepcopy({"m": m, "k": int(k), "axes": tuple(axes)})
    )

    # Input 5: 3D int64 array, k=3, axes=(1, 2)
    m = np.random.randint(-50, 50, size=(4, 5, 6), dtype=np.int64)
    k = 3
    axes = (1, 2)
    list_of_inputs.append(
        copy.deepcopy({"m": m, "k": int(k), "axes": tuple(axes)})
    )

    # Input 6: 3D complex64 array, negative k, axes=(2, 0)
    m = (
        np.random.randn(2, 3, 4) + 1j * np.random.randn(2, 3, 4)
    ).astype(np.complex64)
    k = -2
    axes = (2, 0)
    list_of_inputs.append(
        copy.deepcopy({"m": m, "k": int(k), "axes": tuple(axes)})
    )

    # Input 7: 4D float32 array, axes=(0, 3), k=1
    m = np.random.randn(2, 3, 2, 4).astype(np.float32)
    k = 1
    axes = (0, 3)
    list_of_inputs.append(
        copy.deepcopy({"m": m, "k": int(k), "axes": tuple(axes)})
    )

    # Input 8: 4D uint8 array, negative axes indices, k=-3
    m = np.random.randint(0, 256, size=(2, 4, 3, 5), dtype=np.uint8)
    k = -3
    axes = (-2, -1)
    list_of_inputs.append(
        copy.deepcopy({"m": m, "k": int(k), "axes": tuple(axes)})
    )

    # Input 9: 5D float32 array, large k
    m = np.random.randn(1, 2, 3, 2, 2).astype(np.float32)
    k = 5
    axes = (1, 4)
    list_of_inputs.append(
        copy.deepcopy({"m": m, "k": int(k), "axes": tuple(axes)})
    )

    # Input 10: 2D square matrix with negative axes indexing, k=-1
    m = np.eye(5, dtype=np.float32)
    k = -1
    axes = (-1, -2)
    list_of_inputs.append(
        copy.deepcopy({"m": m, "k": int(k), "axes": tuple(axes)})
    )

    # Input 11: 3D int16 array with mixed positive/negative axes
    m = np.random.randint(-100, 100, size=(3, 3, 3), dtype=np.int16)
    k = 2
    axes = (-3, 1)
    list_of_inputs.append(
        copy.deepcopy({"m": m, "k": int(k), "axes": tuple(axes)})
    )

    return list_of_inputs


generated_inputs["jax.numpy.rot90"] = rot90_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.rot90' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.rot90'.")


check_valid('jax.numpy.rot90', generated_inputs['jax.numpy.rot90'], lib="jax", suffix=0)
