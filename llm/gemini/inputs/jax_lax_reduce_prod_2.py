
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def reduce_prod_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, reduce along axis 0
    operand = np.array([1.5, -2.0, 3.0, -0.5], dtype=np.float32)
    axes = (0,)
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "axes": axes})
    )

    # Input 2: 2D int32 array with negative values, reduce along axis 1
    operand = np.array([[1, -2, 3], [-4, 5, -6]], dtype=np.int32)
    axes = (1,)
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "axes": axes})
    )

    # Input 3: 2D float64 array, reduce along axis 0
    operand = np.random.randn(4, 5).astype(np.float64)
    axes = (0,)
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "axes": axes})
    )

    # Input 4: 3D float32 array, reduce along multiple axes (0, 2)
    operand = np.random.randn(2, 3, 4).astype(np.float32)
    axes = (0, 2)
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "axes": axes})
    )

    # Input 5: 3D int64 array, reduce along axis 1
    operand = np.random.randint(-5, 5, size=(3, 2, 4)).astype(np.int64)
    axes = (1,)
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "axes": axes})
    )

    # Input 6: 4D float32 array, reduce along intermediate axes (1, 2)
    operand = np.random.randn(2, 3, 3, 2).astype(np.float32)
    axes = (1, 2)
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "axes": axes})
    )

    # Input 7: 2D complex64 array, full reduction along all axes (0, 1)
    operand = (np.random.randn(3, 3) + 1j * np.random.randn(3, 3)).astype(
        np.complex64
    )
    axes = (0, 1)
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "axes": axes})
    )

    # Input 8: 0D scalar array, empty axes tuple
    operand = np.array(3.14, dtype=np.float32)
    axes = ()
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "axes": axes})
    )

    # Input 9: 2D int16 array, empty axes reduction (no axes reduced)
    operand = np.array([[2, 3], [-1, 4]], dtype=np.int16)
    axes = ()
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "axes": axes})
    )

    # Input 10: 5D float32 array, reduction over axes (0, 3, 4)
    operand = np.random.randn(2, 2, 2, 2, 2).astype(np.float32)
    axes = (0, 3, 4)
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "axes": axes})
    )

    # Input 11: 3D float16 array, complete reduction
    operand = np.random.randn(2, 4, 3).astype(np.float16)
    axes = (0, 1, 2)
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "axes": axes})
    )

    return list_of_inputs


generated_inputs["jax.lax.reduce_prod_2"] = reduce_prod_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.reduce_prod_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.reduce_prod_2'.")


check_valid('jax.lax.reduce_prod', generated_inputs['jax.lax.reduce_prod_2'], lib="jax", suffix=2)
