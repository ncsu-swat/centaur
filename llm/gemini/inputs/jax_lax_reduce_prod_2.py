
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def generate_reduce_prod_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, reducing axis 0
    operand = np.array([1.5, 2.0, -3.0, 4.0], dtype=np.float32)
    axes = (0,)
    list_of_inputs.append({"operand": operand, "axes": axes})

    # Input 2: 2D float32 array, reducing axis 1
    operand = np.random.randn(3, 4).astype(np.float32)
    axes = (1,)
    list_of_inputs.append({"operand": operand, "axes": axes})

    # Input 3: 2D int32 array, reducing all axes (0, 1)
    operand = np.arange(1, 7, dtype=np.int32).reshape((2, 3))
    axes = (0, 1)
    list_of_inputs.append({"operand": operand, "axes": axes})

    # Input 4: 3D float64 array, reducing axes (0, 2)
    operand = np.random.randn(2, 3, 4).astype(np.float64)
    axes = (0, 2)
    list_of_inputs.append({"operand": operand, "axes": axes})

    # Input 5: 1D int32 array with negative values, reducing axis 0
    operand = np.array([-2, 3, -4, 5], dtype=np.int32)
    axes = (0,)
    list_of_inputs.append({"operand": operand, "axes": axes})

    # Input 6: 4D float32 array, reducing axes (2, 3)
    operand = np.random.randn(2, 2, 3, 3).astype(np.float32)
    axes = (2, 3)
    list_of_inputs.append({"operand": operand, "axes": axes})

    # Input 7: 3D int32 array, empty axes tuple
    operand = np.random.randint(-5, 5, size=(2, 2, 2)).astype(np.int32)
    axes = ()
    list_of_inputs.append({"operand": operand, "axes": axes})

    # Input 8: 2D float32 array containing zeros, reducing axis 0
    operand = np.array([[1.0, 0.0], [2.5, 3.0]], dtype=np.float32)
    axes = (0,)
    list_of_inputs.append({"operand": operand, "axes": axes})

    # Input 9: 5D float32 array, reducing axes (1, 3)
    operand = np.random.randn(2, 2, 2, 2, 2).astype(np.float32)
    axes = (1, 3)
    list_of_inputs.append({"operand": operand, "axes": axes})

    # Input 10: 1D int32 array, reducing axis 0
    operand = np.array([10, 20, 30], dtype=np.int32)
    axes = (0,)
    list_of_inputs.append({"operand": operand, "axes": axes})

    return [copy.deepcopy(x) for x in list_of_inputs]

generated_inputs["jax.lax.reduce_prod_2"] = generate_reduce_prod_inputs()

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
