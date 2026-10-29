
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def rev_inputs():
    list_of_inputs = []

    # Input 1: 1D float32, reverse along axis 0
    operand = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    dimensions = (0,)
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "dimensions": dimensions})
    )

    # Input 2: 2D int32, reverse along axis 0
    operand = np.arange(12, dtype=np.int32).reshape(3, 4)
    dimensions = (0,)
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "dimensions": dimensions})
    )

    # Input 3: 2D float64, reverse along axis 1
    operand = np.random.randn(4, 5).astype(np.float64)
    dimensions = (1,)
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "dimensions": dimensions})
    )

    # Input 4: 2D int64, reverse along both axes (0, 1)
    operand = np.arange(-6, 6, dtype=np.int64).reshape(3, 4)
    dimensions = (0, 1)
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "dimensions": dimensions})
    )

    # Input 5: 3D bool, reverse along axes (0, 2)
    operand = np.random.choice([True, False], size=(2, 3, 4))
    dimensions = (0, 2)
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "dimensions": dimensions})
    )

    # Input 6: 3D float32, reverse no axes (empty tuple)
    operand = np.random.randn(2, 2, 2).astype(np.float32)
    dimensions = ()
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "dimensions": dimensions})
    )

    # Input 7: 4D complex64, reverse along axes (1, 3)
    operand = (
        np.random.randn(2, 3, 2, 4) + 1j * np.random.randn(2, 3, 2, 4)
    ).astype(np.complex64)
    dimensions = (1, 3)
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "dimensions": dimensions})
    )

    # Input 8: 1D int8 with negative values, reverse along axis 0
    operand = np.array([-10, -5, 0, 5, 10], dtype=np.int8)
    dimensions = (0,)
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "dimensions": dimensions})
    )

    # Input 9: 5D uint8, reverse along axes (0, 2, 4)
    operand = np.random.randint(0, 256, size=(2, 1, 3, 1, 2), dtype=np.uint8)
    dimensions = (0, 2, 4)
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "dimensions": dimensions})
    )

    # Input 10: 3D float16, reverse along axis 1
    operand = np.random.randn(3, 5, 2).astype(np.float16)
    dimensions = (1,)
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "dimensions": dimensions})
    )

    # Input 11: 2D float32 with inf and nan values, reverse along axis (0, 1)
    operand = np.array(
        [[np.nan, np.inf], [-np.inf, 0.0]], dtype=np.float32
    )
    dimensions = (0, 1)
    list_of_inputs.append(
        copy.deepcopy({"operand": operand, "dimensions": dimensions})
    )

    return list_of_inputs


generated_inputs["jax.lax.rev_2"] = rev_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.rev_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.rev_2'.")


check_valid('jax.lax.rev', generated_inputs['jax.lax.rev_2'], lib="jax", suffix=2)
