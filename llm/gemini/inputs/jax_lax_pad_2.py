
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def pad_inputs():
    list_of_inputs = []

    # Input 1: 1D int32 array with standard positive low/high padding
    operand = np.array([1, 2, 3, 4], dtype=np.int32)
    padding_value = np.array(0, dtype=np.int32)
    padding_config = ((2, 3, 0),)
    list_of_inputs.append(
        {
            "operand": operand,
            "padding_value": padding_value,
            "padding_config": padding_config,
        }
    )

    # Input 2: 1D float32 array with interior dilation padding
    operand = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    padding_value = np.array(-1.0, dtype=np.float32)
    padding_config = ((0, 0, 1),)
    list_of_inputs.append(
        {
            "operand": operand,
            "padding_value": padding_value,
            "padding_config": padding_config,
        }
    )

    # Input 3: 1D int64 array with negative padding (edge cropping)
    operand = np.array([10, 20, 30, 40, 50, 60], dtype=np.int64)
    padding_value = np.array(0, dtype=np.int64)
    padding_config = ((-1, -2, 0),)
    list_of_inputs.append(
        {
            "operand": operand,
            "padding_value": padding_value,
            "padding_config": padding_config,
        }
    )

    # Input 4: 2D float32 array with asymmetric low and high padding
    operand = np.ones((3, 4), dtype=np.float32)
    padding_value = np.array(0.0, dtype=np.float32)
    padding_config = ((1, 2, 0), (2, 1, 0))
    list_of_inputs.append(
        {
            "operand": operand,
            "padding_value": padding_value,
            "padding_config": padding_config,
        }
    )

    # Input 5: 2D int32 array with interior and exterior padding
    operand = np.arange(6, dtype=np.int32).reshape(2, 3)
    padding_value = np.array(-5, dtype=np.int32)
    padding_config = ((1, 1, 1), (0, 2, 1))
    list_of_inputs.append(
        {
            "operand": operand,
            "padding_value": padding_value,
            "padding_config": padding_config,
        }
    )

    # Input 6: 2D float64 array with negative (cropping) and positive padding
    operand = np.arange(20, dtype=np.float64).reshape(4, 5)
    padding_value = np.array(3.14, dtype=np.float64)
    padding_config = ((-1, 1, 0), (2, -2, 0))
    list_of_inputs.append(
        {
            "operand": operand,
            "padding_value": padding_value,
            "padding_config": padding_config,
        }
    )

    # Input 7: 3D float32 array with mixed zero and non-zero padding
    operand = np.random.randn(2, 3, 4).astype(np.float32)
    padding_value = np.array(0.0, dtype=np.float32)
    padding_config = ((0, 0, 0), (1, 1, 0), (2, 0, 0))
    list_of_inputs.append(
        {
            "operand": operand,
            "padding_value": padding_value,
            "padding_config": padding_config,
        }
    )

    # Input 8: 3D int16 array with interior dilation on one dimension
    operand = np.arange(12, dtype=np.int16).reshape(2, 2, 3)
    padding_value = np.array(99, dtype=np.int16)
    padding_config = ((0, 0, 1), (1, 0, 0), (0, 1, 0))
    list_of_inputs.append(
        {
            "operand": operand,
            "padding_value": padding_value,
            "padding_config": padding_config,
        }
    )

    # Input 9: 4D float32 array (batch, channel, height, width) with spatial padding
    operand = np.random.randn(1, 3, 8, 8).astype(np.float32)
    padding_value = np.array(0.0, dtype=np.float32)
    padding_config = ((0, 0, 0), (0, 0, 0), (2, 2, 0), (2, 2, 0))
    list_of_inputs.append(
        {
            "operand": operand,
            "padding_value": padding_value,
            "padding_config": padding_config,
        }
    )

    # Input 10: 1D boolean array with boolean padding value
    operand = np.array([True, False, True], dtype=np.bool_)
    padding_value = np.array(False, dtype=np.bool_)
    padding_config = ((2, 1, 0),)
    list_of_inputs.append(
        {
            "operand": operand,
            "padding_value": padding_value,
            "padding_config": padding_config,
        }
    )

    # Input 11: 2D complex64 array with complex padding value
    operand = np.array([[1 + 2j, 3 + 4j], [5 + 6j, 7 + 8j]], dtype=np.complex64)
    padding_value = np.array(0 + 0j, dtype=np.complex64)
    padding_config = ((1, 1, 0), (0, 1, 0))
    list_of_inputs.append(
        {
            "operand": operand,
            "padding_value": padding_value,
            "padding_config": padding_config,
        }
    )

    return list_of_inputs


generated_inputs["jax.lax.pad_2"] = pad_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.pad_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.pad_2'.")


check_valid('jax.lax.pad', generated_inputs['jax.lax.pad_2'], lib="jax", suffix=2)
