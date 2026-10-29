
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def pad_inputs():
    list_of_inputs = []

    # Input 1: 1D array with low and high padding
    operand = np.array([1, 2, 3, 4], dtype=np.int32)
    padding_value = np.array(0, dtype=np.int32)
    padding_config = [(2, 3, 0)]
    list_of_inputs.append(
        copy.deepcopy(
            {
                "operand": operand,
                "padding_value": padding_value,
                "padding_config": padding_config,
            }
        )
    )

    # Input 2: 1D array with interior padding
    operand = np.array([1, 2, 3, 4], dtype=np.float32)
    padding_value = np.array(0.0, dtype=np.float32)
    padding_config = [(0, 0, 1)]
    list_of_inputs.append(
        copy.deepcopy(
            {
                "operand": operand,
                "padding_value": padding_value,
                "padding_config": padding_config,
            }
        )
    )

    # Input 3: 2D array with low, high, and no interior padding
    operand = np.arange(6, dtype=np.float64).reshape((2, 3))
    padding_value = np.array(-1.0, dtype=np.float64)
    padding_config = [(2, 2, 0), (2, 2, 0)]
    list_of_inputs.append(
        copy.deepcopy(
            {
                "operand": operand,
                "padding_value": padding_value,
                "padding_config": padding_config,
            }
        )
    )

    # Input 4: 1D array with negative padding (clipping/cropping)
    operand = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    padding_value = np.array(0, dtype=np.int32)
    padding_config = [(-1, -2, 0)]
    list_of_inputs.append(
        copy.deepcopy(
            {
                "operand": operand,
                "padding_value": padding_value,
                "padding_config": padding_config,
            }
        )
    )

    # Input 5: 3D array with various paddings across dimensions
    operand = np.ones((2, 3, 4), dtype=np.float32)
    padding_value = np.array(99.0, dtype=np.float32)
    padding_config = [(1, 0, 0), (0, 2, 1), (1, 1, 0)]
    list_of_inputs.append(
        copy.deepcopy(
            {
                "operand": operand,
                "padding_value": padding_value,
                "padding_config": padding_config,
            }
        )
    )

    # Input 6: Boolean array with padding
    operand = np.array([[True, False], [False, True]], dtype=np.bool_)
    padding_value = np.array(False, dtype=np.bool_)
    padding_config = [(1, 1, 0), (0, 1, 0)]
    list_of_inputs.append(
        copy.deepcopy(
            {
                "operand": operand,
                "padding_value": padding_value,
                "padding_config": padding_config,
            }
        )
    )

    # Input 7: Complex array with interior and edge padding
    operand = np.array([1 + 2j, 3 + 4j], dtype=np.complex64)
    padding_value = np.array(0 + 0j, dtype=np.complex64)
    padding_config = [(1, 1, 2)]
    list_of_inputs.append(
        copy.deepcopy(
            {
                "operand": operand,
                "padding_value": padding_value,
                "padding_config": padding_config,
            }
        )
    )

    # Input 8: 4D array with zero padding configuration
    operand = np.random.randn(2, 2, 2, 2).astype(np.float32)
    padding_value = np.array(0.0, dtype=np.float32)
    padding_config = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 1)]
    list_of_inputs.append(
        copy.deepcopy(
            {
                "operand": operand,
                "padding_value": padding_value,
                "padding_config": padding_config,
            }
        )
    )

    # Input 9: Int8 array with negative and positive padding
    operand = np.arange(10, dtype=np.int8).reshape(5, 2)
    padding_value = np.array(-5, dtype=np.int8)
    padding_config = [(-1, 2, 0), (1, -1, 0)]
    list_of_inputs.append(
        copy.deepcopy(
            {
                "operand": operand,
                "padding_value": padding_value,
                "padding_config": padding_config,
            }
        )
    )

    # Input 10: 0D array (scalar tensor) with empty padding config
    operand = np.array(42, dtype=np.int64)
    padding_value = np.array(0, dtype=np.int64)
    padding_config = []
    list_of_inputs.append(
        copy.deepcopy(
            {
                "operand": operand,
                "padding_value": padding_value,
                "padding_config": padding_config,
            }
        )
    )

    return list_of_inputs


generated_inputs["jax.lax.pad_1"] = pad_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.pad_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.pad_1'.")


check_valid('jax.lax.pad', generated_inputs['jax.lax.pad_1'], lib="jax", suffix=1)
