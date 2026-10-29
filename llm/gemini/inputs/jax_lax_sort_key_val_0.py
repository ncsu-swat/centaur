
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def sort_key_val_inputs():
    list_of_inputs = []

    # Input 1: 1D arrays, float32 keys and int32 values, default dimension
    keys = np.array([3.0, 1.0, 2.0, 5.0, 4.0], dtype=np.float32)
    values = np.array([10, 20, 30, 40, 50], dtype=np.int32)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "keys": keys,
                "values": values,
                "dimension": -1,
                "is_stable": True,
            }
        )
    )

    # Input 2: 1D arrays with negative values, int32 keys and float32 values, dimension=0, unstable
    keys = np.array([-5, 10, -20, 0, 15], dtype=np.int32)
    values = np.array([1.5, -2.5, 3.5, -4.5, 5.5], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "keys": keys,
                "values": values,
                "dimension": 0,
                "is_stable": False,
            }
        )
    )

    # Input 3: 2D arrays, sort along columns (dimension=1)
    keys = np.array(
        [[4.0, 2.0, 1.0], [9.0, 0.0, -3.0], [5.0, 8.0, 2.0]], dtype=np.float32
    )
    values = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int32)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "keys": keys,
                "values": values,
                "dimension": 1,
                "is_stable": True,
            }
        )
    )

    # Input 4: 2D arrays, sort along rows (dimension=0)
    keys = np.array([[10, 50, 30], [20, 40, 60], [5, 45, 15]], dtype=np.int64)
    values = np.array(
        [[100, 200, 300], [400, 500, 600], [700, 800, 900]], dtype=np.int64
    )
    list_of_inputs.append(
        copy.deepcopy(
            {
                "keys": keys,
                "values": values,
                "dimension": 0,
                "is_stable": False,
            }
        )
    )

    # Input 5: 2D float64 keys, boolean values, dimension=-1
    keys = np.array([[1.5, 0.5], [3.2, 2.1], [-1.0, -0.5]], dtype=np.float64)
    values = np.array([[True, False], [False, True], [True, True]], dtype=np.bool_)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "keys": keys,
                "values": values,
                "dimension": -1,
                "is_stable": True,
            }
        )
    )

    # Input 6: 3D float32 keys, int32 values, sort along last dimension (dimension=2)
    keys = np.random.randn(2, 3, 4).astype(np.float32)
    values = np.arange(24, dtype=np.int32).reshape(2, 3, 4)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "keys": keys,
                "values": values,
                "dimension": 2,
                "is_stable": True,
            }
        )
    )

    # Input 7: 3D int32 keys, float32 values, sort along middle dimension (dimension=1)
    keys = np.random.randint(-50, 50, size=(2, 4, 3)).astype(np.int32)
    values = np.random.randn(2, 4, 3).astype(np.float32)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "keys": keys,
                "values": values,
                "dimension": 1,
                "is_stable": False,
            }
        )
    )

    # Input 8: 3D float32 keys, sort along first dimension using negative indexing (dimension=-3)
    keys = np.random.randn(4, 2, 2).astype(np.float32)
    values = np.random.randn(4, 2, 2).astype(np.float32)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "keys": keys,
                "values": values,
                "dimension": -3,
                "is_stable": True,
            }
        )
    )

    # Input 9: 4D arrays, sort along last dimension (dimension=3)
    keys = np.random.randn(2, 2, 3, 4).astype(np.float32)
    values = np.random.randint(0, 100, size=(2, 2, 3, 4)).astype(np.int64)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "keys": keys,
                "values": values,
                "dimension": 3,
                "is_stable": False,
            }
        )
    )

    # Input 10: 1D int32 keys and float32 values, dimension=0
    keys = np.array([400, 100, 300, 200], dtype=np.int32)
    values = np.array([1.1, 2.2, 3.3, 4.4], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "keys": keys,
                "values": values,
                "dimension": 0,
                "is_stable": True,
            }
        )
    )

    # Input 11: 2D arrays with duplicate keys to test stable sort
    keys = np.array([[2, 1, 2, 1], [3, 3, 0, 0]], dtype=np.int32)
    values = np.array([[10, 20, 30, 40], [50, 60, 70, 80]], dtype=np.int32)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "keys": keys,
                "values": values,
                "dimension": -1,
                "is_stable": True,
            }
        )
    )

    return list_of_inputs


generated_inputs["jax.lax.sort_key_val"] = sort_key_val_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.sort_key_val' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.sort_key_val'.")


check_valid('jax.lax.sort_key_val', generated_inputs['jax.lax.sort_key_val'], lib="jax", suffix=0)
