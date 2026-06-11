
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np

def dynamic_update_slice_inputs():
    return [
        {
            "operand": np.zeros((10,), dtype=np.float32),
            "update": np.ones((3,), dtype=np.float32),
            "start_indices": (0,),
            "allow_negative_indices": (True,)
        },
        {
            "operand": np.zeros((10,), dtype=np.float32),
            "update": np.ones((3,), dtype=np.float32),
            "start_indices": (1,),
            "allow_negative_indices": (True,)
        },
        {
            "operand": np.zeros((10,), dtype=np.float32),
            "update": np.ones((3,), dtype=np.float32),
            "start_indices": (2,),
            "allow_negative_indices": (True,)
        },
        {
            "operand": np.zeros((10,), dtype=np.float32),
            "update": np.ones((3,), dtype=np.float32),
            "start_indices": (3,),
            "allow_negative_indices": (True,)
        },
        {
            "operand": np.zeros((10,), dtype=np.float32),
            "update": np.ones((3,), dtype=np.float32),
            "start_indices": (4,),
            "allow_negative_indices": (True,)
        },
        {
            "operand": np.zeros((10,), dtype=np.float32),
            "update": np.ones((3,), dtype=np.float32),
            "start_indices": (5,),
            "allow_negative_indices": (True,)
        },
        {
            "operand": np.zeros((10,), dtype=np.float32),
            "update": np.ones((3,), dtype=np.float32),
            "start_indices": (6,),
            "allow_negative_indices": (True,)
        },
        {
            "operand": np.zeros((10,), dtype=np.float32),
            "update": np.ones((3,), dtype=np.float32),
            "start_indices": (-1,),
            "allow_negative_indices": (True,)
        },
        {
            "operand": np.zeros((10,), dtype=np.float32),
            "update": np.ones((3,), dtype=np.float32),
            "start_indices": (-2,),
            "allow_negative_indices": (True,)
        },
        {
            "operand": np.zeros((10,), dtype=np.float32),
            "update": np.ones((3,), dtype=np.float32),
            "start_indices": (-3,),
            "allow_negative_indices": (True,)
        }
    ]

generated_inputs["jax.lax.dynamic_update_slice_8"] = dynamic_update_slice_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.dynamic_update_slice_8' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.dynamic_update_slice_8'.")


check_valid('jax.lax.dynamic_update_slice', generated_inputs['jax.lax.dynamic_update_slice_8'], lib="jax", suffix=8)
