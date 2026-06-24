
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy
import jax

def gather_inputs():
    list_of_inputs = []

    dim_nums = jax.lax.GatherDimensionNumbers(
        offset_dims=(1,),
        collapsed_slice_dims=(0,),
        start_index_map=(0,)
    )
    slices = (1, 2)

    for i in range(10):
        if i % 2 == 0:
            operand = (np.random.randn(2, 2) * 10).astype(np.float32)
            start_indices = np.array([[0], [1]], dtype=np.int32)
        else:
            operand = np.random.randint(0, 100, size=(2, 2)).astype(np.int32)
            start_indices = np.array([[1], [0]], dtype=np.int32)

        unique = (i % 3 == 0)
        sorted_idx = (i % 4 == 0)
        mode = "clip" if (i % 2 == 0) else "fill"
        fill_val = True if (i % 5 == 0) else False

        input_dict = {
            "operand": operand,
            "start_indices": start_indices,
            "dimension_numbers": dim_nums,
            "slice_sizes": slices,
            "unique_indices": unique,
            "indices_are_sorted": sorted_idx,
            "mode": mode,
            "fill_value": fill_val
        }
        list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.lax.gather_3"] = gather_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.gather_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.gather_3'.")


check_valid('jax.lax.gather', generated_inputs['jax.lax.gather_3'], lib="jax", suffix=3)
