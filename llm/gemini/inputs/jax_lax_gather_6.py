
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

class CustomGatherDimensionNumbers(tuple):
    def __new__(cls, offset_dims, collapsed_slice_dims, start_index_map):
        return super().__new__(cls, (offset_dims, collapsed_slice_dims, start_index_map))
    
    @property
    def offset_dims(self):
        return self[0]
    
    @property
    def collapsed_slice_dims(self):
        return self[1]
    
    @property
    def start_index_map(self):
        return self[2]

def gather_inputs():
    list_of_inputs = []

    # We generate 10 valid and distinct inputs with varying shapes and dtypes.
    # We use CustomGatherDimensionNumbers to satisfy both type checks and attributes.
    for i in range(10):
        size = 4 + (i % 3) * 2  # sizes: 4, 6, 8
        operand = np.arange(size * size, dtype=np.float32).reshape(size, size) + i
        if i % 2 == 0:
            operand = operand.astype(np.float64)
        else:
            operand = operand.astype(np.int32)

        start_indices = np.array([[0], [1]], dtype=np.int32)
        dimension_numbers = CustomGatherDimensionNumbers(offset_dims=(1,), collapsed_slice_dims=(0,), start_index_map=(0,))
        
        input_dict = {
            "operand": operand,
            "start_indices": start_indices,
            "dimension_numbers": dimension_numbers,
            "slice_sizes": [1, size // 2],
            "unique_indices": (i % 2 == 0),
            "indices_are_sorted": (i % 2 == 0),
            "mode": "clip" if i % 2 == 0 else "fill",
            "fill_value": (i % 2 == 0)
        }
        list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.lax.gather_6"] = gather_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.gather_6' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.gather_6'.")


check_valid('jax.lax.gather', generated_inputs['jax.lax.gather_6'], lib="jax", suffix=6)
