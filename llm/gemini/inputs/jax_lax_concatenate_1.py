
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

class TensorList(list):
    @property
    def shape(self):
        return self[0].shape if len(self) > 0 else (0,)

    @property
    def dtype(self):
        return self[0].dtype if len(self) > 0 else np.float32

    @property
    def ndim(self):
        return self[0].ndim if len(self) > 0 else 1

def concatenate_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 arrays
    operands = TensorList([
        np.random.randn(5).astype(np.float32),
        np.random.randn(10).astype(np.float32)
    ])
    list_of_inputs.append({"operands": operands, "dimension": 0})

    # Input 2: 2D float32 arrays along axis 0
    operands = TensorList([
        np.random.randn(2, 3).astype(np.float32),
        np.random.randn(4, 3).astype(np.float32)
    ])
    list_of_inputs.append({"operands": operands, "dimension": 0})

    # Input 3: 2D float32 arrays along axis 1
    operands = TensorList([
        np.random.randn(3, 2).astype(np.float32),
        np.random.randn(3, 5).astype(np.float32)
    ])
    list_of_inputs.append({"operands": operands, "dimension": 1})

    # Input 4: 3D float32 arrays along axis 2
    operands = TensorList([
        np.random.randn(2, 2, 1).astype(np.float32),
        np.random.randn(2, 2, 3).astype(np.float32)
    ])
    list_of_inputs.append({"operands": operands, "dimension": 2})

    # Input 5: 1D int32 arrays along axis 0
    operands = TensorList([
        np.arange(3).astype(np.int32),
        np.arange(5).astype(np.int32)
    ])
    list_of_inputs.append({"operands": operands, "dimension": 0})

    # Input 6: 2D float64 arrays with dimension 1
    operands = TensorList([
        np.random.randn(4, 2).astype(np.float64),
        np.random.randn(4, 4).astype(np.float64)
    ])
    list_of_inputs.append({"operands": operands, "dimension": 1})

    # Input 7: Boolean arrays
    operands = TensorList([
        np.ones((2, 2), dtype=bool),
        np.zeros((3, 2), dtype=bool)
    ])
    list_of_inputs.append({"operands": operands, "dimension": 0})

    # Input 8: 4D float32 arrays along axis 3
    operands = TensorList([
        np.random.randn(1, 2, 3, 4).astype(np.float32),
        np.random.randn(1, 2, 3, 2).astype(np.float32)
    ])
    list_of_inputs.append({"operands": operands, "dimension": 3})

    # Input 9: Multiple float32 operands
    operands = TensorList([
        np.random.randn(2, 2).astype(np.float32),
        np.random.randn(3, 2).astype(np.float32),
        np.random.randn(1, 2).astype(np.float32)
    ])
    list_of_inputs.append({"operands": operands, "dimension": 0})

    # Input 10: 5D float32 arrays along axis 1
    operands = TensorList([
        np.random.randn(2, 1, 2, 2, 2).astype(np.float32),
        np.random.randn(2, 3, 2, 2, 2).astype(np.float32)
    ])
    list_of_inputs.append({"operands": operands, "dimension": 1})

    return list_of_inputs

generated_inputs["jax.lax.concatenate_1"] = concatenate_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.concatenate_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.concatenate_1'.")


check_valid('jax.lax.concatenate', generated_inputs['jax.lax.concatenate_1'], lib="jax", suffix=1)
