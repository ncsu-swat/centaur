
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

class MyDimensionNumbers(tuple):
    def __new__(cls, *args, **kwargs):
        if len(args) == 3:
            return super().__new__(cls, args)
        return super().__new__(cls, *args)
    
    def __deepcopy__(self, memo):
        copied_elements = [copy.deepcopy(x, memo) for x in self]
        return MyDimensionNumbers(*copied_elements)
        
    def __reduce__(self):
        return (MyDimensionNumbers, (self[0], self[1], self[2]))
    
    @property
    def offset_dims(self):
        return self[0]
    
    @property
    def collapsed_slice_dims(self):
        return self[1]
    
    @property
    def start_index_map(self):
        return self[2]
    
    @property
    def operand_batch_dims(self):
        return ()
    
    @property
    def start_indices_batch_dims(self):
        return ()

def gather_inputs():
    list_of_inputs = []

    # Input 1: 2D operand, gathering rows
    operand = np.random.randn(5, 5).astype(np.float32)
    start_indices = np.array([[1], [3], [4]], dtype=np.int32)
    dimension_numbers = MyDimensionNumbers(
        (1,),
        (0,),
        (0,)
    )
    slice_sizes = [1, 5]
    input_dict = {
        'operand': operand,
        'start_indices': start_indices,
        'dimension_numbers': dimension_numbers,
        'slice_sizes': slice_sizes,
        'unique_indices': False,
        'indices_are_sorted': True,
        'mode': 'clip',
        'fill_value': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D operand, gathering columns
    operand = np.random.randn(10, 3).astype(np.float32)
    start_indices = np.array([[0], [2]], dtype=np.int32)
    dimension_numbers = MyDimensionNumbers(
        (0,),
        (1,),
        (1,)
    )
    slice_sizes = [10, 1]
    input_dict = {
        'operand': operand,
        'start_indices': start_indices,
        'dimension_numbers': dimension_numbers,
        'slice_sizes': slice_sizes,
        'unique_indices': False,
        'indices_are_sorted': True,
        'mode': 'fill',
        'fill_value': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D integer operand, unique indices
    operand = np.arange(64).reshape(8, 8).astype(np.int32)
    start_indices = np.array([[0], [2], [4], [6]], dtype=np.int32)
    dimension_numbers = MyDimensionNumbers(
        (1,),
        (0,),
        (0,)
    )
    slice_sizes = [1, 8]
    input_dict = {
        'operand': operand,
        'start_indices': start_indices,
        'dimension_numbers': dimension_numbers,
        'slice_sizes': slice_sizes,
        'unique_indices': True,
        'indices_are_sorted': True,
        'mode': 'promise_in_bounds',
        'fill_value': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 4D operand, gathering blocks
    operand = np.random.randn(5, 5, 5, 5).astype(np.float32)
    start_indices = np.array([[0, 0], [1, 2], [3, 4]], dtype=np.int32)
    dimension_numbers = MyDimensionNumbers(
        (1, 2),
        (0, 1),
        (0, 1)
    )
    slice_sizes = [1, 1, 5, 5]
    input_dict = {
        'operand': operand,
        'start_indices': start_indices,
        'dimension_numbers': dimension_numbers,
        'slice_sizes': slice_sizes,
        'unique_indices': False,
        'indices_are_sorted': False,
        'mode': 'clip',
        'fill_value': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D operand, clip mode
    operand = np.arange(30).reshape(3, 10).astype(np.float32)
    start_indices = np.array([[0], [1], [2]], dtype=np.int32)
    dimension_numbers = MyDimensionNumbers(
        (1,),
        (0,),
        (0,)
    )
    slice_sizes = [1, 10]
    input_dict = {
        'operand': operand,
        'start_indices': start_indices,
        'dimension_numbers': dimension_numbers,
        'slice_sizes': slice_sizes,
        'unique_indices': True,
        'indices_are_sorted': True,
        'mode': 'clip',
        'fill_value': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D int64 operand
    operand = np.arange(36).reshape(6, 6).astype(np.int64)
    start_indices = np.array([[5], [2]], dtype=np.int32)
    dimension_numbers = MyDimensionNumbers(
        (1,),
        (0,),
        (0,)
    )
    slice_sizes = [1, 6]
    input_dict = {
        'operand': operand,
        'start_indices': start_indices,
        'dimension_numbers': dimension_numbers,
        'slice_sizes': slice_sizes,
        'unique_indices': False,
        'indices_are_sorted': False,
        'mode': 'fill',
        'fill_value': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 4D operand, drop mode
    operand = np.ones((3, 3, 3, 3), dtype=np.float32)
    start_indices = np.array([[0, 0], [1, 1], [2, 2]], dtype=np.int32)
    dimension_numbers = MyDimensionNumbers(
        (1, 2),
        (0, 1),
        (0, 1)
    )
    slice_sizes = [1, 1, 3, 3]
    input_dict = {
        'operand': operand,
        'start_indices': start_indices,
        'dimension_numbers': dimension_numbers,
        'slice_sizes': slice_sizes,
        'unique_indices': False,
        'indices_are_sorted': False,
        'mode': 'drop',
        'fill_value': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D operand, unique indices sorted
    operand = np.random.randn(7, 11).astype(np.float32)
    start_indices = np.array([[1], [3], [5]], dtype=np.int32)
    dimension_numbers = MyDimensionNumbers(
        (1,),
        (0,),
        (0,)
    )
    slice_sizes = [1, 11]
    input_dict = {
        'operand': operand,
        'start_indices': start_indices,
        'dimension_numbers': dimension_numbers,
        'slice_sizes': slice_sizes,
        'unique_indices': True,
        'indices_are_sorted': True,
        'mode': 'drop',
        'fill_value': False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D int32 operand, clip mode with sorting
    operand = np.arange(16).reshape(4, 4).astype(np.int32)
    start_indices = np.array([[0], [1], [2], [3]], dtype=np.int32)
    dimension_numbers = MyDimensionNumbers(
        (1,),
        (0,),
        (0,)
    )
    slice_sizes = [1, 4]
    input_dict = {
        'operand': operand,
        'start_indices': start_indices,
        'dimension_numbers': dimension_numbers,
        'slice_sizes': slice_sizes,
        'unique_indices': True,
        'indices_are_sorted': True,
        'mode': 'clip',
        'fill_value': True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D operand, gathering columns, fill mode with unique indices
    operand = np.random.randn(9, 5).astype(np.float32)
    start_indices = np.array([[0], [2], [4]], dtype=np.int32)
    dimension_numbers = MyDimensionNumbers(
        (0,),
        (1,),
        (1,)
    )
    slice_sizes = [9, 1]
    input_dict = {
        'operand': operand,
        'start_indices': start_indices,
        'dimension_numbers': dimension_numbers,
        'slice_sizes': slice_sizes,
        'unique_indices': True,
        'indices_are_sorted': True,
        'mode': 'fill',
        'fill_value': False
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
