
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

class MyScatterDimensionNumbers:
    def __init__(self, update_window_dims, inserted_window_dims, scatter_dims_to_operand_dims):
        self.update_window_dims = update_window_dims
        self.inserted_window_dims = inserted_window_dims
        self.scatter_dims_to_operand_dims = scatter_dims_to_operand_dims
    def __len__(self):
        return 0

def scatter_inputs():
    list_of_inputs = []

    # 1. 2D operand, K=1, index axis 0
    operand = np.ones((3, 3), dtype=np.float32)
    scatter_indices = np.array([[0], [2]], dtype=np.int32)
    updates = np.array([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]], dtype=np.float32)
    dn = MyScatterDimensionNumbers((1,), (0,), (0,))
    list_of_inputs.append({
        'operand': operand,
        'scatter_indices': scatter_indices,
        'updates': updates,
        'dimension_numbers': dn,
        'indices_are_sorted': True,
        'unique_indices': True,
        'mode': 'clip'
    })

    # 2. 2D operand, K=1, index axis 1
    operand = np.zeros((4, 3), dtype=np.float32)
    scatter_indices = np.array([[2]], dtype=np.int32)
    updates = np.ones((1, 4), dtype=np.float32)
    dn = MyScatterDimensionNumbers((1,), (1,), (1,))
    list_of_inputs.append({
        'operand': operand,
        'scatter_indices': scatter_indices,
        'updates': updates,
        'dimension_numbers': dn,
        'indices_are_sorted': False,
        'unique_indices': True,
        'mode': 'drop'
    })

    # 3. 2D operand, K=1, index axis 0, non-unique indices
    operand = np.arange(6, dtype=np.float64).reshape(3, 2)
    scatter_indices = np.array([[1], [1]], dtype=np.int64)
    updates = np.zeros((2, 2), dtype=np.float64)
    dn = MyScatterDimensionNumbers((1,), (0,), (0,))
    list_of_inputs.append({
        'operand': operand,
        'scatter_indices': scatter_indices,
        'updates': updates,
        'dimension_numbers': dn,
        'indices_are_sorted': False,
        'unique_indices': False,
        'mode': 'clip'
    })

    # 4. 4D operand, K=2, index axes 0 & 1
    operand = np.ones((2, 2, 3, 2), dtype=np.float32)
    scatter_indices = np.array([[0, 1], [1, 1]], dtype=np.int32)
    updates = np.zeros((2, 3, 2), dtype=np.float32)
    dn = MyScatterDimensionNumbers((1, 2), (0, 1), (0, 1))
    list_of_inputs.append({
        'operand': operand,
        'scatter_indices': scatter_indices,
        'updates': updates,
        'dimension_numbers': dn,
        'indices_are_sorted': True,
        'unique_indices': True,
        'mode': 'clip'
    })

    # 5. 4D operand, K=2, index axes 2 & 3
    operand = np.zeros((3, 2, 2, 2), dtype=np.float32)
    scatter_indices = np.array([[1, 0], [0, 1]], dtype=np.int32)
    updates = np.ones((2, 3, 2), dtype=np.float32)
    dn = MyScatterDimensionNumbers((1, 2), (2, 3), (2, 3))
    list_of_inputs.append({
        'operand': operand,
        'scatter_indices': scatter_indices,
        'updates': updates,
        'dimension_numbers': dn,
        'indices_are_sorted': False,
        'unique_indices': False,
        'mode': 'drop'
    })

    # 6. 2D operand, K=1, index axis 0, int32
    operand = np.arange(18, dtype=np.int32).reshape(6, 3)
    scatter_indices = np.array([[1], [4]], dtype=np.int32)
    updates = -np.ones((2, 3), dtype=np.int32)
    dn = MyScatterDimensionNumbers((1,), (0,), (0,))
    list_of_inputs.append({
        'operand': operand,
        'scatter_indices': scatter_indices,
        'updates': updates,
        'dimension_numbers': dn,
        'indices_are_sorted': True,
        'unique_indices': True,
        'mode': 'clip'
    })

    # 7. 2D operand, K=1, index axis 1, int64
    operand = np.zeros((3, 4), dtype=np.int64)
    scatter_indices = np.array([[0], [2]], dtype=np.int64)
    updates = np.ones((2, 3), dtype=np.int64)
    dn = MyScatterDimensionNumbers((1,), (1,), (1,))
    list_of_inputs.append({
        'operand': operand,
        'scatter_indices': scatter_indices,
        'updates': updates,
        'dimension_numbers': dn,
        'indices_are_sorted': True,
        'unique_indices': True,
        'mode': 'drop'
    })

    # 8. 4D operand, K=2, index axes 0 & 2
    operand = np.zeros((3, 2, 2, 3), dtype=np.int32)
    scatter_indices = np.array([[1, 1], [2, 0]], dtype=np.int32)
    updates = np.ones((2, 2, 3), dtype=np.int32)
    dn = MyScatterDimensionNumbers((1, 2), (0, 2), (0, 2))
    list_of_inputs.append({
        'operand': operand,
        'scatter_indices': scatter_indices,
        'updates': updates,
        'dimension_numbers': dn,
        'indices_are_sorted': False,
        'unique_indices': True,
        'mode': 'clip'
    })

    # 9. 2D operand, K=1, index axis 0, out of bounds clipping
    operand = np.zeros((4, 2), dtype=np.float32)
    scatter_indices = np.array([[1], [10]], dtype=np.int32)
    updates = np.array([[1.0, 1.0], [2.0, 2.0]], dtype=np.float32)
    dn = MyScatterDimensionNumbers((1,), (0,), (0,))
    list_of_inputs.append({
        'operand': operand,
        'scatter_indices': scatter_indices,
        'updates': updates,
        'dimension_numbers': dn,
        'indices_are_sorted': False,
        'unique_indices': False,
        'mode': 'clip'
    })

    # 10. 4D operand, K=2, index axes 1 & 3, float64
    operand = np.zeros((2, 3, 2, 4), dtype=np.float64)
    scatter_indices = np.array([[1, 2], [0, 1]], dtype=np.int64)
    updates = np.ones((2, 2, 2), dtype=np.float64)
    dn = MyScatterDimensionNumbers((1, 2), (1, 3), (1, 3))
    list_of_inputs.append({
        'operand': operand,
        'scatter_indices': scatter_indices,
        'updates': updates,
        'dimension_numbers': dn,
        'indices_are_sorted': False,
        'unique_indices': False,
        'mode': 'drop'
    })

    return list_of_inputs

generated_inputs["jax.lax.scatter"] = scatter_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.scatter' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.scatter'.")


check_valid('jax.lax.scatter', generated_inputs['jax.lax.scatter'], lib="jax", suffix=0)
