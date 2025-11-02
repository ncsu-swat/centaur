
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy
import numpy as np

def tf_sets_union_inputs():
    list_of_inputs = []
    
    # Input 1, valid
    a = tf.constant([[1, 2, 3], [4, 5, 6]])
    b = tf.constant([[2, 4, -6], [5, 7, 9]])
    validate_indices = True

    input_dict = {
        "a": a,
        "b": b,
        "validate_indices": validate_indices
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2, valid
    a = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    b = tf.constant([[[2, 3], [4, 5]], [[6, 7], [8, 9]]])
    validate_indices = False

    input_dict = {
        "a": a,
        "b": b,
        "validate_indices": validate_indices
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3, valid
    a = tf.sparse.SparseTensor(
        indices=[[0, 0], [0, 1], [1, 0]],
        values=[1, 2, 3],
        dense_shape=[2, 2]
    )
    b = tf.sparse.SparseTensor(
        indices=[[0, 0], [0, 1], [1, 0]],
        values=[2, 4, 5],
        dense_shape=[2, 2]
    )
    validate_indices = True

    input_dict = {
        "a": a,
        "b": b,
        "validate_indices": validate_indices
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4, valid
    a = tf.sparse.SparseTensor(
        indices=[[0, 0], [0, 1], [1, 0]],
        values=[1, 2, 3],
        dense_shape=[2, 2]
    )
    b = tf.sparse.SparseTensor(
        indices=[[0, 0], [0, 1], [1, 0]],
        values=[2, 4, 5],
        dense_shape=[2, 2]
    )
    validate_indices = False

    input_dict = {
        "a": a,
        "b": b,
        "validate_indices": validate_indices
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5, valid - different sizes
    a = tf.sparse.SparseTensor(
        indices=[[0, 0], [0, 1], [1, 0]],
        values=[1, 2, 3],
        dense_shape=[2, 3]
    )
    b = tf.sparse.SparseTensor(
        indices=[[0, 0], [0, 1], [1, 0]],
        values=[2, 4, 5],
        dense_shape=[2, 2]
    )
    validate_indices = True

    input_dict = {
        "a": a,
        "b": b,
        "validate_indices": validate_indices
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6, valid - with negative values
    a = tf.constant([[1, 2, -3], [4, 5, 6]])
    b = tf.constant([[2, 4, -6], [5, 7, 9]])
    validate_indices = True

    input_dict = {
        "a": a,
        "b": b,
        "validate_indices": validate_indices
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7, valid - higher dimensional
    a = tf.constant([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]], [[[9, 10], [11, 12]], [[13, 14], [15, 16]]]])
    b = tf.constant([[[[2, 3], [4, 5]], [[6, 7], [8, 9]]], [[[10, 11], [12, 13]], [[14, 15], [16, 17]]]])
    validate_indices = True

    input_dict = {
        "a": a,
        "b": b,
        "validate_indices": validate_indices
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8, valid - same shape but different values
    a = tf.constant([[1, 2, 3], [4, 5, 6]])
    b = tf.constant([[7, 8, 9], [10, 11, 12]])
    validate_indices = True

    input_dict = {
        "a": a,
        "b": b,
        "validate_indices": validate_indices
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9, valid - sparse with different shapes
    a = tf.sparse.SparseTensor(
        indices=[[0, 0], [0, 1], [1, 0]],
        values=[1, 2, 3],
        dense_shape=[2, 3]
    )
    b = tf.sparse.SparseTensor(
        indices=[[0, 0], [0, 1], [1, 0], [1, 1]],
        values=[2, 4, 5, 6],
        dense_shape=[2, 4]
    )
    validate_indices = True

    input_dict = {
        "a": a,
        "b": b,
        "validate_indices": validate_indices
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10, valid - sparse with overlapping indices
    a = tf.sparse.SparseTensor(
        indices=[[0, 0], [0, 1], [1, 0]],
        values=[1, 2, 3],
        dense_shape=[2, 3]
    )
    b = tf.sparse.SparseTensor(
        indices=[[0, 0], [0, 1], [1, 0]],
        values=[2, 4, 5],
        dense_shape=[2, 3]
    )
    validate_indices = True

    input_dict = {
        "a": a,
        "b": b,
        "validate_indices": validate_indices
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.sets.union"] = tf_sets_union_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.sets.union' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.sets.union'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.sets.union', generated_inputs['tf.sets.union'], lib="tf", suffix=0)
