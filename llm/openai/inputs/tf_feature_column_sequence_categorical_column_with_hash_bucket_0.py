
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy

def sequence_categorical_column_with_hash_bucket_inputs():
    list_of_inputs = []

    key1 = "tokens1"
    hash_bucket_size1 = 100
    dtype1 = tf.string
    input_dict1 = {
        "key": key1,
        "hash_bucket_size": hash_bucket_size1,
        "dtype": dtype1
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    key2 = "tokens2"
    hash_bucket_size2 = 500
    dtype2 = tf.int32
    input_dict2 = {
        "key": key2,
        "hash_bucket_size": hash_bucket_size2,
        "dtype": dtype2
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    key3 = "tokens3"
    hash_bucket_size3 = 200
    dtype3 = tf.string
    input_dict3 = {
        "key": key3,
        "hash_bucket_size": hash_bucket_size3,
        "dtype": dtype3
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    key4 = "tokens4"
    hash_bucket_size4 = 1000
    dtype4 = tf.int64
    input_dict4 = {
        "key": key4,
        "hash_bucket_size": hash_bucket_size4,
        "dtype": dtype4
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    key5 = "tokens5"
    hash_bucket_size5 = 10
    dtype5 = tf.string
    input_dict5 = {
        "key": key5,
        "hash_bucket_size": hash_bucket_size5,
        "dtype": dtype5
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    key6 = "tokens6"
    hash_bucket_size6 = 5
    dtype6 = tf.int32
    input_dict6 = {
        "key": key6,
        "hash_bucket_size": hash_bucket_size6,
        "dtype": dtype6
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    key7 = "tokens7"
    hash_bucket_size7 = 250
    dtype7 = tf.string
    input_dict7 = {
        "key": key7,
        "hash_bucket_size": hash_bucket_size7,
        "dtype": dtype7
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    key8 = "tokens8"
    hash_bucket_size8 = 150
    dtype8 = tf.int64
    input_dict8 = {
        "key": key8,
        "hash_bucket_size": hash_bucket_size8,
        "dtype": dtype8
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))

    key9 = "tokens9"
    hash_bucket_size9 = 300
    dtype9 = tf.string
    input_dict9 = {
        "key": key9,
        "hash_bucket_size": hash_bucket_size9,
        "dtype": dtype9
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))

    key10 = "tokens10"
    hash_bucket_size10 = 75
    dtype10 = tf.int32
    input_dict10 = {
        "key": key10,
        "hash_bucket_size": hash_bucket_size10,
        "dtype": dtype10
    }
    list_of_inputs.append(copy.deepcopy(input_dict10))

    return list_of_inputs

generated_inputs["tf.feature_column.sequence_categorical_column_with_hash_bucket"] = sequence_categorical_column_with_hash_bucket_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.feature_column.sequence_categorical_column_with_hash_bucket' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.feature_column.sequence_categorical_column_with_hash_bucket'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.feature_column.sequence_categorical_column_with_hash_bucket', generated_inputs['tf.feature_column.sequence_categorical_column_with_hash_bucket'], lib="tf", suffix=0)
