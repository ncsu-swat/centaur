
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import tensorflow as tf
import copy

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

def tf_feature_column_categorical_column_with_identity_inputs():
    list_of_inputs = []

    input_dict1 = {
        "key": "video_id",
        "num_buckets": 1000,
        "default_value": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict1))

    input_dict2 = {
        "key": "product_id",
        "num_buckets": 500,
        "default_value": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict2))

    input_dict3 = {
        "key": "user_id",
        "num_buckets": 2000,
        "default_value": 5
    }
    list_of_inputs.append(copy.deepcopy(input_dict3))

    input_dict4 = {
        "key": "item_id",
        "num_buckets": 100,
        "default_value": 99
    }
    list_of_inputs.append(copy.deepcopy(input_dict4))

    input_dict5 = {
        "key": "category",
        "num_buckets": 50,
        "default_value": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict5))

    input_dict6 = {
        "key": "ad_id",
        "num_buckets": 10000,
        "default_value": 1000
    }
    list_of_inputs.append(copy.deepcopy(input_dict6))

    input_dict7 = {
        "key": "location_id",
        "num_buckets": 100,
        "default_value": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict7))

    input_dict8 = {
        "key": "query_id",
        "num_buckets": 5000,
        "default_value": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict8))

    input_dict9 = {
        "key": "campaign_id",
        "num_buckets": 200,
        "default_value": 5
    }
    list_of_inputs.append(copy.deepcopy(input_dict9))

    input_dict10 = {
        "key": "device_id",
        "num_buckets": 1000,
        "default_value": 999
    }
    list_of_inputs.append(copy.deepcopy(input_dict10))
    
    return list_of_inputs

generated_inputs["tf.feature_column.categorical_column_with_identity"] = tf_feature_column_categorical_column_with_identity_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.feature_column.categorical_column_with_identity' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.feature_column.categorical_column_with_identity'.")


tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

check_valid('tf.feature_column.categorical_column_with_identity', generated_inputs['tf.feature_column.categorical_column_with_identity'], lib="tf", suffix=0)
