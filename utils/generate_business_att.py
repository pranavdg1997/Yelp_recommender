# Cleans all the attributes of businesses and converts into a valid json format.
#Input:  business.json
#Output: business_attributes.json

import pandas as pd
import numpy as np
import json

def jsonable(string_text):
    attr_str = string_text 
    attr_str1 = attr_str.replace('"{', '{')
    attr_str2 = attr_str1.replace('}"', '}')
    attr_str3 = attr_str2.replace('"u\'', '\'')
    attr_str4 = attr_str3.replace('"', '')
    attr_str5 = attr_str4.replace('\'', '"')
    attr_str6 = attr_str5.replace('False', '"False"')
    attr_str7 = attr_str6.replace('True', '"True"')
    attr_str8 = attr_str7.replace('""', '"')    
    return attr_str8

def transform_to_dict(json_data):
    main_dict = dict()
    all_keys = list(json_data.keys())
    for key in all_keys:
        if type(json_data[key]) == type(dict()):
            key1 = list(json_data[key].keys())
            for key2 in key1:
                main_dict[key2] = json_data[key][key2]
        else:
            main_dict[key] = json_data[key]
    return main_dict

if __name__=='__main__':
    input_filename = str(input("Input data file name: task1_data.csv"))
    output_filename = str(input("input output file name: data_new.json"))
    
    #Use task1_data.csv
    data = pd.read_csv(input_filename)

    pd.set_option('display.max_columns', None)

    attributes = data[['business_id', 'attributes']]

    data1 = attributes.drop_duplicates(subset='business_id', keep='first', inplace=False)

    data1.reset_index(inplace=True)
    
    num_rows = data1.shape[0]
    for i in range(num_rows):
        clean_str = jsonable(data1['attributes'][i])
        data1.at[i, 'attributes'] = clean_str
        
    for i in range(data1.shape[0]):
        json_data = json.loads(data1['attributes'][i])
        transformed_dict = transform_to_dict(json_data)
        transformed_dict['business_id'] = data1['business_id'][i]
    
        with open(output_filename, 'a', encoding='utf-8') as f:
            f.write(json.dumps(transformed_dict) + '\n' )