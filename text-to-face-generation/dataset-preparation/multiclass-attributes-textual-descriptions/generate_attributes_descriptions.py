import pandas as pd
import numpy as np
import argparse
import sys
import os
from tqdm import tqdm
from textual_description import textual_description

def main(attributes_csv_path, paraphrase = False):
    table = pd.read_csv(attributes_csv_path)

    rows_count = len(table.index)
    
    descriptions = list(table['Description'].values)
    ages = list(table['Old'].values)

    if os.path.exists('last_desc_generated.npy'):
        start = np.load('last_desc_generated.npy')[0]
    else:
        start = 0

    for i in tqdm(range(start,rows_count)):
        attr_record = table.iloc[i]
        textual_description_object = textual_description(attr_record, paraphrase)
        description = textual_description_object.description

        descriptions[i] = description
        ages[i] = textual_description_object.age_attribute

        # after each 1000 descriptions save        
        if i % 10 == 0:
            table['Description'] = descriptions
            table['Old'] = ages
            table.to_csv(attributes_csv_path,index=False)
            np.save('last_desc_generated.npy',[i])



    table['Description'] = descriptions
    table['Old'] = ages
    table.to_csv(attributes_csv_path,index=False)

    # delete last_desc_generated.npy file
    os.remove('last_desc_generated.npy') 


if __name__ == '__main__':
    # arguments parsing
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('-attrcsv', '--path_to_attributes_csv_file', type=str, help='path to attributes csv file', default = './attributes.csv')
    argparser.add_argument("--do_paraphrase", action='store_true')
    args = argparser.parse_args()
    main(args.path_to_attributes_csv_file, args.do_paraphrase)
  