import pandas as pd
import numpy as np
import googletrans
from googletrans import Translator
import argparse
import sys
from tqdm import tqdm
from langdetect import detect

from textual_description import *


def main(attributes_csv_path, paraphrase = False):
    table = pd.read_csv(attributes_csv_path)
    
    # attr_keys = table.keys()
    rows_count = len(table.index)

    # for paraphrases
    translator = Translator()

    descriptions = [0] * len(table)

    for i in tqdm(range(rows_count)):
        attr_record = np.array(table.iloc[i])
        textual_description_object = textual_description(attr_record, translator, paraphrase)
        description = textual_description_object.description

        descriptions[i] = description

        # after each 10 descriptions save        
        if i % 10 == 0:
            table['Description'] = descriptions
            table.to_csv(attributes_csv_path,index=False)

    table['Description'] = descriptions
    table.to_csv(attributes_csv_path,index=False)

    # filtering non-english descriptions
    if paraphrase:
        print('filtering non-english')
        for i in tqdm(range(len(table))):
            try:
                if detect(table.loc[i, "Description"]) != 'en':
                    table = table.drop([i])
                    i -= 1
            except:
                table = table.drop([i])
                i -= 1

        table.to_csv(attributes_csv_path,index=False)


if __name__ == '__main__':
    # arguments parsing
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('-attrcsv', '--path_to_attributes_csv_file', type=str, help='path to attributes csv file', default = './attributes.csv')
    argparser.add_argument('-p', '--paraphrase', type=int, help='paraphrase the generated description or not', default=0)

    args = argparser.parse_args()

    main(args.path_to_attributes_csv_file, bool(args.paraphrase))
  