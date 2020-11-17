import pandas as pd
import numpy as np
from description_class import *
import googletrans
from googletrans import Translator
import argparse
import sys
from tqdm import tqdm
from langdetect import detect

def main(celeb_a_csv_path, paraphrase = False):
    table = pd.read_csv(celeb_a_csv_path)
    table = table.replace(-1, 0)

    # drop unneeded attributes
    unneeded_columns = ['5_o_Clock_Shadow',
                        'Blurry',
                        'Eyeglasses',
                        'Mouth_Slightly_Open',
                        'Wearing_Earrings',
                        'Wearing_Hat',
                        'Wearing_Necklace',
                        'Wearing_Necktie'
    ]
    for col in unneeded_columns:
        table.drop(col, axis='columns', inplace=True, errors='ignore')


    attr_keys = table.keys()
    rows_count = len(table.index)

    # for paraphrases
    languages = ['it', 'ar', 'sv', 'pl', 'sq', 'de', 'zh-cn', 'es', 'ja', 'no', 'ro']
    translator = Translator()

    descriptions = []

    pbar = tqdm(total=rows_count)
    for i in range(rows_count):
        attr_record = np.array(table.iloc[i])
        positive_attributes = np.array(attr_keys[attr_record == 1])
        textual_description_object = textual_description(positive_attributes, languages, translator, paraphrase)
        description = textual_description_object.description
        negative_attributes = np.array(textual_description_object.added_antonyms_attributes)
        unspecified_attributes = list(set(attr_keys[1:]) - set( np.union1d(positive_attributes, negative_attributes)))

        for neg_attr in negative_attributes:
            table.at[i, neg_attr] = -1

        descriptions.append(description)
        pbar.update(1)
    pbar.close()

    table['Description'] = descriptions

    # filtering non-english descriptions
    if paraphrase:
        print('filtering non-english')
        pbar = tqdm(total=len(table))
        for i in range(len(table)):
            if detect(table.loc[i, "Description"]) != 'en':
                table = table.drop([i])
                i -= 1
            pbar.update(1)
        pbar.close()

    table.to_csv('CelebA_with_textual_descriptions.csv',index=False)


if __name__ == '__main__':
    # arguments parsing
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('-celebcsv', '--path_to_celeba_attribute_csv_file', type=str, help='path to CelebA attribute csv file', default = './list_attr_celeba.csv')
    argparser.add_argument('-p', '--paraphrase', type=int, help='paraphrase the generated description or not', default=0)

    args = argparser.parse_args()

    main(args.path_to_celeba_attribute_csv_file, bool(args.paraphrase))
        

''' pronoun+has, with
Arched_Eyebrows
Bangs
Big_Lips //
Big_Nose //
Black_Hair
Blond_Hair
Brown_Hair
Bushy_Eyebrows
Double_Chin
Goatee
Gray_Hair
High_Cheekbones
Mustache //
Narrow_Eyes //
No_Beard // 
Oval_Face
Pale_Skin
Rosy_Cheeks
Sideburns
Straight_Hair
Wavy_Hair
'''

''' pronoun+is,adj
Attractive
Bald
Chubby //
Smiling
'''


'''gender
Male
'''

'''adj, pronoun+is
Young
'''

''' extra added
Bags_Under_Eyes
Heavy_Makeup
Pointy_Nose
Receding_Hairline
Wearing_Lipstick
'''