import pandas as pd
import numpy as np
from description_class import *
import googletrans
from googletrans import Translator
import argparse
import sys
import progressbar

def main(celeb_a_csv_path, paraphrase = False):
    table = pd.read_csv(celeb_a_csv_path)
    attr_keys = table.keys()
    rows_count = len(table.index)

    # for paraphrases
    languages = ['it', 'ar', 'sv', 'pl', 'sq', 'de', 'zh-cn', 'es', 'ja', 'no', 'ro']
    translator = Translator()

    descriptions = []

    bar = progressbar.ProgressBar(maxval=rows_count, \
                widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    for i in range(rows_count):
        attr_record = np.array(table.iloc[i])
        attributes = attr_keys[attr_record == 1]
        description = textual_description(attributes, languages, translator, paraphrase).description
        descriptions.append(description)
        bar.update(i + 1)
    bar.finish()

    table['Description'] = descriptions
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
Big_Lips
Big_Nose
Black_Hair
Blond_Hair
Brown_Hair
Bushy_Eyebrows
Double_Chin
Goatee
Gray_Hair
High_Cheekbones
Mustache
Narrow_Eyes
No_Beard
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
Chubby
Smiling
'''

''' wearing/pronoun+wears
Eyeglasses
Wearing_Hat
'''

'''puts on
Heavy_Makeup

'''

'''gender
Male
'''

'''adj, pronoun+is
Young
'''