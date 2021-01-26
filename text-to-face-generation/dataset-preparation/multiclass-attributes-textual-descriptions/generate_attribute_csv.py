
import random
import pandas as pd
from tqdm import tqdm 
import argparse
import pickle
import numpy as np

def main(number_of_records):
    attributes = {
        # eyebrows
        'Arched_Eyebrows': 3,              # 0-> not mentioned, 1-> slightly arched eyebrows, 2-> arched eyebrows
        'Bushy_Eyebrows': 3,               # 0-> not mentioned, 1-> slightly bushy eyebrows, 2-> bushy eyebrows

        # hair color
        'Black_Hair': 3,                   # 0-> not mentioned, 1-> black hair, 2-> dark black hair
        'Blond_Hair': 2,                   # 0-> not mentioned, 1-> blond hair
        'Brown_Hair': 3,                   # 0-> not mentioned, 1-> brown hair, 2-> dark brown hair
        'Gray_Hair': 3,                    # 0-> not mentioned, 1-> gray hair, 2-> dark gray hair

        # hair style
        'Straight_Hair': 5,                # 0-> not mentioned, 1-> curly hair, 2-> wavy hair, 3-> slightly wavy hair, 4-> straight hair
        'Receding_Hairline': 2,            # 0-> not mentioned, 1-> with receding hairline
        'Bald': 2,                         # 0-> not mentioned, 1-> bald
        'Bangs': 3,                        # 0-> not mentioned, 1-> with bangs

        # hair length
        'Hair_Length': 4,                  # 0-> not mentioned, 1-> short hair, 2-> hair of average-length, 3-> long hair

        # facial hair
        'Goatee': 2,                       # 0-> not mentioned, 1-> with goatee
        'Mustache': 3,                     # 0-> not mentioned, 1-> no mustache, 2-> with mustache
        'Beard': 4,                        # 0-> not mentioned, 1-> no beard, 2-> with small beard, 3-> with heavy beard
        'Sideburns': 2,                    # 0-> not mentioned, 1-> with sideburns

        # skin color
        'Skin_Color': 4,                   # 0-> not mentioned, 1-> black-african, 2-> tanned (between black and white), 3-> white

        # skin type
        'Pale_Skin': 3,                    # 0-> not mentioned, 1-> no pale skin, 2-> pale skin

        # general attributes
        'Chubby': 3,                       # 0-> not mentioned, 1-> thin, 2-> chubby
        'Smiling': 3,                      # 0-> not mentioned, 1-> not smiling, 2-> smiling
        'Attractive': 3,                   # 0-> not mentioned, 1-> ugly, 2-> attractive

        # gender
        'Male': 2,                         # 0-> female, 1-> male

        # age
        'Old': 6,                          # 0-> not mentioned, 1-> kid, 2-> teenager, 3-> middle-aged, 4-> old, 5-> very old

        # eye attributes
        'Bags_Under_Eyes': 2,              # 0-> not mentioned, 1-> bags
        'Wide_Eyes': 6,                    # 0-> not mentioned, 1-> very narrow, 2-> narrow, 3-> normal, 4-> wide, 5-> very wide

        # facial attributes
        'Big_Lips': 4,                     # 0-> not mentioned, 1-> small lips, 2-> middle-sized lips, 3-> big lips
        'Big_Nose': 4,                     # 0-> not mentioned, 1-> small nose, 2-> middle-sized nose, 3-> big nose                  
        'Double_Chin': 2,                  # 0-> not mentioned, 1-> double-chin
        'High_Cheekbones': 2,              # 0-> not mentioned, 1-> high-cheekbones
        'Oval_Face': 2,                    # 0-> not mentioned, 1-> oval face
        'Pointy_Nose': 2,                  # 0-> not mentioned, 1-> pointy nose
        'Rosy_Cheeks': 2,                  # 0-> not mentioned, 1-> rosy cheeks

        # make up
        'Heavy_Makeup': 4,                 # 0-> not mentioned, 1-> no makeup, 2-> simple makeup, 3-> heavy makeup
        'Wearing_Lipstick': 3,             # 0-> not mentioned, 1-> no lipstick, 2-> with lipstick 

        # textual description
        'Description': 'description'
    }

    # save dict to use in normalization after NLP model
    # with open('../../staged-generation/attributes_max.pkl', 'wb') as f:
    #     pickle.dump(attributes[:-1], f, pickle.HIGHEST_PROTOCOL)

    print(len(list(attributes.values())))
    np.save('../../staged-generation/attributes_max.npy', list(attributes.values())[:-1])



    # create dataframe to generate records in
    df = pd.DataFrame(columns = attributes.keys())

    for i in tqdm(range(number_of_records)):
        record = []
        for attribute in list(attributes.keys())[:-1]:
            record.append(random.randint(0, attributes[attribute] - 1))
        record.append('description')
        # some constraints on attributes
        # females cannot have facial hair and cannot be bald
        if record[20] == 0:
            record[8] = 0
            record[11] = 0
            record[12] = 0
            record[13] = 0
            record[14] = 0

        # males cannot have makeup or lipstick
        if record[20] == 1:
            record[31] = 0
            record[32] = 0


        # if bald is mentioned -> delete bald or hair
        if record[8] == 1:
            # 20% keep bald, make hair not mentioned
            if random.random() < 0.2:
                record[2] = 0
                record[3] = 0
                record[4] = 0
                record[5] = 0
                record[6] = 0
                record[7] = 0
                record[9] = 0
                record[10] = 0
            # make bald not mentioned
            else:
                record[8] = 0

        # one hair color no more
        if record[2] or record[3] or record[4] or record[5]:
            # set all zeros
            record[2] = 0
            record[3] = 0
            record[4] = 0
            record[5] = 0

            # set one of them to 1
            record[random.randint(2,5)] = 1

        # receding hairline or bangs
        if record[7] or record[9]:
            if random.random() < 0.5:
                record[7] = 0
                record[9] = 1
            else:
                record[7] = 1
                record[9] = 0


        # add the record to the dataframe
        df.loc[len(df)] = record

    df.to_csv('attributes.csv', index = False)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('-nrecords', '--number_of_records', type=int, help='number of receords to generate', default = 10000)
    args = argparser.parse_args()

    main(args.number_of_records)






