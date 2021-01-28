
import random
import pandas as pd
from tqdm import tqdm 
import argparse
import pickle
import numpy as np
import copy

def main(number_of_records):
    attributes = {
        # eyebrows
        'Arched_Eyebrows': 3,       # 0-> not mentioned, 1-> slightly arched eyebrows, 2-> arched eyebrows
        'Bushy_Eyebrows': 3,        # 0-> not mentioned, 1-> light eyebrows, 2-> bushy eyebrows

        # hair color
        'Black_Hair': 2,            # 0-> not mentioned, 1-> black hair
        'Blond_Hair': 2,            # 0-> not mentioned, 1-> blond hair
        'Brown_Hair': 3,            # 0-> not mentioned, 1-> dark brown hair, 2-> brown hair
        'Gray_Hair': 4,             # 0-> not mentioned, 1-> dark gray hair, 2-> gray hair, 3-> white hair

        # hair style
        'Straight_Hair': 5,         # 0-> not mentioned, 1-> straight hair, 2-> slightly wavy hair, 3-> wavy hair, 4-> curly hair
        'Receding_Hairline': 2,     # 0-> not mentioned, 1-> with receding hairline
        'Bald': 2,                  # 0-> not mentioned, 1-> bald
        'Bangs': 3,                 # 0-> not mentioned, 1-> with bangs

        # hair length
        'Hair_Length': 4,           # 0-> not mentioned, 1-> short hair, 2-> hair of average-length, 3-> long hair

        # facial hair
        'Goatee': 2,                # 0-> not mentioned, 1-> with goatee
        'Mustache': 3,              # 0-> not mentioned, 1-> no mustache, 2-> with mustache
        'Beard': 4,                 # 0-> not mentioned, 1-> no beard, 2-> with small beard, 3-> with heavy beard
        'Sideburns': 2,             # 0-> not mentioned, 1-> with sideburns

        # skin color
        'Skin_Color': 4,            # 0-> not mentioned, 1-> black-african, 2-> tanned (between black and white), 3-> white

        # asian race 
        'Asian': 2,                 # 0-> not mentioned, 1-> asian

        # general attributes
        'Chubby': 3,                # 0-> not mentioned, 1-> thin, 2-> chubby
        
        # gender
        'Male': 2,                  # 0-> female, 1-> male

        # age
        'Old': 7,                   # 0-> not mentioned, 1-> Baby, 2-> kid, 3-> teenager, 4-> middle-aged, 5-> old, 6-> very old

        # eye attributes
        'Bags_Under_Eyes': 2,       # 0-> not mentioned, 1-> bags
        'Wide_Eyes': 6,             # 0-> not mentioned, 1-> very narrow, 2-> narrow, 3-> normal, 4-> wide, 5-> very wide
        'Black_Eyes': 2,            # 0-> not mentioned, 1-> black eyes
        'Green_Eyes': 2,            # 0-> not mentioned, 1-> green eyes
        'Blue_Eyes': 2,             # 0-> not mentioned, 1-> blue eyes
        'Brown_Eyes': 3,            # 0-> not mentioned, 1-> brown eyes


        # facial attributes
        'Big_Lips': 3,              # 0-> not mentioned, 1-> small lips, 2-> big lips
        'Big_Nose': 3,              # 0-> not mentioned, 1-> small nose, 2-> big nose                  
        'Big_Ears': 3,              # 0-> not mentioned, 1-> small ears, 2-> big ears                  
        'Double_Chin': 2,           # 0-> not mentioned, 1-> double-chin
        'High_Cheekbones': 2,       # 0-> not mentioned, 1-> high-cheekbones
        'Pointy_Nose': 2,           # 0-> not mentioned, 1-> pointy nose
        'Rosy_Cheeks': 2,           # 0-> not mentioned, 1-> rosy cheeks

        # make up
        'Heavy_Makeup': 4,          # 0-> not mentioned, 1-> no makeup, 2-> simple makeup, 3-> heavy makeup
        'Wearing_Lipstick': 3,      # 0-> not mentioned, 1-> no lipstick, 2-> with lipstick 

        # textual description
        'Description': 'description'
    }

    # save dict to use in normalization after NLP model
    # with open('../../staged-generation/attributes_max.pkl', 'wb') as f:
    #     pickle.dump(attributes[:-1], f, pickle.HIGHEST_PROTOCOL)

    np.save('../../staged-generation/attributes_max.npy', list(attributes.values())[:-1])



    # create dataframe to generate records in
    df = pd.DataFrame(columns = attributes.keys())

    for i in tqdm(range(number_of_records)):
        record = copy.deepcopy(attributes)
        for attribute in list(attributes.keys())[:-1]:
            record[attribute]= random.randint(0, attributes[attribute] - 1)
        # some constraints on attributes
        # females and babies and children cannot have facial hair and cannot be bald
        if record['Male'] == 0 or record['Old'] == 1 or record['Old'] == 2:
            record['Goatee'] = 0
            record['Mustache'] = 0
            record['Beard'] = 0
            record['Sideburns'] = 0
        # females cannot be bald
        if record['Male'] == 0:
            record['Bald'] = 0

        # males and babies cannot have makeup or lipstick
        if record['Male'] == 1 or record['Old'] == 1:
            record['Heavy_Makeup'] = 0
            record['Wearing_Lipstick'] = 0

        # if bald is mentioned -> delete bald or hair
        if record['Bald'] == 1:
            # 20% keep bald, make hair not mentioned
            if random.random() < 0.4:
                record['Black_Hair'] = 0
                record['Blond_Hair'] = 0
                record['Brown_Hair'] = 0
                record['Gray_Hair'] = 0
                record['Straight_Hair'] = 0
                record['Receding_Hairline'] = 0
                record['Bangs'] = 0
                record['Hair_Length'] = 0
            # make bald not mentioned
            else:
                record['Bald'] = 0

        # one hair color no more
        if record['Black_Hair'] or record['Blond_Hair'] or record['Brown_Hair'] or record['Gray_Hair']:
            # set all zeros
            record['Black_Hair'] = 0
            record['Blond_Hair'] = 0
            record['Brown_Hair'] = 0
            record['Gray_Hair'] = 0

            prob = random.random()
            # set one of them
            if prob < 0.25:
                record['Black_Hair'] = random.randint(1, attributes['Black_Hair'] - 1)
            elif prob < 0.5:
                record['Blond_Hair'] = random.randint(1, attributes['Blond_Hair'] - 1)
            elif prob < 0.75:
                record['Brown_Hair'] = random.randint(1, attributes['Brown_Hair'] - 1)   
            else:
                record['Gray_Hair'] = random.randint(1, attributes['Gray_Hair'] - 1)
        


        # one eye color no more
        if record['Black_Eyes'] or record['Blue_Eyes'] or record['Brown_Eyes'] or record['Green_Eyes']:
            # set all zeros
            record['Black_Eyes'] = 0
            record['Blue_Eyes'] = 0
            record['Brown_Eyes'] = 0
            record['Green_Eyes'] = 0

            prob = random.random()
            # set one of them
            if prob < 0.25:
                record['Black_Eyes'] = random.randint(1, attributes['Black_Eyes'] - 1)
            elif prob < 0.5:
                record['Blue_Eyes'] = random.randint(1, attributes['Blue_Eyes'] - 1)
            elif prob < 0.75:
                record['Brown_Eyes'] = random.randint(1, attributes['Brown_Eyes'] - 1)   
            else:
                record['Green_Eyes'] = random.randint(1, attributes['Green_Eyes'] - 1)


        # receding hairline or bangs
        if record['Bangs'] or record['Receding_Hairline']:
            if random.random() < 0.5:
                record['Bangs'] = 0
                record['Receding_Hairline'] = 1
            else:
                record['Bangs'] = 1
                record['Receding_Hairline'] = 0


        # add the record to the dataframe
        df.loc[len(df)] = record

    df.to_csv('attributes.csv', index = False)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('-nrecords', '--number_of_records', type=int, help='number of receords to generate', default = 10000)
    args = argparser.parse_args()

    main(args.number_of_records)






