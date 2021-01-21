import random
from copy import deepcopy
import numpy as np  
from googletrans import Translator

import pandas as pd

class textual_description:
    def __init__(self, attributes, translator = None, paraphrase = False):
        self.attributes = attributes
        self.with_statements = []
        self.without_statements = []
        self.full_statements = []
        self.adjectives = []
        self.has_full_attributes = []
        self.has_not_full_attributes = []
        self.is_full_attributes = []
        self.is_not_full_attributes = []
        self.putting_full_attributes = []
        self.not_putting_full_attributes = []
        self.added_antonyms_attributes = []

        self.hair_attributes = []
        self.eyebrows_attributes = []

        self.three_way_pos_attribute_dict = dict()
        self.three_way_neg_attribute_dict = dict()
        self.v_tobe_dict = dict()

        self.construct_description()
        # for paraphrasing
        if paraphrase == True:
            self.translator = translator
            self.languages = ['it', 'ar', 'sv', 'pl', 'sq', 'de', 'zh-cn', 'es', 'ja', 'no', 'ro']
            self.description = self.get_cycle_paraphrase(self.description)

    def add_has_with_attribute(self, attribute):
        if random.random() > 0.5:
            self.with_statements.append(attribute)
        else:
            self.has_full_attributes.append(attribute)


        
    def add_adjective(self, adjective, pos_neg):
        '''
        add the adjective in a positive is statement or as an adjective in the first statement, if pos_neg = 0
        add the adjective in a negative is statement, otherwise
        '''
        # positive
        if pos_neg == 0:
            # adjective in first statement
            if random.random() > 0.5:
                self.adjectives.append(adjective)
            # adjective in is statement
            else:
                self.is_full_attributes.append(adjective)
        # negative
        else:
            self.is_not_full_attributes.append(adjective)

    def binary_adjective_processing(self, attr_choice, choice_1_options, choice_2_options):  
        '''
        choose randomly if the adjective will be added positively or negatively
        choose random option from the list of options supported
        '''      
        # attribute = 1
        if attr_choice == 1:
            # 70% add it positively
            if random.random() < 0.7:
                choice = random.choice(choice_1_options)
                self.add_adjective(choice, 0)
                
            # 30% add it negatively
            else:
                choice = random.choice(choice_2_options)
                self.add_adjective(choice, 1)

        # attribute = 2
        elif attr_choice == 2:
            # 70% add it positively
            if random.random() < 0.7:
                choice = random.choice(choice_2_options)
                self.add_adjective(choice, 0)
                
            # 30% add it negatively
            else:
                choice = random.choice(choice_1_options)
                self.add_adjective(choice, 1)

    def adjectives_processing(self):
        '''
        process unary and binary adjectives that can be added to first statement or to is statement
        '''

        ####################################################################################
        # CHUBBY (Binary)
        chubby_attribute = self.attributes[17]
        # options for chubby_attribute = 1
        thin_adjectives = ['thin', 'skinny', 'slim']
        # options for chubby_attribute = 2
        chubby_adjectives = ['chubby', 'fat']
        # process chubby
        self.binary_adjective_processing(chubby_attribute, thin_adjectives, chubby_adjectives)

        ####################################################################################
        # SMILING (Binary)
        smiling_attribute = self.attributes[18]
        # options for smiling_attribute = 1
        not_smiling_adjectives = ['not smiling']
        # options for smiling_attribute = 2
        smiling_adjectives = ['smiling']
        # process chubby
        self.binary_adjective_processing(smiling_attribute, not_smiling_adjectives, smiling_adjectives)

        ####################################################################################
        # ATTRACTIVE (Binary)
        attractive_attribute = self.attributes[19]
        # options for attractive_attribute = 1
        not_attractive_adjectives = ['unattractive', 'ugly']
        # options for attractive_attribute = 2
        attractive_adjectives = ['attractive', 'beautiful', 'gorgeous', 'pretty']
        # process chubby
        self.binary_adjective_processing(attractive_attribute, not_attractive_adjectives, attractive_adjectives)

        ####################################################################################
        # BALD (Unary)
        bald_attribute = self.attributes[8]
        bald_adjective = random.choice(['bald', 'hairless'])
        if bald_attribute == 1:
            # bald in first statement
            if random.random() > 0.5:
                self.adjectives.append(bald_adjective)
            # bald in is statement
            else:
                self.is_full_attributes.append(bald_adjective)


    def age_gender(self):
        '''
        process age and gender
        add correct pronoun and possessive pronoun
        '''
        # age
        age_attribute = self.attributes[21]

        # not mentioned
        if   age_attribute == 0:
            self.age_adj = ''
        
        # kid
        elif age_attribute == 1:
            self.age_adj = random.choice(['kid', 'child'])

        # teenager
        elif age_attribute == 2:
            self.age_adj = random.choice(['teenager', 'teen', 'young'])

        # middle-aged
        elif age_attribute == 3:
            self.age_adj = random.choice(['middle-aged', 'adult', 'grown'])
        
        # old
        elif age_attribute == 4:
            self.age_adj = 'old'
        
        # very old
        elif age_attribute == 5:
            self.age_adj = random.choice(['very old', 'elderly'])


        # add age if mentioned to adjectives or to is statements
        if self.age_adj != '':
            if random.random() > 0.5:
                self.adjectives.append(self.age_adj)
            else:
                self.is_full_attributes.append(self.age_adj)
        
        ############################################################################################
        # gender
        gender_attribute = self.attributes[20]
        # male
        if gender_attribute == 1:
            # middle-aged, old, very old
            if age_attribute > 2:
                self.gender = random.choice(['male', 'guy', 'man'])
            # kid, teen
            else:
                self.gender = random.choice(['male', 'boy'])
            self.pronoun = 'He'
            self.possessive_pronoun = 'his'
        # female
        else:
            # middle-aged, old, very old
            if age_attribute > 2:
                self.gender = random.choice(['female', 'lady', 'woman'])
            # kid, teen
            else:
                self.gender = random.choice(['female', 'girl'])
            self.pronoun = 'She'
            self.possessive_pronoun = 'her'
        
        
    def hair_color_attributes_processing(self):
        '''
        add hair colors  (black, gray, blond, brown)
        '''
        # black hair
        black_hair_attribute = self.attributes[2]
        if black_hair_attribute == 1:
            self.hair_attributes.append('black')
        elif black_hair_attribute == 2:
            self.hair_attributes.append('dark black')


        # brown hair
        brown_hair_attribute = self.attributes[4]
        if brown_hair_attribute == 1:
            self.hair_attributes.append('brown')
        elif brown_hair_attribute == 2:
            self.hair_attributes.append('dark brown')


        # gray hair
        gray_hair_attribute = self.attributes[5]
        if gray_hair_attribute == 1:
            self.hair_attributes.append(random.choice(['gray', 'grey']))
        elif gray_hair_attribute == 2:
            self.hair_attributes.append(random.choice(['dark gray', 'dark grey']))


        # blond hair
        blond_hair_attribute = self.attributes[3]
        if blond_hair_attribute == 1:
            # 50% -> has blond hair - hair is blond - with blond hair (three-way)
            if random.random() > 0.5:
                self.hair_attributes.append(random.choice(['blond', 'blonde']))
            # 50% -> blond "adj" (she is blond)
            else:
                self.add_adjective(random.choice(['blond', 'blonde']), 0)

    def straight_hair_attribute_processing(self):
        straight_hair_attribute = self.attributes[6]
        if straight_hair_attribute == 1:
            self.hair_attributes.append('curly')
        if straight_hair_attribute == 2:
            self.hair_attributes.append('wavy')
        if straight_hair_attribute == 3:
            self.hair_attributes.append('slightly wavy')
        if straight_hair_attribute == 4:
            self.hair_attributes.append(random.choice(['straight', 'smooth']))

    def hair_length_attribute_processing(self):
        hair_length_attribute = self.attributes[10]

        # short hair
        if hair_length_attribute == 1:
            self.hair_attributes.append('short')
            
        # average-length hair
        if hair_length_attribute == 2:
            self.hair_attributes.append(random.choice(['neither short nor long', 'medium-length']))

        # long hair
        if hair_length_attribute == 3:
            self.hair_attributes.append('long')

    def bangs_attribute_processing(self):
        bangs_attribute = self.attributes[9] 
        if bangs_attribute == 1:
            self.add_has_with_attribute('bangs')

    def receding_hairline_attribute_processing(self):
        receding_hairline_attribute = self.attributes[7]
        if receding_hairline_attribute == 1:
            self.add_has_with_attribute('receding hairline')
    
    def hair_attributes_processing(self):
        # PROCESSING
        self.hair_color_attributes_processing()
        self.straight_hair_attribute_processing()
        self.hair_length_attribute_processing()
        self.bangs_attribute_processing()
        self.receding_hairline_attribute_processing()

        # add hair attributes to three way attributes
        self.three_way_pos_attribute_dict['hair'] = self.hair_attributes
        self.v_tobe_dict['hair'] = 'is'


    def eyebrows_attributes_processing(self):
        '''
        add arched and bushy eyebrows attributes
        '''
        # ARCHED EYEBROWS
        arched_eyebrows_attribute = self.attributes[0]
        if arched_eyebrows_attribute == 1:
            self.eyebrows_attributes.append(random.choice(['slightly arched', 'a little arched', 'a bit arched']))
        elif arched_eyebrows_attribute == 2:
            self.eyebrows_attributes.append('arched')


        # BUSHY EYEBROWS
        bushy_eyebrows_attribute = self.attributes[1]
        if bushy_eyebrows_attribute == 1:
            self.eyebrows_attributes.append(random.choice(['slightly bushy', 'a little bushy', 'a bit bushy']))
        elif bushy_eyebrows_attribute == 2:
            self.eyebrows_attributes.append('bushy')

        
        # add eyebrows attributes to three way attributes
        self.three_way_pos_attribute_dict['eyebrows'] = self.eyebrows_attributes
        self.v_tobe_dict['eyebrows'] = 'are'


    def big_nose_attribute_processing(self):
        
        big_nose_attribute = self.attributes[25]

        # small nose
        if big_nose_attribute == 1:
            # 50% add it positively
            if random.random() > 0.5:
                self.three_way_pos_attribute_dict['nose'].append(random.choice(['small', 'tiny']))
            # 50% add it negatively
            else:
                self.three_way_neg_attribute_dict['nose'].append(random.choice(['big', 'large']))

        # medium-sized
        elif big_nose_attribute == 2:
            # 100% add it positively
            self.three_way_pos_attribute_dict['nose'].append(random.choice(['medium-sized', 'neither small nor big']))
        

        # big nose
        elif big_nose_attribute == 3:
            # 50% add it positively
            if random.random() > 0.5:
                self.three_way_pos_attribute_dict['nose'].append(random.choice(['big', 'large']))
            # 50% add it negatively
            else:
                self.three_way_neg_attribute_dict['nose'].append(random.choice(['small', 'tiny']))

    def pointy_nose_attribute_processing(self):
        pointy_nose_attribute = self.attributes[29]
        if pointy_nose_attribute == 1:
            self.three_way_pos_attribute_dict['nose'].append('pointy')

    def nose_attributes_processing(self):
        '''
        add big and pointy nose attributes
        '''
        # empty array for nose 
        self.three_way_pos_attribute_dict['nose'] = []
        self.three_way_neg_attribute_dict['nose'] = []

        # BIG NOSE ATTRIBUTE
        self.big_nose_attribute_processing()

        # POINTY NOSE ATTRIBUTE
        self.pointy_nose_attribute_processing()
        
        # v to-be of nose "is"
        self.v_tobe_dict['nose'] = 'is'


    def big_lips_attribute_processing(self):
        # create empty array for lips
        self.three_way_pos_attribute_dict['lips'] = []
        self.three_way_neg_attribute_dict['lips'] = []

        big_lips_attribute = self.attributes[24]

        # small lips
        if big_lips_attribute == 1:
            # 50% add it positively
            if random.random() > 0.5:
                self.three_way_pos_attribute_dict['lips'].append(random.choice(['small', 'tiny']))
            # 50% add it negatively
            else:
                self.three_way_neg_attribute_dict['lips'].append(random.choice(['big', 'large']))

        # medium-sized
        elif big_lips_attribute == 2:
            # 100% add it positively
            self.three_way_pos_attribute_dict['lips'].append(random.choice(['medium-sized', 'neither small nor big']))
        

        # big lips
        elif big_lips_attribute == 3:
            # 50% add it positively
            if random.random() > 0.5:
                self.three_way_pos_attribute_dict['lips'].append(random.choice(['big', 'large']))
            # 50% add it negatively
            else:
                self.three_way_neg_attribute_dict['lips'].append(random.choice(['small', 'tiny']))

        # v to-be of lips "are"
        self.v_tobe_dict['lips'] = 'are'


    def oval_face_attribute_processing(self):
        # create empty array for face
        self.three_way_pos_attribute_dict['face'] = []

        oval_face_attribute = self.attributes[28]
        if oval_face_attribute == 1:
            self.three_way_pos_attribute_dict['face'].append('oval')
        
        # v to-be of face "is"
        self.v_tobe_dict['face'] = 'is'
        

    def rosy_cheeks_attribute_processing(self):
        # create empty array for cheeks
        self.three_way_pos_attribute_dict['cheeks'] = []

        rosy_cheeks_attribute = self.attributes[30]
        if rosy_cheeks_attribute == 1:
            self.three_way_pos_attribute_dict['cheeks'].append(random.choice(['rosy', 'red']))

        # v to-be of cheeks "are"
        self.v_tobe_dict['cheeks'] = 'are'


    def pale_skin_attribute_processing(self):
        pale_skin_attribute = self.attributes[16]

        # not pale
        if pale_skin_attribute == 1:
            self.three_way_neg_attribute_dict['skin'].append('pale')

        # pale 
        elif pale_skin_attribute == 2:
            self.three_way_pos_attribute_dict['skin'].append('pale')
        
    def skin_color_attribute_processing(self):
        skin_color_attribute = self.attributes[15]

        # black
        if skin_color_attribute == 1:
            # 50% -> adjective
            if random.random() > 0.5:
                self.add_adjective(random.choice(['african', 'black']), 0)
            # 50% -> three-way (black skin)
            else:
                self.three_way_pos_attribute_dict['skin'].append('black')


         # tanned
        if skin_color_attribute == 2:
            self.three_way_pos_attribute_dict['skin'].append(random.choice(['neither white nor black', 'tanned']))


        # white
        if skin_color_attribute == 3:
            # 50% -> adjective
            if random.random() > 0.5:
                self.add_adjective(random.choice(['european', 'white']), 0)
            # 50% -> three-way (white skin)
            else:
                self.three_way_pos_attribute_dict['skin'].append('white')

    def skin_attributes_processing(self):
        # empty array for skin 
        self.three_way_pos_attribute_dict['skin'] = []
        self.three_way_neg_attribute_dict['skin'] = []

        # PALE SKIN
        self.pale_skin_attribute_processing()

        # SKIN COLOR
        self.skin_color_attribute_processing()

        # v to-be of skin "is"
        self.v_tobe_dict['skin'] = 'is'


    def wide_eyes_attribute_processing(self):
        # empty array for eyes 
        self.three_way_pos_attribute_dict['eyes'] = []
        self.three_way_neg_attribute_dict['eyes'] = []


        wide_eyes_attribute = self.attributes[23]

        # very narrow
        if wide_eyes_attribute == 1:
            self.three_way_pos_attribute_dict['eyes'].append(random.choice(['very narrow', 'very small', 'tiny']))

        # narrow
        if wide_eyes_attribute == 2:
            # 50% narrow
            if random.random() > 0.5:
                self.three_way_pos_attribute_dict['eyes'].append(random.choice(['narrow', 'small']))
            # 50% not wide
            else:
                self.three_way_neg_attribute_dict['eyes'].append(random.choice(['big', 'wide']))
        
        # medium-sized
        if wide_eyes_attribute == 3:
            self.three_way_pos_attribute_dict['eyes'].append(random.choice(['neither narrow nor wide', 'neither small nor big', 'medium-sized']))
        
        # wide
        if wide_eyes_attribute == 4:
            # 50% wide
            if random.random() > 0.5:
                self.three_way_pos_attribute_dict['eyes'].append(random.choice(['big', 'wide']))
            # 50% not narrow
            else:
                self.three_way_neg_attribute_dict['eyes'].append(random.choice(['narrow', 'small']))
        
        # very wide
        if wide_eyes_attribute == 5:
            self.three_way_pos_attribute_dict['eyes'].append(random.choice(['very wide', 'very big', 'very large']))

        
        # v to-be of eyes "are"
        self.v_tobe_dict['eyes'] = 'are'

        
    def bags_under_eyes_attribute_processing(self):
        '''
        bags under eyes is has/with attribute
        '''
        bags_under_eyes_attribute = self.attributes[22]
        if bags_under_eyes_attribute == 1:
            self.add_has_with_attribute('bags under eyes')


    def double_chin_attribute_processing(self):
        '''
        double chin is has/with attribute
        '''
        double_chin_attribute = self.attributes[26]
        if double_chin_attribute == 1:
            self.add_has_with_attribute('double chin')


    def high_cheekbones_attribute_processing(self):
        '''
        high cheekbones is has/with attribute
        '''
        high_cheekbones_attribute = self.attributes[27]
        if high_cheekbones_attribute == 1:
            self.add_has_with_attribute('high cheekbones')


    def heavy_makeup_attribute_processing(self):
        heavy_makeup_attribute = self.attributes[31]

        # no makeup
        if heavy_makeup_attribute == 1:
            # 50% -> without makeup
            if random.random() > 0.5:
                self.without_statements.append('makeup')
            # 50% -> not putting makeup
            else:
                self.not_putting_full_attributes.append('makeup')

        # simple makeup
        if heavy_makeup_attribute == 2:
            # 50% -> with simple makeup
            if random.random() > 0.5:
                self.with_statements.append(random.choice(['simple makeup', 'makeup']))
            # 50% -> putting simple makeup
            else:
                self.putting_full_attributes.append(random.choice(['simple makeup', 'makeup']))

        # heavy makeup
        if heavy_makeup_attribute == 3:
            # 50% -> with heavy makeup
            if random.random() > 0.5:
                self.with_statements.append('heavy makeup')
            # 50% -> putting heavy makeup
            else:
                self.putting_full_attributes.append('heavy makeup')


    def wearing_lipstick_attribute_processing(self):
        wearing_lipstick_attribute = self.attributes[32]

        # no lipstick
        if wearing_lipstick_attribute == 1:
            # 50% -> without lipstick
            if random.random() > 0.5:
                self.without_statements.append('lipstick')
            # 50% -> not putting lipstick
            else:
                self.not_putting_full_attributes.append('lipstick')
        
        # lipstick
        if wearing_lipstick_attribute == 2:
            # 50% -> with lipstick
            if random.random() > 0.5:
                self.with_statements.append('lipstick')
            # 50% -> putting lipstick
            else:
                self.putting_full_attributes.append('lipstick')

    
    def goatee_attribute_processing(self):
        goatee_attribute = self.attributes[11]
        if goatee_attribute == 1:
            self.add_has_with_attribute('goatee')


    def mustache_attribute_processing(self):
        mustache_attribute = self.attributes[12]

        # no mustache
        if mustache_attribute == 1:
            r = random.random()
            if r < 0.25:
                self.has_not_full_attributes.append('a mustache')
            elif r < 0.5:
                self.without_statements.append('a mustache')
            elif r < 0.75:
                self.with_statements.append('no mustache')
            else:
                self.has_full_attributes.append('no mustache')

        # with mustache
        if mustache_attribute == 2:
            self.add_has_with_attribute('a mustache')
        

    def beard_attribute_processing(self):
        beard_attribute = self.attributes[13]

        # no beard
        if beard_attribute == 1:
            r = random.random()
            if r < 0.25:
                self.has_not_full_attributes.append('a beard')
            elif r < 0.5:
                self.without_statements.append('a beard')
            elif r < 0.75:
                self.with_statements.append('no beard')
            else:
                self.has_full_attributes.append('no beard')
        
        # with small beard
        if beard_attribute == 2:
            self.add_has_with_attribute(random.choice(['a small beard', 'a light beard']))
        
        # with heavy beard
        if beard_attribute == 3:
            self.add_has_with_attribute(random.choice(['a heavy beard', 'a long beard']))
        

    def sideburns_attribute_processing(self):
        sideburns_attribute = self.attributes[14]
        if sideburns_attribute == 1:
            self.add_has_with_attribute('sideburns')


    # three-way attributes: he has attribute, his attribute is adjective, with attribute
    def three_way_attributes_processing(self):
        
        # negative attributes
        #########################################################################
        for attribute in self.three_way_neg_attribute_dict.keys():
            self.three_way_neg_attribute_dict[attribute] = self.and_combine(self.three_way_neg_attribute_dict[attribute])
        neg_attribute_dict_existed_keys = [attribute for attribute in self.three_way_neg_attribute_dict.keys() if not self.three_way_neg_attribute_dict[attribute] is None]
        for attribute in neg_attribute_dict_existed_keys:
            
            if random.random() > 0.7:
                self.without_statements.append(self.three_way_neg_attribute_dict[attribute] + ' ' + attribute)
            elif random.random() > 0.5:
                self.has_not_full_attributes.append(self.three_way_neg_attribute_dict[attribute] + ' ' + attribute)
            else:
                neg_v_tobe = random.choice([self.v_tobe_dict[attribute] + ' not', self.v_tobe_dict[attribute] + "n't"])
                self.full_statements.append(self.possessive_pronoun + ' ' + attribute + ' ' + neg_v_tobe + ' ' +  self.three_way_neg_attribute_dict[attribute] + '.')


        # positive attributes
        #########################################################################
        for attribute in self.three_way_pos_attribute_dict.keys():
            self.three_way_pos_attribute_dict[attribute] = self.and_combine(self.three_way_pos_attribute_dict[attribute])
        pos_attribute_dict_existed_keys = [attribute for attribute in self.three_way_pos_attribute_dict.keys() if not self.three_way_pos_attribute_dict[attribute] is None]
        if len(pos_attribute_dict_existed_keys) == 0:
            return
        # choose one to be in with statements
        with_attribute = random.choice(pos_attribute_dict_existed_keys)
        self.with_statements.append(self.three_way_pos_attribute_dict[with_attribute] + ' ' + with_attribute)
        
        # make the remaining in the form of he has attribute or his attribute is adjective with 50% probability for each
        for attribute in pos_attribute_dict_existed_keys:
            if attribute != with_attribute:
                if random.random() > 0.5:
                    self.has_full_attributes.append(self.three_way_pos_attribute_dict[attribute] + ' ' + attribute)
                else:
                    self.full_statements.append(self.possessive_pronoun + ' ' + attribute + ' ' + self.v_tobe_dict[attribute] + ' ' +  self.three_way_pos_attribute_dict[attribute] + '.')




    def and_combine(self, words):
        if len(words) == 1:
            return words[0]
        if len(words) == 0:
            return None 
        words_tmp = deepcopy(words)
        words_tmp[-2] = words[-2] + ' and ' + words[-1]
        words_tmp = words_tmp[:-1]
        separator = ', '
        return separator.join(words_tmp)

    def a_an (self, word):
        vowels = ['a', 'A', 'e', 'E', 'i', 'I', 'o', 'O', 'u', 'U']
        if len(word) == 0:
            return word
        if word[0] in vowels:
            return 'an ' + word
        return 'a ' + word 

    def construct_first_sentence(self):
        random.shuffle(self.adjectives)
        random.shuffle(self.with_statements)
        adjective = self.and_combine(self.adjectives)
        if not self.with_statements == []:
            with_stats = ' with ' + self.and_combine(self.with_statements)
        else:
            with_stats = ''

        if not self.without_statements == []:
            without_stats = ' without ' + self.and_combine(self.without_statements)
        else: 
            without_stats = ''
        
        if adjective is None:
            adjective = ''
        else:
            adjective += ' '
        
        self.first_statement = adjective + self.gender + with_stats + without_stats + '.'
       
        self.first_statement = self.a_an(self.first_statement)

    def construct_description(self):
        # preprocessing
        self.age_gender()
        self.adjectives_processing()
        self.hair_attributes_processing()
        self.eyebrows_attributes_processing()
        self.nose_attributes_processing()
        self.big_lips_attribute_processing()
        self.oval_face_attribute_processing()
        self.rosy_cheeks_attribute_processing()
        self.skin_attributes_processing()
        self.wide_eyes_attribute_processing()
        self.bags_under_eyes_attribute_processing()
        self.double_chin_attribute_processing()
        self.high_cheekbones_attribute_processing()
        self.heavy_makeup_attribute_processing()
        self.wearing_lipstick_attribute_processing()
        self.goatee_attribute_processing()
        self.mustache_attribute_processing()
        self.beard_attribute_processing()
        self.sideburns_attribute_processing()
        self.three_way_attributes_processing()


        # first statement
        self.construct_first_sentence()

        # remaining sentences
        # has statement
        if len(self.has_full_attributes) > 0:
            random.shuffle(self.has_full_attributes)
            has_attribute = self.and_combine(self.has_full_attributes)
            has_statement = self.pronoun + ' has ' + has_attribute.lower() + '.'
        else:
            has_statement = ''

        # has not statement
        if len(self.has_not_full_attributes) > 0:
            random.shuffle(self.has_not_full_attributes)
            has_not_attribute = self.and_combine(self.has_not_full_attributes)
            neg_has = random.choice([' has not ', " hasn't "])
            has_not_statement = self.pronoun + neg_has + has_not_attribute.lower() + '.'
        else:
            has_not_statement = ''
        
        # is statement
        if len(self.is_full_attributes) > 0:
            random.shuffle(self.is_full_attributes)
            is_attribute = self.and_combine(self.is_full_attributes)
            is_statement = self.pronoun + ' is ' + is_attribute.lower() + '.'
        else:
            is_statement = ''

        # is not statement
        if len(self.is_not_full_attributes) > 0:
            random.shuffle(self.is_not_full_attributes)
            is_not_attribute = self.and_combine(self.is_not_full_attributes)
            neg_is = random.choice([' is not ', " isn't "])
            is_not_statement = self.pronoun + neg_is + is_not_attribute.lower() + '.'
        else:
            is_not_statement = ''

        # putting statement
        if len(self.putting_full_attributes) > 0:
            random.shuffle(self.putting_full_attributes)
            putting_attribute = self.and_combine(self.putting_full_attributes)
            putting_statement = self.pronoun + ' is putting ' + putting_attribute.lower() + '.'
            puts_statement = self.pronoun + ' puts ' + putting_attribute.lower() + '.'
        else:
            putting_statement = ''
            puts_statement = ''
        
        # not putting statement
        if len(self.not_putting_full_attributes) > 0:
            random.shuffle(self.not_putting_full_attributes)
            not_putting_attribute = self.and_combine(self.not_putting_full_attributes)
            not_putting_statement = self.pronoun + ' is not putting ' + not_putting_attribute.lower() + '.'
        else:
            not_putting_statement = ''


        put_statement = random.choice([puts_statement,putting_statement])

        statements = [not_putting_statement, put_statement, has_statement, has_not_statement, is_statement, is_not_statement] + self.full_statements
        random.shuffle(statements)
        # full description
        separator = ' '
        self.description = self.first_statement
        for sentence in statements:
            self.description = separator.join([self.description, sentence])



    def get_cycle_paraphrase(self, text):
    
        num_languages_per_cycle = random.randint(1,3)
        selected_cycle_languages = random.sample(self.languages, num_languages_per_cycle)
        try:
            result = self.translator.translate(text, dest=selected_cycle_languages[0]).text
        except:
            return text
        del selected_cycle_languages[0]

        for language in selected_cycle_languages:
            try:
                result = self.translator.translate(result, dest=language).text
            except:
                return text

        try:
            result = self.translator.translate(result, dest='en').text
            print('great')
            return result
        except:
            return text
  
    def get_paraphrase(self, text):

    
        num_languages_per_cycle = random.randint(1,3)
        selected_cycle_languages = random.sample(self.languages, num_languages_per_cycle)
        language = random.choice(self.languages)
        try:
            result = self.translator.translate(text, dest=language).text
            result = self.translator.translate(result, dest='en').text
        except:
            return text
        return result


