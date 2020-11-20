import random
from copy import deepcopy
class textual_description:
    def __init__(self, attributes, languages, translator, paraphrase = False):
        self.attributes = attributes
        self.with_statements = []
        self.full_statements = []
        self.adjectives = []
        self.has_full_attributes = []
        self.is_full_attributes = []
        self.putting_full_attributes = []
        self.added_antonyms_attributes = []

        self.construct_description()
        # for paraphrasing
        if paraphrase == True:
            self.translator = translator
            self.languages = languages
            self.description = self.get_cycle_paraphrase(self.description)

        

    def age_gender(self):
        # age
        if 'Young' in self.attributes:
            age = 0
            self.age_adj = 'young'
        elif random.random() > 0.5:
            age = 1
            self.age_adj = random.choice(['old', 'adult', 'grown'])
            # existed negatively
            self.added_antonyms_attributes.append('Young')
        else:
            age = 1
            self.age_adj = ''


        # gender
        if 'Male' in self.attributes:
            if age == 1:
                self.gender = random.choice(['male', 'guy', 'man'])
            else:
                self.gender = random.choice(['male', 'boy'])
            self.pronoun = 'He'
            self.possessive_pronoun = 'his'
        else:
            if age == 1:
                self.gender = random.choice(['female', 'lady', 'woman'])
            else:
                self.gender = random.choice(['female', 'girl'])
            
            # existed negatively
            self.added_antonyms_attributes.append('Male')

            self.pronoun = 'She'
            self.possessive_pronoun = 'her'
        
        if self.age_adj != '':
            if random.random() > 0.5:
                self.adjectives.append(self.age_adj)
            else:
                self.is_full_attributes.append(self.age_adj)

    def adjectives_processing(self):
        adj_attributes = [
            'Attractive',
            'Bald',
            'Chubby',
            'Smiling'
        ]

        adj_attributes_filtered = list(set(self.attributes) & set(adj_attributes))
        # antonyms
        if 'Chubby' not in adj_attributes_filtered:
            if random.random() > 0.5:
                opposites = ['Thin', 'Skinny', 'Slim']
                choice = random.choice(opposites)
                adj_attributes_filtered.append(choice)

                # existed negatively
                self.added_antonyms_attributes.append('Chubby')

        separator = ' '
        for attribute in adj_attributes_filtered:
            attribute = separator.join(attribute.split('_')).lower()
            if random.random() > 0.5:
                self.adjectives.append(attribute)
            else:
                self.is_full_attributes.append(attribute.lower())
        
        

    def has_with_attributes(self):
        '''
            'Arched_Eyebrows',
            'Bangs',
            'Big_Lips',
            'Big_Nose',
            'Black_Hair',
            'Blond_Hair',
            'Brown_Hair',
            'Bushy_Eyebrows',
            'Double_Chin',
            'Goatee',
            'Gray_Hair',
            'High_Cheekbones',
            'Mustache',
            'Narrow_Eyes',
            'No_Beard',
            'Oval_Face',
            'Pale_Skin',
            'Rosy_Cheeks',
            'Sideburns',
            'Straight_Hair',
            'Wavy_Hair'
        '''

        # three-way attributes: he has attribute, his attribute is adjective, with attribute
        self.three_way_attributes_processing()

        # facial hair attributes (beard - mustache - sideburns)
        self.beard_mustache_sideburns_attributes_processing()

        # remaining attributes
        self.remaining_has_attributes_processing()
    
    def three_way_attributes_processing(self):
        three_way_attributes = [
            'Big_Lips',
            'Big_Nose',
            'Pointy_Nose',
            'Narrow_Eyes',
            'Oval_Face',
            'Pale_Skin',
            'Rosy_Cheeks',

            'Arched_Eyebrows',
            'Bushy_Eyebrows',
            
            'Gray_Hair',
            'Black_Hair',
            'Blond_Hair',
            'Brown_Hair',
            'Straight_Hair',
            'Wavy_Hair',
            'Receding_Hairline'
        ]
        three_way_attributes_filtered = list(set(self.attributes) & set(three_way_attributes))
        attribute_dict = dict()
        v_tobe_dict = dict()
        # hair
        hair_attributes = [attribute.split('_')[0] 
                            for attribute in three_way_attributes_filtered 
                            if attribute.endswith('Hair') or attribute.endswith('Hairline')]
        attribute_dict['hair'] = hair_attributes
        v_tobe_dict['hair'] = 'is'

        # eyebrows
        eyebrows_attributes = [attribute.split('_')[0] 
                            for attribute in three_way_attributes_filtered 
                            if attribute.endswith('Eyebrows')]
        attribute_dict['eyebrows'] = eyebrows_attributes
        v_tobe_dict['eyebrows'] = 'are'

        # lips
        attribute_dict['lips'] = []
        if 'Big_Lips' in three_way_attributes_filtered:
            attribute_dict['lips'].append('big')
        else:
            if random.random() > 0.3:
                # existed negatively
                self.added_antonyms_attributes.append('Big_Lips')

                attribute_dict['lips'].append('small')
        v_tobe_dict['lips'] = 'are'

        # nose
        attribute_dict['nose'] = []
        if 'Big_Nose' in three_way_attributes_filtered:
            attribute_dict['nose'].append('big')
        else:
            if random.random() > 0.3:
                # existed negatively
                self.added_antonyms_attributes.append('Big_Nose')

                attribute_dict['nose'].append('small')
        if 'Pointy_Nose' in three_way_attributes_filtered:
            attribute_dict['nose'].append('pointy')
        v_tobe_dict['nose'] = 'is'

        # eyes
        attribute_dict['eyes'] = []
        if 'Narrow_Eyes' in three_way_attributes_filtered:
            attribute_dict['eyes'].append('narrow')
        else:
            if random.random() > 0.3:
                # existed negatively
                self.added_antonyms_attributes.append('Narrow_Eyes')

                attribute_dict['eyes'].append('wide')
        v_tobe_dict['eyes'] = 'are'

        # face
        attribute_dict['face'] = []
        if 'Oval_Face' in three_way_attributes_filtered:
            attribute_dict['face'].append('oval')
        v_tobe_dict['face'] = 'is'
 
        # skin
        attribute_dict['skin'] = []
        if 'Pale_Skin' in three_way_attributes_filtered:
            attribute_dict['skin'].append('pale')
        v_tobe_dict['skin'] = 'is'

        # Rosy Cheeks / Red Cheeks
        attribute_dict['cheeks'] = []
        if 'Rosy_Cheeks' in three_way_attributes_filtered:
            attribute_dict['cheeks'].append(random.choice(['rosy', 'red']))
        v_tobe_dict['cheeks'] = 'are'

        #########################################################################
        for attribute in attribute_dict.keys():
            attribute_dict[attribute] = self.and_combine(attribute_dict[attribute])
        attribute_dict_existed_keys = [attribute for attribute in attribute_dict.keys() if not attribute_dict[attribute] is None]
        if len(attribute_dict_existed_keys) == 0:
            return
        # choose one to be in with statements
        with_attribute = random.choice(attribute_dict_existed_keys)
        self.with_statements.append(attribute_dict[with_attribute] + ' ' + with_attribute)
        
        # make the remaining in the form of he has attribute or his attribute is adjective with 50% probability for each
        for attribute in attribute_dict_existed_keys:
            if attribute != with_attribute:
                if random.random() > 0.5:
                    self.has_full_attributes.append(attribute_dict[attribute] + ' ' + attribute)
                else:
                    self.full_statements.append(self.possessive_pronoun + ' ' + attribute + ' ' + v_tobe_dict[attribute] + ' ' +  attribute_dict[attribute] + '.')

    def beard_mustache_sideburns_attributes_processing(self):
        local_attributes = []
        if 'Male' in self.attributes:
            if 'Mustache' in self.attributes:
                local_attributes.append('a mustache')
            else:
                if random.random() > 0.3:
                    # existed negatively
                    self.added_antonyms_attributes.append('Mustache')

                    local_attributes.append('no mustache')

            if 'Sideburns' in self.attributes:
                local_attributes.append('sideburns')

            if not 'No_Beard' in self.attributes:
                # existed negatively
                self.added_antonyms_attributes.append('No_Beard')

                local_attributes.append('a beard')
            else:
                if random.random() > 0.3:
                    local_attributes.append('no beard')
            
            description = self.and_combine(local_attributes)
            if description is None:
                return
            if random.random() > 0.5:
                self.with_statements += local_attributes
            else:
                self.has_full_attributes.append(description)
            
    def remaining_has_attributes_processing(self):
        remaining_attributes = [
            'Bangs',
            'Double_Chin',
            'Goatee',
            'High_Cheekbones',
            'Bags_Under_Eyes'
        ]
    
        remaining_attributes_filtered = list(set(self.attributes) & set(remaining_attributes))
        separator = ' '
        for attribute in remaining_attributes_filtered:
            attribute = separator.join(attribute.split('_')).lower()
            if random.random() > 0.5:
                self.with_statements.append(attribute)
            else:
                self.has_full_attributes.append(attribute)
    
    def makeup_attributes_processing(self):
        makeup_attributes = [
            'Heavy_Makeup',
            'Wearing_Lipstick'
        ]
    
        makeup_attributes_filtered = list(set(self.attributes) & set(makeup_attributes))
        separator = ' '
        for attribute in makeup_attributes_filtered:
            if attribute.endswith('Lipstick'):
                attribute = 'lipstick'
            else:
                attribute = separator.join(attribute.split('_')).lower()

            if random.random() > 0.5:
                self.with_statements.append(attribute)
            else:
                self.putting_full_attributes.append(attribute)
    
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
        with_stats = self.and_combine(self.with_statements)
        if adjective is None and not with_stats is None:
            self.first_statement = self.gender + ' with ' + with_stats + '.'
        elif with_stats is None and not adjective is None:
            self.first_statement = adjective + ' ' + self.gender + '.'
        elif not with_stats is None and not adjective is None:
            self.first_statement = adjective + ' ' + self.gender + ' with ' + with_stats + '.'
        else:
            self.first_statement = self.gender + '.'
        self.first_statement = self.a_an(self.first_statement)

    def construct_description(self):
        # preprocessing
        self.age_gender()
        self.adjectives_processing()
        self.has_with_attributes()
        self.makeup_attributes_processing()

        # first statement
        self.construct_first_sentence()

        # remaining sentences
        random.shuffle(self.is_full_attributes)
        if len(self.has_full_attributes) > 0:
            random.shuffle(self.has_full_attributes)
            has_attribute = self.and_combine(self.has_full_attributes)
            has_statement = self.pronoun + ' has ' + has_attribute.lower() + '.'
        else:
            has_statement = ''
        
        if len(self.is_full_attributes) > 0:
            random.shuffle(self.is_full_attributes)
            is_attribute = self.and_combine(self.is_full_attributes)
            is_statement = self.pronoun + ' is ' + is_attribute.lower() + '.'
        else:
            is_statement = ''

        if len(self.putting_full_attributes) > 0:
            random.shuffle(self.putting_full_attributes)
            putting_attribute = self.and_combine(self.putting_full_attributes)
            putting_statement = self.pronoun + ' is putting ' + putting_attribute.lower() + '.'
            puts_statement = self.pronoun + ' puts ' + putting_attribute.lower() + '.'
        else:
            putting_statement = ''
            puts_statement = ''

        put_statement = random.choice([puts_statement,putting_statement])

        statements = [put_statement, has_statement, is_statement] + self.full_statements
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
            result = self.translator.translate(text, dest=selected_cycle_languages[0])
        except:
            return text
        del selected_cycle_languages[0]

        for language in selected_cycle_languages:
            try:
                result = self.translator.translate(result.text, dest=language)
            except:
                return text

        try:
            result = self.translator.translate(result.text, dest='en')
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