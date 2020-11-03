"""
    Generate paraphrases of the provided text descriptions,
    By running them through Google translate using different languages
"""

import json
import googletrans
from googletrans import Translator
import random
import progressbar
import argparse
import sys

def get_paraphrases(description, num_generated_paraphrases):
    languages = ['it', 'ar', 'sv', 'pl', 'sq', 'de', 'zh-cn', 'es', 'ja', 'no', 'ro']
    translator = Translator()
    paraphrases = []

    for i in range(num_generated_paraphrases):
        num_languages_per_cycle = random.randint(2,4)
        selected_cycle_languages = random.sample(languages, num_languages_per_cycle)
        try:
            result = translator.translate(description, dest=selected_cycle_languages[0])
        except:
            try:
                result = translator.translate(description, dest='en')
            except:
                continue
        del selected_cycle_languages[0]

        for language in selected_cycle_languages:
            try:
                result = translator.translate(result.text, dest=language)
            except:
                pass

        try:
            result = translator.translate(result.text, dest='en')
            paraphrases.append(result.text)
        except:
            continue
            
    return paraphrases


def main(path_to_json_file, num_generated_paraphrases):
    with open(path_to_json_file) as json_file:
        data = json.load(json_file)
        bar = progressbar.ProgressBar(maxval=len(data), \
                widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()
        for i in range(len(data)):
            description = data[i]['description']
            paraphrases = get_paraphrases(description, num_generated_paraphrases)
            data[i]['paraphrases'] = paraphrases
            bar.update(i + 1)
        bar.finish()

        with open('data_aug.json', 'w') as outfile:
            json.dump(data, outfile)

if __name__ == '__main__':
    # arguments parsing
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('-tj', '--path_to_json_file', type=str, help='path to JSON containing text descriptions', default = './face2text_v1.0/raw.json')
    argparser.add_argument('-pn', '--paraphrases_num', type=int, help='number of paraphrases to be generated per each sentence', default=4)

    args = argparser.parse_args()

    main(args.path_to_json_file, args.paraphrases_num)
        
