import sys
sys.path.insert(0, 'C:\\Users\\kaspr\\Documents\\Studia\\gcpBackend')
import random
import string
import os
import argparse
import pandas as pd
import shutil
from timeit import default_timer
from numpy import random as np_random
# from Algorithm.soundGeneration import generate_morse_code_audio_file
from Algorithm.MorseCodeTranslator import sound_translator
import json
from random import gauss
from scipy.io import wavfile
import numpy as np
from IPython.display import Audio
from pydub import AudioSegment
import soundfile as sf


def translating_test(number_of_sentences: int = 20, save_file_name: str = 'test_results'):

    with open('./Algorithm/data/morse_code.json', "r", encoding='utf-8') as file:
        morse_code = json.load(file)
    
    dir = './Algorithm/data/processed_audios/'

    def generate_random_string(k):
        return ''.join(random.choices(string.ascii_uppercase + string.digits, k=k))

    def delete_file(file_name):
        path_del = './test_tmp/' + file_name # + '.wav'
        os.remove(path_del)
        return

    def calc_accuracy(words_to_compare: dict, scope: str = 'sentence'):
        match scope:
            case 'sentence':
                return 1 if words_to_compare['orig_message'].upper() == words_to_compare['translated_message'].upper() else 0
            case 'word':
                orig_message = words_to_compare['orig_message'].split()
                translation = words_to_compare['translated_message'].split()
                num_hits = 0
                for orig_word, translated_word in zip(orig_message, translation):
                    if orig_word.upper() == translated_word.upper():
                        num_hits += 1
                return num_hits / len(orig_message)
            case 'letter':
                num_hits = 0
                for orig_letter, translated_letter \
                        in zip(words_to_compare['orig_message'], words_to_compare['translated_message']):
                    if orig_letter.upper() == translated_letter.upper():
                        num_hits += 1
                return num_hits / max(len(words_to_compare['orig_message']),
                                      len(words_to_compare['translated_message']))

        
    def get_text_data(morse_code):
        texts = []
        keys = list(morse_code.keys())
        # keys += list(special_characters.keys())
        with open('./Algorithm/data/validation.txt', 'r', encoding='utf-8') as text_file:
            lines = text_file.readlines()
            for line in lines:
                line = line.strip('\n')
                # verify(line, keys)
                texts.append(line)
        return texts
        
    def text_to_morse(text: str) -> str:
        text_morse = []
        # morse_code_dict = dict_morse_to_text()
        text = text.upper()
        text_morse_words = []
        for word in text.split(" "):
            text_morse_letters = []
            for letter in word:
                if letter in morse_code:
                    text_morse_letters.append(morse_code[letter])
            text_morse_words.append("_".join(text_morse_letters))
        return ' '.join(text_morse_words)
    # change Volume
    def volume_manipul(f_list):
        return [f * gauss(1, 0.5) for f in f_list]

    # path 
    def generate_morse_code_audio_file(text: str, path: str):
        code = text_to_morse(text)
        samplerate, data = wavfile.read(path)
        dot_len = int(np.floor((samplerate/10)*np.random.uniform(0.9,1.1)))
        dash_len = int(np.floor(dot_len * gauss(3, 0.1)))

        # sound = []
        if data.ndim == 2:

            sound = sum([(volume_manipul(list(data[:int(dot_len*gauss(3, 0.1)), 0])) if i == '-' else (volume_manipul(list(data[:int(dot_len * gauss(1, 0.1)),0])) if i == '.' else  
                                                (list([0]*int(dot_len*gauss(1, 0.1))) if i == "_" else 
                                                list([0]*int(dot_len*gauss(5, 0.1)))))) + list([0]*int(dot_len * gauss(1, 0.1)))
                                                for i in code],[])
        else: 
            sound = sum([(volume_manipul(list(data[:int(dot_len*gauss(3, 0.1))])) if i == '-' else (volume_manipul(list(data[:int(dot_len * gauss(1, 0.1))])) if i == '.' else  
                                                (list([0]*int(dot_len * gauss(1, 0.1))) if i == "_" else 
                                                list([0]*int(dot_len*gauss(5, 0.1)))))) + list([0]*int(dot_len * gauss(1, 0.1)))
                                                for i in code], [])
        noise  = np.random.normal(0, np.random.uniform(0, 0.1)*np.std(sound), len(sound))
        data =  sound + noise
        # print(data)
        normalized_audio_data = data / np.max(np.abs(data))
        return normalized_audio_data, samplerate


    start = default_timer()
    df = pd.DataFrame(columns=['orig_message', 'message_after_translation', 'source',
                               'word_cnt', 'letter_cnt',
                               'accuracy_sentence', 'accuracy_word', 'accuracy_letter'])
    file_names = [fn for fn in os.listdir(dir) if fn.endswith('.wav')]
    # for i in range(number_of_sentences):
    #     file_names.append(generate_random_string(8))

    sentence_lenghts = np_random.poisson(1, number_of_sentences) + 1
    base_lenghts = [random.randint(100, 1000) for _ in range(number_of_sentences)]

    # print('Beginning sound generation.')
    sentence_num = 1

    texts = get_text_data(morse_code)
    for i, sentence in enumerate(texts):
        for filename in os.listdir(dir): #['beep-01a.wav']:#
            if filename.endswith('.wav'):
        # for fn, sl in zip(file_names, sentence_lenghts):
                # words = [generate_random_string((np_random.poisson(5, 1) + 1)[0]) for _ in range(sl)]
                # sentence = ' '.join(words)
                letter_cnt = len(sentence)
                data, samplerate = generate_morse_code_audio_file(sentence, f"{dir}{filename}")
                # print(f'{filename.replace(".wav", "")}_{i}')
                path = f'./test_tmp/{filename.replace(".wav", "")}_{i}.wav'
                # shutil.copy(fl, path)
                sf.write(path, data, samplerate)
                # a = AudioSegment(data, frame_rate=samplerate, channels=1)
                # a.export(path, format='wav')
                # os.remove(data)
                translated_message = sound_translator(path)
                to_compare = {
                    'orig_message': sentence,
                    'translated_message': translated_message,
                }
                acc_sentence = calc_accuracy(to_compare, 'sentence')
                acc_word = calc_accuracy(to_compare, 'word')
                acc_letter = calc_accuracy(to_compare, 'letter')
                df.loc[len(df)] = {
                    'orig_message': sentence,
                    'message_after_translation': translated_message,
                    'source':filename.replace(".wav", ""),
                    'word_cnt': len(sentence),
                    'letter_cnt': letter_cnt,
                    'accuracy_sentence': acc_sentence,
                    'accuracy_word': acc_word,
                    'accuracy_letter': acc_letter,
                }
                # print(f'Finished sentence {sentence_num}.')
                sentence_num += 1

    # print('Beginning tmp files deletion.')
    for fn  in os.listdir('./test_tmp/') :
        if fn.endswith('.wav'):
            delete_file(fn)
    # print(f'Saving results to \'./test_tmp/{save_file_name}.csv\'.')
    save_path = './test_tmp/' + save_file_name
    df.to_csv(save_path+'.csv', index=False)
    df.to_html(save_path+'.html', index=False)

    end = default_timer()

    avg_sentence_acc = df['accuracy_sentence'].mean()
    avg_word_acc = df['accuracy_word'].mean()
    avg_letter_acc = df['accuracy_letter'].mean()

    # print(f'Test finished in {round(end - start, 2)}s (average of {round((end - start) / number_of_sentences, 2)}s per '
    #       f'sentence). Obtained average accuracies: \nSentence accuracy — {round(avg_sentence_acc, 2)}'
    #       f'\nWord accuracy — {round(avg_word_acc, 2)} \nLetter accuracy — {round(avg_letter_acc, 2)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test the accuracy of our sound-to-text algorithm.')
    parser.add_argument('-nos', '--number_of_sentences', type=int,
                        help='The number of sentences to be created.',
                        default=50)
    parser.add_argument('-sfn', '--save_file_name', type=str,
                        help='The name of the file to which the results should be saved.',
                        default='test_results')
    args = parser.parse_args()
    translating_test(args.number_of_sentences, args.save_file_name)
