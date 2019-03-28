from inference import *


# wav and pinyin to timed phonome
mfcc = process_wav('data/Wave/000001.wav')

phn_timing = get_phoneme_timing(mfcc, ['ka','er','pu','pei','wai','sun','wan','hua','ti'])

for i in phn_timing:
    print(i)
