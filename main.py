from inference import *


# wav and pinyin to timed phonome
#mfcc = process_wav('data/daoshu.wav')


mfcc = process_wav('data/Wave/008890.wav')
pinyin = ['yi', 'ge', 'yue', 'hou', 'di', 'zhi', 'lian', 'meng', 'wa', 'jie']
phn_timing = get_phoneme_timing(mfcc, pinyin)

for i in phn_timing:
    print(i)
