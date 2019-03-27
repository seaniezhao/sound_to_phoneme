import librosa
import matplotlib.pyplot as plt
import numpy as np

hop = 256
sample_rate = 32000

num = 86

def process_wav(path):

    y, osr = librosa.load(path, sr=None)

    sr = sample_rate
    y = librosa.resample(y, osr, sr)

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_fft=1024, hop_length=hop, n_mfcc=24)

    # mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=256)
    # plt.imshow(np.log(mel), aspect='auto', origin='bottom', interpolation='none')
    # plt.show()


    return mfccs



# second to frame:  ceil (sec * sr / hop_length)
def process_label(label_path):

    file = open(label_path, 'r')

    time_phon_list = []
    phon_list = []
    try:
        text_lines = file.readlines()
        #print(type(text_lines), text_lines)
        count = 0
        small_array = []
        for line in text_lines[12:]:
            count+=1
            line = line.replace('\n', '')
            print(line)
            small_array.append(line)
            if count%3 == 0:
                phn = small_array[2].replace('\"','')
                if phn == 'sp1':
                    phn = 'sil'
                print(phn[-1])
                if phn[-1].isdigit():
                    phn = phn[:-1]

                tup = (float(small_array[0]) * sample_rate/hop, float(small_array[1]) * sample_rate/hop, phn)
                time_phon_list.append(tup)
                if phn not in phon_list:
                    phon_list.append(phn)

                small_array.clear()
                print('------------')
    finally:
        file.close()

    return time_phon_list, phon_list


def final_process(time_phon_list, mfcc, all_phon):
    oh_list = []
    idx_list = []
    for i in range(mfcc.shape[1]):
        cur_phn = 0
        for j in range(len(time_phon_list)):
            if time_phon_list[j][0] <= i <= time_phon_list[j][1]:
                cur_phn = all_phon.index(time_phon_list[j][2])



        cur_phn_oh = np.zeros(len(all_phon))
        cur_phn_oh[cur_phn] = 1

        oh_list.append(cur_phn_oh)
        idx_list.append(cur_phn)
        print(cur_phn)
        #print(len(oh_list[-1]), oh_list[-1])

    return idx_list


def restore_phonetic(catagroy_list):

    return None


if __name__ == '__main__':
    save_folder = 'prepared_data/'
    wav_folder = 'Wave/'
    label_folder = 'PhoneLabeling/'

    infos = []
    all_phon = []
    for i in range(10000):
        file_name = str(i+1).zfill(6)

        w_path = wav_folder+file_name+'.wav'
        l_path = label_folder+file_name+'.interval'
        mfcc = process_wav(w_path)
        time_phon_list, phon_list = process_label(l_path)
        for p in phon_list:
            if p not in all_phon:
                all_phon.append(p)

        infos.append((file_name, mfcc, time_phon_list))

    np.save('all_phn.npy', all_phon)
    for item in infos:
        file_name, mfcc, time_phon_list = item
        label = final_process(time_phon_list, mfcc, all_phon)

        np.save(save_folder + file_name + '_data.npy', mfcc)
        np.save(save_folder + file_name + '_label.npy', label)






