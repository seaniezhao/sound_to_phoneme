from model.wavenet_model import *
import hparams
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch.nn.functional as F
from data.preprocess import process_wav

def load_latest_model_from(location):

    files = [location + "/" + f for f in os.listdir(location)]
    newest_file = max(files, key=os.path.getctime)

    print("load model " + newest_file)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    hparam = hparams.create_hparams()

    model = WaveNetModel(hparam, device).to(device)
    if torch.cuda.is_available():
        states = torch.load(newest_file)
    else:
        states = torch.load(newest_file, map_location='cpu')
    model.load_state_dict(states['state_dict'])
    return model


# 1 frame equals 8ms
def get_time(phn_list, all_phn):

    phn_timing = []

    current_idx = -1
    current_start = 0
    for idx, item in enumerate(phn_list):
        if current_idx == -1:
            current_start = idx * 8
            current_idx = item
        elif current_idx != item or idx == len(phn_list)-1:
            end = idx*8
            phn_timing.append((current_start, end, all_phn[current_idx]))
            current_start = end
            current_idx = item

    return phn_timing


def post_process(sorted_outs, avali_phn, all_phn):

    final_out = []
    judge_index = 0
    current_compare = [ all_phn[avali_phn[judge_index]], all_phn[avali_phn[judge_index+1]] ]

    for s_o in sorted_outs:
        if s_o[0] == 'sil':
            final_out.append(all_phn.index(s_o[0]))
            continue
        idxs = []
        for phn in current_compare:
            idx = s_o.index(phn)
            idxs.append(idx)

        if idxs[1] < idxs[0]:
            final_out.append(all_phn.index(current_compare[1]))
            if judge_index+2 < len(avali_phn):
                judge_index += 1
                current_compare = [all_phn[avali_phn[judge_index]], all_phn[avali_phn[judge_index + 1]]]
            continue
        else:
            final_out.append(all_phn.index(current_compare[0]))
            continue

    return final_out

def get_phoneme(pinyins, all_phn):

    pp = pd.read_excel("data/Biaobei_pinyin-phoneme.xlsx")
    pp_dict = pp.set_index("pinyin").to_dict()["phoneme"]
    phns = []
    for py in pinyins:
        ph = pp_dict[py].strip().split('  ')
        for p in ph:
            phns.append(all_phn.index(p))
    return phns


def get_phoneme_timing(mfcc, pinyins=None):

    all_phn = list(np.load('data/all_phn.npy'))

    model = load_latest_model_from('snapshots')
    phn_list, raw_out = model.get_phonetic(mfcc)

    if pinyins!= None:
        sorted_outs = []
        for x in raw_out:
            p_x, s_x = F.softmax(x, dim=0).sort(descending=True)
            phn_array = []
            for item in s_x:
                phn_array.append(all_phn[item])
            sorted_outs.append(phn_array)
            #print(phn_array)
            #print(list(p_x.cpu().numpy()))
        _label = get_phoneme(pinyins, all_phn)
        phn_list = post_process(sorted_outs, _label, list(all_phn))

    phn_timing = get_time(phn_list, all_phn)


    return phn_timing

if __name__ == '__main__':
    all_phn = np.load('data/all_phn.npy')
    model = load_latest_model_from('snapshots')

    #mfcc = process_wav('data/dao_shu.wav')
    data = np.load('data/prepared_data/009885_data.npy')
    phn_list, raw_out = model.get_phonetic(data)

    phn_timing = get_time(phn_list, all_phn)
    for i in phn_timing:
        print(i)


    sorted_outs = []
    for x in raw_out:
        #s_x = sorted(range(len(x)), key=lambda k: -x[k])
        p_x, s_x = F.softmax(x, dim=0).sort(descending=True)
        phn_array = []
        for item in s_x:
            phn_array.append(all_phn[item])

        sorted_outs.append(phn_array)
        print(phn_array)
        print(list(p_x.cpu().numpy()))

    label = np.load('data/prepared_data/009885_label.npy')
    _label = []

    pre = None
    for x in label:
        if x != pre and x != 0:
            pre = x
            _label.append(x)

    phn_list = post_process(sorted_outs, _label, list(all_phn))


    phn_timing = get_time(phn_list, all_phn)
    for i in phn_timing:
        print(i)


