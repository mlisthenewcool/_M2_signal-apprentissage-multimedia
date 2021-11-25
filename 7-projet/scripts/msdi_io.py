# author : Valentin Emiya // Edited by : Debernardi Hippolyte - Fersula Jeremy

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

_msdi_path = './data'  # Change this to configure your path to MSDI dataset


def get_msdi_dataframe(msdi_path=_msdi_path):
    return pd.read_csv(Path(msdi_path) / 'msdi_mapping.csv')


def load_mfcc(entry, msdi_path=_msdi_path):
    x = np.load(Path(msdi_path) / 'sounds' / entry['mfcc'])
    return x[entry['msd_track_id']]


def load_img(entry, msdi_path=_msdi_path):
    return plt.imread(Path(msdi_path) / 'images' / entry['img'])


def load_deep_audio_features(entry, msdi_path=_msdi_path):
    subset_file = 'X_{}_audio_MSD-I.npy'.format(entry['set'])
    x = np.load(Path(msdi_path) / 'sounds/deep_features' / subset_file, mmap_mode='r')
    idx = entry['deep_features']
    return x[idx, :]


def get_set(entry):
    return entry['set']


def get_label(entry):
    return entry['genre']


def get_label_list(msdi_path=_msdi_path):
    df = pd.read_csv(Path(msdi_path) / 'labels.csv', header=None)
    return list(df.iloc[:, 0])


def get_mfcc_deep_features_and_labels(df, msdi_path, n_samples):
    subsets = ["train", "test", "val"]
    mfcc = {subset: [] for subset in subsets}
    deep_features = {subset: [] for subset in subsets}
    labels = {subset: [] for subset in subsets}

    # take whole dataset if n_samples isn't an integer
    # otherwise, randomly choose a subset of the dataset
    if not isinstance(n_samples, int) or n_samples > len(df):
        entry_indexes = df["id"]
    else:
        entry_indexes = np.random.choice(df["id"], n_samples)

    # there is a weird offset of indexes
    # if the whole dataset is chosen, key_errors should be equal to 1
    key_errors = 0
    for entry_idx in entry_indexes:
        try:
            entry = df.loc[entry_idx]
            subset = get_set(entry)

            # load X (mfcc, deep features) and Y
            mfcc[subset].append(load_mfcc(entry, msdi_path))
            deep_features[subset].append(load_deep_audio_features(entry, msdi_path))
            labels[subset].append(get_label(entry))
        except KeyError:
            key_errors += 1

    if key_errors > 0:
        print("Key errors :", key_errors)

    for subset in subsets:
        mfcc[subset] = np.array(mfcc[subset])
        deep_features[subset] = np.array(deep_features[subset])
        labels[subset] = np.array(labels[subset])

    return mfcc, deep_features, labels


def load_all_lyrics(lyrics_path, vocab_size=5000):
    """Loads all lyrics in bow form, from a .txt file as formatted
    in the given project file. Regex could have been used."""
    fil = open(lyrics_path, "r")
    dic = {}
    for sentence in fil.readlines():
        word_count = 0
        music_id = ""
        for word in sentence.split():
          if word_count == 0:  # First word is music_track_id
            dic[word] = ["", np.zeros(vocab_size)]
            music_id = word
          elif word_count == 1:  # Second word is label
            dic[music_id][0] = word
          else:  # Subsequent words are in the form Token:Count
            splitt = word.split(":")
            word_id, count = int(splitt[0]), int(splitt[1])
            dic[music_id][1][word_id-1] = count  # -1 because 0 is reserved
          word_count += 1
    return dic


def get_all_data_with_labels(df, msdi_path, n_samples):
    subsets = ["train", "test", "val"]

    features = {subset: [] for subset in subsets}

    img = {subset: [] for subset in subsets}
    txt = {subset: [] for subset in subsets}
    snd = {subset: [] for subset in subsets}
    labels = {subset: [] for subset in subsets}

    print('Loading lyrics ...', end="")
    txt_dic = load_all_lyrics("./data/lyrics/msx_lyrics_genre.txt", 5000)
    print('Done.')

    # take whole dataset if n_samples isn't an integer
    # otherwise, randomly choose a subset of the dataset
    if not isinstance(n_samples, int):
        entry_indexes = df["id"]
    else:
        entry_indexes = np.random.choice(df["id"], n_samples)

    # there is a weird offset of indexes
    # if the whole dataset is chosen, key_errors should be equal to 1
    key_errors = 0

    ten_percent = n_samples // 10
    for count, entry_idx in enumerate(entry_indexes):
        try:
            entry = df.loc[entry_idx]
            subset = get_set(entry)

            txt_item = txt_dic.get(entry['msd_track_id'])
            if txt_item is None:  # Align data with text
                continue

            img[subset].append(load_img(entry, msdi_path))
            txt[subset].append(txt_item[1])
            snd[subset].append(load_deep_audio_features(entry, msdi_path))

            # load Y (genres)
            labels[subset].append(get_label(entry))

            if count % ten_percent == 0:  # User interface
                print(str(10 * count // ten_percent), "% - ", end="")

        except KeyError:
            key_errors += 1

    if key_errors > 0:
        print("Key errors :", key_errors)
    print("Done.")

    # Merged model need format [np.array]
    for subset in subsets:
        features[subset] = [np.array(img[subset]),
                            np.array(txt[subset]),
                            np.array(snd[subset])]
        labels[subset] = custom_dummies(labels[subset])

    return features, labels


def custom_dummies(labels):
    """Transform str label to one hot encoding"""
    label_list = get_label_list()
    res = np.zeros((len(labels), len(label_list)))
    for i in range(len(labels)):
        for j in range(len(label_list)):
            if label_list[j] == labels[i]:
                res[i, j] = 1
                break
    return res


def dummies_to_labels(dums):
    """Revert one hot encoding to label"""
    res = []
    lab_list = get_label_list()
    for dum in dums:
      res += [lab_list[dum.argmax()]]
    return res



if __name__ == '__main__':
    # Exemple d'utilisation
    msdi = get_msdi_dataframe(_msdi_path)
    print('Dataset with {} entries'.format(len(msdi)))
    print('#' * 80)
    print('Labels:', get_label_list())
    print('#' * 80)

    entry_idx = 23456
    one_entry = msdi.loc[entry_idx]
    print('Entry {}:'.format(entry_idx))
    print(one_entry)
    print('#' * 80)
    mfcc = load_mfcc(one_entry, _msdi_path)
    print('MFCC shape:', mfcc.shape)
    img = load_img(one_entry, _msdi_path)
    print('Image shape:', img.shape)
    deep_features = load_deep_audio_features(one_entry, _msdi_path)
    print('Deep features:', deep_features.shape)
    print('Set:', get_set(one_entry))
    print('Genre:', get_label(one_entry))
