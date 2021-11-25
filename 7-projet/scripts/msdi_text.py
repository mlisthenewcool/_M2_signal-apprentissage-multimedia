import numpy as np
import msdi_io

_msdi_path = './data'  # Change this to configure your path to MSDI dataset

def get_bow_dict(lyrics_path, vocab_size):
    fil = open(lyrics_path, "r")
    dic = {}
    for sentence in fil.readlines():
        word_count = 0
        music_id = ""
        for word in sentence.split():
          if word_count == 0:
            dic[word] = ["", np.zeros(vocab_size)]
            music_id = word
          elif word_count == 1:
            dic[music_id][0] = word
          else:
            splitt = word.split(":")
            word_id, count = int(splitt[0]), int(splitt[1])
            dic[music_id][1][word_id-1] = count  # -1 because 0 is reserved
          word_count += 1
    return dic


def load_bow(df, bow_dic):
    entry_indexes = df["id"]
    keys = ["train", "test", "val"]
    bow = {key: [] for key in keys}
    y = {key: [] for key in keys}
    for entry_idx in entry_indexes:
      try:
        entry = df.loc[entry_idx]
        item = bow_dic.get(entry['msd_track_id'])
        if item:
          bow[entry['set']] += [item[1]]
          y[entry['set']] += [item[0]]
      except KeyError:
        pass
    for key in bow.keys():
      bow[key] = np.array(bow[key])
    for key in y.keys():
      y[key] = msdi_io.custom_dummies(y[key])
    return bow, y
    

def get_bow_x_y(lyrics_path, vocab_size):
    fil = open(lyrics_path, "r")
    x, y = [], []
    for sentence in fil.readlines():
        word_count = 0
        one_bow = np.zeros(vocab_size)
        for word in sentence.split():
          if word_count == 0:
            pass
          elif word_count == 1:
            y += [word]
          else:
            splitt = word.split(":")
            word_id, count = int(splitt[0]), int(splitt[1])
            one_bow[word_id-1] = count  # -1 because 0 is reserved
          word_count += 1
        x += [one_bow]
    x = np.array(x)
    y = msdi_io.custom_dummies(y)
    return x, y
