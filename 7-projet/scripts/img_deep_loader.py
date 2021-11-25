import numpy as np
import msdi_io
import tensorflow as tf


def get_img_dic(start, lim, df):
  dic = {}
  for i in range(start, lim):
      entry = df.loc[i]
      try:
          img = msdi_io.load_img(entry)
          dic[entry["msd_track_id"]] = [msdi_io.get_label(entry), img]
      except FileNotFoundError:
          pass
  return dic


def load_img(start, lim, df):
    x = []
    for i in range(start, lim):
        entry = df.loc[i]
        try:
            x.append(msdi_io.load_img(entry))
        except FileNotFoundError:
            pass
    return np.array(x)


def load_label(start, lim, df):
    y = []
    for i in range(start, lim):
        entry = df.loc[i]
        try:
            y.append(msdi_io.get_label(entry))
        except FileNotFoundError:
            pass
    return msdi_io.custom_dummies(y, msdi_io.get_label_list())


def image_loader(dset, batch_size, df, dset_size):

    L = dset_size[dset]

    # this line is just to make the generator infinite, keras needs that
    while True:

        batch_start = 0
        batch_end = batch_size

        while batch_start < L:
            limit = min(batch_end, L)
            X = load_img(batch_start, limit, dset, df, dset_size)
            Y = load_label(batch_start, limit, dset, df, dset_size)

            yield (X, Y)  # a tuple with two numpy arrays with batch_size samples

            batch_start += batch_size
            batch_end += batch_size
