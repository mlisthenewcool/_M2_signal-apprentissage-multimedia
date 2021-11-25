import numpy as np
import msdi_io
from skimage.transform import resize
from skimage.feature import hog
from sklearn import preprocessing


def get_raw_data(df, data_path, data_path_y, load_data=True):
    X_set = {"val": [], "train": [], "test": []}
    y_set = {"val": [], "train": [], "test": []}

    if load_data:
        for dset in data_path.keys():
            X_set[dset] = np.load(data_path[dset])
            y_set[dset] = np.load(data_path_y[dset])
            print("Loading done.")
    else:
        # Manually retrieve data
        not_found = 0
        for idx in range(df.shape[0]):
            entry = df.loc[idx]
            entry_set = msdi_io.get_set(entry)
            try:
                X_set[entry_set].append(msdi_io.load_img(entry))
                y_set[entry_set].append(msdi_io.get_label(entry))
            except FileNotFoundError:
                not_found += 1
        for dset in X_set.keys():
            X_set[dset] = np.array(X_set[dset])
            y_set[dset] = np.array(y_set[dset])
        print("Raw data loading done. Files not found :", not_found)
        for dset in data_path.keys():
            np.save(data_path[dset], X_set[dset])
            print("Data", dset, "saved at : ", data_path[dset])            

    print("Train : ", final_set["train"].shape)
    print("Test : ", final_set["test"].shape)
    print("Val : ", final_set["val"].shape)
    return X_set, y_set

def get_features(df, raw_path, data_path, data_path_y, load_data=True,
                 split=False, scaling=True):
    X_set = {"val": [], "train": [], "test": []}
    y_set = {"val": [], "train": [], "test": []}

    for dset in data_path.keys():
        y_set[dset] = np.load(data_path_y[dset])

    final_set = {"val": [], "train": [], "test": []}
    final_set2 = {"val": [], "train": [], "test": []}

    if load_data:
        for dset in data_path.keys():
            X_set[dset] = np.load(data_path[dset])
            if scaling:
              X_set[dset] = preprocessing.scale(X_set[dset])
            if split:
              final_set[dset] = X_set[dset][:, :-9]
              final_set2[dset] = X_set[dset][:, -9:]
            else:
              final_set[dset] = X_set[dset]
        print("Loading features done.")
    else:
        for dset in X_set.keys():
            X_set[dset] = np.load(raw_path[dset])
            for i in range(len(X_set[dset])):
                fd = hog(X_set[dset][i], orientations=8, pixels_per_cell=(200 // 3, 200 // 3),
                         cells_per_block=(1, 1), visualize=False, multichannel=True)
                resam = resize(X_set[dset][i], (3, 3)).flatten()
                full = np.hstack((fd, resam)).tolist()
                if split:
                    final_set[dset] += [resam]
                    final_set2[dset] += [fd]
                else:
                    final_set[dset] += [full]

            final_set[dset] = np.array(final_set[dset])
            final_set[dset] = final_set[dset] / final_set[dset].max()

            if split:
                final_set2[dset] = np.array(final_set2[dset])
                final_set2[dset] = final_set2[dset] / final_set2[dset].max()

        print("Extracting features done.")
        for dset in data_path.keys():
            if split:
              np.save(data_path[dset], np.hstack((final_set[dset], final_set2[dset])))
            else:
              np.save(data_path[dset], final_set[dset])
            print("Data", dset, "saved at : ", data_path[dset])
    
    X_set = None  # Free memory
    print("Train : ", final_set["train"].shape)
    print("Test : ", final_set["test"].shape)
    print("Val : ", final_set["val"].shape)
    if split:
        print("\t---")
        print("Train : ", final_set2["train"].shape)
        print("Test : ", final_set2["test"].shape)
        print("Val : ", final_set2["val"].shape)
        return final_set, final_set2, y_set
    return final_set, y_set
