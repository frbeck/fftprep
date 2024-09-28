import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
import pickle
import re
from ROOT import RDataFrame

ind_map = {
    "X": 1,
    "Y": 2,
    "theta": 4,
    "phi": 5,
    1: "X",
    2: "Y",
    4: "theta",
    5: "phi"
}


def make_img(data, obs, params, mode="mean"):
    """
    This function takes physics data in the form [p, x, y, z, theta, phi] and
    the corresponding cluster energy obs and bins it into a n-D image based on
    the information specified by params.

    :param data: physics data in the form [p, x, y, z, theta, phi]
    :param obs: corresponding cluster energy for each physics event
    :param params: dictionary with keys [X, Y, theta, phi], under each key the
    of values for each coordinate is specified. For each coordinate where the
    resolution is specified will be used to create the image, all remaining
    coordinates will be integrated across.
    for example:

    "X_v_theta": {
        "X": {
            "range": (500, 3500),
            "reso": 0.1
        },
        "Y": {
            "range": (500, 2500),
            "reso": None
        }
        ,
        "theta": {
            "range": (0, 0.6),
            "reso": 100
        },
        "phi": {
            "range": (0, 2 * np.pi),
            "reso": None
        }
    }

    Would give an image where one dimension is the X-coordinate and the other is
    the theta-coordinate

    :param mode: function to apply across each bin to get a value (default mean)
    :return: n by m matrix where each entry is a coordinate bin and corresponding
    value is given by mode applied to all data points in that bin.
    """
    ind_map = {
        "X": 1,
        "Y": 2,
        "theta": 4,
        "phi": 5
    }

    print(f"[1/2] binning data...")
    data = data.transpose()
    subset = (data[1] > params["X"]["range"][0]) * (data[1] < params["X"]["range"][1]) * \
             (data[2] > params["Y"]["range"][0]) * (data[2] < params["Y"]["range"][1]) * \
             (data[4] > params["theta"]["range"][0]) * (data[4] < params["theta"]["range"][1]) * \
             (data[5] > params["phi"]["range"][0]) * (data[5] < params["phi"]["range"][1])
    data = data.transpose()[subset].transpose()
    binned_data = {"E": obs[subset]}
    for key in tqdm(params.keys()):
        if params[key]["reso"]:
            bins = data[ind_map[key]] * params[key]["reso"]
            binned_data[key] = bins.astype('int')
    print("[2/2] creating image...")
    df = pd.DataFrame(binned_data)
    if mode == "mean":
        df_binned = df.groupby(list(binned_data.keys())[1:]).mean()
    elif mode == "std":
        df_binned = df.groupby(list(binned_data.keys())[1:]).std()
    elif mode == "counts":
        df_binned = df.groupby(list(binned_data.keys())[1:]).count()
    elif mode == "max":
        df_binned = df.groupby(list(binned_data.keys())[1:]).max()
    elif mode == "min":
        df_binned = df.groupby(list(binned_data.keys())[1:]).min()
    else:
        print(f"ERROR: parameter 'mode' must be either 'mean' or 'std'. Specified value: {mode}")
    E_by_bin = np.array(df_binned.E)
    return E_by_bin.reshape(
        [int(max(binned_data[key]) - min(binned_data[key]) + 1) for key in list(binned_data.keys())[1:]])


def fft_filter(data, n_sigma=1, make_plots=False, make_sparse=False, no_filter=False):
    """
    Apply a fourier filter to physics data after the make_img preprocessing step.

    :param data: nxm matrix of binned data produced by the make_img function
    :param n_sigma: number of standard deviations to use in filtering (default 1)
    :param make_plots: whether to plot the filtered and unfiltered data for
    comparison (default False)
    :param make_sparse: whether to return the full filtered fourier transformed
    data or just the values that pass the filtering (default False)
    :param no_filter: whether to not perform filtering (default False)
    :return: filtered and fourier transformed version of data, if make_sparse=True
    this function will return a list of tuples that can be use to reconstruct the
    data later (see: load_from_sparse)
    """
    print(f"[1/2] performing fast Fourier transform along {len(data.shape)} axes...")
    data_fft = np.fft.fftn(data)
    if no_filter:
        return data_fft
    cutoff = np.mean(np.log(abs(data_fft.flatten()))) + n_sigma * np.std(np.log(abs(data_fft.flatten())))
    print(f"[2/2] filtering Fourier transformed data with cutoff e^{cutoff}")
    if make_sparse:
        saved_points = []
        for index, E in tqdm(np.ndenumerate(data_fft)):
            if np.log(abs(E)) > cutoff:
                saved_points.append((index, E))
        return saved_points
    else:
        for index, E in tqdm(np.ndenumerate(data_fft)):
            if np.log(abs(E)) < cutoff:
                data_fft[index] = 1 + 0j
            else:
                data_fft[index] = E
    if make_plots:
        plt.figure(figsize=(12, 8))
        plt.imshow(data, cmap='gray', origin="lower")
        plt.show()
        plt.figure(figsize=(12, 8))
        plt.imshow(abs(np.fft.ifftn(data_fft)), cmap='gray', origin="lower")
        plt.show()
    return data_fft


def fft_filter_pairwise(data, obs, params_pairwise, n_sigma=1, make_plots=False, make_sparse=False, no_filter=False,
                        mode="mean"):
    """
    Apply make_img and then fft_filter to physics data of the form
    [p, x, y, z, theta, phi] and the corresponding cluster energy obs
    for each coordinate pair.

    :param data: physics data in the form [p, x, y, z, theta, phi]
    :param obs: corresponding cluster energy for each physics event
    :param params_pairwise: pairwise parameters for pre-processing
    (see: params in make_img function) this is the same style of
    parameter but for each specific coordinate pair.
    :param n_sigma: cutoff for filtering (default 1)
    :param make_plots: whether to plot the filtered and unfiltered data for
    comparison (default False)
    :param make_sparse: whether to return the full filtered fourier transformed
    data or just the values that pass the filtering (default False)
    :param no_filter: whether to not perform filtering (default False)
    :param mode: function to apply across each bin to get a value (default mean)
    :return: dict of outputs from fft_filter where each key corresponds to a
    coordinate pair.
    """
    fft_filt_pw = {}
    for key in params_pairwise.keys():
        data_binned = make_img(data, obs, params_pairwise[key], mode=mode)
        if make_plots:
            print(f"Plotting {key}")
            print(f"{key.split("_")[0]} range is {params_pairwise[key][key.split("_")[0]]["range"]}")
            print(f"{key.split("_")[0]} bin scaling is {1 / params_pairwise[key][key.split("_")[0]]["reso"]}")
            print(f"{key.split("_")[2]} range is {params_pairwise[key][key.split("_")[2]]["range"]}")
            print(f"{key.split("_")[2]} bin scaling is {1 / params_pairwise[key][key.split("_")[2]]["reso"]}")
            fft_filt_pw[key] = fft_filter(data_binned, n_sigma=n_sigma, make_plots=True, no_filter=no_filter)
        elif make_sparse:
            fft_filt_pw[key] = fft_filter(data_binned, n_sigma=n_sigma, make_sparse=True, no_filter=no_filter)
        else:
            fft_filt_pw[key] = fft_filter(data_binned, n_sigma=n_sigma, no_filter=no_filter)
    return fft_filt_pw


def reconstruct_img(sparse_data, shape, background=None, apply_ifft=False):
    """
    given sparse data in the form of (coordinate_1, coordinate_2, value) build an
    nxm matrix

    :param sparse_data: sparse data in the form of (coordinate_1, coordinate_2, value)
    as given by fft_filter
    :param shape: tuple (n,m) specifying the shape of the output matrix
    :param background: nxm matrix of background data to impute for the missing
     values (optional, default is that 1+0i is imputed for each missing value)
    :param apply_ifft:
    :return: nxm matrix of reconstructed data
    """
    if background is None:
        a = np.ones(shape, dtype='complex')
    else:
        a = background
    for e in sparse_data:
        a[e[0]] = e[1]
    if apply_ifft:
        a = abs(np.fft.ifftn(a))
    return a


def complete_data_pw(data, background_pw=None):
    """
    apply the reconstruct_img function to each coordinate pair.

    :param data: dict with each key corresponding to a coordinate pair and the entry
    is the sparse data to be completed using reconstruct_img
    :param background_pw: nxm matrix of background data to impute for the missing
    values (optional, default is that 1+0i is imputed for each missing value)
    :return: dict with each key corresponding to a coordinate pair and each entry
    is the completed matrix
    """
    completed_data = {}
    for key in data:
        var1_min = min([data[key][i][0][0] for i in range(len(data[key]))])
        var2_min = min([data[key][i][0][1] for i in range(len(data[key]))])
        var1_max = max([data[key][i][0][0] for i in range(len(data[key]))])
        var2_max = max([data[key][i][0][1] for i in range(len(data[key]))])
        completed_data[key] = reconstruct_img(
            data[key],
            (var1_max - var1_min + 1, var2_max - var2_min + 1),
            background=background_pw[key] if background_pw else None
        )
    return completed_data


def load_from_sparse(return_full=False, a=1, n_sigma=1, background_pw=None, p=None):
    """
    load sparse data from pickle files

    :param return_full: whether to return the full filtered or unfiltered data (default False)
    :param a: float scaler to multiply by (default 1)
    :param n_sigma: cutoff for filtering (default 1)
    :param background_pw: background_pw: nxm matrix of background data to impute for the missing
    values (optional, default is that 1+0i is imputed for each missing value)
    :param p: momentum value of the physics data
    :return: dict with each key corresponding to a coordinate pair and each entry
    is the completed matrix
    """
    keys = ['X_v_Y', 'X_v_theta', 'X_v_phi', 'Y_v_theta', 'Y_v_phi', 'theta_v_phi']
    fft_filt_pw_sparse = {}
    for key in keys:
        if p:
            with open(f"fftprep/data/{key}_n_sigma={n_sigma}_p={p}gev_sparse.pkl", "rb") as f:
                fft_filt_pw_sparse[key] = pickle.load(f)
        else:
            with open(f"fftprep/data/{key}_n_sigma={n_sigma}_sparse.pkl", "rb") as f:
                fft_filt_pw_sparse[key] = pickle.load(f)

    if return_full:
        fft_filt_pw = complete_data_pw(fft_filt_pw_sparse, background_pw=background_pw)
        return {key: abs(np.fft.ifftn(a * fft_filt_pw[key])) for key in fft_filt_pw.keys()}
    else:
        return fft_filt_pw_sparse


def make_sparse_dataset(data, obs, params, n_sigma, p=None):
    """
    fourier filter dataset and save in sparse format.

    :param data: physics data in the form [p, x, y, z, theta, phi]
    :param obs: corresponding cluster energy for each physics event
    :param params: parameters for preprocessing (see: make_img)
    :param n_sigma: cutoff for filtering (default 1)
    :param p: momentum value of the physics data

    saves sparse data under
    "fftprep/data/[coordinate pair]_n_sigma=[n_sigma]_p=[p]_sparse.pkl"
    """
    fft_sparse = fft_filter_pairwise(data, obs, params, n_sigma=n_sigma, make_sparse=True)
    for key in fft_sparse.keys():
        if p is None:
            with open(f"fftprep/data/{key}_n_sigma={n_sigma}_sparse.pkl", "wb") as output_file:
                pickle.dump(fft_sparse[key], output_file)
        else:
            with open(f"fftprep/data/{key}_n_sigma={n_sigma}_p={p}gev_sparse.pkl", "wb") as output_file:
                pickle.dump(fft_sparse[key], output_file)


class fftpredictor():
    def __init__(self, data=None, params=None, n_sigma=1, bins=None):
        if bins is None:
            self.bins = [1000, 5000, 10000]
        if data is None:
            self.data = load_from_sparse(return_full=True, n_sigma=n_sigma, background_pw=None)
        else:
            self.data = data
        self.params = params
        if params is None:
            self.params = {
                "X": {
                    "range": (500, 3500),
                    "reso": 0.1
                },
                "Y": {
                    "range": (500, 2500),
                    "reso": 0.1
                }
                ,
                "theta": {
                    "range": (0, 0.05),
                    "reso": 100
                },
                "phi": {
                    "range": (0, 2 * np.pi),
                    "reso": 10
                }
            }
        else:
            self.params = params

    def predict(self, vec, p, mode="reg"):

        if p in self.bins:
            predictions_pw = []
            coords = list(combinations(vec, 2))
            bin_ind = self.bins.index(p)
            data = self.data[bin_ind]
            for i, key in enumerate(data.keys()):
                var1 = key.split("_")[0]
                var2 = key.split("_")[2]
                coords_scales = (
                    int((coords[i][0] - self.params[var1]["range"][0]) * self.params[var1]["reso"]),
                    int((coords[i][1] - self.params[var2]["range"][0]) * self.params[var2]["reso"])
                )
                predictions_pw.append(data[key][coords_scales])
            if mode == "all":
                return predictions_pw
            return np.sum(np.concatenate((predictions_pw, vec)) * self.coef_[bin_ind]) + self.intercept_[bin_ind]

        else:
            nearest_bins_ind = (np.digitize(p, self.bins) - 1, np.digitize(p, self.bins))
            nearest_bins = (self.bins[nearest_bins_ind[0]], self.bins[nearest_bins_ind[1]])
            scale = abs(p - nearest_bins[0]) / (abs(p - nearest_bins[0]) + abs(p - nearest_bins[1]))
            predictions_pw = []
            coords = list(combinations(vec, 2))
            data_lower = self.data[nearest_bins_ind[0]]
            data_upper = self.data[nearest_bins_ind[1]]
            for i, key in enumerate(data_lower.keys()):
                var1 = key.split("_")[0]
                var2 = key.split("_")[2]
                coords_scaled = (
                    int((coords[i][0] - self.params[var1]["range"][0]) * self.params[var1]["reso"]),
                    int((coords[i][1] - self.params[var2]["range"][0]) * self.params[var2]["reso"])
                )
                predictions_pw.append(
                    data_lower[key][coords_scaled] * (1 - scale) + data_upper[key][
                        coords_scaled] * scale)
            inputs = np.concatenate((predictions_pw, vec))
            if mode == "all":
                return predictions_pw
            return (np.sum(inputs * self.coef_[nearest_bins_ind[0]]) + self.intercept_[nearest_bins_ind[0]]) * (
                        1 - scale) \
                + (np.sum(inputs * self.coef_[nearest_bins_ind[1]]) + self.intercept_[nearest_bins_ind[1]]) * (
                            1 - scale)

    def fit_linear(self, inputs, outputs):
        self.coef_ = []
        self.intercept_ = []
        for i, p in enumerate(self.bins):
            print("[1/2] Getting predictions for each coordinate pair from FFT data...")
            preds = []
            for row in tqdm(inputs[i]):
                preds.append(self.predict((row[1], row[2], row[4], row[5]), p, mode="all"))
            print("[2/2] computing linear regression coefficients...")
            preds = np.array(preds)
            inputs_featurized = np.concatenate((preds, np.delete(inputs[i], [0, 3], axis=1)), axis=1)
            reg = LinearRegression().fit(inputs_featurized, outputs[i])
            self.coef_.append(reg.coef_)
            self.intercept_.append(reg.intercept_)


def generate_params_pairwise(params):
    """
    given a set of parameters, generates a set of pairwise parameters to
    be use by the fft_filt_pairwise function or any other function that uses the
    pairwise parameter format.

    :param params: params for make_image, fill out the resolution value for each
    coordinate.
    :return: dictionary with each key corresponding to a coordinate pair and each
    entry the set of parameters for that pair
    """
    params_pw = {}
    key_pairs = list(combinations(params, 2))
    for pair in key_pairs:
        params_pw[pair[0] + "_v_" + pair[1]] = {
            pair[0]: {
                "range": params[pair[0]]["range"],
                "reso": params[pair[0]]["reso"]
            },
            pair[1]: {
                "range": params[pair[1]]["range"],
                "reso": params[pair[1]]["reso"]
            },
            list(params.keys() - list(pair))[0]: {
                "range": params[list(params.keys() - list(pair))[0]]["range"],
                "reso": None
            },
            list(params.keys() - list(pair))[1]: {
                "range": params[list(params.keys() - list(pair))[1]]["range"],
                "reso": None
            }
        }
    return params_pw


def assemble_sparse_data(files, params, n_sigma=1):
    """
    from a list of files assemble a collection of sparse datasets to be saved locally

    :param files: list of file names
    :param params: params for make_image, fill out the resolution value for each
    :param n_sigma: cutoff for filtering (default 1)

    saves sparse data in fftprep/data in pickle format
    """
    for file in files:
        print(f"[1/2] Loading file: {file} (this may take a while) ...")
        p = int(re.search(r'\d+', file).group())
        data = RDataFrame("t;1", file).AsNumpy()
        X = np.array(
            [
                data["part_p"],
                data["part_x"],
                data["part_y"],
                data["part_z"],
                data["part_theta"],
                data["part_phi"]
            ]
        ).transpose()
        obs = data["cl_E_ecal"]
        print(f"[2/3] Subsetting data based on parameters: {params} ...")
        subset = (X.transpose()[1] > params["X"]["range"][0]) \
                 * (X.transpose()[1] < params["X"]["range"][1]) \
                 * (X.transpose()[2] > params["Y"]["range"][0]) \
                 * (X.transpose()[2] < params["Y"]["range"][1]) \
                 * (X.transpose()[3] > 12280) \
                 * (X.transpose()[3] < 12300) \
                 * (X.transpose()[4] > params["theta"]["range"][0]) \
                 * (X.transpose()[4] < params["theta"]["range"][1]) \
                 * (X.transpose()[5] > params["phi"]["range"][0]) \
                 * (X.transpose()[5] < params["phi"]["range"][1])
        X, obs = X[subset], obs[subset]

        print(f"[3/3] Making sparse datasets at fftprep/data ...")
        make_sparse_dataset(X, obs, generate_params_pairwise(params), n_sigma, p)


def load_multi_from_sparse(bins, n_sigma=1, background_pw=None):
    """
    load multiple sparse datasets to be passed to the fft_predictor model.

    :param bins: list of momentum bins for which data is provided
    :param n_sigma: cutoff for filtering (default 1)
    :param background_pw: background data to be imputed (optional, see: load_from_sparse)
    :return: list of matrices constructed from the sparse data for each momentum bin
    """
    return [
        load_from_sparse(return_full=True, n_sigma=n_sigma, background_pw=background_pw, p=int(p / 1000)) \
        for p in bins
    ]


def get_training_data(files, params):
    """
    get training data for the fit_linear method of the fft_predictor model

    :param files: list of file names with the training data
    :param params: params for make_image, fill out the resolution value for each
    :return: training data for the fit_linear method of the fft_predictor model in the
    form ([X_1, X_2,...], [obs_1, obs_2,...]). (X_i is input data, obs is output data)
    """
    X_set = []
    obs_set = []
    for file in files:
        print(f"[1/2] Loading file: {file} (this may take a while) ...")
        p = int(re.search(r'\d+', file).group())
        data = RDataFrame("t;1", file).AsNumpy()
        X = np.array(
            [
                data["part_p"],
                data["part_x"],
                data["part_y"],
                data["part_z"],
                data["part_theta"],
                data["part_phi"]
            ]
        ).transpose()
        obs = data["cl_E_ecal"]
        print(f"[2/2] Subsetting data based on parameters: {params} ...")
        subset = (X.transpose()[1] > params["X"]["range"][0]) \
                 * (X.transpose()[1] < params["X"]["range"][1]) \
                 * (X.transpose()[2] > params["Y"]["range"][0]) \
                 * (X.transpose()[2] < params["Y"]["range"][1]) \
                 * (X.transpose()[3] > 12280) \
                 * (X.transpose()[3] < 12300) \
                 * (X.transpose()[4] > params["theta"]["range"][0]) \
                 * (X.transpose()[4] < params["theta"]["range"][1]) \
                 * (X.transpose()[5] > params["phi"]["range"][0]) \
                 * (X.transpose()[5] < params["phi"]["range"][1])
        X_set.append(X[subset])
        obs_set.append(obs[subset])
    return X_set, obs_set


params_pairwise = {
    "X_v_Y": {
        "X": {
            "range": (500, 3500),
            "reso": 0.1
        },
        "Y": {
            "range": (500, 2500),
            "reso": 0.1
        }
        ,
        "theta": {
            "range": (0, 0.6),
            "reso": None
        },
        "phi": {
            "range": (0, 2 * np.pi),
            "reso": None
        }
    },
    "X_v_theta": {
        "X": {
            "range": (500, 3500),
            "reso": 0.1
        },
        "Y": {
            "range": (500, 2500),
            "reso": None
        }
        ,
        "theta": {
            "range": (0, 0.6),
            "reso": 100
        },
        "phi": {
            "range": (0, 2 * np.pi),
            "reso": None
        }
    },
    "X_v_phi": {
        "X": {
            "range": (500, 3500),
            "reso": 0.1
        },
        "Y": {
            "range": (500, 2500),
            "reso": None
        }
        ,
        "theta": {
            "range": (0, 0.6),
            "reso": None
        },
        "phi": {
            "range": (0, 2 * np.pi),
            "reso": 10
        }
    },
    "Y_v_theta": {
        "X": {
            "range": (500, 3500),
            "reso": None
        },
        "Y": {
            "range": (500, 2500),
            "reso": 0.1
        }
        ,
        "theta": {
            "range": (0, 0.6),
            "reso": 100
        },
        "phi": {
            "range": (0, 2 * np.pi),
            "reso": None
        }
    },
    "Y_v_phi": {
        "X": {
            "range": (500, 3500),
            "reso": None
        },
        "Y": {
            "range": (500, 2500),
            "reso": 0.1
        }
        ,
        "theta": {
            "range": (0, 0.6),
            "reso": None
        },
        "phi": {
            "range": (0, 2 * np.pi),
            "reso": 10
        }
    },
    "theta_v_phi": {
        "X": {
            "range": (500, 3500),
            "reso": None
        },
        "Y": {
            "range": (500, 2500),
            "reso": None
        }
        ,
        "theta": {
            "range": (0, 0.6),
            "reso": 100
        },
        "phi": {
            "range": (0, 2 * np.pi),
            "reso": 10
        }
    }
}
