# Documentation

## `make_img(data, obs, params, mode="mean")`
This function takes physics data in the form `[p, x, y, z, theta, phi]` and the corresponding cluster energy `obs`, and bins it into an n-D image based on the information specified by `params`.

- **Parameters:**
  - `data`: Physics data in the form `[p, x, y, z, theta, phi]`
  - `obs`: Corresponding cluster energy for each physics event
  - `params`: Dictionary with keys `X`, `Y`, `theta`, and `phi`. For each coordinate where the resolution is specified, it will be used to create the image, and all remaining coordinates will be integrated across.
  - `mode`: Function to apply across each bin to get a value (default `mean`)

- **Returns:**
  - `n x m` matrix where each entry is a coordinate bin, and the corresponding value is given by `mode` applied to all data points in that bin.

---

## `fft_filter(data, n_sigma=1, make_plots=False, make_sparse=False, no_filter=False)`
Applies a Fourier filter to physics data after the `make_img` preprocessing step.

- **Parameters:**
  - `data`: `n x m` matrix of binned data produced by `make_img`
  - `n_sigma`: Number of standard deviations to use in filtering (default `1`)
  - `make_plots`: Whether to plot the filtered and unfiltered data for comparison (default `False`)
  - `make_sparse`: Whether to return the full filtered Fourier transformed data or just the values that pass the filtering (default `False`)
  - `no_filter`: Whether to skip the filtering process (default `False`)

- **Returns:**
  - Filtered and Fourier-transformed version of `data`. If `make_sparse=True`, returns a list of tuples that can be used to reconstruct the data later.

---

## `fft_filter_pairwise(data, obs, params_pairwise, n_sigma=1, make_plots=False, make_sparse=False, no_filter=False, mode="mean")`
Applies `make_img` and then `fft_filter` to physics data of the form `[p, x, y, z, theta, phi]` and the corresponding cluster energy `obs` for each coordinate pair.

- **Parameters:**
  - `data`: Physics data in the form `[p, x, y, z, theta, phi]`
  - `obs`: Corresponding cluster energy for each physics event
  - `params_pairwise`: Pairwise parameters for pre-processing (see: `params` in `make_img`)
  - `n_sigma`: Cutoff for filtering (default `1`)
  - `make_plots`: Whether to plot the filtered and unfiltered data for comparison (default `False`)
  - `make_sparse`: Whether to return the full filtered Fourier transformed data or just the values that pass the filtering (default `False`)
  - `no_filter`: Whether to skip the filtering process (default `False`)
  - `mode`: Function to apply across each bin to get a value (default `mean`)

- **Returns:**
  - Dictionary of outputs from `fft_filter`, where each key corresponds to a coordinate pair.

---

## `reconstruct_img(sparse_data, shape, background=None, apply_ifft=False)`
Given sparse data in the form of `(coordinate_1, coordinate_2, value)`, builds an `n x m` matrix.

- **Parameters:**
  - `sparse_data`: Sparse data in the form of `(coordinate_1, coordinate_2, value)`
  - `shape`: Tuple `(n, m)` specifying the shape of the output matrix
  - `background`: `n x m` matrix of background data to impute for the missing values (optional, default is that `1 + 0i` is imputed for each missing value)
  - `apply_ifft`: Whether to apply an inverse Fourier transform

- **Returns:**
  - `n x m` matrix of reconstructed data.

---

## `complete_data_pw(data, background_pw=None)`
Applies the `reconstruct_img` function to each coordinate pair.

- **Parameters:**
  - `data`: Dictionary where each key corresponds to a coordinate pair and the entry is the sparse data to be completed using `reconstruct_img`
  - `background_pw`: `n x m` matrix of background data to impute for the missing values (optional)

- **Returns:**
  - Dictionary where each key corresponds to a coordinate pair and each entry is the completed matrix.

---

## `load_from_sparse(return_full=False, a=1, n_sigma=1, background_pw=None, p=None)`
Loads sparse data from pickle files.

- **Parameters:**
  - `return_full`: Whether to return the full filtered or unfiltered data (default `False`)
  - `a`: Float scalar to multiply by (default `1`)
  - `n_sigma`: Cutoff for filtering (default `1`)
  - `background_pw`: `n x m` matrix of background data to impute for the missing values (optional)
  - `p`: Momentum value of the physics data

- **Returns:**
  - Dictionary with each key corresponding to a coordinate pair, where each entry is the completed matrix.

---

## `make_sparse_dataset(data, obs, params, n_sigma, p=None)`
Applies Fourier filter to dataset and saves in sparse format.

- **Parameters:**
  - `data`: Physics data in the form `[p, x, y, z, theta, phi]`
  - `obs`: Corresponding cluster energy for each physics event
  - `params`: Parameters for preprocessing (see: `make_img`)
  - `n_sigma`: Cutoff for filtering (default `1`)
  - `p`: Momentum value of the physics data

- **Saves:**
  - Sparse data under `fftprep/data/[coordinate pair]_n_sigma=[n_sigma]_p=[p]_sparse.pkl`.

---

## `fftpredictor`
A class for predicting outputs using FFT-filtered physics data.

### `__init__(self, data=None, params=None, n_sigma=1, bins=None)`
Initializes the predictor with data, parameters, and optional filtering options.

- **Parameters:**
  - `data`: Physics data (optional)
  - `params`: Dictionary with the resolution for each coordinate (optional)
  - `n_sigma`: Cutoff for filtering (default `1`)
  - `bins`: List of bins for momentum (default `[1000, 5000, 10000]`)

### `predict(self, vec, p, mode="reg")`
Predicts the output for a given vector of coordinates and a momentum value `p`.

- **Parameters:**
  - `vec`: Vector of coordinates
  - `p`: Momentum value of the physics data
  - `mode`: Prediction mode, either `"reg"` or `"all"` (default `"reg"`)

- **Returns:**
  - Predicted value based on the FFT-filtered data.

### `fit_linear(self, inputs, outputs)`
Fits a linear regression model to the inputs and outputs using FFT-filtered data.

- **Parameters:**
  - `inputs`: Input data for the regression model
  - `outputs`: Target data for the regression model

---

## `generate_params_pairwise(params)`
Generates a set of pairwise parameters for use in the `fft_filt_pairwise` function.

- **Parameters:**
  - `params`: Dictionary of parameters for `make_img`

- **Returns:**
  - Dictionary with each key corresponding to a coordinate pair and each entry the set of parameters for that pair.

---

## `assemble_sparse_data(files, params, n_sigma=1)`
Assembles a collection of sparse datasets from a list of files and saves them locally.

- **Parameters:**
  - `files`: List of file names
  - `params`: Parameters for `make_img`
  - `n_sigma`: Cutoff for filtering (default `1`)

- **Saves:**
  - Sparse data in `fftprep/data` in pickle format.
