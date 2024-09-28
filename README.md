```python
from fftprep.fftprep import fftpredictor, assemble_sparse_data, load_multi_from_sparse, get_training_data
import numpy as np
```

# 1. Generating Data

First we need to generate sparse datasets for our algorithm to run off of. To this end we specify coordinate ranges and resolutions that give use the corresponding binning for the data. The sparse data will be saved in `fftprep/data` and be labeled according to which physics experiment it is from as well as the thresholding parameter $\sigma$ used.


```python
# set parameters
params = {
    "X": {
        "range": (500,3500),
        "reso": 0.1
    }, 
    "Y": {
        "range": (500,2800),
        "reso": 0.1
    }
    , 
    "theta": {
        "range": (0,0.6),
        "reso": 100
    }, 
    "phi": {
        "range": (0,2*np.pi),
        "reso": 10
    }
}
```

Here we can specify all of the files we want to use in order to generate sparse data. As we generate more data we can extend the list of files to for each additional momentum bin. Make sure the files are in the format `FullSim_[p]GeV.root` where `p` is the associated momentum value of the data in GeV. The `assemble_sparse_data` will take the given files and parameters and construct a full set of sparse data using our fourier filtering method to be used by the `fftpredictor` class. Once you have run this block once -- unless you have a new data set to add of course -- it won't be need to run again.


```python
# add all files in order of momentum
files = ["FullSim_1GeV.root", "FullSim_5GeV.root", "FullSim_10GeV.root"] 

# generate sparse data for storage
assemble_sparse_data(files, params)

# Note that this block might run for a bit and do nothing, there is a bug with 
# jupyter that blocks the print statements while a function is being called.
```

    [1/2] Loading file: FullSim_1GeV.root (this may take a while) ...
    [2/3] Subsetting data based on parameters: {'X': {'range': (500, 3500), 'reso': 0.1}, 'Y': {'range': (500, 2800), 'reso': 0.1}, 'theta': {'range': (0, 0.6), 'reso': 100}, 'phi': {'range': (0, 6.283185307179586), 'reso': 10}} ...
    [3/3] Making sparse datasets at fftprep/data ...
    [1/2] binning data...


    100%|█████████████████████████████████████████████| 4/4 [00:00<00:00,  5.78it/s]


    [2/2] creating image...
    [1/2] performing fast Fourier transform along 2 axes...
    [2/2] filtering Fourier transformed data with cutoff e^5.653977110014694


    69000it [00:00, 1744792.43it/s]

    [1/2] binning data...


    
    100%|█████████████████████████████████████████████| 4/4 [00:00<00:00, 19.72it/s]


    [2/2] creating image...
    [1/2] performing fast Fourier transform along 2 axes...
    [2/2] filtering Fourier transformed data with cutoff e^4.565716845313729


    18000it [00:00, 2015200.51it/s]

    [1/2] binning data...


    
    100%|█████████████████████████████████████████████| 4/4 [00:00<00:00, 22.06it/s]


    [2/2] creating image...
    [1/2] performing fast Fourier transform along 2 axes...
    [2/2] filtering Fourier transformed data with cutoff e^4.654498856673975


    18900it [00:00, 1929189.98it/s]

    [1/2] binning data...


    
    100%|█████████████████████████████████████████████| 4/4 [00:00<00:00, 24.22it/s]


    [2/2] creating image...
    [1/2] performing fast Fourier transform along 2 axes...
    [2/2] filtering Fourier transformed data with cutoff e^4.144427013004881


    13800it [00:00, 2023754.25it/s]

    [1/2] binning data...


    
    100%|█████████████████████████████████████████████| 4/4 [00:00<00:00, 24.64it/s]


    [2/2] creating image...
    [1/2] performing fast Fourier transform along 2 axes...
    [2/2] filtering Fourier transformed data with cutoff e^4.173029798646244


    14490it [00:00, 1937498.88it/s]

    [1/2] binning data...


    
    100%|█████████████████████████████████████████████| 4/4 [00:00<00:00, 21.98it/s]


    [2/2] creating image...
    [1/2] performing fast Fourier transform along 2 axes...
    [2/2] filtering Fourier transformed data with cutoff e^2.980630964991211


    3780it [00:00, 2032363.69it/s]

    [1/2] Loading file: FullSim_5GeV.root (this may take a while) ...


    


    [2/3] Subsetting data based on parameters: {'X': {'range': (500, 3500), 'reso': 0.1}, 'Y': {'range': (500, 2800), 'reso': 0.1}, 'theta': {'range': (0, 0.6), 'reso': 100}, 'phi': {'range': (0, 6.283185307179586), 'reso': 10}} ...
    [3/3] Making sparse datasets at fftprep/data ...
    [1/2] binning data...


    100%|█████████████████████████████████████████████| 4/4 [00:00<00:00,  5.87it/s]


    [2/2] creating image...
    [1/2] performing fast Fourier transform along 2 axes...
    [2/2] filtering Fourier transformed data with cutoff e^6.537277372475305


    69000it [00:00, 245489.62it/s]


    [1/2] binning data...


    100%|█████████████████████████████████████████████| 4/4 [00:00<00:00, 20.23it/s]


    [2/2] creating image...
    [1/2] performing fast Fourier transform along 2 axes...
    [2/2] filtering Fourier transformed data with cutoff e^5.575899717807056


    18000it [00:00, 2023193.05it/s]

    [1/2] binning data...


    
    100%|█████████████████████████████████████████████| 4/4 [00:00<00:00, 24.66it/s]


    [2/2] creating image...
    [1/2] performing fast Fourier transform along 2 axes...
    [2/2] filtering Fourier transformed data with cutoff e^5.698401779898349


    18900it [00:00, 1993721.12it/s]

    [1/2] binning data...


    
    100%|█████████████████████████████████████████████| 4/4 [00:00<00:00, 26.57it/s]


    [2/2] creating image...
    [1/2] performing fast Fourier transform along 2 axes...
    [2/2] filtering Fourier transformed data with cutoff e^5.168410811385815


    13800it [00:00, 1784425.05it/s]

    [1/2] binning data...


    
    100%|█████████████████████████████████████████████| 4/4 [00:00<00:00, 24.00it/s]


    [2/2] creating image...
    [1/2] performing fast Fourier transform along 2 axes...
    [2/2] filtering Fourier transformed data with cutoff e^5.1939860558600195


    14490it [00:00, 1931464.60it/s]

    [1/2] binning data...


    
    100%|█████████████████████████████████████████████| 4/4 [00:00<00:00, 25.54it/s]


    [2/2] creating image...
    [1/2] performing fast Fourier transform along 2 axes...
    [2/2] filtering Fourier transformed data with cutoff e^3.873938157007859


    3780it [00:00, 1723686.58it/s]

    [1/2] Loading file: FullSim_10GeV.root (this may take a while) ...


    


    [2/3] Subsetting data based on parameters: {'X': {'range': (500, 3500), 'reso': 0.1}, 'Y': {'range': (500, 2800), 'reso': 0.1}, 'theta': {'range': (0, 0.6), 'reso': 100}, 'phi': {'range': (0, 6.283185307179586), 'reso': 10}} ...
    [3/3] Making sparse datasets at fftprep/data ...
    [1/2] binning data...


    100%|█████████████████████████████████████████████| 4/4 [00:00<00:00, 43.91it/s]


    [2/2] creating image...
    [1/2] performing fast Fourier transform along 2 axes...
    [2/2] filtering Fourier transformed data with cutoff e^7.312362483214127


    69000it [00:00, 2012300.02it/s]

    [1/2] binning data...


    
    100%|█████████████████████████████████████████████| 4/4 [00:00<00:00, 63.21it/s]


    [2/2] creating image...
    [1/2] performing fast Fourier transform along 2 axes...
    [2/2] filtering Fourier transformed data with cutoff e^6.280883229660546


    18000it [00:00, 2017192.72it/s]

    [1/2] binning data...


    
    100%|█████████████████████████████████████████████| 4/4 [00:00<00:00, 55.36it/s]


    [2/2] creating image...
    [1/2] performing fast Fourier transform along 2 axes...
    [2/2] filtering Fourier transformed data with cutoff e^6.3874940474698985


    18900it [00:00, 2033302.01it/s]

    [1/2] binning data...


    
    100%|█████████████████████████████████████████████| 4/4 [00:00<00:00, 57.68it/s]


    [2/2] creating image...
    [1/2] performing fast Fourier transform along 2 axes...
    [2/2] filtering Fourier transformed data with cutoff e^5.923635968339074


    13800it [00:00, 1738493.28it/s]

    [1/2] binning data...


    
    100%|█████████████████████████████████████████████| 4/4 [00:00<00:00, 55.01it/s]


    [2/2] creating image...
    [1/2] performing fast Fourier transform along 2 axes...
    [2/2] filtering Fourier transformed data with cutoff e^5.958348598402122


    14490it [00:00, 1842127.33it/s]

    [1/2] binning data...


    
    100%|█████████████████████████████████████████████| 4/4 [00:00<00:00, 57.72it/s]


    [2/2] creating image...
    [1/2] performing fast Fourier transform along 2 axes...
    [2/2] filtering Fourier transformed data with cutoff e^4.640366336696195


    3780it [00:00, 2092999.22it/s]


# 2. Initializing the model

Now that we have the necessary data we can intialize our `fftpredictor` model with this data. The `bins` parameter should have all of the momentum bins you want to load in MeV specified. Also make sure the `params` parameter matches with the one you used to generate the data originally, so the algorithm knows with what range and resoltuion the binning was done. Once we have loaded the model we can fit a linear regression to the pairwise coordinate predictions it gives, allowing it to give predictions for any set of coordinate $p, x, y, \theta, \phi$. The `files` parameter in `get_training_data` should match the files parameter used in `assemble_sparse_data` in the previous section.


```python
# first we initilize the model from the files we generated in the previous code block
# if you have issues here ensure that your file system is set up correctly and that 
# the sparse data is saved under fftprep/data
model = fftpredictor(load_multi_from_sparse(bins = [1000,5000,10000]), params=params)

# Now we load the training data and fit a linear regression to the data 
model.fit_linear(*get_training_data(files, params))
```

    [1/2] Loading file: FullSim_1GeV.root (this may take a while) ...
    [2/2] Subsetting data based on parameters: {'X': {'range': (500, 3500), 'reso': 0.1}, 'Y': {'range': (500, 2800), 'reso': 0.1}, 'theta': {'range': (0, 0.6), 'reso': 100}, 'phi': {'range': (0, 6.283185307179586), 'reso': 10}} ...
    [1/2] Loading file: FullSim_5GeV.root (this may take a while) ...
    [2/2] Subsetting data based on parameters: {'X': {'range': (500, 3500), 'reso': 0.1}, 'Y': {'range': (500, 2800), 'reso': 0.1}, 'theta': {'range': (0, 0.6), 'reso': 100}, 'phi': {'range': (0, 6.283185307179586), 'reso': 10}} ...
    [1/2] Loading file: FullSim_10GeV.root (this may take a while) ...
    [2/2] Subsetting data based on parameters: {'X': {'range': (500, 3500), 'reso': 0.1}, 'Y': {'range': (500, 2800), 'reso': 0.1}, 'theta': {'range': (0, 0.6), 'reso': 100}, 'phi': {'range': (0, 6.283185307179586), 'reso': 10}} ...
    [1/2] Getting predictions for each coordinate pair from FFT data...


    100%|████████████████████████████| 60070237/60070237 [12:54<00:00, 77545.33it/s]


    [2/2] computing linear regression coefficients...
    [1/2] Getting predictions for each coordinate pair from FFT data...


    100%|████████████████████████████| 60403703/60403703 [13:09<00:00, 76476.40it/s]


    [2/2] computing linear regression coefficients...
    [1/2] Getting predictions for each coordinate pair from FFT data...


    100%|████████████████████████████| 25487788/25487788 [05:21<00:00, 79333.08it/s]


    [2/2] computing linear regression coefficients...


Once the linear regression coefficients have been determined as well as the intercepts we can simply save them (just copy the plain text output from this block or your command line). This allows us to just use the model without doing any fitting on every future use. Again unless you have new data to add you can just use the same coefficients on every run, instead of having to run this block every time the model is intialized.


```python
# these are the coeficients determined by the linear regression
coef = model.coef_
coef
```




    [array([2.06760854e-01, 2.24318204e-01, 7.27489415e-01, 5.77515270e-01,
            6.82154470e-01, 1.99721306e-01, 1.28499286e-05, 1.53433174e-05,
            4.11704027e-02, 2.51585091e-02]),
     array([-5.68759353e-02,  4.58101256e-01,  7.74284749e-01,  7.95634036e-01,
             8.37574140e-01, -2.51841410e-01,  4.06784754e-05,  1.40460751e-04,
             8.57536271e-02,  6.60613376e-02]),
     array([ 1.61857691e-01,  4.24262543e-01,  7.06796277e-01,  8.09580389e-01,
             8.57487536e-01, -2.26864829e-01,  1.35914080e-04,  4.50624295e-04,
             4.22329996e-01,  6.73391904e-02])]




```python
# these are the intercepts determined by the linear regression

intercept = model.intercept_
intercept
```




    [-182.24575061427902, -988.6440171217222, -2261.3892557711224]



This is how you set the linear regression coefficents and intercepts in the case you already have determined them:


```python
# now that we have determined the coefficients and intercepts every 
# time we use the model we can initialize it using these as follows:

model = fftpredictor(load_multi_from_sparse([1000,5000,10000]), params=params)
model.coef_ = coef # write whatever coeficients you want here, but keeping the same format as the outputs above
model.intercept_ = intercept# write whatever intercepts you want here, but keeping the same format as the outputs above
```

And now running the model to make predictions is as easy as follows:


```python
x,y,theta,phi = 1000,1000,0.1,1.2
model.predict(vec = (x,y,theta,phi), p = 5893)
```




    626.1013379545213



Here the `vec` input is just the coordinates of the particle you want to simulate. Here we are saying that the particle begins at $(1000,1000)$ where the $x,y$ values are given in millimeters relative to the center of the calorimeter. Then the particle travels according to $\theta,\phi$ and creates a shower. Last we specify the mometum of the particle with the `p` parameter. The function returns the predicted cluster energy given the specified parameters.
