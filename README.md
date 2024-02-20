# DL-HRGNSS
DL-HRGNSS

### GNSS_Magnitude_V1

This is our preliminary DL model based on a convolutional neural network 
for magnitude estimation from HR-GNSS data (1 Hz sampling rate).

We have trained the model for three cases:

- Magnitude estimation from 3 stations, 181 seconds, 3 components.
- Magnitude estimation from 7 stations, 181 seconds, 3 components.
- Magnitude estimation from 7 stations, 501 seconds, 3 components.


## Related work:

Claudia Quinteros-Cartaya, Jonas Köhler, Wei Li, Johannes Faber,
Nishtha Srivastava, Exploring a CNN model for earthquake magnitude
estimation using HR-GNSS data, Journal of South American
Earth Sciences, 2024, 104815, ISSN 0895-9811,
https://doi.org/10.1016/j.jsames.2024.104815.


## Getting Started

# Clone the repository

```
git clone https://github.com/srivastavaresearchgroup/DL-HRGNSS
```

# Install dependencies (with python 3.8)

(virtualenv is recommended)

```
pip install -r requirements.txt
```

# Data

The database for each configuration/case is in `./data`.
For example the data folder: `./data/GNSS_M3S_181` contains the data for 
3 stations, 181 seconds.

The data is in numpy format, previously selected from the open-access 
database published by Lin et al., 2020 
(https://doi.org/10.5281/zenodo.4008690).

You can find the information related to the data (ID, Hypocenter, 
Magnitude) in the dataframes info_data.csv located in the same folder as 
the respective data.

You can use Data_plot.ipynb to plot the waveforms.

Refer to Quinteros et al., 2024 (https://doi.org/10.1016/j.jsames.2024.104815) for 
more details about the data configuration.

# Pretrained models and results

The pre-trained  models are located in `./trained_models`.
These models are described in Quinteros et al., 2024 
(https://doi.org/10.1016/j.jsames.2024.104815).
You can find the respective results in `./predictions`.

# Processing and output

If you want to change the default configuration, you can edit the 
variables in `main.py`.

Configuration parameters you should change, depending on your choice:
- Number of stations (nst)
- Time window length (nt)
- Paths/folder names

To split, train and test the data, run:

```
python main.py
```

The outputs will be located in the path that you set.

By default the outputs will be saved in `./tests`, and in the subfolders:

- `./data_inf`: you can find the numpy files with the data index for 
train, validation, and test dataset, associated with the initial 
xdata.npy file.
- `./models`: trained models are saved in this folder
- `./predictions`: the predicted magnitude and error values are saved in 
text files in this folder.
- `./out_log`: the output from logging is saved in this folder.

