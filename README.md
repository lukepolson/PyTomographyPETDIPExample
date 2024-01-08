# Deep Image Prior (PyTomography Example)
The following repository contains all necessary files to perform the DIPRecon algorithm in PyTomography. The phantom corresponds to an ultra high resolution PET/MRI brain scan. PET simulation was performed via GATE.

The files are ran in the following order:

* `compute_eta.py` computes the normalization factor $\eta$ needed for PET reconstruction. It uses data from a cylindrical calibration scan performed using the same system geometry.
* `open_data.py` loads and processes the raw ROOT coincidence data and converts it into a readable `.npy` file.
* `init_recon.py` performs an initial reconstruction of the data using OSEM
* `dip_init_net.py` trains a DIP network to predict the output of the `init_recon.py` file using an MRI image as input. This is used to get an initial network configuration before running the next script.
* `dip_recon.py` performs DIP reconstruction.

Other files include:

* `simulation_parameters.py` controls all parameters used in the code.
* `misc.py` contains all miscellaneous functions used for image alignment/etc required
* `networks.py` contains the neural network architecture used here.
* `analysis.ipynb` contains code for analyzing the reconstructions/statistics.
