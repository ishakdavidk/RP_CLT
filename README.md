# RP_CLT - Co-Location Testing using Magnetometer Data and Autoencoders

A research project for **Co-Location Testing (CLT)** that determines whether two or more mobile devices are in the same physical location by comparing their magnetometer sensor readings using autoencoders.

## Overview

The core idea is to train an autoencoder on one device's magnetometer signal from a known location/time window, then test reconstruction on another device's data. **Low reconstruction error (MSE)** indicates the devices are co-located, while **high reconstruction error** indicates they are at different locations.

Two autoencoder architectures are explored:
- **Model 1 (Dense AE)**: A fully connected autoencoder operating on 1D magnetometer time-series (480 -> 60 -> latent -> 60 -> 480)
- **Model 2 (Convolutional AE)**: A convolutional autoencoder operating on 32x32 grayscale images generated from signal plots (with line-only and filled-area variants)

## Project Structure

```
├── main_model1.py                             # Main script for Model 1 end-to-end pipeline
├── data_prep.py                               # Data loading, preprocessing, sliding window, image generation
├── data_plotting.py                           # Visualization utilities and result plotting
├── train.py                                   # Autoencoder model definitions and training (Dense AE & CAE)
├── V1_20200709-data_visualization.ipynb       # V1 dataset exploration (2 devices)
├── V2_20200716-data_visualization.ipynb       # V2 dataset exploration (3 devices, with activity labels)
├── main_visualization_model1.ipynb            # Full pipeline for Dense AE with visualizations
├── main_visualization_model2-fill.ipynb       # Convolutional AE with filled signal plots
├── main_visualization_model2-line.ipynb       # Convolutional AE with line-only signal plots
├── main_visualization_stft_analysis.ipynb     # STFT frequency analysis of magnetometer data
└── conference_paper_plot.ipynb                # Publication-quality plots for conference paper
```

## Requirements

```
tensorflow
numpy
matplotlib
scikit-learn
scikit-image
```

## Usage

Run the Model 1 pipeline:

```bash
python main_model1.py
```

This loads V2 data (3 devices), computes magnetic intensity, downsamples to 250ms intervals, trains the dense autoencoder, and tests reconstruction on same-location vs. different-location data with MSE output.

For detailed analysis and visualizations, use the Jupyter notebooks.

## Data

Magnetometer data is collected from multiple mobile devices (identified by Bluetooth MAC addresses) with the following fields:
- `timestamp`, `mag_x`, `mag_y`, `mag_z`, `mag_acc`
- `activity`: still (st), walking (wk), running (r), bus (b), subway (sb), etc.
- `pos_state`: indoor (i) / outdoor (o)

Total magnetic field intensity is computed as: **F = sqrt(H^2 + Z^2)**, where **H = sqrt(X^2 + Y^2)**

## Key Parameters

| Parameter | Value |
|-----------|-------|
| Sampling interval | 250ms (after downsampling) |
| Window size | 480 samples (~2 minutes) |
| Data augmentation | 10x via sliding window |
| Loss function | MSE |
| Optimizer | Adam |
| Evaluation metrics | MSE (both models), SSIM (Model 2) |

## Datasets

- **V1** (2020-07-09): 2 devices
- **V2** (2020-07-16): 3 devices with activity labels
