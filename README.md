# Kepler - Stochastic Resonance - RMS + KNN, MLP and XGB Classification

This repository contains an experiment in which Gaussian noise is injected into RMS features extracted from Kepler light curves that were transformed into audio files. The impact of this noise on classification performance is evaluated using a KNN classifier, an MLP and XGB.

This experiment is part of the Bachelor's thesis **"Machine Learning for Exoplanet Detection: Investigating Feature Engineering Approaches and Stochastic Resonance Effects"** by Alicia Martinelli (2025).

## Folder Structure

```
kepler_audio_rms_xgb_stochres/
├── convert_arrayfiles.py      # Converts the light curves from the raw folder and saves the transformed audio files into the dataset folder
├── data_aug.py                # Adds Gaussian noise to the audio files from the dataset folder and saves them into the aug_dataset folder
├── feature_extractor.py       # Extracts the RMS feature from the audio files from the aug_dataset folder and saves the .npy files into the rms folder
├── calculate_snr.py           # Calculates the SNR
├── train_KNN.py               # KNN training
├── train_MLP.py               # MLP training
├── train_XGB.py               # XGB training
├── run_pipeline.sh            # Run the pipeline to add noise, extract RMS and train the models for each noise intensity parameter sigma
└── README.md                  # This file
└── .gitignore                 # Git ignore rules
```

## Preprocessed Kepler dataset
The preprocessed Kepler dataset used in this project is based on the public release from Shallue & Vanderburg (2018) and is available via the AstroNet GitHub repository (Google Drive) [https://drive.google.com/drive/folders/1Gw-o7sgWC1Y_mlaehN85qH5XC161EHSE](https://drive.google.com/drive/folders/1Gw-o7sgWC1Y_mlaehN85qH5XC161EHSE)

Download the TFRecords from the Google Drive, convert them into .npy files and save them in the raw folder.

## Setup

1. Clone the repository:
    ```bash
    git clone https://github.com/AliciaMartinelli/kepler_audio_rms_xgb_stochres.git
    cd kepler_audio_rms_xgb_stochres
    ```

2. Create and activate a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

3. Install dependencies:
    You may need to install `scikit-learn`, `tsfresh`, `matplotlib`, `numpy`, and `tensorflow` (and more).

## Usage

1. Convert the light curves into audio files:
```bash
python convert_arrayfiles.py
```
This converts the light curves from the raw folder into audio files and saves them into the dataset folder.

2. Run the pipeline:
```bash
./run_pipeline.sh
```
Start the pipeline to add noise to the audio files and train the different models. The results will be saved into the results folder.

3. Plot the AUC vs noise intensity parameter sigma:
```bash
python visualize_results.py
```
This will visualize the results in a plot with AUC vs. noise intensity parameter sigma

## Thesis Context

This repository corresponds to the experiment described in:
- **Section 6.2**: Audio-based RMS feature noise injection: Evaluation using XGB, MLP and KNN

**Author**: Alicia Martinelli  
**Email**: alicia.martinelli@stud.unibas.ch  
**Year**: 2025