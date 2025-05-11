import warnings
warnings.filterwarnings('ignore')

import os
import sys
import numpy as np
import librosa
import pathlib
from tqdm.auto import tqdm

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def extract_rms_features(dataset_path, sigma, features_root="rms"):
    print(f"Extracting RMS features for sigma={sigma}...")
    save_path = os.path.join(features_root, f"rms_sigma_{sigma:.5f}")
    files = sorted(list(pathlib.Path(dataset_path).rglob("*.wav")))

    for f in tqdm(files):
        signal, sr = librosa.load(f, duration=5.0)
        S, _ = librosa.magphase(librosa.stft(signal))
        rms = librosa.feature.rms(S=S, frame_length=2048, hop_length=512, center=True, pad_mode='constant')
        class_folder = os.path.join(save_path, f.parts[-2])
        ensure_dir(class_folder)
        np.save(os.path.join(class_folder, f.stem + ".npy"), rms)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python features_extractor.py <sigma>")
        sys.exit(1)

    try:
        sigma = float(sys.argv[1])
    except ValueError:
        print("Error: sigma must be a float.")
        sys.exit(1)

    DATASET_PATH = f"aug_dataset/sigma_{sigma:.5f}"
    extract_rms_features(DATASET_PATH, sigma)
