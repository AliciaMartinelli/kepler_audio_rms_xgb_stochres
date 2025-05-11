import warnings
warnings.filterwarnings('ignore')

import os
import sys
import pathlib
import numpy as np
import librosa
from tqdm import tqdm
import soundfile as sf

class audioPreprocessing:
    def __init__(self, sample_rate=22050):
        self.sample_rate = sample_rate

    def readAudio(self, fileName):
        signal, sr = librosa.load(fileName, sr=self.sample_rate)
        return signal

    def audioAugmentationGaussian(self, signal, sigma):
        noise = sigma * np.random.normal(0, 1, size=signal.shape)
        return signal + noise

obj = audioPreprocessing()
targetSampleRate = 22050

def load_data(path):
    audioFiles = sorted(list(pathlib.Path(path).rglob("*.wav")))
    classes = [str(f.parent).split(os.sep)[-1] for f in audioFiles]
    return audioFiles, classes

def augment_data(audioFileNames, classes, sigma, save_path_root="aug_dataset"):
    global obj
    save_path = os.path.join(save_path_root, f"sigma_{sigma:.5f}")

    for idx, x in tqdm(enumerate(audioFileNames), total=len(audioFileNames)):
        signal_org = obj.readAudio(x)
        class_label = classes[idx]

        class_save_path = os.path.join(save_path, class_label)
        pathlib.Path(class_save_path).mkdir(parents=True, exist_ok=True)

        augmented_signal = obj.audioAugmentationGaussian(signal_org, sigma)
        augmented_file_name = f"{x.stem}_augmented{x.suffix}"
        augmented_path = os.path.join(class_save_path, augmented_file_name)
        sf.write(augmented_path, augmented_signal, targetSampleRate)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python data_aug.py <sigma>")
        sys.exit(1)

    try:
        sigma = float(sys.argv[1])
    except ValueError:
        print("Error: sigma must be a float.")
        sys.exit(1)

    np.random.seed(42)
    X, y = load_data("dataset")
    augment_data(X, y, sigma)
