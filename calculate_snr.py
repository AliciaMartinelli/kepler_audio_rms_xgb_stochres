import os
import sys
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

if len(sys.argv) < 2:
    print("Usage: python calculate_snr.py <sigma>")
    sys.exit(1)

try:
    sigma = float(sys.argv[1])
except ValueError:
    print("Error: sigma must be a float.")
    sys.exit(1)

original_path = "dataset"
augmented_path = os.path.join("aug_dataset", f"sigma_{sigma:.5f}")
result_csv = os.path.join("results", "snr", f"snr_sigma_{sigma:.5f}.csv")
os.makedirs(os.path.dirname(result_csv), exist_ok=True)

results = []

for class_label in ["0", "1"]:
    original_class_path = os.path.join(original_path, class_label)
    augmented_class_path = os.path.join(augmented_path, class_label)

    for file in tqdm(os.listdir(original_class_path), desc=f"Class {class_label}"):
        if not file.endswith(".wav"):
            continue

        original_file = os.path.join(original_class_path, file)
        augmented_file = os.path.join(augmented_class_path, file.replace(".wav", "_augmented.wav"))

        if not os.path.exists(augmented_file):
            print(f"Warning: Augmented file not found for {file}")
            continue

        original_audio, _ = sf.read(original_file)
        augmented_audio, _ = sf.read(augmented_file)

        if original_audio.shape != augmented_audio.shape:
            print(f"Warning: Shape mismatch in {file}")
            continue

        noise = augmented_audio - original_audio
        signal_var = np.var(original_audio)
        noise_var = np.var(noise)

        snr = 10 * np.log10(signal_var / noise_var) if noise_var != 0 else float('inf')

        results.append({
            "filename": file,
            "class": class_label,
            "snr": snr
        })

df = pd.DataFrame(results)
df.to_csv(result_csv, index=False)
print(f"\nSNR-Ergebnisse gespeichert in {result_csv}")
