import os
import pandas as pd
import matplotlib.pyplot as plt

size = 20
plt.rcParams.update({
    "font.size": size,
    "axes.titlesize": size,
    "axes.labelsize": size,
    "xtick.labelsize": size,
    "ytick.labelsize": size,
    "legend.fontsize": 15
})

sigmas = [0.00001, 0.0001, 0.001, 0.003, 0.005, 0.007, 0.01, 0.015, 0.02, 0.03, 0.1, 0.3, 1]

def load_metrics(model_name):
    result_dir = os.path.join("results", model_name)
    all_rows = []

    for sigma in sigmas:
        filename = f"results_{model_name}_sigma_{sigma:.5f}.csv"
        filepath = os.path.join(result_dir, filename)

        if not os.path.exists(filepath):
            print(f"warning: missing file {filepath}")
            continue

        df = pd.read_csv(filepath)
        df["sigma"] = sigma
        all_rows.append(df)

    if not all_rows:
        raise RuntimeError(f"no data loaded for model: {model_name}")

    return pd.concat(all_rows, ignore_index=True)

def load_snr_data():
    result_dir = os.path.join("results", "snr")
    snr_rows = []

    for sigma in sigmas:
        filename = f"snr_sigma_{sigma:.5f}.csv"
        filepath = os.path.join(result_dir, filename)

        if not os.path.exists(filepath):
            print(f"warning: missing snr file {filepath}")
            continue

        df = pd.read_csv(filepath)
        df["sigma"] = sigma
        snr_rows.append(df)

    if not snr_rows:
        raise RuntimeError("no SNR data found.")

    return pd.concat(snr_rows, ignore_index=True)

def summarize(df, value_col, model_name=None):
    grouped = df.groupby("sigma")[value_col].agg(["mean", "std"]).reset_index()
    if model_name:
        grouped["model"] = model_name
    return grouped

def plot_combined_auc_snr(xgb, knn, mlp, snr):
    plt.figure(figsize=(10, 6))

    for df, color in zip([xgb, knn, mlp], ['tab:blue', 'tab:orange', 'tab:green']):
        plt.errorbar(
            df["sigma"], df["mean"], yerr=df["std"],
            label=f"AUC ({df['model'].iloc[0]})",
            marker='o', capsize=4, color=color
        )

    ax1 = plt.gca()
    ax1.set_xscale("log")
    ax1.set_xlabel("noise level (sigma)")
    ax1.set_ylabel("AUC")
    ax1.tick_params(axis='y')

    ax2 = ax1.twinx()
    snr_summary = snr.groupby("sigma")["snr"].agg(["mean", "std"]).reset_index()
    ax2.errorbar(
        snr_summary["sigma"], snr_summary["mean"], yerr=snr_summary["std"],
        label="SNR", marker='s', color='tab:red', capsize=4
    )
    ax2.set_ylabel("SNR (dB)", color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    combined = list(zip(handles1 + handles2, labels1 + labels2))

    final_handles = []
    final_labels = []
    seen = set()

    for h, l in combined:
        if l not in seen:
            final_handles.append(h)
            final_labels.append(l)
            seen.add(l)

    ax1.legend(final_handles, final_labels, loc='lower left')

    plt.title("AUC (XGB, MLP, KNN) and SNR vs. noise level (sigma)")
    plt.tight_layout()
    os.makedirs("results/plots", exist_ok=True)
    plt.savefig("results/plots/auc_snr_all_models_vs_sigma.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    print("Loading data...")
    df_xgb = summarize(load_metrics("xgb"), "auc", "XGB")
    df_knn = summarize(load_metrics("knn"), "auc", "KNN")
    df_mlp = summarize(load_metrics("mlp"), "auc", "MLP")
    df_snr = load_snr_data()

    print("Generating combined plot...")
    plot_combined_auc_snr(df_xgb, df_knn, df_mlp, df_snr)
    print("Plot saved as results/plots/auc_snr_all_models_vs_sigma.png")
