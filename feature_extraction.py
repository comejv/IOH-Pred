from concurrent.futures import ThreadPoolExecutor
from os import listdir, makedirs
from os.path import join
from sys import argv

import matplotlib.pyplot as plt
import pandas as pd
from progress.bar import ChargingBar
from scipy import signal

from utils import *


def beat_segmentation(art_wf: pd.DataFrame) -> pd.DataFrame:
    sys_peaks = signal.find_peaks(art_wf.values, distance=40, prominence=10)[0]
    dia_valley = signal.find_peaks(-art_wf.values, distance=40, prominence=10)[0]

    features = pd.DataFrame(
        columns=[
            "sys",
            "dia",
            "pulse_pressure",
            "pp_var",
            "pulse_length",
            "bpm",
        ]
    )

    # Ensure we start with a valley
    if sys_peaks[0] < dia_valley[0]:
        sys_peaks = sys_peaks[1:]

    for i in range(len(dia_valley) - 1):
        pulse_pressure = art_wf.iloc[sys_peaks[i]] - art_wf.iloc[dia_valley[i]]
        pp_var = (
            1 - pulse_pressure / features.loc[i - 1]["pulse_pressure"] if i > 0 else 0
        )
        pulse_length_ms = (
            art_wf.index[dia_valley[i + 1]] - art_wf.index[dia_valley[i]]
        ) * 10
        bpm = 60 / pulse_length_ms * 1000
        features.loc[i] = [
            sys_peaks[i],
            dia_valley[i],
            pulse_pressure,
            pp_var,
            pulse_length_ms,
            bpm,
        ]

    print("Beat segmentation complete")

    return features


def plot_comparison(df: pd.DataFrame) -> None:
    art_wf = case["SNUADC/ART"]

    features = beat_segmentation(art_wf)

    plt.plot(art_wf, color="gray")
    plt.plot(art_wf.iloc[features["sys"]], "-", color="green")
    plt.plot(case["Solar8000/ART_SBP"], "-", color="teal")
    plt.plot(art_wf.iloc[features["dia"]], "-", color="red")
    plt.plot(case["Solar8000/ART_DBP"], "-", color="orange")
    plt.plot(features["sys"] + art_wf.index[0], features["bpm"], "-", color="purple")
    plt.plot(case["Solar8000/HR"], "-", color="pink")
    plt.legend(
        [
            "Arterial waveform",
            "SBP (computed)",
            "SBP (Solar8000)",
            "SBP (computed)",
            "DBP (Solar8000)",
            "BPM (computed)",
            "HR (Solar8000)",
        ]
    )
    print("Plotting case...")
    plt.show()
    print("Done.")


def transpose(ifile: str, ofile: str, tf: int, bar: ChargingBar = None) -> None:
    """Transpose a dataframe into windows of size tf seconds

    Args:
        ifile (str): input file
        ofile (str): output file
        tf (int): timeframe in seconds
    """
    df = pd.read_pickle(ifile)
    df["timestamp"] = pd.to_datetime(df.index * 10, unit="ms")
    grouper = pd.Grouper(key="timestamp", freq=f"{tf}s")

    reshaped_df = (
        df.assign(
            window=(g := df.groupby(grouper)).ngroup().add(1), row=g.cumcount().add(1)
        )
        .drop(columns="timestamp")
        .set_index(["window", "row"])
    )

    # Keep only windows with all data points
    final_df = reshaped_df.groupby("window").filter(lambda group: len(group) == 2000)

    final_df.to_pickle(ofile)
    bar.next()


def multithreaded_transpose(
    ifolder: str, ofolder: str, tf: int, n_files: int = 0
) -> pd.DataFrame:
    makedirs(ifolder, exist_ok=True)
    makedirs(ofolder, exist_ok=True)
    ifiles = []
    ofiles = []
    n = 0
    for file in listdir(ifolder):
        if file.endswith(".gz"):
            ifiles.append(join(ifolder, file))
            ofiles.append(join(ofolder, file))
            n += 1
            if n == n_files:
                break

    with ChargingBar(
        "Transposing\t",
        max=len(ifiles),
        suffix="%(index)d/%(max)d - ETA %(eta)ds",
        color=133,
    ) as bar:
        with ThreadPoolExecutor(max_workers=4) as executor:
            executor.map(
                transpose, ifiles, ofiles, [tf] * len(ifiles), [bar] * len(ifiles)
            )


def label_events(input: str, output: str, bar: ChargingBar = None) -> None:
    df_data = pd.read_pickle(input)

    # # Drop windows where values are missing (done in transpose function)
    # n_rows = df_data.groupby("window").size()
    # to_drop = n_rows[n_rows < env.SAMPLING_RATE * env.WINDOW_SIZE].index
    # df_data.drop(to_drop, inplace=True)

    # Change to false where less than 1 minute of consecutive event
    mask_grouped = (df_data["Solar8000/ART_MBP"] < 65).groupby("window").all()
    df_labels = mask_grouped.reset_index(level="window")
    df_labels["block"] = (
        df_labels["Solar8000/ART_MBP"]
        .ne(df_labels["Solar8000/ART_MBP"].shift())
        .cumsum()
    )
    df_labels["counts"] = df_labels.groupby("block")["Solar8000/ART_MBP"].transform(
        "sum"
    )
    df_labels["Solar8000/ART_MBP"] = df_labels["Solar8000/ART_MBP"].mask(
        df_labels["counts"] < 3, False
    )
    df_labels = df_labels.set_index("window")["Solar8000/ART_MBP"]

    df_labels.to_pickle(output)
    bar.next()


def multithreaded_label_events(ifolder: str, ofolder: str) -> None:
    makedirs(ofolder, exist_ok=True)
    ifiles = []
    ofiles = []
    for file in listdir(ifolder):
        if file.endswith(".gz"):
            ifiles.append(join(ifolder, file))
            ofiles.append(join(ofolder, file[:-3] + "_labels.gz"))

    with ChargingBar(
        "Labelling\t",
        max=len(ifiles),
        suffix="%(index)d/%(max)d - ETA %(eta)ds",
        color=133,
    ) as bar:
        with ThreadPoolExecutor(max_workers=env.CORES + 1) as executor:
            executor.map(label_events, ifiles, ofiles, [bar] * len(ifiles))


def plot_labels(case: str):
    multi_case_df = pd.read_pickle(
        join(env.DATA_FOLDER, "ready", "cases", case + ".gz")
    )
    case_df = multi_case_df.reset_index().drop(columns=["window", "row"], axis=1)
    labels_df = pd.read_pickle(
        join(env.DATA_FOLDER, "ready", "labels", case + "_labels.gz")
    )
    labels_shift = labels_df.shift(-30).fillna(False)
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(
        multi_case_df.index.to_list(),
        multi_case_df["Solar8000/ART_MBP"].values,
        color="blue",
    )
    ax.plot(
        [group.index[0] for _, group in multi_case_df.groupby(level=0)],
        labels_shift.astype("int32").values,
        color="red",
    )
    plt.show()


if __name__ == "__main__":
    if len(argv) < 2:
        argv.append("-" + input("T, L or P? "))
    if "-T" in argv:
        n_files = input("How many files? 0 for all: ")
        if n_files:
            n_files = int(n_files)
        multithreaded_transpose(
            ifolder=join(env.DATA_FOLDER, "preprocessed", "all"),
            ofolder=join(env.DATA_FOLDER, "ready", "cases"),
            tf=env.WINDOW_SIZE,
            n_files=n_files,
        )
    if "-L" in argv:
        multithreaded_label_events(
            ifolder=join(env.DATA_FOLDER, "ready", "cases"),
            ofolder=join(env.DATA_FOLDER, "ready", "labels"),
        )
    if "-P" in argv:
        plot_labels(input("Which case? "))
