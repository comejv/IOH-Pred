import matplotlib.pyplot as plt
from scipy import signal
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from os.path import join
from os import makedirs, listdir
from sys import argv


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

    verbose("Beat segmentation complete")

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
    verbose("Plotting case...")
    plt.show()
    verbose("Done.")


def transpose(file: str, tf: int, pickle: bool) -> pd.DataFrame | None:
    """Transpose a dataframe into windows of size tf seconds

    Args:
        df (pd.DataFrame): input dataframe
        tf (int): timeframe in seconds

    Returns:
        pd.DataFrame: transposed dataframe
    """
    df = pd.read_pickle(file)
    df["timestamp"] = pd.to_datetime(df.index * 10, unit="ms")
    grouper = pd.Grouper(key="timestamp", freq=f"{tf}s")

    final_df = (
        df.assign(
            window=(g := df.groupby(grouper)).ngroup().add(1), row=g.cumcount().add(1)
        )
        .drop(columns="timestamp")
        .set_index(["window", "row"])
    )

    verbose("Transposed dataframe", file)
    if pickle:
        final_df.to_pickle(file)
    else:
        return final_df


def multithreaded_transpose(folder: str, tf: int) -> pd.DataFrame:
    makedirs(folder, exist_ok=True)
    files = []
    for file in listdir(folder):
        if file.endswith(".gz"):
            files.append(join(folder, file))

    with ThreadPoolExecutor(max_workers=4) as executor:
        executor.map(transpose, files, [tf] * len(files), [True] * len(files))


def label_events(input: str, output: str) -> None:
    df = pd.read_pickle(input)

    # Check for each window if all MAP values are under 65
    mask_grouped = (df["Solar8000/ART_MBP"] < 65).groupby("window").all()

    # Change to false where less than 1 minute of consecutive event
    df = mask_grouped.reset_index(level='window')
    df['block'] = df["Solar8000/ART_MBP"].ne(df['Solar8000/ART_MBP'].shift()).cumsum()    
    df['counts'] = df.groupby('block')['Solar8000/ART_MBP'].transform('sum')
    df['Solar8000/ART_MBP'] = df['Solar8000/ART_MBP'].mask(df['counts'] < 3, False)
    df = df.set_index('window')['Solar8000/ART_MBP']

    df.to_pickle(output)


def multithreaded_label_events(input: str, output: str) -> None:
    makedirs(output, exist_ok=True)
    ifiles = []
    ofiles = []
    for file in listdir(input):
        if file.endswith(".gz"):
            ifiles.append(join(input, file))
            ofiles.append(join(output, file[:-3] + "_labels.gz"))
            

    with ThreadPoolExecutor(max_workers=env.CORES + 1) as executor:
        executor.map(label_events, ifiles, ofiles)


if __name__ == "__main__":
    if "-T" in argv:
        multithreaded_transpose(join(env.DATA_FOLDER, "transpose"), 20)
    if "-L" in argv:
        multithreaded_label_events(input=join(env.DATA_FOLDER, "transpose"), output=join(env.DATA_FOLDER, "ready"))
