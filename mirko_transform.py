import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from concurrent.futures import ThreadPoolExecutor
from os.path import join
from os import makedirs, listdir

from utils import *


def group_beat_unstack(filename, output):
    df = pd.read_pickle(filename)
    mask = df["Solar8000/ART_MBP"] < 65
    mask_diff = mask.diff()
    changes = mask_diff[mask_diff == True]
    if mask.iloc[0] == True:
        start_seg = [df.index[0]] + list(changes.index[1::2])
        end_seg = list(changes.index[0::2])
        if len(start_seg) > len(end_seg):
            end_seg.append(df.index[-1])
    else:
        start_seg = list(changes.index[0::2])
        end_seg = list(changes.index[1::2])
        if len(start_seg) > len(end_seg):
            end_seg += [-1]

    event_segments = [
        list(range(a, b))
        for a, b in zip(start_seg, end_seg)
        if b - a > 60 * env.SAMPLING_RATE
    ]
    flat_segments = [item for sublist in event_segments for item in sublist]
    # fig, ax = plt.subplots(figsize=(15, 6))
    # ax.plot(df["Solar8000/ART_MBP"])
    # for start, end in event_segments:
    #     ax.axvspan(start, end, alpha=0.1, color="red")
    # plt.show()

    art_wf = df["SNUADC/ART"]
    dia_valley = signal.find_peaks(-art_wf.values, distance=40, prominence=10)[0]

    df = df[["SNUADC/ART"]]

    df["beat"] = df.index.isin(dia_valley)
    df["beat"] = df["beat"].ne(df["beat"].shift()).cumsum().add(-1)

    df["event"] = df.index.isin(flat_segments)

    out = (
        df.assign(value=df.groupby("beat").cumcount())
        .set_index(["beat", "value"])
        .unstack()
    )

    compression_set = {"method": "gzip", "compresslevel": 1, "mtime": 1}

    verbose("Beat segmentation complete")
    out.to_pickle(output, compression=compression_set)


def group_beat_unstack_multithreaded(ifolder, ofolder):
    makedirs(ofolder, exist_ok=True)
    ifiles = []
    ofiles = []
    for file in listdir(ifolder):
        if file.endswith(".gz"):
            ifiles.append(join(ifolder, file))
            ofiles.append(join(ofolder, file))

    with ThreadPoolExecutor(max_workers=env.CORES + 1) as executor:
        executor.map(group_beat_unstack, ifiles, ofiles)


group_beat_unstack(
    join(env.DATA_FOLDER, "preprocessed", "event", "111.gz"),
    join(env.DATA_FOLDER, "mirko", "111.gz"),
)
group_beat_unstack_multithreaded(
    join(env.DATA_FOLDER, "preprocessed", "event"), join(env.DATA_FOLDER, "mirko")
)
