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
            end_seg.append(df.index[-1])

    event_segments = [
        (a, b) for a, b in zip(start_seg, end_seg) if b - a > 60 * env.SAMPLING_RATE
    ]

    event_segments_shift = [
        (
            a - env.PRED_WINDOW * 60 * env.SAMPLING_RATE,
            b - env.PRED_WINDOW * 60 * env.SAMPLING_RATE,
        )
        for a, b in event_segments
    ]

    # flat_segments = [item for sublist in event_segments for item in sublist]
    # fig, ax = plt.subplots(figsize=(15, 6))
    # ax.plot(df["Solar8000/ART_MBP"])
    # for start, end in event_segments:
    #     ax.axvspan(start, end, alpha=0.1, color="red")
    # for start, end in event_segments_shift:
    #     ax.axvline(start, color="purple", alpha=0.5)
    # plt.show()

    art_wf = df["SNUADC/ART"]
    dia_valley = signal.find_peaks(-art_wf, distance=40, prominence=10)[0]

    dia_valley = [art_wf.index[i] for i in dia_valley]

    beat_event = []
    i, j = 0, 0
    if event_segments:
        while j < len(dia_valley) - 1:
            while (
                j < len(dia_valley) - 1 and dia_valley[j] < event_segments_shift[i][0]
            ):
                beat_event.append(False)
                j += 1
            while j < len(dia_valley) - 1 and dia_valley[j] < event_segments[i][1]:
                beat_event.append(True)
                j += 1
            i += 1
            if i == len(event_segments):
                break
        beat_event += [False] + [False] * (len(dia_valley) - len(beat_event))
    else:
        beat_event = [False] * (len(dia_valley) + 1)
    # fig, ax = plt.subplots(figsize=(15, 6))
    # ax.plot(df["Solar8000/ART_MBP"])
    # i = 0
    # for start, end in event_segments:
    #     if not i:
    #         ax.axvspan(start, end, alpha=0.1, color="red", label="IOH")
    #         i += 1
    #     ax.axvspan(start, end, alpha=0.1, color="red")
    # ax.plot(dia_valley, [50 if x else 40 for x in beat_event], "-", label=f"{env.PRED_WINDOW} min before IOH")
    # ax.axhline(65, color="green", alpha=0.5, label="IOH Threshold")
    # ax.legend()
    # plt.show()

    df = df[["SNUADC/ART"]]

    df["beat"] = df.index.isin(dia_valley)
    df["beat"] = df["beat"].cumsum()

    out = (
        df.assign(value=df.groupby("beat").cumcount())
        .set_index(["beat", "value"])
        .unstack()
    )

    out.insert(loc=0, column="label", value=beat_event)

    compression_set = {"method": "gzip", "compresslevel": 1, "mtime": 1}

    verbose("Beat segmentation complete")
    out.to_pickle(output, compression=compression_set)


def group_beat_unstack_multithreaded(ifolder, ofolder, N):
    makedirs(ofolder, exist_ok=True)
    ifiles = []
    ofiles = []
    n = 0
    for file in listdir(ifolder):
        if n == N:
            break
        if file.endswith(".gz"):
            ifiles.append(join(ifolder, file))
            ofiles.append(join(ofolder, file))
        n += 1

    with ThreadPoolExecutor(max_workers=env.CORES + 1) as executor:
        executor.map(group_beat_unstack, ifiles, ofiles)


if __name__ == "__main__":
    group_beat_unstack_multithreaded(
        join(env.DATA_FOLDER, "preprocessed", "nonevent"),
        join(env.DATA_FOLDER, "mirko", "nonevent"),
        50,
    )
