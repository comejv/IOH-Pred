from os.path import exists
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from vitaldb import VitalFile

import json

from utils import *

timeframe = env.SAMPLING_RATE * 60
plotting = True

while plotting:
    # Case selection
    try:
        caseid = int(input("Case ID: "))
    except ValueError:
        break

    if not exists(f"{env.DATA_FOLDER}vital/{caseid}.vital"):
        print("Case does not exist (missing vital file)")
        continue

    if exists(f"{env.DATA_FOLDER}preprocessed/event/{caseid}.gz"):
        event = "event"
    elif exists(f"{env.DATA_FOLDER}preprocessed/nonevent/{caseid}.gz"):
        event = "nonevent"
    else:
        print("Not pickled")
        continue

    # read pickled dataframes
    vtf = VitalFile(f"{env.DATA_FOLDER}vital/{caseid}.vital")
    original = vtf.to_pandas(["SNUADC/ART", "Solar8000/ART_MBP", "Solar8000/ART_DBP", "Solar8000/ART_SBP"], 1 / env.SAMPLING_RATE)
    original.ffill(inplace=True)

    processesed = pd.read_pickle(f"{env.DATA_FOLDER}preprocessed/{event}/{caseid}.gz")

    if input("Stacked? (y/n)") == "y":
        original_pre_ag = original[original.index < processesed.index.min()]
        original_post_ag = original[original.index > processesed.index.max()]

        # Find IOH
        mask = processesed["Solar8000/ART_MBP"].lt(65)
        idxs = processesed.index[
            mask.rolling(window=60 * env.SAMPLING_RATE, axis=0).apply(
                lambda x: x.all(), raw=True
            )
            == True
        ]

        fig, ax = plt.subplots(figsize=(15, 6))

        original_pre_ag.plot(ax=ax, y="SNUADC/ART", color="gray", label=None)
        original_post_ag.plot(ax=ax, y="SNUADC/ART", color="gray", label=None)

        processesed.plot(
            ax=ax,
            y=[
                "SNUADC/ART",
                "Solar8000/ART_MBP",
                "Solar8000/ART_DBP",
                "Solar8000/ART_SBP",
            ],
            label=[
                "Arterial pressure waveform",
                "Mean arterial pressure",
                "DBP",
                "SBP",
            ],
        )

        # Highlight all segments with hypotensive (IOH) events
        hypotensive_event_patch = mpatches.Patch(
            color="red", alpha=0.3, label="Hypotensive event (1 minute wide)"
        )

        if not idxs.empty:
            idx_prev = 0
            for i, idx in enumerate(idxs):
                if idx - idx_prev >= 60 * env.SAMPLING_RATE:
                    label = "Hypotensive event (1 minute wide)" if i == 0 else None
                    ax.axvspan(
                        idx - 60 * env.SAMPLING_RATE,
                        idx,
                        color="red",
                        alpha=0.3,
                        label=label,
                    )
                    idx_prev = idx
        # Add the custom legend
        current_handles, current_labels = ax.get_legend_handles_labels()
        current_handles.append(hypotensive_event_patch)
        ax.legend(handles=current_handles, labels=current_labels, loc="upper right")
        plt.xlabel("Time (ms)")
        plt.ylabel("Arterial pressure (mmHg)")
        plt.ylim((-50, 200))

    else:
        fig, axs = plt.subplots(2, 1, sharex=True, figsize=(15, 6))
        original.plot(
            ax=axs[0],
            y=[
                "SNUADC/ART"
            ],
            title="Original",
            label=[
                "Arterial pressure waveform"
            ],
            ylim=(-50, 200),
            xlabel="Time (ms)",
            ylabel="Arterial pressure (mmHg)",
        )
        processesed.plot(
            ax=axs[1],
            y=["SNUADC/ART", "Solar8000/ART_MBP", "Solar8000/ART_DBP", "Solar8000/ART_SBP"],
            title="Processed",
            label=["Arterial pressure waveform", "Mean arterial pressure", "DBP", "SBP"],
            ylim=(-50, 200),
            xlabel="Time (ms)",
            ylabel="Arterial pressure (mmHg)",
        )

    plt.show()
