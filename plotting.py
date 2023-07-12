import json
from os.path import exists

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
from vitaldb import VitalFile

from utils import *


def plot():
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
        preprocessed = pd.read_pickle(
            f"{env.DATA_FOLDER}preprocessed/{event}/{caseid}.gz"
        ).reset_index(drop=True)

        if input("IOH events? (y/n)") == "y":  # Find IOH
            mask = preprocessed["Solar8000/ART_MBP"].lt(65)
            idxs = mask.rolling(window=60 * env.SAMPLING_RATE, axis=0).apply(
                lambda x: x.all(), raw=True
            )
            preprocessed["IOH"] = idxs
            preprocessed.fillna(0, inplace=True)
            preprocessed["IOH"] *= 20

            fig, ax = plt.subplots(figsize=(15, 6))

            preprocessed.plot(
                ax=ax,
                y=["SNUADC/ART", "Solar8000/ART_MBP"],
                label=["Arterial pressure waveform", "MAP"],
            )
            preprocessed.plot(ax=ax, y="IOH", label="IOH", color="red")

            plt.title(f"IOH for {caseid}")
            plt.xlabel("Time (ms)")
            plt.ylabel("Arterial pressure (mmHg)")
            plt.ylim((-50, 200))

        else:
            vtf = VitalFile(f"{env.DATA_FOLDER}vital/{caseid}.vital")
            original = vtf.to_pandas(
                [
                    "SNUADC/ART",
                    "Solar8000/ART_MBP",
                    "Solar8000/ART_DBP",
                    "Solar8000/ART_SBP",
                ],
                1 / env.SAMPLING_RATE,
            )
            original.ffill(inplace=True)
            fig, axs = plt.subplots(2, 1, sharex=True, figsize=(15, 6))
            original.plot(
                ax=axs[0],
                y=["SNUADC/ART"],
                title="Original",
                label=["Arterial pressure waveform"],
                ylim=(-50, 200),
                xlabel="Time (ms)",
                ylabel="Arterial pressure (mmHg)",
            )
            preprocessed.plot(
                ax=axs[1],
                y=[
                    "SNUADC/ART",
                    "Solar8000/ART_MBP",
                    "Solar8000/ART_DBP",
                    "Solar8000/ART_SBP",
                ],
                title="Processed",
                label=[
                    "Arterial pressure waveform",
                    "Mean arterial pressure",
                    "DBP",
                    "SBP",
                ],
                ylim=(-50, 200),
                xlabel="Time (ms)",
                ylabel="Arterial pressure (mmHg)",
            )

        plt.show()
