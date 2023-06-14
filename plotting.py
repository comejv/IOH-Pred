from os.path import exists
import pandas as pd
import matplotlib.pyplot as plt
from vitaldb import VitalFile

ECH_RATE_HZ = 100
timeframe = ECH_RATE_HZ * 60
plotting = True

while plotting:
    # Case selection
    try:
        caseid = int(input("Case ID: "))
    except ValueError:
        break

    if not exists(f"/local/vincenco/IOH-Pred/data/vital/{caseid}.vital"):
        print("Case does not exist (missing vital file)")
        continue

    if exists(f"/local/vincenco/IOH-Pred/data/preprocessed/event/{caseid}.pkl"):
        event = "event"
    elif exists(f"/local/vincenco/IOH-Pred/data/preprocessed/nonevent/{caseid}.pkl"):
        event = "nonevent"
    else:
        print("Not pickled")
        continue

    # read pickled dataframes
    vtf = VitalFile(f"/local/vincenco/IOH-Pred/data/vital/{caseid}.vital")
    original = vtf.to_pandas(["SNUADC/ART", "Solar8000/ART_MBP"], 1 / 100)

    processesed = pd.read_pickle(
        f"/local/vincenco/IOH-Pred/data/preprocessed/{event}/{caseid}.pkl"
    )

    if input("Stacked? (y/n)") == "y":

        processesed.reset_index(inplace=True, drop=True)

        mask = original["SNUADC/ART"] > 40
        rolling_sum = mask.rolling(window=timeframe).sum()
        ag_start = rolling_sum[rolling_sum >= 0.9 * timeframe].idxmin()

        ag_end = ag_start + len(processesed)

        processesed.index = processesed.index + ag_start

        mask = processesed["Solar8000/ART_MBP"].lt(65)
        idxs = processesed.index[
            mask.rolling(window=60 * ECH_RATE_HZ, axis=0).apply(lambda x: x.all(), raw=True)
            == True
        ]

        original_pre_ag = original.iloc[0:ag_start]
        original_post_ag = original.iloc[ag_end:]
        plot = original_pre_ag.plot(y="SNUADC/ART", color="gray", legend=False)
        original_post_ag.plot(ax=plot, y="SNUADC/ART", color="gray", legend=False)

        processesed.plot(ax=plot, y=["SNUADC/ART", "Solar8000/ART_MBP"])
        # Highlight all segments with < 4
        if idxs.empty:
            print("No hypo")
        else:
            idx_prev = 0
            for idx in idxs:
                if idx - idx_prev >= 60 * ECH_RATE_HZ:
                    plot.axvspan(idx - 60 * ECH_RATE_HZ, idx, color="red", alpha=0.3)
                idx_prev = idx

    else:
        fig, axs = plt.subplots(2, 1, sharex=True)
        original.plot(ax=axs[0], y=["SNUADC/ART", "Solar8000/ART_MBP"], title="Original")
        processesed.plot(ax=axs[1], y=["SNUADC/ART", "Solar8000/ART_MBP"], title="Processed")

    plt.show()
