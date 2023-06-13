from os.path import exists
import pandas as pd
import matplotlib.pyplot as plt
from vitaldb import VitalFile

ECH_RATE_HZ = 100
timeframe = ECH_RATE_HZ
plotting = True

while plotting:
    # Case selection
    try:
        caseid = int(input("Case ID: "))
    except ValueError:
        break

    if not exists(f"/local/vincenco/IOH-Pred/data/vital/{caseid}.vital") or not exists(
        f"/local/vincenco/IOH-Pred/data/preprocessed/event/{caseid}.pkl"
    ):
        print("Case does not exist (missing vital or pickled data)")
        continue

    # read pickled dataframes
    vtf = VitalFile(f"/local/vincenco/IOH-Pred/data/vital/{caseid}.vital")
    original = vtf.to_pandas(["SNUADC/ART", "Solar8000/ART_MBP"], 1 / 100)

    processesed = pd.read_pickle(
        f"/local/vincenco/IOH-Pred/data/preprocessed/event/{caseid}.pkl"
    )
    ag_start = processesed.index[0]
    ag_end = processesed.index[-1]
    mask = processesed["Solar8000/ART_MBP"].lt(65)
    idxs = processesed.index[
        mask.rolling(window=60 * ECH_RATE_HZ, axis=0).apply(lambda x: x.all(), raw=True)
        == True
    ]

    original_pre_ag = original.iloc[0:ag_start]
    original_post_ag = original.iloc[ag_end:]
    plot = original_pre_ag.plot(
        y=["SNUADC/ART", "Solar8000/ART_MBP"]
    )
    original_post_ag.plot(
        ax=plot, y=["SNUADC/ART", "Solar8000/ART_MBP"]
    )

    processesed.plot(
        ax=plot, y=["SNUADC/ART", "Solar8000/ART_MBP"]
    )
    # Highlight all segments with < 4
    if idxs.empty:
        print("No hypo")
    else:
        idx_prev = 0
        for idx in idxs:
            if idx - idx_prev >= 60 * ECH_RATE_HZ:
                plot.axvspan(idx - 60 * ECH_RATE_HZ, idx, color="red", alpha=0.3)
            idx_prev = idx
    plt.show()
