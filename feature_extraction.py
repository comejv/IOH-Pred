import matplotlib.pyplot as plt
from scipy import signal
from numpy import less_equal, greater_equal
import pandas as pd


def beat_segmentation(art_wave: pd.DataFrame) -> pd.DataFrame:
    sys_peaks = signal.find_peaks(art_wf.values, distance=40, prominence=10)[0]
    dia_valley = signal.find_peaks(-art_wf.values, distance=40, prominence=10)[0]

    features = pd.DataFrame(
        columns=[
            "sys",
            "dia",
            "pulse_pressure",
            "pp_var",
            "pulse_length",
        ]
    )

    # Ensure we start with a valley
    if sys_peaks[0] < dia_valley[0]:
        sys_peaks = sys_peaks[1:]

    for i in range(len(dia_valley) - 1):
        pulse_pressure = art_wave.iloc[sys_peaks[i]] - art_wave.iloc[dia_valley[i]]
        pp_var = 1 - pulse_pressure / features.loc[i - 1]["pulse_pressure"] if i > 0 else 0
        pulse_length = art_wave.index[dia_valley[i + 1]] - art_wave.index[dia_valley[i]]
        features.loc[i] = [sys_peaks[i], dia_valley[i], pulse_pressure, pp_var, pulse_length]

    return features


case = pd.read_pickle("/local/vincenco/IOH-Pred/data/preprocessed/event/641.gz")
art_wf = case["SNUADC/ART"]
features = beat_segmentation(art_wf)
plt.plot(art_wf)
plt.plot(hp_art_wf)
plt.plot(art_wf.iloc[features["sys"]], "v", color="green")
plt.plot(art_wf.iloc[features["dia"]], "^", color="red")
plt.show()
