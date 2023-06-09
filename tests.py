import pandas as pd
import matplotlib.pyplot as plt
from vitaldb import VitalFile
from sys import argv

# read pickled dataframes
caseid = argv[1]
vtf = VitalFile(f"data/vital/{caseid}.vital").to_pandas(["SNUADC/ART"], 1 / 100)
df = pd.read_pickle(f"data/preprocessed/{caseid}.pkl")
plot = vtf.plot(y="SNUADC/ART", kind="line", color="blue")
df.plot(ax=plot, y="SNUADC/ART", kind="line", color="red")
plot.legend(loc="best")
plt.show()
