from os import listdir, makedirs, remove, rename
from os.path import exists, join, basename
from sys import argv
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import vitaldb as vdb


def verbose(*args, **kwargs):
    if VERBOSE:
        print(*args, **kwargs)


def find_cases(track_names: list[str], ops: list[str]) -> list[int]:
    df_cases = pd.read_csv("https://api.vitaldb.net/cases")

    # Cases selection
    cases_tracks = vdb.find_cases(track_names)
    op_set = (
        pd.concat([df_cases[df_cases["opname"] == op]["caseid"] for op in ops], axis=1)
        .stack()
        .values
    )
    cases_ag = df_cases[df_cases["ane_type"] == "General"]["caseid"]

    return list(set(cases_tracks) & set(op_set) & set(cases_ag))


def download_case(
    case_id: int,
    track_names: list[str],
    interval: float = None,
    index: tuple[int, int] = None,
) -> None:
    """Download a single case from the database

    Args:
        case_id (int): case id
        track_names (list[str]): list of track names
        index (int, optional): Counter of the current case being downloaded. Defaults to None.
    """
    if exists(f"data/vital/{case_id}.vital"):
        verbose(f"Case {case_id} already exists")
        return
    try:
        case = vdb.VitalFile(case_id, track_names, interval)
        # Rename tracks
        # case.rename_tracks(mapping)
        case.to_vital(opath=f"data/vital/{case_id}.vital")
        if index:
            verbose(f"Downloaded case {case_id} : {index[0]+1}/{index[1]}")
        else:
            verbose(f"Downloaded case {case_id}")
    except KeyboardInterrupt:
        print("Download interrupted")
        return


def download_cases(
    track_names: list[str],
    case_ids: list[int],
    interval: float = None,
    max_cases: int = None,
) -> None:
    """Download a list of cases from the VitalDB database ; multithreaded.

    Args:
        track_names (list): list of track names
        case_ids (list): list of case ids
    """
    makedirs("data/vital", exist_ok=True)
    if max_cases:
        case_ids = case_ids[: min(len(case_ids), max_cases)]
    with ThreadPoolExecutor(max_workers=10) as executor:
        for i, case_id in enumerate(case_ids):
            executor.submit(
                download_case, case_id, track_names, interval, (i, len(case_ids))
            )


def vital_to_csv(ipath: str, opath: str, interval: float = None) -> None:
    """Convert vital file to csv
    NB : csv files are ~25 times bigger than vital files

    Args:
        ipath (str): path to the vital file to convert
        opath (str): path where to save the csv
        interval (float): interval resolution in seconds. Defaults to None (max res)
    """
    try:
        vital = vdb.VitalFile(ipath)
        track_names = vital.get_track_names()
        with open(opath, "w") as tmp:
            df = vital.to_pandas(track_names, interval)
            df.to_csv(tmp, index=False)
        verbose(ipath, "converted to csv")
    except Exception as e:
        print(f"Could not convert {ipath} to csv : {e}")
        if exists(opath):
            remove(opath)


def folder_vital_to_csv(ifolder: str, ofolder: str, interval: float = None) -> None:
    """Convert all vital files in a folder to csv ; multithreaded
    NB : csv files are ~25 times bigger than vital files

    Args:
        ifolder (str): path to the folder containing the vital files
        ofolder (str): path where to save the csv files
        interval (float): interval resolution in seconds. Defaults to None (max res)
    """
    makedirs(ofolder, exist_ok=True)

    ipaths = []
    opaths = []

    for file in listdir(ifolder):
        if not file.endswith("vital"):
            continue

        ipath = join(ifolder, file)
        ofilename = basename(ipath)
        ofilename = ofilename[: ofilename.index(".")]
        opath = join(ofolder, ofilename)
        opath += ".csv"

        if exists(opath):
            verbose("File", opath, "already converted, skipping")
            continue

        ipaths.append(ipath)
        opaths.append(opath)

    with ThreadPoolExecutor(max_workers=10) as executor:
        executor.map(vital_to_csv, ipaths, opaths, [interval] * len(ipaths))


def check_and_move(condition: bool, error_msg: str, src: str, dst: str) -> bool:
    if condition:
        verbose(error_msg)
        if not exists(dst):
            rename(src=src, dst=dst)
        else:
            verbose(
                f"Warning : file {dst} already exists, not removed from input folder"
            )
        return False
    return True


def preprocessing(ifile: str, ofile: str) -> bool:
    """Preprocessing of a vital file

    Args:
        vital (str): path to the vital file to preprocess

    Returns:
        bool: True if preprocessing was successful
    """
    vital = vdb.VitalFile(ifile)
    track_names = vital.get_track_names()
    df = vital.to_pandas(track_names, INTERVAL)

    # Delete rows with only nan values
    df.dropna(axis="index", subset=["SNUADC/ART"], inplace=True)

    # Number of rows in a minute
    timeframe = int(1 / INTERVAL * 120)

    # Delete rows from begining to first minute with 80% of ART values above 40mmHg
    mask = df["SNUADC/ART"] > 40
    rolling_sum = mask.rolling(window=timeframe).sum()
    if (rolling_sum >= timeframe).sum() < timeframe * 30:
        rename(ifile, "data/unfit/" + basename(ifile))
        verbose("File", ifile, "unfit")
        return False
    start_of_ag = rolling_sum[rolling_sum >= 0.9 * timeframe].idxmin()

    # Delete rows after last minute with 80% of ART values above 40mmHg
    end_of_ag = (rolling_sum[rolling_sum >= 0.8 * timeframe]).iloc[::-1].idxmax() + timeframe

    df = df.iloc[start_of_ag : end_of_ag - start_of_ag]

    # Fill missing values from undersampled columns
    df.fillna(method="ffill", inplace=True)
    df.fillna(method="bfill", inplace=True)

    # Pickle the dataframe
    df.to_pickle(ofile)

    verbose("File", ifile, "preprocessed and pickled")

    return True


def folder_preprocessing(ifolder: str, ofolder: str, force: bool = False) -> None:
    """Quality control and preprocessing of all vital files in a folder ; multithreaded

    Args:
        ifolder (str): path to the folder containing the vital files
        ofolder (str): path where to save the vital files that don't pass QC
    """
    makedirs(ofolder, exist_ok=True)
    makedirs("data/unfit", exist_ok=True)
    futures = []

    ipaths = []
    opaths = []

    for file in listdir(ifolder):
        if not file.endswith("vital"):
            continue
        ipath = join(ifolder, file)
        ofilename = basename(ipath)[:-5] + "pkl"
        opath = join(ofolder, ofilename)

        if exists(opath) and not force:
            verbose("File", opath, "already preprocessed, skipping")
            continue
        ipaths.append(ipath)
        opaths.append(opath)

    assert len(ipaths) == len(opaths), "Number of files does not match"

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = list(executor.map(preprocessing, ipaths, opaths))

    verbose(f"Valid cases : {sum(futures)}/{len(ipaths)}")


VERBOSE = False
INTERVAL = 1 / 100

TRACKS = [
    "ART",
    "Solar8000/PLETH_SPO2",
    "Orchestra/PPF20_CP",
    "Primus/CO2",
]
OPS = [
    "Cholecystectomy",
    "Distal gastrectomy",
    "Distal pancreatectomy",
    "Exploratory laparotomy",
    "Hemicolectomy",
    "Hernia repair",
    "Low anterior resection",
    "Lung lobectomy",
    "Lung segmentectomy",
    "Lung wedge resection",
    "Thyroid lobectomy",
    "Total thyroidectomy",
]


if __name__ == "__main__":
    VERBOSE = "-v" in argv
    if "-dl" in argv:
        max_cases = int(input("Number of cases to download : "))
        case_ids = find_cases(TRACKS, OPS)
        download_cases(TRACKS, case_ids, max_cases=max_cases)
    if "-csv" in argv:
        folder_vital_to_csv("data/vital/", "data/csv/", interval=INTERVAL)
    if "-pre" in argv:
        folder_preprocessing("data/vital", "data/preprocessed")
    elif "-pref" in argv:
        folder_preprocessing("data/vital", "data/preprocessed", force=True)
