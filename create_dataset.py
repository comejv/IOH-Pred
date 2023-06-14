import json
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
    if exists(join(env["dataFolder"], f"vital/{case_id}.vital")):
        verbose(f"Case {case_id} already exists")
        return
    try:
        case = vdb.VitalFile(case_id, track_names, interval)
        # Rename tracks
        # case.rename_tracks(mapping)
        case.to_vital(opath=join(env["dataFolder"], f"vital/{case_id}.vital"))
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
    makedirs(join(env["dataFolder"], "vital"), exist_ok=True)
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


def test_hypothension(case_name: str, case: pd.DataFrame, ofolder: str) -> None:
    """Test for hypohetnsion in a case

    Args:
        case (pd.DataFrame): case
    """
    # Hypotension event if more than 1 min of consecutive MAP below 65mmHg
    # MAP is mean average of SNUADC/ART, moving window of 2 seconds
    mask = case["Solar8000/ART_MBP"].lt(65)
    # idxs = case.index[mask.rolling(window=60 * SAMPLING_RATE, axis=0).apply(lambda x: x.all(), raw=True) == True]
    for i in range(len(mask) - 60 * SAMPLING_RATE):
        if all(mask[i : i + 60 * SAMPLING_RATE]):
            case.to_pickle(join(ofolder, f"event/{case_name}.pkl"))
            break
    else:
        case.to_pickle(join(ofolder, f"nonevent/{case_name}.pkl"))


def preprocessing(ifile: str, ofolder: str) -> bool:
    """Preprocessing and pickling of a vital file

    Args:
        ifile (str): path to the vital file
        ofolder (str): path where to save the csv

    Returns:
        bool: True if preprocessing was successful
    """
    vital = vdb.VitalFile(ifile)
    track_names = vital.get_track_names()
    df = vital.to_pandas(track_names, 1 / SAMPLING_RATE)

    # Delete rows with only nan values
    df.dropna(axis="index", subset=["SNUADC/ART"], inplace=True)

    # Number of rows in a minute
    timeframe = SAMPLING_RATE * 60

    mask_art = (df["SNUADC/ART"] > 30) & (df["SNUADC/ART"] < 160)
    rolling_sum = mask_art.rolling(window=timeframe).sum()
    # If less than 30 minutes of valid data or less than 70% of MAP, case is unfit
    # Note : only 1 MAP value every 1.7 * SAMPLING_RATE rows
    if (rolling_sum == timeframe).sum() < timeframe * 30 or (
        df["Solar8000/ART_MBP"].isna().sum()/len(df) < 0.7 * 1/(1.7*SAMPLING_RATE)
    ):
        rename(ifile, join(join(env["dataFolder"], "unfit/"), basename(ifile)))
        verbose("File", ifile, "unfit")
        return False
    start_of_ag = rolling_sum[rolling_sum >= 0.9 * timeframe].idxmin()

    # Delete rows after last minute with 80% of ART values above 40mmHg
    end_of_ag = (rolling_sum[rolling_sum >= 0.8 * timeframe]).iloc[
        ::-1
    ].idxmax() + timeframe

    df = df.iloc[start_of_ag : end_of_ag - start_of_ag]

    # Fill missing values from undersampled columns
    df.fillna(method="ffill", inplace=True)

    # Deleting outliers using MAP
    # Outliers are defined as measure where derivative of MAP is greater than 5mmHg
    derivative_map = df["Solar8000/ART_MBP"].diff()
    derivative_map.replace(0, None, inplace=True)
    derivative_map.fillna(method="bfill", inplace=True)
    mask_map = ~(derivative_map.abs() > 2)
    df = df[mask_map & mask_art]

    # df.reset_index(inplace=True, drop=True)

    # Test for hypothension
    test_hypothension(basename(ifile)[:-6], df, ofolder)

    verbose("File", ifile, "preprocessed and pickled")

    return True


def folder_preprocessing(ifolder: str, ofolder: str, force: bool = False) -> None:
    """Quality control and preprocessing of all vital files in a folder ; multithreaded

    Args:
        ifolder (str): path to the folder containing the vital files
        ofolder (str): path where to save the vital files that don't pass QC
    """
    evt_folder = join(ofolder, "event")
    makedirs(evt_folder, exist_ok=True)
    nonevt_folder = join(ofolder, "nonevent")
    makedirs(nonevt_folder, exist_ok=True)
    makedirs(join(env["dataFolder"], "unfit"), exist_ok=True)
    futures = []

    ipaths = []
    opaths = []

    for file in listdir(ifolder):
        if not file.endswith("vital"):
            continue
        ipath = join(ifolder, file)
        ofilename = basename(ipath)[:-5] + "pkl"

        if (
            exists(join(evt_folder, ofilename))
            or exists(join(nonevt_folder, ofilename))
        ) and not force:
            verbose("File", ofilename, "already preprocessed, skipping")
            continue
        ipaths.append(ipath)
        opaths.append(ofolder)

    assert len(ipaths) == len(opaths), "Number of files does not match"

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = list(executor.map(preprocessing, ipaths, opaths))

    verbose(f"Valid cases : {sum(futures)}/{len(ipaths)}")


env_file = open("env.json", "r")
env = json.load(env_file)
env_file.close()

TRACKS = env["tracks"]
OPS = env["operations"]
SAMPLING_RATE = env["samplingRate"]
VERBOSE = env["verbose"]

if __name__ == "__main__":
    if "-dl" in argv:
        max_cases = int(input("Number of cases to download : "))
        case_ids = find_cases(TRACKS, OPS)
        download_cases(TRACKS, case_ids, max_cases=max_cases)
    if "-csv" in argv:
        folder_vital_to_csv(
            join(env["dataFolder"], "vital/"),
            join(env["dataFolder"], "csv/"),
            interval=SAMPLING_RATE,
        )
    if "-pre" in argv:
        folder_preprocessing(
            join(env["dataFolder"], "vital"), join(env["dataFolder"], "preprocessed")
        )
    elif "-pref" in argv:
        folder_preprocessing(
            join(env["dataFolder"], "vital"),
            join(env["dataFolder"], "preprocessed"),
            force=True,
        )
