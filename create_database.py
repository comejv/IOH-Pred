from os import listdir, makedirs, remove, rename
from os.path import exists, join, basename
from sys import argv
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import vitaldb as vdb


def verbose(*args, **kwargs):
    if VERBOSE:
        print(*args, **kwargs)


def find_cases(track_names: list[str]) -> list[int]:
    return vdb.find_cases(track_names)


def download_case(case_id: int, track_names: list[str], index: int = None) -> None:
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
        case = vdb.VitalFile(case_id, track_names)
        case.to_vital(opath=f"data/vital/{case_id}.vital")
        if index:
            verbose(f"Downloaded case {case_id} : {index+1}/{len(case_ids)}")
        else:
            verbose(f"Downloaded case {case_id}")
    except KeyboardInterrupt:
        print("Download interrupted")
        return


def download_cases(track_names: list[str], case_ids: list[int]) -> None:
    """Download a list of cases from the VitalDB database ; multithreaded.

    Args:
        track_names (list): list of track names
        case_ids (list): list of case ids
    """
    makedirs("data/vital", exist_ok=True)
    with ThreadPoolExecutor(max_workers=10) as executor:
        for i, case_id in enumerate(case_ids):
            executor.submit(download_case, case_id, track_names, i)


def vital_to_csv(ipath: str, opath: str, interval: float = None) -> None:
    """Convert vital file to csv

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

    Args:
        ifolder (str): path to the folder containing the vital files
        ofolder (str): path where to save the csv files
        interval (float): interval resolution in seconds. Defaults to None (max res)
    """
    makedirs(ofolder, exist_ok=True)
    with ThreadPoolExecutor(max_workers=30) as executor:
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

            executor.submit(vital_to_csv, ipath, opath, interval)


def quality_control(vital_path: str, ofolder: str) -> bool:
    """Quality control of a vital file

    Args:
        vital (vdb.VitalFile): vital file to check

    Returns:
        bool: True if ok, False otherwise
    """
    vital = vdb.VitalFile(vital_path)
    df = vital.to_pandas(vital.get_track_names(), None)

    # Delete rows with only nan values
    df.dropna(axis="index", how="all", inplace=True)

    # No more than 20% missing data for ART and ART_MPB
    if df["ART"].isna().sum() > 0.2 * len(df):
        verbose(f"More than 20% of missing values for ART ({basename(vital_path)})")
        return False
    # TODO : Check for ART_MBP, taking into account that it's only
    # present every 1.7 seconds instead of every 1/500 seconds like ART

    # No more than 20% of values under 50mmHg for ART_MBP
    if (df["ART_MBP"] < 50).sum() > 0.2 * len(df):
        verbose("More than 20% of MAP under 50mmHg")
        return False

    # Move file to ofolder
    rename(src=vital_path, dst=join(ofolder, basename(vital_path)))

    return True


def folder_quality_control(ifolder: str, ofolder: str, force: bool = False) -> None:
    """Quality control of all vital files in a folder ; multithreaded

    Args:
        ifolder (str): path to the folder containing the vital files
        ofolder (str): path where to save the csv files
    """
    makedirs(ofolder, exist_ok=True)
    futures = []
    valid_cases = 0

    with ThreadPoolExecutor(max_workers=4) as executor:
        for file in listdir(ifolder):
            if not file.endswith("vital"):
                continue
            ipath = join(ifolder, file)
            ofilename = basename(ipath)
            opath = join(ofolder, ofilename)

            if exists(opath) and not force:
                verbose("File", opath, "already controlled, skipping")
                continue

            futures.append(executor.submit(quality_control, ipath, opath))

        for future in as_completed(futures):
            if future.result():
                valid_cases += 1

    verbose(f"Valid cases : {valid_cases}/{len(listdir(ifolder))}")


def load_cases(
    track_names: list[str],
    dir_path: str,
    case_ids: list[int] = None,
    num_cases: int = None,
) -> tuple[dict[pd.DataFrame], list[int]]:
    """Load a list of cases from the local database

    Args:
        track_names (list[str]): list of track names
        dir_path (str): path of files folder
        case_ids (list[int]): list of case ids
        num_cases (int): maximum number of cases to load

    Returns:
        dict[pd.DataFrame]: dictionnary with case id as key and case dataframe as value
        list[int]: list of all case_id loaded
    """
    cases: dict[pd.DataFrame] = dict()
    ids = []
    i = 0

    for file in listdir(dir_path):
        i += 1
        if not file.endswith(".vital"):
            continue
        try:
            case_id = int(file[: file.index(".")])
        except ValueError:  # file n'a pas d'extension visible, on ignore
            continue

        if case_ids and case_id not in case_ids:
            continue

        if num_cases and i > num_cases:
            break

        filename = f"data/vital/{case_id}.vital"
        if exists(filename):
            verbose(f"Loading case {case_id}")
            vital = vdb.VitalFile(ipath=filename, track_names=track_names)
            case_df = vital.to_pandas(track_names=track_names, interval=INTERVAL)
            cases[case_id] = case_df
            ids.append(case_id)
        else:
            verbose(f"Case {case_id} does not exist locally")
    return cases, ids


VERBOSE = False
INTERVAL = None  # for max res

TRACKS = ["ART", "ART_MBP", "CI", "SVI", "SVRI", "SVV", "ART_SBP", "ART_DBP"]

if __name__ == "__main__":
    VERBOSE = input("Verbose ? (y/[n]) ").lower() == "y"
    if input("Download cases ? (y/[n]) ").lower() == "y":
        case_ids = find_cases(TRACKS)
        download_cases(TRACKS, case_ids)

    if "-csv" in argv:
        folder_vital_to_csv("data/vital/", "data/csv/", interval=INTERVAL)
    if "-quality" in argv:
        folder_quality_control("data/vital/", "data/quality/", force=True)
