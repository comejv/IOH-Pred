from os import listdir, makedirs, remove
from os.path import exists, join, basename
from concurrent.futures import ThreadPoolExecutor
from tempfile import NamedTemporaryFile

import pandas as pd
import vitaldb as vdb


def verbose(*args, **kwargs):
    if VERBOSE:
        print(*args, **kwargs)


def find_cases(track_names: list[str]) -> list[int]:
    return vdb.find_cases(track_names)


def download_cases(track_names: list, case_ids: list) -> None:
    """Download a list of cases from the VitalDB database

    Args:
        track_names (list): list of track names
        case_ids (list): list of case ids
    """
    makedirs("data/cases", exist_ok=True)
    for i, case_id in enumerate(case_ids):
        if exists(f"data/cases/{case_id}.vital"):
            verbose(f"Case {case_id} already exists")
            continue
        try:
            verbose(f"Downloading case {case_id} : {i+1}/{len(case_ids)}")
            case = vdb.VitalFile(case_id, track_names)
            case.to_vital(opath=f"data/cases/{case_id}.vital")
        except KeyboardInterrupt:
            print("Download interrupted")
            break


def vital_to_csv(ipath, opath, interval):
    try:
        vital = vdb.VitalFile(ipath)
        track_names = vital.get_track_names()
        with open(opath, "w") as tmp:
            df = vital.to_pandas(track_names, interval)
            df.to_csv(tmp)
        verbose(ipath, "converted to csv")
    except Exception as e:
        print(f"Could not convert {ipath} to csv : {e}")
        if exists(opath):
            remove(opath)


def folder_vital_to_csv(ifolder, ofolder, interval):
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


def quality_control(df: pd.DataFrame) -> bool:
    # No more than 20% missing data for ART and ART_MPB
    if df["ART"].isna().sum() > 0.3 * len(df):
        verbose("More than 20% of missing values")
        return False
    if df["ART_MBP"].isna().sum() > 0.3 * len(df):
        verbose("More than 20% of missing values")
        return False

    # No more than 20% of values under 50mmHg for ART_MBP
    if (df["ART_MBP"] < 50).sum() > 0.2 * len(df):
        verbose("More than 20% of MAP under 50mmHg")
        return False

    return True


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
INTERVAL = None # for max res

TRACKS = ["ART", "ART_MBP", "CI", "SVI", "SVRI", "SVV", "ART_SBP", "ART_DBP"]

if __name__ == "__main__":
    VERBOSE = input("Verbose ? (y/[n]) ").lower() == "y"
    if input("Download cases ? (y/[n]) ").lower() == "y":
        case_ids = find_cases(TRACKS)
        download_cases(TRACKS, case_ids)

    # cases, loaded_ids = load_cases(track_names=TRACKS, dir_path="data/cases", num_cases=15)

    # print(sum([quality_control(cases[cid]) for cid in loaded_ids]), "/", len(loaded_ids), "pass")

    folder_vital_to_csv("data/vital/", "data/csv/", interval=INTERVAL)
