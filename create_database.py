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
    op_set = pd.concat([df_cases[df_cases["opname"] == op]["caseid"] for op in ops], axis=1).stack().values
    cases_ag = df_cases[df_cases["ane_type"] == "General"]["caseid"]

    return list(set(cases_tracks) & set(op_set) & set(cases_ag))


def download_case(
    case_id: int, track_names: list[str], interval: float = None, index: int = None
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
        case.to_vital(opath=f"data/vital/{case_id}.vital")
        if index:
            verbose(f"Downloaded case {case_id} : {index+1}/{len(case_ids)}")
        else:
            verbose(f"Downloaded case {case_id}")
    except KeyboardInterrupt:
        print("Download interrupted")
        return


def download_cases(
    track_names: list[str], case_ids: list[int], interval: float = None, max_cases: int = None
) -> None:
    """Download a list of cases from the VitalDB database ; multithreaded.

    Args:
        track_names (list): list of track names
        case_ids (list): list of case ids
    """
    makedirs("data/vital", exist_ok=True)
    if max_cases:
        case_ids = case_ids[:min(len(case_ids), max_cases)]
    with ThreadPoolExecutor(max_workers=10) as executor:
        for i, case_id in enumerate(case_ids):
            executor.submit(download_case, case_id, track_names, interval, i)


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


def quality_check(vital_path: str, opath: str) -> bool:
    """Quality control of a vital file, if unfit moves file to opath

    Args:
        vital (vdb.VitalFile): vital file to check

    Returns:
        bool: True if ok, False otherwise
    """
    vital = vdb.VitalFile(vital_path)
    track_names = [item.split("/")[-1] for item in vital.get_track_names()]
    try:
        df = vital.to_pandas(track_names, None)
    except ValueError:
        verbose(f"Could not convert {vital_path} to pandas")
        rename(src=vital_path, dst=opath)
        return False

    # Delete rows with only nan values
    df.dropna(axis="index", how="all", inplace=True)

    # No more than 20% missing data for ART and ART_MPB
    if not check_and_move(
        condition=df["ART"].isna().sum() > 0.2 * len(df),
        error_msg="More than 20% of ART missing",
        src=vital_path,
        dst=opath,
    ):
        return False
    # TODO : Check for ART_MBP, taking into account that it's only
    # present every 1.7 seconds instead of every 1/500 seconds like ART

    # No more than 20% of values under 50mmHg for ART_MBP
    if not check_and_move(
        condition=(df["ART_MBP"] < 50).sum() > 0.2 * len(df),
        error_msg="More than 20% of MAP under 50mmHg",
        src=vital_path,
        dst=opath,
    ):
        return False
    return True


def folder_quality_check(ifolder: str, ofolder: str, force: bool = False) -> None:
    """Quality control of all vital files in a folder ; multithreaded

    Args:
        ifolder (str): path to the folder containing the vital files
        ofolder (str): path where to save the vital files that don't pass QC
    """
    makedirs(ofolder, exist_ok=True)
    futures = []

    ipaths = []
    opaths = []

    for file in listdir(ifolder):
        if not file.endswith("vital"):
            continue
        ipath = join(ifolder, file)
        ofilename = basename(ipath)
        opath = join(ofolder, ofilename)

        if exists(opath) and not force:
            verbose("File", opath, "already controlled, skipping")
            continue
        ipaths.append(ipath)
        opaths.append(opath)

    assert len(ipaths) == len(opaths)

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = list(executor.map(quality_check, ipaths, opaths))

    verbose(f"Valid cases : {sum(futures)}/{len(ipaths)}")


VERBOSE = False
INTERVAL = None  # for max res

TRACKS = [
    "ART",
    "ART_MBP",
    "ART_SBP",
    "ART_DBP",
    "PLETH_SPO2",
    "RR_CO2",
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
    if "-qc" in argv:
        folder_quality_check("data/vital/", "data/unfit/", force=False)
