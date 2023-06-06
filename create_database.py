from os.path import exists
from os import makedirs, listdir
import vitaldb as vdb
import pandas as pd


def verbose(msg):
    if VERBOSE:
        print(msg)


def find_cases(track_names: list[str]) -> list[int]:
    return vdb.find_cases(track_names)


def download_cases(track_names: list, caseids: list) -> None:
    """Download a list of cases from the VitalDB database

    Args:
        track_names (list): list of track names
        caseids (list): list of case ids
    """
    makedirs("data/cases", exist_ok=True)
    for i, caseid in enumerate(caseids):
        if exists(f"data/cases/{caseid}.vital"):
            verbose(f"Case {caseid} already exists")
            continue
        try:
            verbose(f"Downloading case {caseid} : {i+1}/{len(caseids)}")
            case = vdb.VitalFile(caseid, track_names)
            case.to_vital(opath=f"data/cases/{caseid}.vital")
        except KeyboardInterrupt:
            print("Download interrupted")
            break


def load_cases(
    track_names: list[str], dir_path: str, caseids: list[int] = None
) -> pd.DataFrame:
    """Load a list of cases from the local database

    Args:
        track_names (list[str]): list of track names
        caseids (list[int]): list of case ids

    Returns:
        pd.DataFrame: dataframe of cases
    """
    cases: dict[pd.DataFrame] = dict()
    for file in listdir(dir_path):
        if not file.endswith(".vital"):
            continue
        try:
            caseid = int(file[: file.index(".")])
        except ValueError:  # file n'a pas d'extension visible, on ignore
            continue
        if caseids:
            if caseid not in caseids:
                continue

        filename = f"data/cases/{caseid}.vital"
        if exists(filename):
            verbose(f"Loading case {caseid}")
            vital = vdb.VitalFile(ipath=filename, track_names=track_names)
            case_df = vital.to_pandas(track_names=track_names, interval=INTERVAL)
            cases[caseid] = case_df
        else:
            verbose(f"Case {caseid} does not exist locally")
    return cases


def quality_control(dataframe: pd.DataFrame) -> pd.DataFrame:
    # Less than 20% missing data for ART_MPB
    pass


VERBOSE = False
INTERVAL = 1 / 500

TRACKS = ["ART", "ART_MBP", "CI", "SVI", "SVRI", "SVV", "ART_SBP", "ART_DBP"]

if __name__ == "__main__":
    VERBOSE = input("Verbose ? (y/n) ").lower() == "y"
    if input("Download cases ? (y/n)").lower() == "y":
        case_ids = find_cases(TRACKS)
        download_cases(TRACKS, case_ids)

    cases = load_cases(track_names=TRACKS, dir_path="data/cases", caseids=[1025, 4612, 1029])
