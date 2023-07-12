from concurrent.futures import ThreadPoolExecutor
from os import listdir, makedirs, remove, rename
from os.path import basename, exists, join
from sys import argv

import pandas as pd
import vitaldb as vdb
from progress.bar import ChargingBar

from utils import *


def find_cases(track_names: list[str], ops: list[str] = None) -> list[int]:
    if not exists("cases_preop.csv"):
        df_cases = pd.read_csv("https://api.vitaldb.net/cases")
        df_cases.to_csv("cases_preop.csv", index=True)
    else:
        df_cases = pd.read_csv("cases_preop.csv")
    # Cases selection
    final_set = set(vdb.find_cases(track_names))

    if ops:
        op_set = (
            pd.concat(
                [df_cases[df_cases["opname"] == op]["caseid"] for op in ops], axis=1
            )
            .stack()
            .values
        )
        final_set &= set(op_set)

    cases_ag = df_cases[df_cases["ane_type"] == "General"]["caseid"]
    final_set &= set(cases_ag)

    return list(final_set)


def download_case(
    case_id: int,
    track_names: list[str],
    bar: ChargingBar = None,
) -> None:
    """Download a single case from the database

    Args:
        case_id (int): case id
        track_names (list[str]): list of track names
        bar (ChargingBar): progress bar

    """
    if exists(join(env.DATA_FOLDER, f"vital/{case_id}.vital")):
        bar.next()
        return

    case = vdb.VitalFile(case_id, track_names)

    case.to_vital(opath=join(env.DATA_FOLDER, f"vital/{case_id}.vital"))
    if bar:
        bar.next()


def download_cases(
    track_names: list[str],
    case_ids: list[int],
    max_cases: int = None,
) -> None:
    """Download a list of cases from the VitalDB database ; multithreaded.

    Args:
        track_names (list): list of track names
        case_ids (list): list of case ids
    """
    makedirs(join(env.DATA_FOLDER, "vital"), exist_ok=True)
    if max_cases > 0:
        case_ids = case_ids[: min(len(case_ids), max_cases)]

    with ChargingBar(
        "Downloading\t",
        max=len(case_ids),
        suffix="%(index)d/%(max)d - ETA %(eta)ds",
        color=162,
    ) as bar:
        with ThreadPoolExecutor(max_workers=env.CORES) as executor:
            executor.map(
                download_case,
                case_ids,
                [track_names] * len(case_ids),
                [bar] * len(case_ids),
            )


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
            print("File", opath, "already converted, skipping")
            continue

        ipaths.append(ipath)
        opaths.append(opath)

    with ThreadPoolExecutor(max_workers=10) as executor:
        executor.map(vital_to_csv, ipaths, opaths, [interval] * len(ipaths))


def check_and_move(condition: bool, error_msg: str, src: str, dst: str) -> bool:
    if condition:
        print(error_msg)
        if not exists(dst):
            rename(src=src, dst=dst)
        else:
            print(
                f"Warning : file {dst} already exists, not removed from input folder"
            )
        return False
    return True


def test_hypothension(case_name: str, case: pd.DataFrame, ofolder: str) -> bool:
    """Test for hypohetnsion in a case

    Args:
        case (pd.DataFrame): case
    """
    # Hypotension event if more than 1 min of consecutive MAP below 65mmHg
    # MAP is mean average of SNUADC/ART, moving window of 2 seconds
    mask = case["Solar8000/ART_MBP"].lt(65)
    # idxs = case.index[mask.rolling(window=60 * env.SAMPLING_RATE, axis=0).apply(lambda x: x.all(), raw=True) == True]
    for i in range(len(mask) - 60 * env.SAMPLING_RATE):
        if all(mask[i : i + 60 * env.SAMPLING_RATE]):
            return True
    else:
        return False


def preprocessing(ifile: str, ofile: str, bar: ChargingBar = None) -> bool:
    """Preprocessing and pickling of a vital file

    Args:
        ifile (str): path to the vital file
        ofile (str): path where to save the gzipped file

    Returns:
        bool: True if preprocessing was successful
    """
    vital = vdb.VitalFile(ifile)
    track_names = vital.get_track_names()
    df = vital.to_pandas(track_names, 1 / env.SAMPLING_RATE)

    # Delete rows with only nan values
    df.dropna(axis="index", subset=["SNUADC/ART"], inplace=True)

    # Number of rows in a minute
    timeframe = env.SAMPLING_RATE * 60

    mask_art = df["SNUADC/ART"].between(30, 160)
    rolling_sum = mask_art.rolling(window=timeframe).sum()
    # If less than 30 minutes of valid data or less than 70% of MAP, case is unfit
    # Note : only 1 MAP value every 1.7 * env.SAMPLING_RATE rows
    if (rolling_sum == timeframe).sum() < timeframe * 30 or (
        df["Solar8000/ART_MBP"].isna().sum() / len(df)
        < 0.7 * 1 / (1.7 * env.SAMPLING_RATE)
    ):
        rename(ifile, join(join(env.DATA_FOLDER, "unfit/"), basename(ifile)))
        bar.next()
        return False
    start_of_ag = rolling_sum[rolling_sum >= 0.9 * timeframe].idxmin()

    # Delete rows after last minute with 80% of ART values above 40mmHg
    end_of_ag = (rolling_sum[rolling_sum >= 0.9 * timeframe]).iloc[
        ::-1
    ].idxmin() - timeframe

    df = df.iloc[start_of_ag:end_of_ag]

    # Fill missing values from undersampled columns
    df.fillna(method="ffill", inplace=True)

    # Deleting outliers using MAP
    # Outliers are defined as measure where derivative of MAP is greater than 5mmHg
    derivative_map = df["Solar8000/ART_MBP"].diff()
    derivative_map.replace(0, None, inplace=True)
    derivative_map.fillna(method="bfill", inplace=True)
    mask_map = (
        (derivative_map.abs() < 3)
        & (df["Solar8000/ART_MBP"] > 40)
        & (df["Solar8000/ART_MBP"] < 150)
    )

    total_mask = mask_map & df["SNUADC/ART"].between(30, 160)

    df = df[total_mask]

    # Don't reset index to keep timestamps
    # df.reset_index(inplace=True, drop=True)

    compression_set = {"method": "gzip", "compresslevel": 1, "mtime": 1}

    # Test for hypothension
    case_name = basename(ifile)[:-6]
    # if test_hypothension(case_name, df, ofolder):
    #     df.to_pickle(
    #         join(ofolder, f"event/{case_name}.gz"),
    #         compression=compression_set,
    #     )
    # else:
    #     df.to_pickle(
    #         join(ofolder, f"nonevent/{case_name}.gz"),
    #         compression=compression_set,
    #     )
    df.to_pickle(
        path=ofile,
        compression=compression_set,
    )

    bar.next()

    return True


def folder_preprocessing(
    ifolder: str, ofolder: str, force: bool = False, N: int = -1
) -> None:
    """Quality control and preprocessing of all vital files in a folder ; multithreaded

    Args:
        ifolder (str): path to the folder containing the vital files
        ofolder (str): path where to save the vital files that don't pass QC
    """
    makedirs(ofolder, exist_ok=True)
    makedirs(join(env.DATA_FOLDER, "unfit"), exist_ok=True)
    futures = []

    ipaths = []
    opaths = []

    n = 0
    for file in listdir(ifolder):
        if not file.endswith("vital"):
            continue
        ipath = join(ifolder, file)
        case_id = basename(ipath)[:-6]
        opath = join(ofolder, case_id + ".gz")

        if exists(opath) and not force:
            continue
        ipaths.append(ipath)
        opaths.append(opath)
        n += 1
        if n == N:
            break

    assert len(ipaths) == len(opaths), "Number of files does not match"

    with ChargingBar(
        "Preprocessing\t",
        max=len(ipaths),
        suffix="%(index)d/%(max)d - ETA %(eta)ds",
        color=162,
    ) as bar:
        with ThreadPoolExecutor(max_workers=env.CORES - 1) as executor:
            futures = list(
                executor.map(preprocessing, ipaths, opaths, [bar] * len(ipaths))
            )

    print(f"Valid cases : {sum(futures)}/{len(ipaths)}")


if __name__ == "__main__":
    if "-dl" in argv:
        max_cases = int(input("Number of cases to download : "))
        case_ids = find_cases(env.TRACKS)
        download_cases(env.TRACKS, case_ids, max_cases=max_cases)
    if "-csv" in argv:
        folder_vital_to_csv(
            join(env.DATA_FOLDER, "vital/"),
            join(env.DATA_FOLDER, "csv/"),
            interval=env.SAMPLING_RATE,
        )
    if "-pre" in argv:
        folder_preprocessing(
            ifolder=join(env.DATA_FOLDER, "vital"),
            ofolder=join(env.DATA_FOLDER, "preprocessed", "all"),
        )
    elif "-pref" in argv:
        folder_preprocessing(
            join(env.DATA_FOLDER, "vital"),
            join(env.DATA_FOLDER, "preprocessed", "all"),
            force=True,
        )
