import json
from os import cpu_count

# Basic python module providing functions to output
# formatted text using ANSI escape sequences.

# Author : Côme VINCENT, 2023
# github.com/comejv/utils-and-games/


def clear() -> None:
    """Clears the terminal screen, deletes scrollback buffer and\
moves the cursor to the top left corner of the screen."""
    print("\x1b[3J\x1b[H\x1b[J", end="")


def pitalic(*args, **kwargs) -> None:
    """Prints given arguments in italic. Stdout unless specified."""
    print("\x1b[3m", end="")
    print(*args, **kwargs)
    print("\x1b[23m", end="")


def pbold(*args, **kwargs) -> None:
    """Prints given arguments in bold. Stdout unless specified."""
    print("\x1b[1m", end="")
    print(*args, **kwargs)
    print("\x1b[22m", end="")


def pwarn(*args, **kwargs) -> None:
    """Prints given arguments in bold yellow. Stdout unless specified."""
    print("\x1b[1;33m", end="")
    print(*args, **kwargs)
    print("\x1b[22;39m", end="")


def perror(*args, **kwargs) -> None:
    """Prints given arguments in bold red. Stdout unless specified."""
    print("\x1b[1;31m", end="")
    print(*args, **kwargs)
    print("\x1b[22;39m", end="")


def pblink(s: str, **kwargs) -> None:
    """Prints a string in blink. Stdout unless specified."""
    print("\x1b[5m" + s + "\x1b[25m", **kwargs)


def binput(prompt: str) -> bool:
    """Asks the user to input a boolean value.
    Cancel with Ctrl+C."""

    # Get the user input
    bold_prompt = "\x1b[1m" + prompt + "\x1b[0m"
    try:
        str_input = input(bold_prompt).lower()
    except KeyboardInterrupt:
        print()
        exit(0)

    # Check if the input is valid
    bool_input = ["true", "1", "t", "y", "yes", "o", "false", "0", "f", "n", "no", "n"]

    while True:
        try:
            if bool_input.index(str_input) < 6:
                return True
            else:
                return False
        except ValueError:
            try:
                str_input = input(bold_prompt).lower()
            except KeyboardInterrupt:
                print()
                exit(0)


def print_table(data: list[str], headers: list[str]) -> None:
    """Shows a formatted table in the terminal."""
    max_length_column = [0] * len(headers)
    for i in range(len(data)):
        for j in range(len(data[i])):
            if len(str(data[i][j])) > max_length_column[j]:
                max_length_column[j] = len(str(data[i][j]))

    print("|", end="")
    for i, header in enumerate(headers):
        if len(header) > max_length_column[i]:
            max_length_column[i] = len(header)
        pbold(" " + header.ljust(max_length_column[i]) + " |", end="")
    print()

    print("*" * (sum(max_length_column) + 3 * len(headers) + 1))

    for row in data:
        print("|", end="")
        for i in range(len(row)):
            print(" " + str(row[i]).ljust(max_length_column[i]) + " |", end="")
        print()


def verbose(*args, **kwargs):
    """Calls print in verbose mode (defined by env.VERBOSE)"""
    if env.VERBOSE:
        print(*args, **kwargs)


def init_venv_pip():
    import subprocess
    from os.path import exists
    from sys import executable, prefix, base_prefix
    from time import sleep

    if prefix != base_prefix:
        print(
            "Running from inside a virtual environment. Installing dependencies..."
        )
        if (
            subprocess.call(
                [executable, "-m", "pip", "install", "-r", "requirements.txt"]
            )
            == 0
        ):
            pbold("Dependencies installed successfully. Proceeding in 3 seconds...")
            sleep(3)
            clear()
        else:
            perror(
                "Could not install dependencies, try manually installing them with:\npip install -r requirements.txt"
            )
            exit(1)
    else:
        print("Not running from inside a virtual environment.")
        if exists(".venv"):
            pbold(
                "A virtual environment is detected (.venv). Try activating it before running this script again:\nsource .venv/bin/activate"
            )
            exit(0)
        else:
            print("No virtual environment detected. Creating one now...")
            if subprocess.call([executable, "-m", "venv", ".venv"]) != 0:
                perror(
                    f"Could not create virtual environment, try manually creating it with:\n{executable} -m venv .venv"
                )
                exit(1)
            pbold(
                "Virtual environment created. Activate it by running :\nsource .venv/bin/activate\nThen run this script again to install dependencies."
            )
            exit(0)


def env(env_file: str = "env.json"):
    """Load environment variables into the function's __dict__. They can be accessed as attributes.

    Args:
        env_file (str, optional): Path to the environment file . Defaults to "env.json".
    """
    env_file = open(env_file, "r")
    env_json = json.load(env_file)
    env_file.close()

    env.__dict__ |= env_json

    env.CORES = cpu_count() if cpu_count() > 1 else 4


env()


if __name__ == "__main__":
    clear()
    pitalic("This is italic")
    pbold("This is bold")
    pwarn("This is a warning")
    perror("This is an error")
    pblink("This is blinking")
    print(binput("Do you like this module ?"))
    print_table(
        data=[["Hello", "World"], ["This", "Is", "A", "Table"]],
        headers=["A", "B", "C", "D"],
    )
