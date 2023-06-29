import json
from os import cpu_count


def verbose(*args, **kwargs):
    """Calls print in verbose mode (defined by env.VERBOSE)
    """
    if env.VERBOSE:
        print(*args, **kwargs)


def env(env_file: str="env.json"):
    """Load environment variables into the function's dictionary

    Args:
        env_file (str, optional): Path to the environment file . Defaults to "env.json".
    """
    env_file = open(env_file, "r")
    env_json = json.load(env_file)
    env_file.close()

    env.__dict__ |= env_json

    env.CORES = cpu_count() if cpu_count() > 1 else 4

env()
