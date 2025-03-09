from pathlib import Path

def _test_exists(path):
    assert path.exists(), f"{path} does not exist"

def _get_mount(fileshare: str, cluster: bool) -> Path:
    mounts = {
        "nrs": {
            True: Path("/nrs/funke"),
            False: Path("/Volumes/funke"),
        },
        "groups": {
            True: Path("/groups/funke/home"),
            False: Path("/Volumes/funke$"),
        },
    }
    if fileshare not in mounts.keys():
        raise ValueError(f"Fileshare {fileshare} not in supported set {list(mounts.keys())}")
    
    return mounts[fileshare][cluster]
