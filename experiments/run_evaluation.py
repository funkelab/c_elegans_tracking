import argparse
from pathlib import Path

from c_elegans_utils.experiment import Experiment
from c_elegans_utils.tracking.evaluate import evaluate

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_dir")
    parser.add_argument("--time-range", nargs=2, type=int, default=None)
    parser.add_argument("-c", "--cluster", action="store_true")
    args = parser.parse_args()

    mount_path = Path("/groups/funke") if args.cluster else Path("/Volumes/funke$")
    base_path = mount_path / "malinmayorc/experiments/c_elegans_tracking"

    exp_dir = base_path / args.exp_dir
    exp = Experiment.from_dir(exp_dir, cluster=args.cluster)
    evaluate(exp)
