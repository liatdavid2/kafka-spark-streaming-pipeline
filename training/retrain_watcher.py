import time
from pathlib import Path
import subprocess

DATA_DIR = Path("/app/output/unsw_stream")

known_partitions = set()


def get_partitions():
    return sorted([p.name for p in DATA_DIR.glob("date=*")])


def main():

    global known_partitions

    print("Retraining watcher started...")

    while True:

        partitions = set(get_partitions())

        new_partitions = partitions - known_partitions

        if new_partitions:

            print("New partitions detected:", new_partitions)

            sorted_parts = sorted(partitions)

            if len(sorted_parts) > 1:

                previous_partition = sorted_parts[-2]

                print("Training on partition:", previous_partition)

                subprocess.run(["python", "train.py"])

            known_partitions = partitions

        time.sleep(60)


if __name__ == "__main__":
    main()