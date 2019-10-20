import time


def display_progress(iteration, total, total_length=50):

    progress = f"{iteration}/{total}"
    length = int(total_length * iteration // total)
    bar = "█" * length + "-" * (total_length - length)
    print(f"\rProgress: |{bar}| {progress}", end="\r")

    if iteration == total:
        print()


def display_runtime(config):

    total_runtime = config.runtime + time.time() - config.start_time

    min, sec = divmod(total_runtime, 60)
    hr, min = divmod(min, 60)
    print(f"Runtime: {hr:.0f} hours {min:.0f} minutes {sec:.2f} seconds")