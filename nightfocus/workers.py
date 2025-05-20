import os


def get_num_workers() -> int:
    num_workers = os.cpu_count()
    if num_workers is None:
        num_workers = 1
    else:
        if num_workers > 1:
            num_workers -= 1
    return num_workers
