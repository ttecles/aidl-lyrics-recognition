def print_metadata(metadata, src=None):
    if src:
        print("-" * 10)
        print("Source:", src)
        print("-" * 10)
    print(" - sample_rate:", metadata.sample_rate)
    print(" - num_channels:", metadata.num_channels)
    print(" - num_frames:", metadata.num_frames)
    print(" - bits_per_sample:", metadata.bits_per_sample)
    print(" - encoding:", metadata.encoding)
    print()


def time2sample(time: float, sample_rate: int) -> int:
    """converts time to number of sample

    Args:
        time (float): time in seconds to be converted into samples
        sample_rate (int): sample rate to use

    Returns:
        int: sample
    """
    return round(time * sample_rate)


def sample2time(sample: int, sample_rate: int) -> float:
    """Converts sample into seconds
    Args:
        sample (int): sample to be converted into seconds
        sample_rate (int): sample rate to use

    Returns:
        float: second corresponding to the sample
    """
    return sample / sample_rate
