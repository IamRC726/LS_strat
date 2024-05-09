def frequency(interval):
    if interval == "1d":
        return 252
    elif interval == "1w":
        return 52
    elif interval == "1m":
        return 12
    else:
        raise RuntimeError("Unsupported interval")
