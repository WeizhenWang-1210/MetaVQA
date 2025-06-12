def divide_list_into_n_chunks(lst, n):
    """
    Divides a list `lst` into `n` nearly equal-sized chunks.

    Args:
        lst (List): The list to be divided.
        n (int): The number of chunks to create.

    Returns:
        List[List]: A list of n chunks, where each chunk is a sublist of the original list.
    """
    # Ensure n is a positive integer
    if n <= 0:
        raise ValueError("n must be a positive integer")

    # Calculate the size of each chunk
    total_length = len(lst)
    chunk_size, remainder = divmod(total_length, n)

    chunks = []
    start_index = 0

    for i in range(n):
        # Calculate the end index for the current chunk, adding one if there's a remainder
        end_index = start_index + chunk_size + (1 if i < remainder else 0)
        # Slice the list to create the chunk and add it to the list of chunks
        chunks.append(lst[start_index:end_index])
        # Update the start index for the next chunk
        start_index = end_index

    return chunks


def divide_into_intervals_exclusive(total, n, start=0):
    """
    Divides a `[start, total)` range into `n` exclusive intervals, starting from a given start point.

    Args:
        total (int): The total size of the range to be divided.
        n (int): The number of intervals to create.
        start (int): The starting point of the range. Default is 0.
    Returns:
        intervals (List[Tuple[int, int]]): A list of tuples representing the start and exclusive end of each interval.
    """
    # Calculate the basic size of each interval and the remainder
    interval_size, remainder = divmod(total, n)
    intervals = []
    for i in range(n):
        # Determine the exclusive end of the current interval
        # If there's a remainder, distribute it among the first few intervals
        end = start + interval_size + (1 if i < remainder else 0)
        # Add the interval to the list, note that the 'end' is now exclusive
        intervals.append((start, end))
        # Update the start for the next interval
        start = end
    return intervals