import random


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


def majority_true(things, criteria=lambda x: x, threshold=0.8):
    num_things = len(things)
    num_true = 0
    for thing in things:
        if criteria(thing):
            num_true += 1
    return num_true / num_things >= threshold


def find_label(obj_id, label2id):
    for label, id in label2id.items():
        if id == obj_id:
            return label
    return None


def split_list(lst, num_chunks):
    """# Check if num_chunks is greater than 0 to avoid division errors
    if num_chunks <= 0:
        raise ValueError("Number of chunks must be a positive integer.")

    # Calculate the approximate chunk size
    chunk_size = len(lst) // num_chunks
    remainder = len(lst) % num_chunks

    # Create the chunks, distributing the remainder across chunks
    chunks = []
    start = 0
    for i in range(num_chunks):
        end = start + chunk_size + (1 if i < remainder else 0)
        chunks.append(lst[start:end])
        start = end"""
    chunks = divide_list_into_n_chunks(lst, num_chunks)
    return chunks


def select_not_from(space, forbidden, population=1):
    unique_types = set(space)
    unique_forbiddens = set(forbidden)
    assert len(space)>len(forbidden)
    diff = unique_types.difference(unique_forbiddens)
    assert len(diff) >= population
    return random.sample(list(diff), population)
