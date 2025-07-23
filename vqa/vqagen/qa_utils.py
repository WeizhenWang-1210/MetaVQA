import re

import numpy as np


def replace_substrs(original_string, mapping):
    # Define the pattern to match <n> where n is an integer
    pattern = r'<(\d+)>'

    # Function to use for substitution
    def replacer(match):
        # Extract the number n from <n>
        n = int(match.group(1))
        # Check if n is in the mapping and replace with its mapped value
        return f"<{str(mapping.get(n, match.group(0)))}>"

    # Substitute using re.sub with the replacer function
    replaced_string = re.sub(pattern, replacer, original_string)

    return replaced_string


def create_options(present_values, num_options, answer, namespace, transform=None):
    if len(present_values) < num_options:
        space = set(namespace)
        result = set(present_values)
        diff = space.difference(result)
        choice = np.random.choice(np.array(list(diff)), size=num_options - len(present_values), replace=False)
        result = list(result) + list(choice)
    elif len(present_values) == num_options:
        result = present_values
    else:
        answer = {answer}
        space = set(present_values)
        diff = space.difference(answer)
        choice = np.random.choice(np.array(list(diff)), size=num_options - 1, replace=False)
        result = list(answer) + list(choice)
    if transform:
        if callable(transform):
            result = [transform(o) for o in result]
        elif isinstance(transform, dict):
            result = [transform[o] for o in result]
    #paired_list = list(enumerate(result))
    np.random.shuffle(result)
    #shuffled_list = [element for _, element in paired_list]
    return result  #, {answer: index for index, answer in paired_list}


def create_multiple_choice(options):
    assert len(options) < 26, "no enough alphabetic character"
    result = []
    answer_to_choice = {}
    for idx, option in enumerate(options):
        label = chr(idx + 64 + 1)
        result.append(
            "({}) {}".format(label, option)
        )
        answer_to_choice[option] = label
    return "; ".join(result) + ".", answer_to_choice


def split_list(lst, num_chunks):
    # Check if num_chunks is greater than 0 to avoid division errors
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
        start = end

    return chunks


def find_label(obj_id, label2id):
    for label, id in label2id.items():
        if id == obj_id:
            return label
    return None
