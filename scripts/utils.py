

def read_resolutions_as_int_list(file_path):
    """Reads a file containing resolutions and returns them as a list of integers.

    Args:
    file_path: The path to the file containing the resolutions.

    Returns:
    A list of integers representing the resolutions found in the file.
    """

    with open(file_path, 'r') as file:
        resolutions = [int(line.strip()) for line in file]

    return resolutions