def decoder_function(character_idx_seq: str, idx_to_char_dict: dict) -> str:
    """
    Convert the string of comma seperated idx to
    the respective character string(word).

    Args:
        character_idx_seq (str): string of index seperated by comma
        idx_to_char_dict (dict): Corresponding dict of the language that maps idx to char

    Returns:
        str: The string
    """
    char_idx_list = str_idx_to_list(character_idx_seq)
    return "".join([idx_to_char_dict[i] for i in char_idx_list])


def str_idx_to_list(
    character_idx_seq: str,
) -> list[int]:
    """
    Convert string of idx to list of idx

    Args:
        character_idx_seq (str): string of index seperated by comma

    Returns:
        list[int]: list of idx
    """

    return [int(i) for i in character_idx_seq.split(",")]
