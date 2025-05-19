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


def color_code_text(row) -> str:
    """
    COlor code the prediction and the actual text

    Args:
        row (pandas row): Row from the prediction table

    Returns:
        str: HTML
    """
    input_text = row["Input"]
    actual_text = row["Actual_Y"]
    pred_text = row["Prediction"]

    colored = []
    # Compare up to the shorter length
    for i in range(min(len(actual_text), len(pred_text))):
        char = pred_text[i]
        color = "green" if char == actual_text[i] else "red"
        colored.append(f"<span style='color:{color}'>{char}</span>")

    # Extra predicted chars (over-prediction)
    for char in pred_text[len(actual_text) :]:
        colored.append(f"<span style='color:red'>{char}</span>")

    # Missing chars (under-prediction)
    for _ in actual_text[len(pred_text) :]:
        colored.append(f"<span style='color:red'>_</span>")

    return "".join(colored)
