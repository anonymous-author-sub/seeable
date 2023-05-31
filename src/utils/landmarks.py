import numpy as np
from typing import Dict
from functools import lru_cache

SYM_1_5_DICT = {1: 2, 3: 3, 4: 5}

# START AT 1
SYM_1_68_DICT = {
    1: 17,
    2: 16,
    3: 15,
    4: 14,
    5: 13,
    6: 12,
    7: 11,
    8: 10,
    #
    51: 53,
    50: 54,
    49: 55,
    60: 56,
    59: 57,
    #
    62: 64,
    61: 65,
    68: 66,
    #
    33: 35,
    32: 36,
    #
    37: 46,
    38: 45,
    39: 44,
    40: 43,
    41: 48,
    42: 47,
    #
    18: 27,
    19: 26,
    20: 25,
    21: 24,
    22: 23,
    #
    # id
    9: 9,
    58: 58,
    67: 67,
    63: 63,
    52: 52,
    34: 34,
    31: 31,
    30: 30,
    29: 29,
    28: 28,
}

SYM_1_81_DICT = {
    **SYM_1_68_DICT,
    69: 81,
    70: 80,
    71: 79,
    72: 78,
    73: 77,
    74: 76,
    75: 75,
}


def reorder_landmarks(landmarks):
    """Reorder the landmark to the correct order
    d81 keeps the same order as dlib68 for the first 68
    but the last 13 doesnt form a closed loop
    """
    if landmarks.shape[0] == 81:
        ord_src = np.arange(68, 68 + 13)
        ord_dst = np.array([77, 75, 76, 68, 69, 70, 71, 80, 72, 73, 79, 74, 78])
        landmarks[ord_src] = landmarks[ord_dst]
    return landmarks


def get_symetric_landmarks(n: int) -> Dict[int, int]:
    """Get the symetric landmarks"""
    if n == 5:
        return SYM_1_5_DICT
    elif n == 68:
        return SYM_1_68_DICT
    elif n == 81:
        return SYM_1_81_DICT
    else:
        raise ValueError(f"Invalid number of landmarks {n}")


@lru_cache
def get_symmetric_permutation(n: int) -> np.array:
    """return the permutation array from landmarks to its vertical symetric landmarks
    raise ValueError if the number of landmarks is not 68 or 81
    """
    sym_dict = get_symetric_landmarks(n)
    perm = np.arange(n)
    for i, j in sym_dict.items():
        perm[i - 1], perm[j - 1] = perm[j - 1], perm[i - 1]
    return perm


if __name__ == "__main__":
    x = get_symmetric_permutation(68)
    # print(x)
    x = np.arange(81).repeat(2).reshape(81, 2)
    y = reorder_landmarks(x.copy())
  