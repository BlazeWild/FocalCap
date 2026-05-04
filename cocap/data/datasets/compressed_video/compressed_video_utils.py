# -*- coding: utf-8 -*-
# @Project : Hav-CoCap
# @File    : compressed_video_utils.py

import numpy as np

from cocap.utils.image import byte_imread, byte_imwrite


def serialize(data_to_serialize, quality):
    """
    Compress RGB I-frames by converting to JPEG format.
    Hav-CoCap Note: rgb_full and residual processing have been 
    stripped out to enforce the lightweight Delta-Distillation architecture.
    
    :param data_to_serialize: python dict
    :param quality: quality parameter for JPEG format
    :return: compressed python dict
    """
    data = {}
    for k, v in data_to_serialize.items():
        # ONLY compress the I-frames. Motion vectors pass through unmodified.
        if k == "rgb_gop":
            data[k] = [byte_imwrite(img, quality=quality) for img in v]
        else:
            data[k] = v
    return data


def deserialize(serialized_data):
    """
    Reverse version of serialize()
    """
    data = {}
    for k, v in serialized_data.items():
        if k == "rgb_gop":
            data[k] = [np.array(byte_imread(img)) for img in v]
        else:
            data[k] = v
    return data