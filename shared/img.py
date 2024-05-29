import base64
from pathlib import Path


__all__ = (
    "img",
)

import cv2
import numpy as np


class Image(object):

    def __init__(self):
        super().__init__()

    def read_base64_from_file(
            self,
            img_path: str | Path,
    ):
        img_path = (
            img_path
            if isinstance(img_path, Path)
            else Path(img_path).absolute()
        )
        if not img_path.exists():
            raise FileNotFoundError(img_path)
        with img_path.open(mode="rb") as f:
            content = base64.b64encode(f.read())
            return str(content, "utf8")

    def parser_base64_to_numpy(
            self,
            b64: bytes | str,
            **kwargs,
    ):
        img = base64.b64decode(b64)
        ar = np.fromstring(img, np.uint8)
        img = cv2.imdecode(ar, cv2.IMREAD_COLOR)
        return img


img = Image()
