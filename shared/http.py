from typing import Any, Dict

import requests


__all__ = (
    "http",
)


class Http(object):

    def __init__(self):
        super(Http, self).__init__()

    def post(
            self,
            url: str,
            data: Any = None,
            headers: Dict | None = None,
            **kwargs,
    ):
        ret = requests.post(
            url=url,
            json=data,
            headers=headers,
        )
        return ret.json()


http = Http()
