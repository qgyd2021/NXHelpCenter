#!/usr/bin/python3
# -*- coding: utf-8 -*-
import requests


def main():
    url = "http://10.75.27.247:8050/render.html"

    params = {
        "url": "https://help.nxcloud.com/nxlink/docs/qOtoWc",
        "wait": 3,
        "images": 0,
    }

    response = requests.get(
        url=url,
        params=params,
    )
    print(response.text)

    return


if __name__ == '__main__':
    main()
