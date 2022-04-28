#!/usr/bin/env python3
"""
prints the location of a user
"""
import sys
import requests as rq
import time


if __name__ == "__main__":
    url = sys.argv[1]
    pay = {'Accept': "application/vnd.github.v3+json"}
    r = rq.get(url, params=pay)
    if r.status_code == 403:
        lim = r.headers["X-Ratelimit-Reset"]
        x = (int(lim) - int(time.time())) / 60
        print("Reset in {} min".format(int(x)))
    if r.status_code == 200:
        loc = r.json()["location"]
        print(loc)
    if r.status_code == 404:
        print("Not found")
