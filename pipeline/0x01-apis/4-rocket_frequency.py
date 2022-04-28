#!/usr/bin/env python3
"""
Script prints how many times a rocket has launched
"""
import sys
import requests as rq
import time


if __name__ == '__main__':
    url = "https://api.spacexdata.com/v4/launches"
    r = rq.get(url)
    laun_count = {}

    for i in r.json():
        if i['rocket'] not in laun_count:
            laun_count[i['rocket']] = 1
        else:
            laun_count[i['rocket']] += 1

    url = "https://api.spacexdata.com/v4/rockets/"
    r = rq.get(url)
    rocks = []

    for i in r.json():
        if i['id'] in laun_count:
            rocks.append({'rocket': i['name'],
                          'launches': laun_count[i['id']]})
        else:
            continue

    laun_count = sorted(rocks, key=lambda i: i['rocket'])
    laun_count = sorted(rocks, key=lambda i: i['launches'], reverse=True)

    for i in laun_count:
        print("{}: {}".format(i['rocket'], i['launches']))
