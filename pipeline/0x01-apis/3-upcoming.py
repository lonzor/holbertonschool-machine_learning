#!/usr/bin/env python3
"""
script that display space upcoming ship launching information
"""
import sys
import requests as rq
import time

if __name__ == '__main__':
    url = "https://api.spacexdata.com/v4/launches/upcoming"
    r = rq.get(url)
    laun = sorted(r.json(), key=lambda i: i['date_unix'])
    date = laun[0]['date_unix']

    for d in r.json():
        if d['date_unix'] == date:
            name = d['name']
            l_date = d['date_local']
            rock = d['rocket']
            pad = d['launchpad']
            break

    url = "https://api.spacexdata.com/v4/rockets/{}".format(rock)
    r = rq.get(url)
    r_name = r.json()['name']

    url = "https://api.spacexdata.com/v4/launchpads/{}".format(pad)
    r = rq.get(url)
    p_name = r.json()['name']
    pad_loc = r.json()['locality']

    print("{} ({}) {} - {} ({})".format(name, l_date, r_name, p_name, pad_loc))
