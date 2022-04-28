#!/usr/bin/env python3
"""
Contains function sentientPlanets()

"""
from numpy import append
import requests as rq


def sentientPlanets():
    """Returns list of planets that have sentient beings
    """
    planets = []
    url = "https://swapi-api.hbtn.io/api/species/?format=json"

    while url:
        r = rq.get(url).json()
        for animals in r['results']:
            if (animals['designation'] == 'sentient' or
               animals['classification'] == 'sentient') and \
               animals['homeworld'] is not None:
                planets.append(rq.get(
                    animals['homeworld']
                ).json()['name'])
        url = r.get('next')

    return planets
