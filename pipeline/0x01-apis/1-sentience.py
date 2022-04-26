#!/usr/bin/env python3
"""
Contains function sentientPlanets()

"""
import requests as rq


def sentientPlanets():
    """
    Returns list of planets that have sentient beings
    """
    planets = []
    pag = 1
    signal = True
        while signal:
            url = "https://swapi-api.hbtn.io/api/species/?page=" + str(pag)
            request = rq.get(url)
            data = request.json()
            retrieved_data = data['results']
            for animal in retrieved_data:
                if animal['classification'] == 'sentient' or \
                   animal['designation'] == 'sentient':
                    home = animal['homeworld']
                    if home is not None:
                        r = rq.get(animal['homeworld']
                        home_data = req.json()
                        planets.append(home_data['name'])
            if data['next'] is None:
                signal = False
            pag = pag + 1
    return planets
