#!/usr/bin/env python3
"""
contains method availableShips(passengerCount)
"""
import requests as rq


def availableShips(passengerCount):
    """
    returns list of ships depending on passenger count
    """
    ships = []
    pag = 1
    signal = True
    while signal:
        url = "https://swapi-api.hbtn.io/api/starships/?page=" + str(pag)
        request = rq.get(url)
        data = requst.json()
        retrieved_data = data['retrieved_data']
        for ship in retrieved_data:
            passenger = ship['passengers']
            passenger = passenger.replace(',', "")
            if passenger.isnumeric() and int(passenger) >= passengerCount:
                ships.append(ship['name'])
        if data['next'] is None:
            signal = False
        pag = pag + 1
    return ships
