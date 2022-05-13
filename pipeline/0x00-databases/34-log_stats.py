#!/usr/bin/env python3
"""
Contains script that provides log information
"""
from pymongo import MongoClient


if __name__ == "__main__":
    client = MongoClient('mongodb://127.0.0.1:27017')
    logs = client.logs.nginx
    count = logs.count_documents({})
    print("{} logs".format(count))
    print("Methods:")
    methods = ["GET", "POST", "PUT", "PATCH", "DELETE"]
    for m in methods:
        count2 = logs.count_documents({"method": m})
        print("\tmethod {}: {}".format(m, count2))
    fpath = {"method": "GET", "path": "/status"}
    count2 = logs.count_documents(fpath)
    print("{} status check".format(count2))