#!/usr/bin/env python3
"""
Contains script that provides log information
"""
from pymongo import MongoClient


if __name__ == "__main__":
    """
    Provides stats on MongoDB
    """
    client = MongoClient('mongodb://127.0.0.1:27017')
    logs = client.logs.nginx
    doc = logs.count_documents({})
    print("{} logs".format(doc))
    print("Methods:")
    lst = ["GET", "POST", "PUT", "PATCH", "DELETE"]
    for i in lst:
        method2 = logs.count_documents({"method": i})
        print("\tmethod {}: {}".format(i, method2))
    fpath = {"method": "GET", "path": "/status"}
    path2 = logs.count_documents(fpath)
    print("{} status check".format(path2))
