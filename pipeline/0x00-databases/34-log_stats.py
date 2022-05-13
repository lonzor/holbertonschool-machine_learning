#!/usr/bin/env python3
"""shows stats about Nginx logs"""


if __name__ == "__main__":
    """
    Script provides stats
    """
    client = MongoClient('mongodb://127.0.0.1:27017')
    logs = client.logs.nginx
    count = logs.count_documents({})
    print("{} logs".format(count))
    print("Methods:")
    methods = ["GET", "POST", "PUT", "PATCH", "DELETE"]
    for m in methods:
        meth = logs.count_documents({"method": meth})
        print("\tmethod {}: {}".format(m, meth))
    fpath = {"method": "GET", "path": "/status"}
    path2 = logs.count_documents(fpath)
    print("{} status check".format(path2))