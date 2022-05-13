#!/usr/bin/env python3
"""
Contains function schools_by_topic()
"""
from pymongo import MongoClient


def schools_by_topic(mongo_collection, topic):
    """
    Lists schools with a specified topic
    """
    school = mongo_collection.find({'topics': topic})
    return school
