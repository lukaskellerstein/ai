from typing import Dict
from abc import ABC, abstractmethod


class DBObject(object):
    _id: str
    _moduleName: str
    _className: str

    def __init__(self, moduleName: str, className: str):
        self._moduleName = moduleName
        self._className = className
