# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import threading

"""
Implementation of a thread-safe counter with many producers and many consumers.
"""
class Counter:
    def __init__(self, initial_count: int) -> None:
        self.count: int = initial_count
        self.cv = threading.Condition()

    def decrement(self) -> None:
        self.cv.acquire()
        self.count -= 1
        self.cv.notify_all()
        self.cv.release()

    def wait(self) -> None:
        self.cv.acquire()
        while self.count > 0:
            self.cv.wait()
        self.cv.release()
