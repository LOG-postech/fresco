# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import threading
import typing

if typing.TYPE_CHECKING:
    import torch

"""
Implementation of a thread-safe queue with one producer and one consumer.
"""
class Queue:
    def __init__(self) -> None:
        self.queue: list[torch.Tensor] = []
        self.cv = threading.Condition()

    def add(self, tensor: torch.Tensor) -> None:
        self.cv.acquire()
        self.queue.append(tensor)
        self.cv.notify()
        self.cv.release()

    def remove(self) -> torch.Tensor:
        self.cv.acquire()
        while len(self.queue) == 0:
            self.cv.wait()
        tensor = self.queue.pop(0)
        self.cv.release()
        return tensor
