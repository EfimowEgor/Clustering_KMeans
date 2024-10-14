from typing import Protocol


class IGifCreator(Protocol):
    def add_frame(self, X, labels, centroids, iteration): ...

    def save(self): ...
