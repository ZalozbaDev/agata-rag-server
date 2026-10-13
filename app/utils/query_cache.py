from __future__ import annotations

from collections import OrderedDict
from typing import Generic, TypeVar

V = TypeVar('V')


class LruCache(Generic[V]):
    """Small in-process LRU cache for repeated questions."""

    def __init__(self, max_size: int) -> None:
        self.max_size = max(0, max_size)
        self._items: OrderedDict[str, V] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> V | None:
        if self.max_size == 0:
            return None
        value = self._items.get(key)
        if value is None:
            self.misses += 1
            return None
        self._items.move_to_end(key)
        self.hits += 1
        return value

    def put(self, key: str, value: V) -> None:
        if self.max_size == 0:
            return
        self._items[key] = value
        self._items.move_to_end(key)
        while len(self._items) > self.max_size:
            self._items.popitem(last=False)

    def __len__(self) -> int:
        return len(self._items)
