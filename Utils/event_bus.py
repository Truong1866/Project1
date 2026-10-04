from __future__ import annotations

import threading
import time
from collections import defaultdict
from typing import Callable

from Utils.logger import get_logger

log = get_logger("EventBus")

# Tên chủ đề chuẩn
PERSON_KNOWN = "person_known"      # nhận ra người quen
PERSON_UNKNOWN = "person_unknown"  # phát hiện người lạ
ALL = "*"


class EventBus:
    def __init__(self):
        self._subs: dict[str, list[Callable[[dict], None]]] = defaultdict(list)
        self._lock = threading.Lock()

    def subscribe(self, topic: str, callback: Callable[[dict], None]) -> Callable[[], None]:
        with self._lock:
            self._subs[topic].append(callback)

        def unsubscribe():
            with self._lock:
                if callback in self._subs[topic]:
                    self._subs[topic].remove(callback)

        return unsubscribe

    def publish(self, topic: str, **payload) -> dict:
        event = {"topic": topic, "ts": time.time(), **payload}
        with self._lock:
            callbacks = list(self._subs.get(topic, ())) + list(self._subs.get(ALL, ()))
        for cb in callbacks:
            try:
                cb(event)
            except Exception:  # một subscriber lỗi không được làm hỏng pipeline
                log.exception("Subscriber lỗi ở chủ đề %s", topic)
        return event