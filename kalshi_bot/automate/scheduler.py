from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Dict


@dataclass
class Task:
    name: str
    interval_sec: int
    fn: Callable[[], None]
    last_run: float = 0.0


class Scheduler:
    def __init__(self) -> None:
        self.tasks: Dict[str, Task] = {}

    def add(self, name: str, interval_sec: int, fn: Callable[[], None]) -> None:
        self.tasks[name] = Task(name=name, interval_sec=interval_sec, fn=fn, last_run=0.0)

    def run_forever(self, sleep_sec: int = 1) -> None:
        while True:
            now = time.time()
            for task in self.tasks.values():
                if now - task.last_run >= task.interval_sec:
                    task.fn()
                    task.last_run = now
            time.sleep(sleep_sec)
