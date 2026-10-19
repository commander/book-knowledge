"""Stage 8: the time log. Records each stage session, and checks the cap.

A background thread writes a "last_seen" time every 60 s while a stage runs.
If Colab disconnects, the next run counts the interrupted session up to its
last_seen time. The time of a disconnect is not lost.
"""
import threading
import time

from .common import gpu_name, read_json, write_json

HEARTBEAT = 60


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


class TimeLog:
    def __init__(self, path):
        self.path = path
        self.log = read_json(path) or {"first_start": now(), "first_start_epoch": time.time(), "sessions": []}
        for s in self.log["sessions"]:          # sessions cut by a disconnect
            if s.get("end_epoch") is None:
                s["end_epoch"] = s["last_seen_epoch"]
                s["status"] = "interrupted"
        self._lock = threading.Lock()
        self._stop = None
        self._save()

    def _save(self):
        with self._lock:
            write_json(self.path, self.log)

    def gpu_seconds(self):
        return sum(s["end_epoch"] - s["start_epoch"] for s in self.log["sessions"])

    def wall_seconds(self):
        return time.time() - self.log["first_start_epoch"]

    def start(self, stage):
        t = time.time()
        self.cur = {"stage": stage, "gpu": gpu_name(), "start": now(), "start_epoch": t,
                    "last_seen_epoch": t, "end_epoch": None, "status": "running"}
        self.log["sessions"].append(self.cur)
        self._save()
        self._stop = threading.Event()

        def beat():
            while not self._stop.wait(HEARTBEAT):
                with self._lock:
                    self.cur["last_seen_epoch"] = time.time()
                self._save()
        threading.Thread(target=beat, daemon=True).start()

    def end(self, status):
        self._stop.set()
        t = time.time()
        with self._lock:
            self.cur.update(end=now(), end_epoch=t, last_seen_epoch=t, status=status,
                            seconds=round(t - self.cur["start_epoch"], 1))
        self._save()

    def summary(self):
        by_stage = {}
        for s in self.log["sessions"]:
            by_stage[s["stage"]] = by_stage.get(s["stage"], 0) + s["end_epoch"] - s["start_epoch"]
        self.log["seconds_by_stage"] = {k: round(v, 1) for k, v in by_stage.items()}
        self.log["gpu_seconds_total"] = round(self.gpu_seconds(), 1)
        self.log["wall_seconds_total"] = round(self.wall_seconds(), 1)
        self._save()
        return self.log
