import time

class TokenMetrics:
    """Simple collector for token‑saving metrics.

    The real ETK/RTK tools provide token counts before and after a command.
    This stub stores a list of records ``(command, before, after, duration)``
    and can compute total savings.
    """

    def __init__(self):
        self.records = []

    def record(self, command: str, before: int, after: int):
        duration = time.time()  # placeholder for elapsed time
        self.records.append({
            "command": command,
            "before": before,
            "after": after,
            "saved": max(0, before - after),
            "duration": duration,
        })

    def total_saved(self) -> int:
        return sum(r["saved"] for r in self.records)

    def summary(self) -> str:
        return f"Recorded {len(self.records)} commands, total tokens saved: {self.total_saved()}"
