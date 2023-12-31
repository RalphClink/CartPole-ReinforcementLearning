import time


class timer:
    def __init__(self):
        self._start_time = None

    def start(self):
        # Start a new timer
        if self._start_time is not None:
            print(f"Timer is running. Use .stop() to stop it")
        self._start_time = time.perf_counter()

    def stop(self):
        # Stop the timer, and report the elapsed time"""
        if self._start_time is None:
            print(f"Timer is not running. Use .start() to start it")
        elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None

        # print(f"Elapsed time: {elapsed_time:0.4f} seconds")
        return float(round(elapsed_time, 4))

