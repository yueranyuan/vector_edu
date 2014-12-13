from __future__ import division
import sys


def winalert(count=3, length=1000, freq=600, interval=500):
    if not sys.platform.startswith('win'):
        raise Exception("trying to use winsound on non-windows system")
    import winsound
    import time
    for i in range(count - 1):
        winsound.Beep(freq, length)
        time.sleep(interval / 1000)
    else:
        winsound.Beep(freq, length)

if __name__ == "__main__":
    winalert()
