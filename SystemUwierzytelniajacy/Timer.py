import time
from threading import Thread
import queue

global q
q = queue.Queue()


def setTimeout(delayInSeconds):
    time.sleep(delayInSeconds)
    q.put("STOP")


def startTimer(delayInSeconds):
    with q.mutex:
        q.queue.clear()
    Ts = Thread(target=setTimeout, args=(delayInSeconds,))
    Ts.setDaemon(True)
    Ts.start()