import multiprocessing
from time import sleep
from random import random


def Log(txt):
    print(txt)


class _PopJob():
    def __init__(self, job_queue, timeout=1):
        self.job_queue = job_queue
        self.timeout = timeout

    def __enter__(self):
        return self.job_queue.get(timeout=self.timeout)

    def __exit__(self, exc_type, exc_value, traceback):
        self.job_queue.task_done()


class JobConsumer(multiprocessing.Process):
    def __init__(self, job_queue, func, params=None, id='[no_name]'):
        multiprocessing.Process.__init__(self)
        self.job_queue = job_queue
        self.id = id
        self.func = func
        self.params = params or {}
        print self.params, "PARAMS"

    def run(self):
        while True:
            with _PopJob(self.job_queue) as job:
                if job is None:
                    break
                Log('ec2 {id} is doing task {job.id}'.format(
                    id=self.id, job=job))
                self.func(**dict(self.params, **job.params))
        self.shutdown()

    def shutdown(self):
        Log('ec2 {id} is shutting down'.format(id=self.id))


class Job():
    def __init__(self, params, id='[no name]'):
        self.id = id
        self.params = params

    def __call__(self):
        self.func()


def _slow_print(num, consumer='[some consumer]'):
    sleep(random() * 2)
    Log('doing task {num} on {consumer}'.format(num=num, consumer=consumer))

if __name__ == '__main__':
    job_queue = multiprocessing.JoinableQueue()

    consumers = [JobConsumer(job_queue, _slow_print,
                             params={'consumer': str(i)}, id=str(i))
                 for i in range(5)]
    for c in consumers:
        c.start()

    jobs = [Job({'num': i}, id=str(i)) for i in range(15)]
    for j in jobs:
        job_queue.put(j)
    for i in range(len(consumers)):
        job_queue.put(None)

    job_queue.join()
    Log('finished')
