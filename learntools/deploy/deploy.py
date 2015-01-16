from time import sleep
import multiprocessing

from fabric.api import run, hide, execute, env, cd

from learntools.libs.multijob import Job, JobConsumer
from learntools.libs.logger import gen_log_name
from learntools.libs.ec2 import (connect, terminate_all, add_instances,
                                 get_active_instance)
from learntools.deploy.config import all_param_set_keys


env.key_filename = "cmu-east-key1.pem"
env.connection_attempts = 5


def install_all():
    with hide('output'):
        run('sudo apt-get update')
        run('sudo apt-get install -y gcc g++ gfortran build-essential git wget libopenblas-dev python-dev python-pip python-nose python-numpy python-scipy awscli')
        run('sudo pip install theano')
        run('git clone https://github.com/yueranyuan/vector_edu.git')
        with cd('vector_edu'):
            run('aws s3 cp --recursive --region us-east-1 s3://cmu-data/vectoredu/data/ data/')


def run_experiment(param_set, task_num=0):
    with cd('vector_edu'):
        log_name = gen_log_name()
        run('python kt_driver.py -p {param_set} -o {log_name} -tn {task_num}'.format(
            param_set=param_set, log_name=log_name, task_num=task_num))
        run('aws s3 cp --region us-east-1 {log_name} s3://cmu-data/vectoredu/results/'.format(
            log_name=log_name))


def deploy(addr, func=run_experiment, **kwargs):
    host_list = ['ubuntu@{0}'.format(addr)]
    return execute(func, hosts=host_list, **kwargs)


class TempWorkers():
    def __init__(self, **kwargs):
        self.args = kwargs

    def __enter__(self):
        self.ids = add_instances(**self.args)
        # wait to launch. if the server doesn't know about the instance yet
        # it might cause boto to crash
        sleep(5)
        return self.ids

    def __exit__(self, type, value, traceback):
        terminate_all(ids=self.ids)
        pass


class AWSConsumer(JobConsumer):
    def __init__(self, job_queue, addr=None, no_install=False, kill_after_running=True, **kwargs):
        JobConsumer.__init__(self, job_queue, **kwargs)
        self.addr = addr
        self.inner_func = self.func
        self.func = self.wrapper
        self.no_install = no_install
        self.kill_after_running = kill_after_running

    def wrapper(self, **kwargs):
        self.inner_func(addr=self.addr, **kwargs)

    def run(self):
        if not self.no_install:
            deploy(self.addr, func=install_all)
        JobConsumer.run(self)

    def shutdown(self):
        if self.kill_after_running:
            terminate_all(ids=[self.id])
        JobConsumer.shutdown(self)


def run_on_one(**kwargs):
    conn = connect()
    ids = [i.id for i in conn.get_only_instances() if i.state != 'terminated']
    if not ids:
        raise Exception("no instances were started")
    use_worker(conn=conn, ids=ids[:1], **kwargs)


def use_worker(param_set, conn, ids, no_install=False, num_jobs=1, **kwargs):
    job_queue = multiprocessing.JoinableQueue()

    # setup a AWSConsumer for every worker
    consumers = [AWSConsumer(job_queue,
                             func=deploy,
                             addr=get_active_instance(ids=[id], conn=conn, **kwargs).ip_address,
                             no_install=no_install,
                             id=id)
                 for id in ids]
    for c in consumers:
        c.start()

    # setup a bunch of jobs
    jobs = [Job({'param_set': param_set, 'task_num': i}, id=str(i)) for i in range(num_jobs)]
    for j in jobs:
        job_queue.put(j)
    for i in range(len(consumers)):
        job_queue.put(None)

    job_queue.join()
    print('finished')


def run_batch(param_set, **kwargs):
    conn = connect()
    with TempWorkers(conn=conn, **kwargs) as ids:
        use_worker(param_set, conn, ids, **kwargs)


if __name__ == "__main__":
    handlers = {
        'start': add_instances,
        'run': run_on_one,
        'batch': run_batch,
        'terminate': terminate_all
    }

    import argparse
    parser = argparse.ArgumentParser(description='Deploy some workers to do some ML')
    parser.add_argument('mode', metavar="mode", type=str, default='exp',
                        choices=handlers.keys(),
                        help='choices are {modes}'.format(modes='/'.join(handlers.keys())))
    parser.add_argument('-p', dest='param_set', type=str, default='default',
                        choices=all_param_set_keys,
                        help='the name of the parameter set to use')
    parser.add_argument('--wait', dest='wait', default=600, type=int,
                        help='how long to wait for a connection')
    parser.add_argument('-i', dest='instance_cnt', metavar="i", default=1,
                        type=int, help='how many instances')
    parser.add_argument('-it', dest='instance_type', default='c3.2xlarge',
                        help='what type of ec2 instance to use')
    parser.add_argument('--ni', dest='no_install', action='store_true',
                        help='when running, do not do install phase')
    parser.add_argument('-x', dest='num_jobs', type=int, default=1,
                        help='how many experiments to run')

    cmd_args = vars(parser.parse_args())
    # if we're running an experiment, we need the parameter set
    if cmd_args['mode'] in ('batch', 'run') and not cmd_args['param_set']:
        raise Exception('batch mode requires a parameter set')
    if cmd_args['num_jobs'] < cmd_args['instance_cnt']:
        raise Exception("don't launch more instances than jobs to run")
    handlers[cmd_args['mode']](**cmd_args)
