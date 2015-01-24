from time import sleep

from fabric.api import run as fabrun
from fabric.api import hide, execute, env, cd, local

from learntools.libs.multijob import Job, JobConsumer, do_jobs
from learntools.libs.logger import gen_log_name
from learntools.libs.ec2 import (terminate_all, add_instances,
                                 get_active_instance, get_available_instances)
from learntools.libs.ec2 import connect as ec2_connect
from learntools.deploy.config import all_param_set_keys


env.key_filename = "cmu-east-key1.pem"
env.connection_attempts = 5


ONLINE = False
_CONNECTION = None


def run(*args, **kwargs):
    '''run commandline either locally or online depending on the ONLINE global variable'''
    if ONLINE:
        return fabrun(*args, **kwargs)
    return local(*args, **kwargs)


def connect():
    '''connects aws ec2 services and cache the connection inside the module'''
    global _CONNECTION
    if _CONNECTION is None:
        _CONNECTION = ec2_connect()
    return _CONNECTION


def install_all():
    '''fabric function to install necessary dependences, download the git repo and data'''
    with hide('output'):
        run('sudo apt-get update')
        run('sudo apt-get install -y gcc g++ gfortran build-essential git wget libopenblas-dev python-dev python-pip python-nose python-numpy python-scipy awscli')
        run('sudo pip install theano')
        run('git clone https://github.com/yueranyuan/vector_edu.git')
        with cd('vector_edu'):
            run('aws s3 cp --recursive --region us-east-1 s3://cmu-data/vectoredu/data/ data/')


def deploy(addr, func, **kwargs):
    '''run a given function on an aws ec2 instance

    Args:
        addr (string): the ip address of the aws ec2 instance
        func (function): the fabric function to execute on the ec2
        **kwargs (dict): arguments to func
    '''
    host_list = ['ubuntu@{0}'.format(addr)]
    execute(func, hosts=host_list, **kwargs)


class TempWorkers():
    '''a context to initialize a group of ec2 workers. On exiting the context, the ec2
    workers are automatically terminated
    '''
    def __init__(self, n, **kwargs):
        self.n = n
        self.args = kwargs

    def __enter__(self):
        self.ids = add_instances(instanec_cnt=self.n, conn=connect(), **self.args)
        # wait to launch. if the server doesn't know about the instance yet
        # it might cause boto to crash
        sleep(5)
        return self.ids

    def __exit__(self, type, value, traceback):
        terminate_all(ids=self.ids)
        pass


class AWSConsumer(JobConsumer):
    '''JobConsumer which can be run on AWS. It handles installing dependencies, downloading
        the git repo, downloading the data, deploying a fabric job to aws, and (optionally) shutting
        down the aws instance'''
    def __init__(self, job_queue, addr=None, no_install=False, kill_after_running=True,
                 wait=180, **kwargs):
        JobConsumer.__init__(self, job_queue, **kwargs)
        if addr is None:
            self.addr = get_active_instance(ids=[self.id], conn=connect(), wait=wait).ip_address
        else:
            self.addr = addr
        self.inner_func = self.func
        self.func = self._wrapper
        self.no_install = no_install
        self.kill_after_running = kill_after_running

    @classmethod
    def make_factory(cls, **kwargs):
        '''generate a consumer factory with the same signature as the JobConsumer constructor.
        This allows us to specify various AWSConsumer specific parameters and still use this
        factory anywhere the JobConsumer constructor is used'''
        def _factory(job_queue=None, id=None, func=None):
            return cls(job_queue=job_queue, id=id, func=func, **kwargs)
        return _factory

    def _wrapper(self, **kwargs):
        deploy(self.addr, func=self.inner_func, **kwargs)

    def run(self):
        if not self.no_install:
            deploy(self.addr, func=install_all)
        JobConsumer.run(self)

    def shutdown(self):
        if self.kill_after_running:
            terminate_all(ids=[self.id])
        JobConsumer.shutdown(self)


def run_experiment_aws(*args, **kwargs):
    '''wrapper for using the run_experiment function on aws. Uploads results to s3.'''
    with cd('vector_edu'):
        log_name = run_experiment(*args, **kwargs)
        if log_name:
            run('aws s3 cp --region us-east-1 {log_name} s3://cmu-data/vectoredu/results/'.format(
                **locals()))


def run_experiment(driver, param_set, task_num=0, **kwargs):
    '''use the commandline to run an experiment

    Args:
        driver (string): name of the driver file. File location relative to the repo root.
        param_set (string): name of the parameter set to run.
            See learntools.deploy.config.get_config()
        task_num (int): this number maintains a counter to distinguish the various runs
            in a batch operation. The first run is 0, the second is 1, etc. The task_num can
            be used to distinguish cross-validation folds etc.

    Returns:
        (string): the name of the log file produced by this experiment. File location relative
            to the repo root.
    '''
    log_name = gen_log_name()
    run('python {driver} -p {param_set} -o {log_name} -tn {task_num}'.format(
        **locals()))
    return log_name


def run_batch(param_set, n_workers, driver, existing_instances=False, no_install=False,
              wait=180, num_jobs=1, **kwargs):
    jobs = [Job({'driver': driver, 'param_set': param_set, 'task_num': i}, id=str(i))
            for i in range(num_jobs)]
    if ONLINE:
        kill_after = not existing_instances
        aws_consumer_factory = AWSConsumer.make_factory(no_install=no_install,
                                                        kill_after_running=kill_after,
                                                        wait=wait)
        if existing_instances:
            ids = [inst.id for inst in get_available_instances()]
            do_jobs(ids, jobs=jobs, func=run_experiment_aws,
                    consumer_factory=aws_consumer_factory)
        else:
            with TempWorkers(n=n_workers) as ids:
                do_jobs(ids, jobs=jobs, func=run_experiment_aws,
                        consumer_factory=aws_consumer_factory)
    else:
        ids = ['worker{}'.format(i) for i in xrange(n_workers)]
        do_jobs(ids, jobs=jobs, func=run_experiment,
                consumer_factory=JobConsumer)

if __name__ == "__main__":
    handlers = {
        'start': add_instances,
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
    parser.add_argument('-i', dest='n_workers', metavar="i", default=1,
                        type=int, help='how many workers/instances to use')
    parser.add_argument('-it', dest='instance_type', default='c3.2xlarge',
                        help='what type of ec2 instance to use')
    parser.add_argument('-ni', '--no_install', dest='no_install', action='store_true',
                        help='when running, do not do install phase')
    parser.add_argument('-x', dest='num_jobs', type=int, default=1,
                        help='how many experiments to run')
    parser.add_argument('-aws', dest='aws', action='store_true',
                        help='run on aws vs local computer')
    parser.add_argument('-d', '--driver', dest='driver', type=str, default='kt_driver.py',
                        help='the file location of the driver relative to the repo root')
    parser.add_argument('-ei', '--existing_instances', dest='existing_instances',
                        action='store_true',
                        help='set to use existing instances')

    cmd_args = vars(parser.parse_args())
    # if we're running an experiment, we need the parameter set
    if cmd_args['mode'] in ('batch', 'run') and not cmd_args['param_set']:
        raise Exception('batch mode requires a parameter set')
    if cmd_args['num_jobs'] < cmd_args['n_workers']:
        raise Exception("don't launch more instances/workers than jobs to run")
    if cmd_args['aws']:
        ONLINE = True
    handlers[cmd_args['mode']](**cmd_args)
