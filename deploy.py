import boto.ec2
from fabric.api import run, hide, execute, env, cd
from fabric.network import disconnect_all
from time import sleep
import multiprocessing
from multijob import Job, JobConsumer
from config import all_param_set_keys

env.key_filename = "cmu-east-key1.pem"


def connect(region="us-east-1"):
    return boto.ec2.connect_to_region(region)


def terminate_all(ids=None, conn=None, **kwargs):
    conn = conn or connect()
    instances = conn.get_only_instances(instance_ids=ids)
    to_stop = [i.id for i in instances if i.state != 'terminated']
    if to_stop:
        print 'terminating: {0}'.format(to_stop)
        conn.terminate_instances(instance_ids=to_stop)


def reserve(conn, free=False, **kwargs):
    print 'launching instance'
    size = 't2.micro' if free else 't2.medium'
    reserve = conn.run_instances('ami-9eaa1cf6', key_name='cmu-east-key1',
                                 instance_type=size, security_groups=['Aaron-CMU-East'],
                                 instance_profile_arn='arn:aws:iam::999933667566:instance-profile/Worker')
    print 'launched instance ' + reserve.instances[0].id
    return reserve.instances[0].id


def wait_till_running(instance, wait_length, interval=3):
    print instance.state
    if instance.state == 'running':
        return True
    interval = 3
    for i in range(0, wait_length, interval):
        sleep(interval)
        instance.update()
        if instance.state == 'running':
            return True
    return False


def ssh_connect(instance):
    print 'connecting to instance {0}'.format(instance.id)
    return instance.ip_address


class NoInstanceException(BaseException):
    pass


def get_active_instance(conn=None, wait=180, interval=10, ids=None, **kwargs):
    conn = conn or connect()

    def check_for_active():
        print('looking for usable instance...')
        statuses = {s.id: s.instance_status.status
                    for s in conn.get_all_instance_status()}
        instances = [i for i in conn.get_only_instances(instance_ids=ids)
                     if i.state != 'terminated']
        if not instances:
            raise NoInstanceException("no instances have been started")
        for instance in instances:
            instance.update()
            if (instance.state == 'running' and
                    statuses.get(instance.id, '') == 'ok'):
                return instance
        print 'none found'
        return None

    inst = check_for_active()
    if inst:
        return inst
    for i in range(0, wait, interval):
        sleep(interval)
        inst = check_for_active()
        if inst:
            return inst
    raise NoInstanceException("no ready instances after waiting {0} seconds".format(wait))


def install_all():
    with hide('output'):
        run('sudo apt-get update')
        run('sudo apt-get install -y gcc g++ gfortran build-essential git wget libopenblas-dev python-dev python-pip python-nose python-numpy python-scipy awscli')
        run('sudo pip install theano')
        run('git clone https://github.com/yueranyuan/vector_edu.git')
        with cd('vector_edu'):
            run('aws s3 cp --recursive --region us-east-1 s3://cmu-data/vectoredu/data/ data/')


def run_experiment(param_set):
    with hide('output'):
        with cd('vector_edu'):
            run('python driver.py {param_set}'.format(param_set=param_set))
            run('aws s3 cp --region us-east-1 *.log s3://cmu-data/vectoredu/results/')


def deploy(addr, **kwargs):
    host_list = ['ubuntu@{0}'.format(addr)]
    print host_list, kwargs
    return execute(test_command, hosts=host_list, **kwargs)


def start_over(**kwargs):
    conn = connect()
    terminate_all(conn)  # clear all old reservations
    reserve(conn, **kwargs)


def add_instances(instance_cnt=1, conn=None, **kwargs):
    conn = conn or connect()
    return [reserve(conn, **kwargs) for c in range(instance_cnt)]


def use_worker(commands=[install_all, run_experiment], **kwargs):
    addr = ssh_connect(get_active_instance(**kwargs))
    deploy(commands, addr)
    disconnect_all()


def test_command(txt):
    run('echo {txt}'.format(txt=txt))


def full_command(param_set):
    install_all()
    run_experiment(param_set)


class TempWorkers():
    def __init__(self, **kwargs):
        self.args = kwargs

    def __enter__(self):
        self.ids = add_instances(**self.args)
        return self.ids

    def __exit__(self, type, value, traceback):
        terminate_all(ids=self.ids)
        pass


def run_batch(param_set, **kwargs):
    conn = connect()
    with TempWorkers(conn=conn, **kwargs) as ids:
        job_queue = multiprocessing.JoinableQueue()

        # setup a jobConsumer for every worker
        def gen_work(addr):
            def _work(*args):
                deploy([test_command], addr)
            return _work
        consumers = [JobConsumer(job_queue,
                                 deploy,
                                 params={'addr': get_active_instance(ids=[id], conn=conn, **kwargs).ip_address},
                                 id=id)
                     for id in ids]
        for c in consumers:
            c.start()

        # setup a bunch of jobs
        jobs = [Job({'param_set': param_set}, id=str(i)) for i in range(15)]
        for j in jobs:
            job_queue.put(j)
        for i in range(len(consumers)):
            job_queue.put(None)

        job_queue.join()
        print('finished')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Deploy some workers to do some ML')
    parser.add_argument('mode', metavar="mode", type=str, default='exp',
                        choices=['restart', 'start', 'run', 'batch', 'terminate'],
                        help='choices are start/restart/run/terminate')
    parser.add_argument('param_set', type=str, default=None,
                        choices=all_param_set_keys,
                        help='the name of the parameter set to use')
    parser.add_argument('--wait', dest='wait', default=600, type=int,
                        help='how long to wait for a connection')
    parser.add_argument('-i', dest='instance_cnt', metavar="i", default=1,
                        type=int, help='how many instances')
    parser.add_argument('--free', dest='free', action='store_true',
                        help='[true] to start free micro instance')

    cmd_args = vars(parser.parse_args())
    if cmd_args['mode'] == 'batch' and not cmd_args['param_set']:
        raise Exception('batch mode requires a parameter set')
    handlers = {
        'restart': start_over,
        'start': add_instances,
        'run': use_worker,
        'batch': run_batch,
        'terminate': terminate_all
    }
    handlers[cmd_args['mode']](**cmd_args)
