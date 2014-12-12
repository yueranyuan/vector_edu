import boto.ec2
from fabric.api import run, hide, execute, env, cd
from fabric.network import disconnect_all
from time import sleep


def connect(region="us-east-1"):
    return boto.ec2.connect_to_region("us-east-1")


def stop_all(conn=None):
    conn = conn or connect()
    reserves = conn.get_all_reservations()
    for r in reserves:
        to_stop = [inst.id for inst in r.instances if inst.state == 'running']
        if to_stop:
            print 'stopping: ', to_stop
            conn.stop_instances(instance_ids=to_stop)


def terminate_all(conn=None):
    conn = conn or connect()
    reserves = conn.get_all_reservations()
    for r in reserves:
        to_stop = [inst.id for inst in r.instances if inst.state != 'terminated']
        if to_stop:
            print 'terminating: ', to_stop
            conn.terminate_instances(instance_ids=to_stop)


def reserve(conn):
    print 'launching instance'
    reserve = conn.run_instances('ami-9eaa1cf6', key_name='cmu-east-key1',
                                 instance_type='t2.medium', security_groups=['Aaron-CMU-East'],
                                 instance_profile_arn='arn:aws:iam::999933667566:instance-profile/Worker')
    print 'launched instance ' + reserve.instances[0].id
    return reserve


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


def get_active_instance(conn, wait_length=180, interval=10):
    def check_for_active():
        statuses = {s.id: s.instance_status.status
                    for s in conn.get_all_instance_status()}
        for instance in conn.get_only_instances():
            instance.update()
            if (instance.state == 'running' and
                    statuses.get(instance.id, '') == 'ok'):
                return instance
        return None

    inst = check_for_active()
    if inst:
        return inst
    for i in range(0, wait_length, interval):
        sleep(interval)
        inst = check_for_active()
        if inst:
            return inst
    raise Exception("no ready instances after waiting {0} seconds".format(wait_length))


def install_all():
    with hide('output'):
        run('sudo apt-get update')
        run('sudo apt-get install -y gcc g++ gfortran build-essential git wget libopenblas-dev python-dev python-pip python-nose python-numpy python-scipy awscli')
        run('sudo pip install theano')
        run('git clone https://github.com/yueranyuan/vector_edu.git')
        with cd('vector_edu'):
            run('aws s3 cp --recursive --region us-east-1 s3://cmu-data/vectoredu/data/ data/')


def run_experiment():
    with hide('output'):
        with cd('vector_edu'):
            run('python driver.py > run_log.txt')
            run('aws s3 cp --region us-east-1 *.log s3://cmu-data/vectoredu/results/')


def deploy(tasks, addr):
    host_list = ['ubuntu@{0}'.format(addr)]
    results = []
    for task in tasks:
        results.append(execute(task, hosts=host_list))
    return results


def start_over():
    conn = connect()
    terminate_all(conn)  # clear all old reservations
    reserve(conn)


def run_something(wait_time=180):
    addr = ssh_connect(get_active_instance(connect(), wait_time))
    env.key_filename = "cmu-east-key1.pem"
    deploy([install_all, run_experiment], addr)
    disconnect_all()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Deploy some workers to do some ML')
    parser.add_argument('mode', metavar="mode", type=str,
                        choices=['start', 'run', 'terminate'],
                        help='choices are start/run/terminate')
    parser.add_argument('--wait', dest='wait', default=180, type=int,
                        help='how long to wait for a connection')

    args = parser.parse_args()
    if args.mode == 'start':
        start_over()
    elif args.mode == 'run':
        run_something()
    elif args.mode == 'terminate':
        terminate_all()
