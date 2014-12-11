import boto.ec2
from fabric.api import run, hide, execute, env, cd
from fabric.network import disconnect_all
from time import sleep


def connect(region="us-east-1"):
    return boto.ec2.connect_to_region("us-east-1")


def stop_all(conn):
    reserves = conn.get_all_reservations()
    for r in reserves:
        to_stop = [inst.id for inst in r.instances if inst.state == 'running']
        if to_stop:
            print 'stopping: ', to_stop
            conn.stop_instances(instance_ids=to_stop)


def terminate_all(conn):
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


def ssh_connect(instance, wait=30):
    if not wait_till_running(instance, wait):
        raise Exception("instance {0} not running after {1} seconds".format(instance.id, wait))
    print 'connecting to instance {0}'.format(instance.id)
    return instance.ip_address


def get_active_instance(conn):
    for res in conn.get_all_reservations():
        if res.instances[0].state in ("pending", "running"):
            return res.instances[0]
    raise Exception("no more running instances")


def install_all():
    with hide('output'):
        run('sudo apt-get update')
        run('sudo apt-get install -y gcc g++ gfortran build-essential git wget libopenblas-dev python-dev python-pip python-nose python-numpy python-scipy awscli')
        run('sudo pip install theano')
        run('git clone https://github.com/yueranyuan/vector_edu.git')
        with cd('vector_edu'):
            run('aws s3 cp --recursive --region us-east-1 s3://cmu-data/vectoredu/data/ data/')


def deploy(task, addr):
    host_list = ['ubuntu@{0}'.format(addr)]
    results = execute(task, hosts=host_list)
    return results


def start_over():
    conn = connect()
    terminate_all(conn)  # clear all old reservations
    reserve(conn)


def run_something():
    addr = ssh_connect(get_active_instance(connect()))
    env.key_filename = "cmu-east-key1.pem"
    deploy(install_all, addr)
    disconnect_all()

#start_over()
run_something()
#terminate_all()
