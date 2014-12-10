import boto.ec2
from fabric.api import run, execute, env
from fabric.network import disconnect_all

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
    reserve = conn.run_instances('ami-b66ed3de', key_name='cmu-east-key1',
                                 instance_type='t2.micro', security_groups=['Aaron-CMU-East'])
    print 'launched instance ' + reserve.instances[0].id
    return reserve


def ssh_connect(instance):
    instance.update()
    if instance.state != 'running':
        raise Exception("instance {0} not running yet".format(instance.id))
    print 'connecting to instance {0}'.format(instance.id)
    return instance.ip_address

#stop_all(connect())
#terminate_all(connect())
#conn = connect()
#res = reserve(conn)
conn = connect()
res = conn.get_all_reservations()[0]
addr = ssh_connect(res.instances[0])
env.key_filename = "cmu-east-key1.pem"


def do_work(cmd):
    return run(cmd)


def deploy(cmd, addr):
    host_list = ['ec2-user@{0}'.format(addr)]
    results = execute(do_work, cmd, hosts=host_list)
    print results

deploy('pwd', addr)
disconnect_all()
