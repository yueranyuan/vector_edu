import boto.ec2
from time import sleep


def connect(region="us-east-1"):
    return boto.ec2.connect_to_region(region)


def get_available_instances(conn=None, ids=None):
    conn = conn or connect()

    return [i for i in conn.get_only_instances(instance_ids=ids)
            if i.state != 'terminated' and i.state != 'shutting-down']


def terminate_all(ids=None, conn=None, **kwargs):
    conn = conn or connect()
    instances = conn.get_only_instances(instance_ids=ids)
    to_stop = [i.id for i in instances if i.state != 'terminated']
    if to_stop:
        print 'terminating: {0}'.format(to_stop)
        conn.terminate_instances(instance_ids=to_stop)


def reserve(conn, instance_type, **kwargs):
    print 'launching instance'
    if instance_type == 'free':
        instance_type = 't2.micro'
    reserve = conn.run_instances('ami-9eaa1cf6', key_name='cmu-east-key1',
                                 instance_type=instance_type, security_groups=['Aaron-CMU-East'],
                                 instance_profile_arn='arn:aws:iam::999933667566:instance-profile/Worker')
    print 'launched instance ' + reserve.instances[0].id
    return reserve.instances[0].id


def add_instances(instance_cnt=1, conn=None, **kwargs):
    conn = conn or connect()
    return [reserve(conn, **kwargs) for c in range(instance_cnt)]


class NoInstanceException(BaseException):
    pass


def get_active_instance(conn=None, wait=180, interval=10, ids=None, **kwargs):
    conn = conn or connect()

    def check_for_active():
        print('looking for usable instance...')
        statuses = {s.id: s.instance_status.status
                    for s in conn.get_all_instance_status()}
        instances = get_available_instances(conn, ids=ids)
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
