import boto.ec2
import time
import sys
from boto.manage.cmdshell import sshclient_from_instance as sshclient

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

#stop_all(connect())
terminate_all(connect())

conn = boto.ec2.connect_to_region("us-east-1")

print 'launching instance'
reserve = conn.run_instances('ami-b66ed3de', key_name='cmu-east-key1',
                    instance_type='t2.micro', security_groups=['Aaron-CMU-East'])
print 'launched instance'

def run_stuff(inst):
    ssh_client = sshclient(inst,
                            ssh_key_file='cmu-east-key1.pem',
                            user_name='root')
    status, stdout, stderr = ssh_client.run('pwd')
    print status, stdout, stderr
    conn.stop_instances(instance_ids=[inst.id])

for i in range(20):
    inst = reserve.instances[0]
    inst.update()
    print inst.state
    if inst.state == "running":
        run_stuff(inst)
        break
    time.sleep(15)

#conn.stop_instances(reserve.instances)
