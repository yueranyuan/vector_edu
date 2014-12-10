from fabric.api import run, execute, task, env

env.key_filename = 'cmu-east-key1.pem'


def do_work():
    return run("pwd")


@task
def deploy():
    host_list = ['ec2-user@54.173.222.18']
    results = execute(do_work, hosts=host_list)
    print results
