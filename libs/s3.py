import boto
import re
import os


def connect_to_bucket(bucket_name, conn=None):
    conn = conn or boto.connect_s3()
    return conn.get_bucket(bucket_name)


def get_key_name(key):
    return os.path.split(key.key)[1]


def load_key(key, cache_dir=None):
    key_name = get_key_name(key)
    try:
        with open(os.path.join(cache_dir, key_name)) as f:
            content = f.read()
    except IOError:
        content = key.get_contents_as_string()
        if cache_dir:
            with open(os.path.join(cache_dir, key_name), 'w') as f:
                f.write(content)
    return content


def load_all(bucket_name, sub_folder, cache_dir=None):
    bucket = connect_to_bucket(bucket_name)
    bucket_list = bucket.list()
    keys = [k for k in bucket_list
            if re.match(r'{}/.+'.format(sub_folder), str(k.key))]
    for key in keys:
        content = load_key(key, cache_dir)
        lines = content.split('\n')
        results = lines[-2]
        acc, best_epoch, all_iter, run_time = eval(results)
        print get_key_name(key), acc, best_epoch, all_iter, run_time


def load_one(bucket_name, key_name, cache_dir=None):
    bucket = connect_to_bucket(bucket_name)
    key = bucket.get_key(key_name)
    return load_key(key, cache_dir)


if __name__ == '__main__':
    load_all('cmu-data', 'vectoredu/results', cache_dir='results')
