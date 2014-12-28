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
    if cache_dir:
        try:
            with open(os.path.join(cache_dir, key_name)) as f:
                content = f.read()
        except IOError:
            content = key.get_contents_as_string()
            with open(os.path.join(cache_dir, key_name), 'w') as f:
                f.write(content)
    else:
        content = key.get_contents_as_string()
    return content


def load_all(bucket_name, sub_folder, cache_dir=None):
    bucket = connect_to_bucket(bucket_name)
    bucket_list = bucket.list()
    keys = [k for k in bucket_list
            if re.match(r'{}/.+'.format(sub_folder), str(k.key))]
    return {get_key_name(key): load_key(key, cache_dir) for key in keys}


def load_one(bucket_name, key_name, cache_dir=None):
    bucket = connect_to_bucket(bucket_name)
    key = bucket.get_key(key_name)
    return load_key(key, cache_dir)
