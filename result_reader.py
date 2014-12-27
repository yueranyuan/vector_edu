import boto
import re
import os


def load_all():
    bucket_name = 'cmu-data'

    conn = boto.connect_s3()
    bucket = conn.get_bucket(bucket_name)
    bucket_list = bucket.list()
    keys = [k for k in bucket_list
            if re.match(r'vectoredu/results/.+', str(k.key))]
    CACHE_DIR = 'results'
    cached_keys = os.listdir(CACHE_DIR)
    for key in keys:
        key_name = os.path.split(key.key)[1]
        if key_name in cached_keys:
            with open(os.path.join(CACHE_DIR, key_name)) as f:
                content = f.read()
        else:
            content = key.get_contents_as_string()
            with open(os.path.join(CACHE_DIR, key_name), 'w') as f:
                f.write(content)
        lines = content.split('\n')
        # params = lines[0]
        results = lines[-2]
        acc, best_iter, all_iter, run_time = eval(results)
        print key_name, acc, all_iter, run_time


def load_one(key_name):
    bucket_name = 'cmu-data'

    conn = boto.connect_s3()
    bucket = conn.get_bucket(bucket_name)
    key = bucket.get_key(key_name)
    print key.get_contents_as_string()


load_all()
