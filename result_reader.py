import boto
import re


def load_all():

    bucket_name = 'cmu-data'

    conn = boto.connect_s3()
    bucket = conn.get_bucket(bucket_name)
    keys = [k for k in bucket.list()
            if re.match(r'vectoredu/results/.+', str(k.key))]
    for key in keys:
        content = key.get_contents_as_string()
        lines = content.split('\n')
        # params = lines[0]
        results = lines[-2]
        acc, best_iter, all_iter, run_time = eval(results)
        print key.key, acc, all_iter, run_time


def load_one(key_name):
    bucket_name = 'cmu-data'

    conn = boto.connect_s3()
    bucket = conn.get_bucket(bucket_name)
    key = bucket.get_key(key_name)
    print key.get_contents_as_string()


load_one('vectoredu/results/2014_12_16_23_09_25_8606.log')
