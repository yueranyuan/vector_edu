import boto
import re

bucket_name = 'cmu-data'

conn = boto.connect_s3()
bucket = conn.get_bucket(bucket_name)
bucket_list = bucket.list()
keys = [k for k in bucket.list()
        if re.match(r'vectoredu/results/.+', str(k.key))]
for key in keys:
    content = key.get_contents_as_string()
    lines = content.split('\n')
    params = lines[0]
    results = lines[-2]
    acc, best_iter, test_acc, all_iter = eval(results)
    print key.key, acc
