import os
import sys

if sys.platform.startswith('win'):
    # yup, windows does not have 'rm -r'
    os.system("del C:\Users\yuera_000\AppData\Local\Theano\compiledir_Windows-8-6.2.9200-Intel64_Family_6_Model_60_Stepping_3_GenuineIntel-2.7.8-64\lock_dir\lock")
    os.system("rmdir C:\Users\yuera_000\AppData\Local\Theano\compiledir_Windows-8-6.2.9200-Intel64_Family_6_Model_60_Stepping_3_GenuineIntel-2.7.8-64\lock_dir")
else:
    raise NotImplemented()
