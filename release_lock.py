import os
import sys


def release():
    if sys.platform.startswith('win'):
        # yup, windows does not have 'rm -r'
        try:
            os.system("del C:\Users\yuera_000\AppData\Local\Theano\compiledir_Windows-8-6.2.9200-Intel64_Family_6_Model_60_Stepping_3_GenuineIntel-2.7.8-64\lock_dir\lock")
        except:
            pass
        try:
            os.system("rmdir C:\Users\yuera_000\AppData\Local\Theano\compiledir_Windows-8-6.2.9200-Intel64_Family_6_Model_60_Stepping_3_GenuineIntel-2.7.8-64\lock_dir")
        except:
            pass
    else:
        # for some reason this is only a problem on windows
        pass


if __name__ == "__main__":
    release()