import os
import sys

if sys.platform.startswith('win'):
    os.system('del *.log')
else:
    os.system('rm *.log')
