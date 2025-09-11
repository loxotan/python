import os
import sys

os.environ['OMP_NUM_THREADS'] = '1'

if getattr(sys, 'frozen', False):
    torch_lib_path = os.path.join(sys._MEIPASS, 'torch', 'lib')
    if os.path.exists(torch_lib_path):
        os.environ['PATH'] = torch_lib_path + os.pathsep + os.environ['PATH']
