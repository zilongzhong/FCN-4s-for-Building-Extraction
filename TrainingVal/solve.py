import caffe
import surgery, score
import sys
import numpy as np
import os

import setproctitle
setproctitle.setproctitle(os.path.basename(os.getcwd()))

weights = '/path/to/FCN4s_building.caffemodel'

# init
#caffe.set_device(int(sys.argv[1]))
caffe.set_device(0)
caffe.set_mode_gpu()

solver = caffe.SGDSolver('/path/to/solver.prototxt')
solver.net.copy_from(weights)

# surgeries
interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
surgery.interp(solver.net, interp_layers)

# scoring
val = np.loadtxt('/home/finoa/Desktop/Mat_dataset/Train_Val/val.txt', dtype=str)

for _ in range(25):
    solver.step(2000)
    score.seg_tests(solver, False, val, layer='score')

#/home/finoa/Caffe Models/snapshot/FCN8s_building_1e12_iter_20000.caffemodel
