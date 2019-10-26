#
# PROGRAM NAME: PROJECT 2 - IMPLEMENTATION OF K-MEANS ALGORITHM FOR A GIVEN
#               2D DATASET
# AUTHOR NAME: NEHA RAJENDRA VADNERE
# EMAIL: nvadnere@asu.edu
# DATE: 25-OCT-2019
#

import scipy.io
import matplotlib.pyplot as plot_inst
import numpy
import random

def main():
    '''print "--------------------------------------------------------------------"
    print "PROGRAM NAME: PROJECT 1- DENSITY EXTIMATION AND CLASSIFICATION \n" \
        "AUTHOR NAME: NEHA RAJENDRA VADNERE\n" \
        "EMAIL: nvadnere@asu.edu\n" \
        "DATE: 14-SEPT-2019"
    print "--------------------------------------------------------------------"'''
    #Read the input .matlab file
    input_file = 'AllSamples.mat'
    data = scipy.io.loadmat(input_file)
    samples = (data['AllSamples'])
    plot_inst.style.use('seaborn-whitegrid')
    for i in range(len(samples)):
        plot_inst.plot(samples[i][0], samples[i][1], 'o', color = 'blue')
    k = 10 #number of cluster
    c = [] #cluster center
    for i in range(k):
        r = random.choice(samples)
        c.append(r)
    print (c)

main()
print("END")
