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
import pdb
import random
import math

def initial_cluster_centers_1(samples,k):
    c = {}
    for i in range(k):
        r = random.choice(samples)
        c[i] = numpy.array([])
        c[i] = numpy.append(c[i],r)
    return c

def initial_cluster_centers_2(samples,k):
    c = {}
    r = random.choice(samples)
    c[0] = numpy.array([])
    c[0] = numpy.append(c[0],r)
    other = []
    for i in range(1,k):
        temp = 0
        eu_dist = []
        eu_dist1 = []
        c[i] = numpy.array([])
        for j in range(len(samples)):
            eu_dist = []
            for m in range(0,i):
                temp = math.sqrt(((samples[j][0]-c[m][0])**2) + ((samples[j][1]-c[m][1])**2))
                eu_dist.append(temp)
            #val_avg = numpy.mean((val) for (val) in enumerate(eu_dist))
            val_avg = numpy.mean(eu_dist)
            eu_dist1.append(val_avg)
        val, index = max((val, index) for (index, val) in enumerate(eu_dist1))
        other.append(val)
        c[i] = numpy.append(c[i],samples[index])
    #plot initial points
    '''plot_inst.style.use('seaborn-whitegrid')
    for i in range(len(samples)):
        plot_inst.plot(samples[i][0], samples[i][1], 'o', color = 'blue')
    for i in range(k):
        plot_inst.plot(c[i][0], c[i][1], '*', color = 'red')
    plot_inst.show()
    plot_inst.clf()
    plot_inst.cla()
    plot_inst.close()
    #plot_inst.clf()'''
    return c

def main():
    #Extract data
    input_file = 'AllSamples.mat'
    data = scipy.io.loadmat(input_file)
    samples = (data['AllSamples'])

    #Strategy 1
    iteration = 0
    iterate = 1

    #initial cluster center
    obj_fun_new = {}
    for k in range(2,10):
        c = {}
        #Strategy 1: Random cluster centers
        c = initial_cluster_centers_1(samples,k)
        print ("OLD = ", c, "K = ", k)

        while (iterate):
            #Calculate Euclidean Distance and Objective function
            eu_dist = []
            obj_fun = 0
            out = []
            C = []
            for i in range(len(samples)):
                temp = 0
                eu_dist = []
                for j in range(k):
                    temp = math.sqrt(((samples[i][0]-c[j][0])**2) + ((samples[i][1]-c[j][1])**2))
                    eu_dist.append(temp)
                    #obj_fun = obj_fun + (temp**2)
                val, index = min((val, index) for (index, val) in enumerate(eu_dist))
                obj_fun = obj_fun + (val**2)
                out.append((samples[i][0],samples[i][1],index))
            #print(out)
            #print (obj_fun)

            #CLuster formation for first time
            cluster={}
            for k_i in range(k):
                cluster[k_i]=numpy.array([]).reshape(2,0)
            for i in range(len(samples)):
                for j in range(k):
                    if(out[i][2]==j):
                        cluster[j]=numpy.append(cluster[j],(samples[i][0],samples[i][1]))
                    cluster[j] = cluster[j].reshape(-1,2)
            print (cluster[0])
            new_mean = {}
            for i in range(k):
                new_mean[i] = numpy.array([])
                new_mean[i] = numpy.append(new_mean[i],numpy.mean(cluster[i], axis = 0))

            iteration = iteration + 1
            #stopping criteria
            for i in range(k):
                if(numpy.array_equal(c[i],new_mean[i]) is False):
                    c = new_mean
                    iterate = 1
                    break
                else:
                    iterate = 0
            #print ("NEW = ", new_mean)
        obj_fun_new[k-2] = numpy.array([])
        #obj_fun = []
        temp = 0
        for i in range(len(samples)):
            for j in range(k):
                temp += ((samples[i][0]-c[j][0])**2) + ((samples[i][1]-c[j][1])**2)
        #obj_fun.append(temp)
        obj_fun_new[k-2] = numpy.append(obj_fun_new[k-2],temp)
        print ("Iteration = ", iteration)
        print ("Objective function = ", obj_fun_new[k-2])
    obj_f = []
    for key in obj_fun_new:
        obj_f.extend(obj_fun_new[key].tolist())
    print (obj_f)
    print ("Neha : ", obj_fun_new)
    K_array=numpy.arange(2,10,1)
    plot_inst.plot(K_array, obj_f)
    plot_inst.xlabel('Number of Clusters')
    plot_inst.ylabel('Objective Function')
    plot_inst.title('Strategy 1')
    plot_inst.show()

    #Strategy 2:
    iteration = 0
    iterate = 1

    #plot initial points
    '''plot_inst.style.use('seaborn-whitegrid')
    for i in range(len(samples)):
        plot_inst.plot(samples[i][0], samples[i][1], 'o', color = 'blue')'''

    #initial cluster center :
    obj_fun_new = {}
    for k in range(2,10):
        c = {}

        #Strategy 2:  cluster centers with maximum distance
        c = initial_cluster_centers_2(samples,k)
        print ("OLD = ", c, "K = ", k)

        while (iterate):
            #Calculate Euclidean Distance and Objective function
            eu_dist = []
            obj_fun = 0
            out = []
            C = []
            for i in range(len(samples)):
                temp = 0
                eu_dist = []
                for j in range(k):
                    temp = math.sqrt(((samples[i][0]-c[j][0])**2) + ((samples[i][1]-c[j][1])**2))
                    eu_dist.append(temp)
                    #obj_fun = obj_fun + (temp**2)
                val, index = min((val, index) for (index, val) in enumerate(eu_dist))
                obj_fun = obj_fun + (val**2)
                out.append((samples[i][0],samples[i][1],index))
            #print(out)
            #print (obj_fun)

            #CLuster formation for first time
            cluster={}
            for k_i in range(k):
                cluster[k_i]=numpy.array([]).reshape(2,0)
            for i in range(len(samples)):
                for j in range(k):
                    if(out[i][2]==j):
                        cluster[j]=numpy.append(cluster[j],(samples[i][0],samples[i][1]))
                    cluster[j] = cluster[j].reshape(-1,2)
            #print (cluster)
            new_mean = {}
            for i in range(k):
                new_mean[i] = numpy.array([])
                new_mean[i] = numpy.append(new_mean[i],numpy.mean(cluster[i], axis = 0))

            iteration = iteration + 1
            #stopping criteria
            for i in range(k):
                if(numpy.array_equal(c[i],new_mean[i]) is False):
                    c = new_mean
                    iterate = 1
                    break
                else:
                    iterate = 0
            #print ("NEW = ", new_mean)
        obj_fun_new[k-2] = numpy.array([])
        #obj_fun = []
        temp = 0
        for i in range(len(samples)):
            for j in range(k):
                temp += ((samples[i][0]-c[j][0])**2) + ((samples[i][1]-c[j][1])**2)
        #obj_fun.append(temp)
        obj_fun_new[k-2] = numpy.append(obj_fun_new[k-2],temp)
        print ("Iteration = ", iteration)
        print ("Objective function = ", obj_fun_new[k-2])
    obj_f = []
    for key in obj_fun_new:
        obj_f.extend(obj_fun_new[key].tolist())

    K_array=numpy.arange(2,10,1)
    plot_inst.plot(K_array, obj_f)
    plot_inst.xlabel('Number of Clusters')
    plot_inst.ylabel('Objective Function')
    plot_inst.title('Strategy 2')
    plot_inst.show()

main()
print ("END")
