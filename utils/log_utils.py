import sys
import os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class Logger(object):
    def __init__(self,logfile):
        self.terminal = sys.stdout
        self.log = open(logfile, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass


def plot(name,loss_total,loss_class,loss_regress,tpr,tnr):
    fig,(ax1,ax2)=plt.subplots(2,1)
    fig.set_size_inches(18,10)

    ax1.plot(list(range(1,len(loss_total)+1)),loss_total,color='r')
    ax1.plot(list(range(1,len(loss_class)+1)),loss_class,color='b')
    #ax1.plot(list(range(1,len(loss_regress[:,3])+1)),r_c,color='g')

    ax2.plot(list(range(1,len(tpr)+1)),tpr,color='r')
    ax2.plot(list(range(1,len(tnr)+1)),tnr,color='b')

    ax1.set_title('loss_curve, r=loss_total, b=loss_class')
    ax2.set_title('metric_curve, r=tpr, b=tnr')
    plt.savefig(name)
