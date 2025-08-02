from src.ResultVisualizer1 import ResultVisualizer, MultipleCompare
import matplotlib.pyplot as plt
import numpy as np
import yaml
import os

#visualizer = ResultVisualizer("./results/default.yaml", "../DProxSGT/results/logs_FashionMNIST_Size4_LeNet5_epochs_10_batchsize8_lr00.01_data_dividelabell10.0001_methodDProxSGT.tsv")
#visualizer.visualize()
#f1 = "./results/FashionMNIST-Compressed-Topk-DistributedAdam.yaml"
#f2 = "./results/FashionMNIST-Compressed-Topk-AdaGrad.yaml"
#f3 = "./results/FashionMNIST-Compressed-Topk-AMSGrad.yaml"
#f4 = "./results/FashionMNIST-Compressed-Topk-DProxAGT.yaml"
#f5 = "./results/FashionMNIST-DProxAGT.yaml"
#f6="./results/logs_FashionMNIST_Size4_LeNet5_epochs_100_batchsize8_lr00.01_DataDivide_label_methodCDProxSGT_QxTop30_QyTop30_gamma_x0.5_gamma_y0.5.tsv"


#change this path to results you want to compute
DIR_PATH = './results/comp2/none_label/'

file_list = (os.listdir(DIR_PATH))


f_list = []
for f in file_list:
	if f != '.DS_Store' and ('.yaml' in f or '.tsv' in f):
	    f_list.append(DIR_PATH+f)


file_type = []
for file in f_list:
	file_type.append(file.split('.')[-1])


#list of plots we want
#Options currently supported - 'train_acc','test_acc','train_loss','test_loss','cons_error'


plot_list = ['test_acc','train_loss','cons_error']

#this is one way to do it 
for opt in plot_list:
	visualizer2 = MultipleCompare(f_list,file_type,"results_fmnist_comp_label_none"+'_'opt,"FMNIST Comp Label ",epochs=100,plot_list=[opt],figsize=(40,8))
	visualizer2.visualize_results()


#Initialize and run visualizer
visualizer2 = MultipleCompare(f_list,file_type,"results_fmnist_comp_label_none","FMNIST Comp Label ",epochs=100,plot_list=plot_list,figsize=(40,8))
visualizer2.visualize_results()
