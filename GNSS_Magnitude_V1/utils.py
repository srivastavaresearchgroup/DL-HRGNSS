# -*- coding: utf-8 -*-
import logging
import numpy as np
import os
from sklearn.model_selection import train_test_split

#
def make_outfolders(dir_out,dir_log,dir_info,dir_trmodel,dir_pred):
    if not os.path.exists(dir_out):
        os.mkdir(dir_out)

    if not os.path.exists(dir_log):
        os.mkdir(dir_log)
        
    if not os.path.exists(dir_info):
        os.mkdir(dir_info)
        
    if not os.path.exists(dir_trmodel):
        os.mkdir(dir_trmodel)
    
    if not os.path.exists(dir_pred):
        os.mkdir(dir_pred)


# Load and split Data *********************************************************
def make_data(dir_info,dir_case):
    dirdata = './data/'+dir_case+'/'
    dirout = dir_info+dir_case+'/'
    
    x=np.load(dirdata+'xdata.npy')
    y=np.load(dirdata+'ydata.npy')

    # Split (1): Train-Val & Test Data *********************************************************
    ntest=0.1 # 10%
    ix=np.arange(0, len(y), 1, dtype=int) #index
    x_train1, x_test, ix_train1, ix_test = train_test_split(x, ix,
                                                            test_size=ntest,
                                                            random_state=1)# random_state for reproducible output
    y_train1=y[ix_train1]
    y_test=y[ix_test]
    
    # Split (2): Train and Validation data
    n_val=0.2 # 2% of 90% (1/5 of data train)
    x_train, x_val, ix_train, ix_val = train_test_split(x_train1, ix_train1,
                                                        test_size=n_val,
                                                        random_state=1)# random_state for reproducible output
    y_train=y[ix_train]
    y_val=y[ix_val]
    
    # Save index in numpy files
    if not os.path.exists(dirout):
        os.mkdir(dirout)
    np.save(dirout+'index_datatrain.npy',ix_train)
    np.save(dirout+'index_dataval.npy',ix_val)
    np.save(dirout+'index_datatest.npy',ix_test)
    
    # Print shapes *********************************************************
    logging.info('Train & Validation data '+str(100-(100*ntest))+'%:')
    logging.info('x_train: '+str(x_train.shape))
    logging.info('y_train: '+str(y_train.shape))
    logging.info('x_val: '+str(x_val.shape))
    logging.info('y_val: '+str(y_val.shape))
    logging.info('Test_data'+str(100*ntest)+'%:')
    logging.info('x_test: '+str(x_test.shape))
    logging.info('y_test: '+str(y_test.shape)+'\n')

    return (x_train,y_train), (x_val,y_val), (x_test,y_test)


