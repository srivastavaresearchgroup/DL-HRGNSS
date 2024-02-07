# -*- coding: utf-8 -*-
import numpy as np
import os
import logging
import tensorflow as tf
import tensorflow.keras as keras
import statistics as statis

# Prediction & Errors **************************************************************
def predict(x_test,y_test,model,dirout,dir_case):
    pred = model.predict(x_test)
    y_pred = pred.reshape(len(y_test),)
    yr_pred = np.array([round(valy,1) for valy in y_pred])#round to 1 decimal

    # Statistics value prediction
    pred_error = np.array([round(valerr,1) for valerr in (yr_pred - y_test)])#round to 1 decimal
    abs_pred_error = np.array([round(valea,1) for valea in np.abs(yr_pred - y_test)])#round to 1 decimal
    mean_error = np.mean(abs_pred_error)
    min_error = np.min(abs_pred_error)
    max_error = np.max(abs_pred_error)
    std_error = np.std(pred_error)
    mode_error = statis.mode(abs_pred_error)
    rms_error = np.sqrt(((pred_error)**2).mean())
    
    logging.info('\nPrediction report:')
    logging.info('Mean Absolute Error: %10.3f', round(mean_error,3))
    logging.info('Min Absolute Error: %10.3f', min_error)
    logging.info('Max Absolute Error: %10.3f', max_error)
    logging.info('Std Dev: %10.3f', round(std_error,3))
    logging.info('Mode Absolute Error: %10.3f', mode_error)
    logging.info('RMSE: %10.3f', round(rms_error,3))

    # Save files with prediction results       
    if os.path.exists(dirout+dir_case):
        logging.warning('The folder '+dirout+dir_case+' already exist! It will be removed and made again.\n')
        shutil.rmtree(dirout+dir_case)
    os.mkdir(dirout+dir_case)
    
    f=open(dirout+dir_case+'/Results_Magnitude.dat','w')
    f.write('Magnitude, Predicted Mag\n')
    for ix in range(len(yr_pred)):
        f.write(str(y_test[ix])+' '+ str(yr_pred[ix])+'\n')
    f.close()

    np.savetxt(dirout+dir_case+'/Predict_Eval.txt',(mean_error,min_error,max_error,std_error,
                                                           mode_error,rms_error),
               fmt='%10.5f',header='mean_error, min_error, max_error, std_error, mode_error, rms_error',
               delimiter=' ')