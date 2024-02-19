# -*- coding: utf-8 -*-
import logging
import warnings
import numpy as np
from GNSS_Magnitude_V1.utils import make_outfolders
from GNSS_Magnitude_V1.utils import make_data
from GNSS_Magnitude_V1.training import trainer
from GNSS_Magnitude_V1.prediction import predict


# dataset shape --> Set dimension!!!
nt = 181 # Time windows [seconds] #Set 181 or 501
nst = 3 # Nunmber of stations #Set 3 or 7
nc = 3 # Channels. Don't change it.
case_nm = 'GNSS_M'+str(nst)+'S_'+str(nt)  #'GNSS_M3S_181/' #'GNSS_M7S_181/' #'GNSS_M7S_501/'

# Paths # Set the folder and file names!
dir_out = './tests' 
dir_datinf = dir_out+'/data_info_1/'
dir_trmodel = dir_out+'/models_1/'
dir_pred = dir_ou+'/predictions_1/'
dir_log = dir_out+'/out_log/'
filelog = case_nm+'.log'

# ****************************************************************************
def main():
    
    # Create the output folders
    make_outfolders(dir_out,dir_log,dir_datinf,dir_trmodel,dir_pred)
        
    logging.basicConfig(
        filename=dir_log+filelog,
        level=logging.DEBUG,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%H:%M:%S'
    )

    # Make datasets
    (x_train,y_train), (x_val,y_val), (x_test,y_test) = make_data(dir_datinf,case_nm)
    
    # Training Model
    model = trainer(nst,nt,nc,dir_trmodel,case_nm,x_train,y_train,x_val,y_val)
    
    # Predictions
    predict(x_test,y_test,model,dir_pred,case_nm)
    
    logging.info('Finished *********************************************************************')
    
if __name__ == '__main__':

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", np.ComplexWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)
        
        main()
