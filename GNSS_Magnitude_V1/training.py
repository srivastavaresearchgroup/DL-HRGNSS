# -*- coding: utf-8 -*-
import numpy as np
import pickle
import os
import shutil
import logging
import tensorflow as tf
import tensorflow.keras as keras
from GNSS_Magnitude_V1.model import build_model6


# Tracking learning rate
def get_lr_metric(optimizer):
    def lr(y_true,y_esti):
        curLR = optimizer._decayed_lr(tf.float32) #is the current Learning Rate.
        return curLR
    return lr

# Call checkpoint - Early stopping ****************
def trainCallback(checkpoint_filepath):
    
    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=True)

    model_early_stopping_callback = keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        min_delta=0, 
        patience=20, 
        verbose=0,
        mode='min', 
        baseline=None, 
        restore_best_weights=False)
    
    return model_checkpoint_callback,model_early_stopping_callback


def trainer(nst,nt,nc,dirout,dir_case,x1,y1,x2,y2):
    tf.random.set_seed(2)
    
    # Paths ****************************************************************
    if os.path.exists(dirout+dir_case+'/'):
        logging.warning('The folder '+dirout+dir_case+' already exist! It will be removed and made again.\n')
        shutil.rmtree(dirout+dir_case+'/')
    os.mkdir(dirout+dir_case+'/')

    checkpoint_filepath = dirout+dir_case+'/cp.ckpt'
    hist_filepath = dirout+dir_case+'/history.p'
    model_path = dirout+dir_case+'/model.h5'
    
    # Parameters ****************************************************************
    batch_size = 128
    epochs = 200
    
    logging.info('Total epochs = %i',epochs)
    logging.info('Batch size = %i', batch_size)
 
    # Build Model ****************************************************************
    model=build_model6(nst,nt,nc)
    
    # Learning rate & Optimizer ****************************************************************
    logging.info("\n Using 'standard' learning rate decay...")
    initial_learning_rate = 1e-2
    decay = 1e-1 / epochs
    logging.info('initial learning rate: %5.4f', initial_learning_rate)
    logging.info('initial decay: %5.4f \n', decay)

    opt = keras.optimizers.Adam(learning_rate=initial_learning_rate,decay=decay)
    lr_metric = get_lr_metric(opt)

    # Compile ****************************************************************
    model.compile(loss='mse', optimizer=opt, metrics=['mae',lr_metric])
    
    #Checkpoint ****************************************************************
    model_checkpoint_callback,model_early_stopping_callback=trainCallback(checkpoint_filepath)
    
    # Train ****************************************************************
    hist = model.fit(x1, y1,validation_data=(x2, y2),epochs=epochs,
                       batch_size=batch_size,verbose=2,
                       callbacks=[model_checkpoint_callback, model_early_stopping_callback],
                       shuffle=True)
          
    # Evaluate ****************************************************************
    loss, mae, lr = model.evaluate(x2, y2, verbose=2)
    logging.info('\nValidation:')
    logging.info('loss: %5.5f', loss)
    logging.info('mae: %5.5f', mae)
    logging.info('lr: %5.5f', lr)
 
    # Save validation info in file    
    np.savetxt(dirout+dir_case+'/Validation_values.txt',(loss,mae,lr),
           fmt='%5.5f',header='loss, mae, lr',
           delimiter=' ')
    
    # Save Model ****************************************************************
    model.load_weights(checkpoint_filepath)
    model.save(model_path)

    hist_h = hist.history
    
    logging.info('History attributes:')
    logging.info(hist_h.keys())

    with open(hist_filepath, 'wb') as file_pi:
        pickle.dump(hist_h, file_pi)
        
    # Model summary    ****************************************************************
    with open(dirout+dir_case+'/report_model.txt','w') as fh:
        model.summary(print_fn=lambda x: fh.write(x + '\n'))
        
    dot_img_file = dirout+dir_case+'/model.pdf'
    keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)
    
    return model