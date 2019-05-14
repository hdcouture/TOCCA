import os
import sys
import argparse
import random
import numpy as np
import sklearn.preprocessing
import sklearn.svm
import sklearn.model_selection
#import rcca
from sklearn.cross_decomposition import CCA
from keras.callbacks import LearningRateScheduler
from keras.wrappers.scikit_learn import KerasClassifier

import tocca

def train_test_model( X_train, y_train, X_val, y_val, X_test, y_test, model_type, layers, layer_size, l2dist_weight, l2_weight, momentum, learning_rate, batch_size, epochs, sd_weight=0, zca_r=1e-4 ):

    classes = np.unique(y_train)
    np.sort(classes)
    nclasses = len(classes)
    out_train = sklearn.preprocessing.label_binarize( y_train, classes )
    out_train = [ out_train, out_train, np.ones((X_train[0].shape[0],1)) ]
    out_val = sklearn.preprocessing.label_binarize( y_val, classes )
    out_val = [ out_val, out_val, np.ones((X_val[0].shape[0],1)) ]
    out_test = sklearn.preprocessing.label_binarize( y_test, classes )
    out_test = [ out_test, out_test, np.ones((X_test[0].shape[0],1)) ]
    
    input_dims = [ len(Xt[0]) for Xt in X_train ]
    model = tocca.create_model( model_type, nclasses, input_dims, layers, layer_size, shared_size, learning_rate, l2dist_weight, momentum, l2_weight, sd_weight, zca_r )

    #lr = lambda epoch: learning_rate * 0.5**( epoch//20 )
    #lrate = LearningRateScheduler(lr)
    #callbacks = [lrate]
    callbacks = []

    #print([X.shape for X in X_train])
    model.fit( X_train, out_train, batch_size=batch_size, epochs=epochs, verbose=2, validation_data=(X_val,out_val), shuffle=True, callbacks=callbacks )
    #self.model.fit( X_train, out_train, batch_size=self.batch_size, epochs=self.epochs, verbose=2, shuffle=True, callbacks=callbacks )
    p_train = model.predict( X_train )
    shared_train = [ p_train[2][:,:shared_size], p_train[2][:,shared_size:] ]

    # cross-modal classification
    svm = sklearn.svm.LinearSVC()
    params = { 'C':[float(2**e) for e in range(-10,5)] }
    grid_search = sklearn.model_selection.GridSearchCV( estimator=svm, param_grid=params, scoring='accuracy', cv=5, refit=True, verbose=1, n_jobs=4 )
    grid_search.fit( shared_train[0], y_train )
    svm = grid_search.best_estimator_
    #self.svm.fit( shared_train[0], y_train )

    p_train = svm.predict( shared_train[1] )
    acc_train = ( y_train == p_train ).mean()

    p_val = model.predict( X_val )
    shared_val = [ p_val[2][:,:shared_size], p_val[2][:,shared_size:] ]
    p_val = svm.predict( shared_val[1] )
    acc_val = ( y_val == p_val ).mean()

    if np.isinf(X_test[0]).flatten().sum() > 0 or np.isinf(X_test[1]).flatten().sum() > 0:
        print('************inf')
        return 0,0,0
    print(np.isnan(X_test[0]).flatten().sum())
    print(np.isnan(X_test[1]).flatten().sum())
    p_test = model.predict( X_test )
    shared_test = [ p_test[2][:,:shared_size], p_test[2][:,shared_size:] ]
    print(shared_test[1])
    print(np.isnan(shared_test[1]).flatten().sum())
    p_test = svm.predict( shared_test[1] )
    acc_test = ( y_test == p_test ).mean()

    return acc_train,acc_val,acc_test

if __name__ == "__main__":

    parser = argparse.ArgumentParser( description='Compute CNN features.' )
    parser.add_argument('--dataset', '-d', help='data set name', default='MNISTsplit' )
    parser.add_argument('--training_size', help='training set size' )
    parser.add_argument('--model_type', '-m', required=True, help='model type (w, sd, nd)' )
    parser.add_argument('--param_search', '-p', help='random search for best parameters' )
    parser.add_argument('--cv', help='cross-validation folds' )
    parser.add_argument('--layers', '-l', help='number of layers', default=2 )
    parser.add_argument('--layer_size', help='layer size', default=200 )
    parser.add_argument('--shared_size', '-s', help='number of shared features', default=50 )
    parser.add_argument('--l2dist', help='l2 distance weight', default=1.0 )
    parser.add_argument('--momentum', help='momentum', default=0.99 )
    parser.add_argument('--l2', help='l2 weight', default=1e-4 )
    parser.add_argument('--sd', help='soft decorrelation weight', default=0 )
    parser.add_argument('--zca', help='zca regularization', default=1e-4 )
    parser.add_argument('--learning_rate', '-r', help='learning rate', default=1e-3 )
    parser.add_argument('--batch_size', '-b', help='batch size', default=128 )
    parser.add_argument('--epochs', '-e', help='epochs', default=100 )
    parser.add_argument('--semi', help='semi supervised: % missing labels' )
    parser.add_argument('--out_file', '-o', help='output file' )
    args = parser.parse_args()
    dataset = args.dataset
    training_size = args.training_size
    model_type = args.model_type
    param_search = args.param_search
    cv = args.cv
    layers = int(args.layers)
    layer_size = int(args.layer_size)
    shared_size = int(args.shared_size)
    l2dist_weight = float(args.l2dist)
    momentum = float(args.momentum)
    l2_weight = float(args.l2)
    sd_weight = float(args.sd)
    zca_r = float(args.zca)
    learning_rate = float(args.learning_rate)
    batch_size = int(args.batch_size)
    epochs = int(args.epochs)
    semi = args.semi
    out_file = args.out_file

    if dataset == 'MNISTsplit':
        from keras.datasets import mnist
        (X_train,y_train),(X_test,y_test) = mnist.load_data()
        X_train = X_train.astype('float')
        X_test = X_test.astype('float')
    
        #random.seed(1000)
        #randidx = np.arange(X_train.shape[0])
        #random.shuffle(randidx)

        # split train into train+val
        #data_train = [ X_train[i,:,:].squeeze() for i in range(50000) ]
        #data_val = [ X_train[i,:,:].squeeze() for i in range(50000,X_train.shape[0]) ]
        data_train = [ X_train[i,:,:].squeeze() for i in range(X_train.shape[0]) ]
        data_test = [ X_test[i,:,:].squeeze() for i in range(X_test.shape[0]) ]

        # create left and right modalities by split image in half
        X_train = [ [ dt[:,:14].flatten() for dt in data_train ], [ dt[:,14:].flatten() for dt in data_train ] ]
        #X_val = [ [ dt[:,:14].flatten() for dt in data_val ], [ dt[:,14:].flatten() for dt in data_val ] ]
        X_test = [ [ dt[:,:14].flatten() for dt in data_test ], [ dt[:,14:].flatten() for dt in data_test ] ]
        X_train = [ np.array(X) for X in X_train ]
        #X_val = [ np.array(X) for X in X_val ]
        X_test = [ np.array(X) for X in X_test ]
        #y_train,y_val = y_train[:50000],y_train[50000:]

        print(len(X_train[0]),X_train[0][0].shape,y_train.shape)
    else:
        print('Unsupported data set: '+dataset)
        sys.exit(1)

    if semi is not None:
        # semi-supervised: randomly remove X% of labels on training set
        semi = float(semi)
        pass
    
    if param_search is not None:
        param_search = int(param_search)
        if training_size is None and cv is None:
            training_size = len(y_train)//5
        elif training_size is None:
            training_size = len(train)//int(cv)
        else:
            training_size = int(training_size)
            if cv is None:
                cv = 5
            else:
                cv = int(cv)

        params = { 'layers':[1,2,3,4], 'layer_size':[200], 'l2dist_weight':[1e-3,1e-2,1e-1,1e0,1e1,1e2,1e3], 'l2_weight':[1e-5,1e-4,1e-3,1e-2,1e-1,0], 'momentum':[0.99,0.95,0.9], 'learning_rate':[1e-2,1e-3,1e-4], 'batch_size':[100,1000] }
        if model_type == 'sd':
            params.update( { 'sd_weight':[1e-5,1e-4,1e-3,1e-2,1e-1] } )
        elif model_type == 'w':
            params.update( { 'zca_r':[1e-4,1e-3,1e-2] } )
    else:
        param_search = 1
        training_size = len(y_train)
        params = { 'layers':[layers], 'layer_size':[layer_size], 'l2dist_weight':[l2dist_weight], 'l2_weight':[l2_weight], 'momentum':[momentum], 'learning_rate':[learning_rate], 'batch_size':[batch_size] }
        if model_type == 'sd':
            params.update( { 'sd_weight':[sd_weight] } )
        elif model_type == 'w': 
            params.update( { 'zca_r':[zca_r] } )
    
    param_sampler = sklearn.model_selection.ParameterSampler( params, param_search )

    X_train_all = X_train
    y_train_all = y_train

    for p in param_sampler:
        print(p)

        acc_train = []
        acc_val = []
        acc_test = []
        for f in range(cv):
            print('Fold %d of %d'%(f+1,cv))
            
            idx = np.arange(len(y_train_all))
            random.shuffle(idx)
            X_train = [ X[idx[:training_size],:] for X in X_train_all ]
            y_train = y_train_all[idx[:training_size]]
            X_val = [ X[idx[training_size:],:] for X in X_train_all ]
            y_val = y_train_all[idx[training_size:]]

            # normalize
            mean = [ X.mean(axis=0) for X in X_train ]
            std = [ X.std(axis=0)+1e-6 for X in X_train ]
            X_train = [ (X-m)/s for X,m,s in zip(X_train,mean,std) ]
            X_val = [ (X-m)/s for X,m,s in zip(X_val,mean,std) ]
            X_test = [ (X-m)/s for X,m,s in zip(X_test,mean,std) ]
        
            a_train,a_val,a_test = train_test_model( X_train, y_train, X_val, y_val, X_test, y_test, model_type=model_type, epochs=epochs, **p )#, layers, layer_size, l2dist_weight, l2_weight, sd_weight, zca_r, momentum, learning_rate, batch_size, epochs )
            #else:
            #    a_train,a_val,a_test = train_test_model( X_train, y_train, X_val, y_val, X_test, y_test, model_type, layers, layer_size, l2dist_weight, l2_weight, momentum, learning_rate, batch_size, epochs, sd_weight, zca_r )

            acc_train.append( a_train )
            acc_val.append( a_val )
            acc_test.append( a_test )

            if cv > 1:
                print('train %f val %f test %f'%(a_train,a_val,a_test))

        acc_train = np.array(acc_train)
        acc_val = np.array(acc_val)
        acc_test = np.array(acc_test)
        if cv > 1:
            print('train %f (%f) val %f (%f) test %f (%f)'%(acc_train.mean(),acc_train.std(),acc_val.mean(),acc_val.std(),acc_test.mean(),acc_test.std()))
        else:
            print('train %f val %f test %f'%(acc_train,acc_val,acc_test))
        if out_file is not None:
            fd = open( out_file, 'a' )
            if cv > 1:
                print('train %f (%f) val %f (%f) test %f (%f)'%(acc_train.mean(),acc_train.std(),acc_val.mean(),acc_val.std(),acc_test.mean(),acc_test.std()),file=fd)
            else:
                print('train %f val %f test %f'%(acc_train,acc_val,acc_test),file=fd)
            fd.close()

    if False:#True:
        # apply linear CCA
        #cca = rcca.CCA( kernelcca=False, numCC=ncomp0, reg=reg )
        #cca.train( [ X_train[0], X_train[1] ] )
        #W1,W2 = cca.ws
        cca = CCA( shared_size )
        cca.fit( shared_train[0], shared_train[1] )
        W1 = cca.x_weights_
        W2 = cca.y_weights_
        S_train = [ np.dot( shared_train[0], W1 ), np.dot( shared_train[1], W2 ) ]
        S_val = [ np.dot( shared_val[0], W1 ), np.dot( shared_val[1], W2 ) ]
        S_test = [ np.dot( shared_test[0], W1 ), np.dot( shared_test[1], W2 ) ]
    
        correlation = lambda X: np.array( [ np.dot( X[0][:,c].T, X[1][:,c] ) / np.sqrt( np.dot( X[0][:,c].T, X[0][:,c] ) * np.dot( X[1][:,c].T, X[1][:,c] ) ) for c in range(X[0].shape[1]) ] ).sum()
        corr_train = correlation(S_train)
        corr_val = correlation(S_val)
        corr_test = correlation(S_test)

        print('corr %f %f %f'%(corr_train,corr_val,corr_test))
