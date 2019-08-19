import os
import sys
import argparse
import random
import numpy as np
import sklearn.preprocessing
import sklearn.svm
import sklearn.model_selection
from sklearn.cross_decomposition import CCA
from keras.callbacks import LearningRateScheduler
from keras.wrappers.scikit_learn import KerasClassifier

import tocca

def train_test_model( X_train, y_train, X_val, y_val, X_test, y_test, model_type, layers, layer_size, l2dist_weight, l2_weight, momentum, learning_rate, batch_size, epochs, sd_weight=0, zca_r=1e-4 ):

    classes = np.unique(y_train)
    np.sort(classes)
    nclasses = len(classes)

    # format labels
    out_train = sklearn.preprocessing.label_binarize( y_train, classes )
    out_train = [ out_train, out_train, np.ones((X_train[0].shape[0],1)) ]
    out_val = sklearn.preprocessing.label_binarize( y_val, classes )
    out_val = [ out_val, out_val, np.ones((X_val[0].shape[0],1)) ]
    out_test = sklearn.preprocessing.label_binarize( y_test, classes )
    out_test = [ out_test, out_test, np.ones((X_test[0].shape[0],1)) ]

    # train model
    input_dims = [ len(Xt[0]) for Xt in X_train ]
    model = tocca.create_model( model_type, nclasses, input_dims, layers, layer_size, shared_size, learning_rate, l2dist_weight, momentum, l2_weight, sd_weight, zca_r )
    model.fit( X_train, out_train, batch_size=batch_size, epochs=epochs, verbose=2, validation_data=(X_val,out_val), shuffle=True )

    p_train = model.predict( X_train )
    shared_train = [ p_train[2][:,:shared_size], p_train[2][:,shared_size:] ]

    # cross-modal classification
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        svm = sklearn.svm.LinearSVC()
        params = { 'C':[float(2**e) for e in range(-10,5)] }
        grid_search = sklearn.model_selection.GridSearchCV( estimator=svm, param_grid=params, scoring='accuracy', cv=5, refit=True, verbose=1, n_jobs=4 )
        grid_search.fit( shared_train[0], y_train )
    svm = grid_search.best_estimator_

    # training accuracy
    p_train = svm.predict( shared_train[1] )
    acc_train = ( y_train == p_train ).mean()

    # validation accuracy
    p_val = model.predict( X_val )
    shared_val = [ p_val[2][:,:shared_size], p_val[2][:,shared_size:] ]
    p_val = svm.predict( shared_val[1] )
    acc_val = ( y_val == p_val ).mean()

    # test accuracy
    p_test = model.predict( X_test )
    shared_test = [ p_test[2][:,:shared_size], p_test[2][:,shared_size:] ]
    p_test = svm.predict( shared_test[1] )
    acc_test = ( y_test == p_test ).mean()

    # sum correlation for train, val, test
    correlation = lambda X: np.array( [ np.dot( X[0][:,c].T, X[1][:,c] ) / np.sqrt( np.dot( X[0][:,c].T, X[0][:,c] ) * np.dot( X[1][:,c].T, X[1][:,c] ) ) for c in range(X[0].shape[1]) ] ).sum()
    print('corr',correlation(shared_train),correlation(shared_val),correlation(shared_test))

    ortho = lambda X: [ np.dot( X[:,c1].T, X[:,c2] ) / np.sqrt( np.dot( X[:,c1].T, X[:,c1] ) * np.dot( X[:,c2].T, X[:,c2] ) ) for c1 in range(X.shape[1]) for c2 in range(X.shape[1]) if c1 != c2 ]
    print(ortho(shared_train[0]))
    print(ortho(shared_train[1]))
    #print(ortho(shared_val[0]))
    #print(ortho(shared_val[1]))
    #print(ortho(shared_test[0]))
    #print(ortho(shared_test[1]))

    return acc_train,acc_val,acc_test

if __name__ == "__main__":

    parser = argparse.ArgumentParser( description='Compute CNN features.' )
    parser.add_argument('--dataset', '-d', help='data set name', default='MNISTsplit' )
    parser.add_argument('--training_size', help='training set size' )
    parser.add_argument('--model_type', '-m', required=True, help='model type (w, sd, nd)' )
    parser.add_argument('--param_search', '-p', help='random search for best parameters' )
    parser.add_argument('--cv', help='cross-validation folds' )
    parser.add_argument('--layers', '-l', help='number of layers' )
    parser.add_argument('--layer_size', help='layer size' )
    parser.add_argument('--shared_size', '-s', help='number of shared features', default=50 )
    parser.add_argument('--l2dist', help='l2 distance weight' )
    parser.add_argument('--momentum', help='momentum' )
    parser.add_argument('--l2', help='l2 weight decay' )
    parser.add_argument('--sd', help='soft decorrelation weight' )
    parser.add_argument('--zca', help='zca regularization' )
    parser.add_argument('--learning_rate', '-r', help='learning rate' )
    parser.add_argument('--batch_size', '-b', help='batch size' )
    parser.add_argument('--epochs', '-e', help='epochs', default=100 )
    parser.add_argument('--semi', help='semi supervised: % missing labels' )
    parser.add_argument('--out_file', '-o', help='output file' )
    args = parser.parse_args()
    dataset = args.dataset
    training_size = args.training_size
    model_type = args.model_type
    param_search = args.param_search
    cv = args.cv
    layers = args.layers
    layer_size = args.layer_size
    shared_size = int(args.shared_size)
    l2dist_weight = args.l2dist
    momentum = args.momentum
    l2_weight = args.l2
    sd_weight = args.sd
    zca_r = args.zca
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    epochs = int(args.epochs)
    semi = args.semi
    out_file = args.out_file
    
    if dataset == 'MNISTsplit':
        # load data
        from keras.datasets import mnist
        (X_train,y_train),(X_test,y_test) = mnist.load_data()
        X_train = X_train.astype('float')
        X_test = X_test.astype('float')
    
        data_train = [ X_train[i,:,:].squeeze() for i in range(X_train.shape[0]) ]
        data_test = [ X_test[i,:,:].squeeze() for i in range(X_test.shape[0]) ]

        # create left and right modalities by splitting image in half
        X_train = [ [ dt[:,:14].flatten() for dt in data_train ], [ dt[:,14:].flatten() for dt in data_train ] ]
        X_test = [ [ dt[:,:14].flatten() for dt in data_test ], [ dt[:,14:].flatten() for dt in data_test ] ]
        X_train = [ np.array(X) for X in X_train ]
        X_test = [ np.array(X) for X in X_test ]
    else:
        print('Unsupported data set: '+dataset)
        sys.exit(1)

    if cv is None:
        cv = 5
    else:
        cv = int(cv)
    if training_size is None:
        training_size = (len(y_train)//cv)*(cv-1)
    else:
        training_size = int(training_size)
    
    if param_search is not None:
        param_search = int(param_search)
        layers = [1,2,3,4] if layers is None else [int(layers)]
        layer_size = [200] if layer_size is None else [int(layer_size)]
        l2dist_weight = [1e-3,1e-2,1e-1,1e0,1e1,1e2,1e3] if l2dist_weight is None else [float(l2dist_weight)]
        l2_weight = [1e-5,1e-4,1e-3,1e-2,1e-1,0] if l2_weight is None else [float(l2_weight)]
        momentum = [0.99,0.95,0.9] if momentum is None else [float(momentum)]
        learning_rate = [1e-2,1e-3,1e-4] if learning_rate is None else [float(learning_rate)]
        batch_size = [1000,100] if batch_size is None else [int(batch_size)]
        params = { 'layers':layers, 'layer_size':layer_size, 'l2dist_weight':l2dist_weight, 'l2_weight':l2_weight, 'momentum':momentum, 'learning_rate':learning_rate, 'batch_size':batch_size }
        if model_type == 'sd':
            params.update( { 'sd_weight':[1e-5,1e-4,1e-3,1e-2,1e-1] if sd_weight is None else [float(sd_weight)] } )
        elif model_type == 'w':
            params.update( { 'zca_r':[1e-4,1e-3,1e-2] if zca_r is None else [float(zca_r)] } )
    else:
        param_search = 1
        layers = 2 if layers is None else [int(layers)]
        layer_size = [200] if layer_size is None else [int(layer_size)]
        l2dist_weight = [1.0] if l2dist_weight is None else [float(l2dist_weight)]
        l2_weight = [1e-4] if l2_weight is None else [float(l2_weight)]
        momentum = [0.99] if momentum is None else [float(momentum)]
        learning_rate = [1e-3] if learning_rate is None else [float(learning_rate)]
        batch_size = [1000] if batch_size is None else [int(batch_size)]
        params = { 'layers':layers, 'layer_size':layer_size, 'l2dist_weight':l2dist_weight, 'l2_weight':l2_weight, 'momentum':momentum, 'learning_rate':learning_rate, 'batch_size':batch_size }
        if model_type == 'sd':
            params.update( { 'sd_weight':[0] if sd_weight is None else [float(sd_weight)] } )
        elif model_type == 'w':
            params.update( { 'zca_r':[1e-4] if zca_r is None else [float(zca_r)] } )

    if semi is not None:
        # TODO: semi-supervised: randomly retain only X% of labels on training set
        semi = float(semi)
        pass
    
    param_sampler = sklearn.model_selection.ParameterSampler( params, param_search )

    X_train_all = X_train
    y_train_all = y_train
    X_test_all = X_test

    for p in param_sampler:
        print(p)

        acc_train = []
        acc_val = []
        acc_test = []
        for f in range(cv):
            print('Fold %d of %d'%(f+1,cv))

            # create train/val split
            idx = np.arange(len(y_train_all))
            random.shuffle(idx)
            X_train = [ X[idx[:training_size],:] for X in X_train_all ]
            y_train = y_train_all[idx[:training_size]]
            X_val = [ X[idx[training_size:],:] for X in X_train_all ]
            y_val = y_train_all[idx[training_size:]]
            X_test = X_test_all[:]

            # normalize
            mean = [ X.mean(axis=0) for X in X_train ]
            std = [ X.std(axis=0)+1e-6 for X in X_train ]
            X_train = [ (X-m)/s for X,m,s in zip(X_train,mean,std) ]
            X_val = [ (X-m)/s for X,m,s in zip(X_val,mean,std) ]
            X_test = [ (X-m)/s for X,m,s in zip(X_test,mean,std) ]
        
            a_train,a_val,a_test = train_test_model( X_train, y_train, X_val, y_val, X_test, y_test, model_type=model_type, epochs=epochs, **p )

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

