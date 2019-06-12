from Ldata_helper import *
from Lglobal_defs import *
from sklearn.svm import SVC
from sklearn.metrics import *
import Llog as log
from sklearn.decomposition import PCA
from sklearn.preprocessing import *
from sklearn.neural_network import MLPClassifier
from LCNN2 import CNN


def test(datasetA, clf, use_pca=True, do_norm2=True, rate=0.99):
    data_tr, data_te = read_dataset_A(datasetA)
    if do_norm2:
        ss = Normalizer().fit(data_tr[0])
        data_tr[0] = ss.transform(data_tr[0])
        data_te[0] = ss.transform(data_te[0])
    if use_pca:
        #pca = PCA(0.99).fit(data_tr[0])
        pca = PCA(rate).fit(data_tr[0])
        data_tr[0] = pca.transform(data_tr[0])
        data_te[0] = pca.transform(data_te[0])
        print(data_tr[0].shape)
        exit(1)
    clf.fit(data_tr[0],data_tr[1])
    pred_tr = clf.predict(data_tr[0])
    pred_te = clf.predict(data_te[0])
    acc_tr = accuracy_score(data_tr[1], pred_tr)
    acc_te = accuracy_score(data_te[1], pred_te)
    precision_te = precision_score(data_te[1], pred_te, average='weighted')
    recall_te = recall_score(data_te[1], pred_te, average='weighted')
    f1_te = f1_score(data_te[1], pred_te, average='weighted')
    svs = len(clf.support_vectors_) if hasattr(clf, 'support_vectors_') else -1
    print('\n\n\n\nacc_tr %.5f\t acc_te %.5f\t precision_te %.5f\t recall_te %.5f\t f1_te %.5f\t svs %d'
         % (acc_tr, acc_te, precision_te, recall_te, f1_te, svs) + '\n' + 
        str(data_tr[0].shape) + '\t' + str(data_te[0].shape) + '\t' + str(datasetA) + '\tuse_pca ' + str(use_pca) + '   do_norm2 ' + str(do_norm2))
    print(str(clf))
    return acc_te


def svc_accord(C, kernel):
    if kernel == 'linear':
        yield SVC(C=C,kernel=kernel)
    elif kernel == 'poly':
        for gamma in ['scale', 'auto']:
            for degree in [3,2,4]:
                # for coef0 in [0.0,0.5, 2.0]:
                for coef0 in [0.0,0.5]:
                    yield  SVC(C=C,kernel=kernel, gamma=gamma, coef0=coef0, degree=degree)
    elif kernel == 'sigmoid':
        for gamma in ['scale', 'auto']:
            #for coef0 in [0.0, 0.5, 2.0]:
            for coef0 in [0.0, 0.5]:
                yield  SVC(C=C,kernel=kernel, gamma=gamma, coef0=coef0)
    elif kernel == 'rbf':
        for gamma in ['scale', 'auto']:
            yield  SVC(C=C,kernel=kernel, gamma=gamma)


def SVM_test():
    datasets = [
        DatasetsA.a1a, DatasetsA.a7a, DatasetsA.a8a, 
               DatasetsA.pendigits, DatasetsA.usps
               ]
    datasets = [DatasetsA.mnist, DatasetsA.cifar10]
    datasets = [DatasetsA.cifar10]
    for ds in datasets:
        acc_te_best = 0.0
        for kernel in ['rbf', 'linear', 'poly', 'sigmoid']:
            bad_kernel = False
            #for C in [1.0, 0.1, 0.5, 3.0, 10.0]:
            for C in [1.0, 0.5, 3.0]:
                # for svc in svc_accord(C, kernel):
                for svc in [SVC(C=C, kernel=kernel, gamma='scale')]:
                    for prep in [True]:
                        for rate in [0.99,0.95,0.90] if ds == DatasetsA.mnist else [0.85]:
                            acc_te = test(ds, svc, use_pca=prep, do_norm2=prep, rate=rate)
                            print("rate " + str(rate))
                            if acc_te + 0.03 < acc_te_best:
                                bad_kernel = True
                                break
                            acc_te_best = max(acc_te_best, acc_te)
                        if bad_kernel: break
                    if bad_kernel: break
                if bad_kernel: break


def MLP_test():
    for ds in [
        DatasetsA.a1a, DatasetsA.a7a, DatasetsA.a8a, 
               DatasetsA.pendigits, DatasetsA.usps
               ]:
        acc_te_best = 0.0
        for activation in ['relu', 'logistic', 'tanh', 'identity']:
            bad_act = False
            for alpha in [0.0001,0.001,0.00001]:
                for hidden_layer_sizes in [(100,),(128,96)]:
                    for prep in [False, True]:
                        mlp = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes,activation=activation,alpha=alpha,batch_size=128,
                                            learning_rate_init=0.0004)
                        acc_te = test(ds, mlp, use_pca=prep, do_norm2=prep)
                        if acc_te + 0.05 < acc_te_best:
                            bad_act = True
                            break
                        if acc_te + 0.02 < acc_te_best: break
                        acc_te_best = max(acc_te_best, acc_te)
                    if acc_te + 0.02 < acc_te_best: break
                if bad_act: break
                        

def CNN_test():
    import tensorflow as tf
    for ds in [
            DatasetsA.pendigits, DatasetsA.usps
            ]:
        acc_te_best = 0.0
        for act in [tf.nn.relu, tf.sigmoid, tf.tanh, None]:
            for kp in [0.5,0.9]:
                for cnns in [
                    [],
                    [[16,5],2,[32,5],2],
                    [[16,5],2],
                    [[16,3],2,[32,3],2],
                    ]:
                    for fcs in [[512], [128], [256,96]]:
                        cnn = CNN(fcs=fcs,act=act,kp=kp,cnns=cnns,batch_size=128,lr_init=0.01, epoch=25)
                        acc_te = test(ds, cnn, use_pca=False, do_norm2=False)
                        if acc_te + 0.03 < acc_te_best: break
                        acc_te_best = max(acc_te_best, acc_te)
                    if acc_te + 0.06 < acc_te_best: break
                if acc_te + 0.08 < acc_te_best: break


if __name__ == '__main__':
    #logger = log.logger(__name__, 'SVM_B_')
    #print = logger.info
    SVM_test()
    # MLP_test()
    # CNN_test()
    #test(DatasetsA.pendigits, SVC())

