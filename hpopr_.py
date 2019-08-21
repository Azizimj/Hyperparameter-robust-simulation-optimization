import All_models as am
from hyperopt import hp
from hyperopt import fmin, tpe, space_eval

X = None
y = None
num_classes = None
nn_hold_ = None

# images_dir = "F:/Acad/research/fafar/RSO/nd_code/alderley/images"
images_dir = "F:/Acad/research/fafar/RSO/nd_code/alderley/images[100,200]"
# images_dir = "images"
tr_dir = images_dir + "_" + str(am.test_precs) + "/tr/"
num_epoch = 2

def objective(hyps):
    hidden_dim = hyps['hidden_dim']
    learningrate_ = hyps['learningrate_']
    lrdecay_ = hyps['lrdecay_']
    weightdecay_ = hyps['weightdecay_']

    net = nn_hold_.nn_run(hidden_dim, num_epoch, learningrate_, lrdecay_, weightdecay_, num_classes, X, y)

    eval = am.eval(tr_dir, "hpopt", am.test_precs, am.model_name,
                   X, y, net, am.svm, am.f, tr_=True)

    return eval[0] # acc


# define a search space
space = hp.choice('a',
    [
        ('case 1', 1 + hp.lognormal('c1', 0, 1)),
        ('case 2', hp.uniform('c2', -10, 10))
    ])

# minimize the objective over the space
best = fmin(objective, space, algo=tpe.suggest, max_evals=100)

print(best)
# -> {'a': 1, 'c2': 0.01420615366247227}
print(space_eval(space, best))
# -> ('case 2', 0.01420615366247227}


if __name__=='__main___':

    images_dir = "F:/Acad/research/fafar/RSO/nd_code/alderley/images"
    images_dir = "F:/Acad/research/fafar/RSO/nd_code/alderley/images[100,200]"
    # images_dir = "images"

    tr_dir = images_dir + "_" + str(am.test_precs) + "/tr/"
    X, y, num_classes = am.read_files(tr_dir, am.model_name)

    nn_hold_ = am.nn_hold(am.bias_, am.hiddenclass_, am.outclass_, am.momentum_, am.batchlearning_)


    nn_hold_ = am.nn_hold()



    f = open("res/result_"+str(test_precs) +"_"+model_name+"_epo"+str(num_epoch)+ ".txt", "a")
    f.write(model_name + "\n")

    f_all = open("res/result_" + str(test_precs) + "_" + model_name + "_epo" + str(num_epoch) + ".csv", 'a')
    writer_f_all = csv.writer(f_all)




    X_train, y_train = None, None
    #hpopt done

    tes_dir = images_dir + "_" + str(test_precs) + "/" + "tes/"
    X_test, y_test, num_classes = am.read_files(tes_dir, model_name)



