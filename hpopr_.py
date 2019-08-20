import All_models as am

# images_dir = "F:/Acad/research/fafar/RSO/nd_code/alderley/images"
# images_dir = "F:/Acad/research/fafar/RSO/nd_code/alderley/images[100,200]"
images_dir = "images"

def


if __name__=='__main___':
    tes_dir = images_dir + "_" + str(test_precs) + "/"+"tes/"
    X_test, y_test, num_classes = read_files(tes_dir, model_name)

    f = open("res/result_"+str(test_precs) +"_"+model_name+"_epo"+str(num_epoch)+ ".txt", "a")
    f.write(model_name + "\n")

