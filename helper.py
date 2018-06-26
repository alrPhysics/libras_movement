def print_results(y_test, y_pred, name, acc, con_mat, class_rep):
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    print 'Results for {}:'.format(name)
    if acc:
        print 'Accuracy = {:.4f}'.format(accuracy_score(y_test,y_pred))
    if con_mat:
        print 'Confusion matrix:'
        print confusion_matrix(y_test, y_pred)
    if class_rep:
        print classification_report(y_test, y_pred)
    

def train_predict(X_train, X_test, y_train, y_test, models, acc=True, con_mat=True, class_rep=False):
    preds = {}
    for clf in models:
        name = clf.__class__.__name__
        clf.fit(X_train, y_train)
        preds[name] = clf.predict(X_test)
        
        print_results(y_test, preds[name], name, acc=acc, con_mat=con_mat, class_rep=class_rep)
 

def optimize_models(X_train, X_test, y_train, y_test, models, params, seed, cv_n_splits=10, show_best_params = True, acc=True, con_mat=True, class_rep=False, use_kfold=True):
    from sklearn.model_selection import GridSearchCV, KFold
    
    if use_kfold:
        kfold=KFold(n_splits=cv_n_splits, random_state=seed)

    for model,param in zip(models,params):
        if use_kfold:
            grid_obj = GridSearchCV(model, param,cv=kfold)
        else:
            grid_obj = GridSearchCV(model, param, cv=cv_n_splits)
        grid_fit = grid_obj.fit(X_train, y_train)
        best_clf = grid_fit.best_estimator_
        best_pred = best_clf.predict(X_test)
        model_name = best_clf.__class__.__name__
        
        best_pred_prob = best_clf.predict_proba(X_test)[:,1]
        
        print_results(y_test, best_pred, model_name, acc=acc,con_mat=con_mat,class_rep=class_rep)
        if show_best_params:
            print grid_fit.best_params_
        
def pca_hist(data, pca, figsize):
    
    #Copied and modified from code provided for Udacity's MLND
    
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    
    dim = ['Dimension {}'.format(i) for i in range(1, len(pca.components_)+1)]
    comp = pd.DataFrame(np.round(pca.components_,4), columns = data.keys())
    comp.index = dim
    ratios = pca.explained_variance_ratio_.reshape(len(pca.components_),1)
    var_ratios = pd.DataFrame(np.round(ratios, 4), columns = ['Explained Varaince'])

    fig, ax = plt.subplots(figsize = figsize)
    comp.plot(ax = ax, kind='bar');
    ax.set_ylabel("Feature Weights")
    ax.set_xticklabels(dim,rotation=0)
    ax.legend(loc = 2, bbox_to_anchor = (1,1));

    for i, ev in enumerate(pca.explained_variance_ratio_):
        ax.text(i-0.40, ax.get_ylim()[1] + 0.05, "Explained Variance\n          %.4f"%(ev))