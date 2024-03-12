import argparse
import os
import warnings
from os import listdir
from os.path import isfile, join
from random import seed

import pandas as pd
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB as Naive

import src.utils as ut
from src.ssl.ensemble import Ensemble
from src.ssl.self_flexcon import SelfFlexCon


crs = [0.05]
thresholds = [0.95]

warnings.simplefilter("ignore")

parser = argparse.ArgumentParser(description="Escolha um classificador para criar um cômite")
parser.add_argument('classifier', metavar='c', type=int, help='Escolha um classificador para criar um cômite. Opções: 1 - Naive Bayes, 2 - Tree Decision, 3 - Knn, 4 - Heterogeneous')
parent_dir = "path_for_results"
datasets_dir = "../FlexConC/datasets"
datasets = sorted(os.listdir(datasets_dir))
init_labelled = [0.03, 0.05, 0.08, 0.10, 0.13, 0.15, 0.18, 0.20, 0.23, 0.25]

args = parser.parse_args()

fold_result_acc_final = []
fold_result_f1_score_final = []


for threshold in thresholds:
    
    comite = "Comite_Naive_" if args.classifier == 1 else "Comite_Tree_" if args.classifier == 2 else 'Comite_KNN_' if args.classifier == 3 else "Comite_Heterogeneo_"

    path = os.path.join(parent_dir)
    
    folder_check_csv = f'path_for_results'
    os.makedirs(folder_check_csv, exist_ok=True)

    file_check = f'{comite}.csv'
    check = os.path.join(folder_check_csv, file_check)

    if not os.path.exists(check):
        with open(f'{folder_check_csv}/{file_check}', 'a') as f:
            f.write(
                f'"ROUNDS", "DATASET","LABELLED-LEVEL","ACC","F1-SCORE"'
            )

    file_check = f'{comite}F.csv'
    check = os.path.join(folder_check_csv, file_check)

    if not os.path.exists(check):
        with open(f'{folder_check_csv}/{file_check}', 'a') as f:
            f.write(
                f'"DATASET","LABELLED-LEVEL","ACC-AVERAGE","STANDARD-DEVIATION-ACC","F1-SCORE-AVERAGE","STANDARD-DEVIATION-F1-SCORE"'
        )

    for cr in crs:
        for labelled_level in init_labelled:
            for dataset in datasets:
                comite = Ensemble(SelfFlexCon, cr=cr, threshold=threshold)


                fold_result_acc = []
                fold_result_f1_score = []
                df = pd.read_csv('datasets/'+dataset, header=0)
                seed(214)
                kfold = StratifiedKFold(n_splits=10)
                fold = 1
                flag = 1
                _instances = df.iloc[:,:-1].values #X
                _target_unlabelled = df.iloc[:,-1].values #Y
                # _target_unlabelled_copy = _target_unlabelled.copy()

                # round counter
                rounds = 0
                for train, test in kfold.split(_instances, _target_unlabelled):
                    X_train, X_test = _instances[train], _instances[test]
                    y_train, y_test = _target_unlabelled[train], _target_unlabelled[test]
                    labelled_instances = round(len(X_train)*labelled_level)

                    rounds += 1

                    if args.classifier != 1 and args.classifier != 2 and args.classifier != 3 and args.classifier != 4:
                        print('\nOpção inválida! Escolha corretamente...\nOpções: 1 - Naive Bayes, 2 - Tree Decision, 3 - Knn, 4 - Heterogeneous\nEx: python main.py 1\n')
                        exit()
                    else:
                        # DISPLAY QUE INFORMA PARA O USUÁRIO COMO PROCEDER
                        if(flag == 1):
                            flag += 1
                            print(f"\n\nO sistema irá selecionar instâncias da base {dataset}. Para o treinamento, será usado {round(labelled_level, 4) * 100}% das instâncias rotuladas de um total de {len(_instances)}.\n\n")
                        instanciasRot = labelled_instances
                        instanciasRotPCento = (round(labelled_level, 4) * 100)
                        tInstanciasRot = "Inst. Rot.: " + str(labelled_instances)
                        tInstanciasRotPCento = " Uso: " + str(instanciasRotPCento) + "% das Inst. Rot."
                        if args.classifier == 1:
                            if(fold == 1):
                                fold += 1
                            y = ut.select_labels(y_train, X_train, labelled_instances)
                            for i in range(9):
                                comite.add_classifier(Naive(var_smoothing=float(f'1e-{i}')))
                            comite.fit_ensemble(X_train, y)

                        elif args.classifier == 2:
                            if(fold == 1):
                                fold += 1
                            y = ut.select_labels(y_train, X_train, labelled_instances)
                            for i in ut.list_tree:
                                comite.add_classifier(i)
                            comite.fit_ensemble(X_train, y)

                        elif args.classifier == 3:
                            if(fold == 1):
                                fold += 1
                            y = ut.select_labels(y_train, X_train, labelled_instances)
                            for i in ut.list_knn:
                                comite.add_classifier(i)
                            comite.fit_ensemble(X_train, y)

                        elif args.classifier == 4:
                            if(fold == 1):
                                fold += 1
                            y = ut.select_labels(y_train, X_train, labelled_instances)
                            for i in range(9):
                                comite.add_classifier(Naive(var_smoothing=float(f'1e-{i}')))
                            for i in ut.list_tree:
                                comite.add_classifier(i)
                            for i in ut.list_knn:
                                comite.add_classifier(i)
                            comite.fit_ensemble(X_train, y)

                        y_pred = comite.predict(X_test)
                        
                        result_acc = round(accuracy_score(y_test, y_pred) * 100, 4)

                        # Adds new accuracy to fold_result_acc
                        fold_result_acc.append(result_acc)

                        # Adds new accuracy to fold_result_acc_final
                        fold_result_acc_final.append(result_acc)

                        # Save data to .csv
                        result_f1 = ut.result(
                            args.classifier,
                            dataset,
                            y_test,
                            y_pred,
                            path,
                            labelled_level,
                            rounds
                            )
                        
                        fold_result_f1_score.append(result_f1)

                        fold_result_f1_score_final.append(result_f1)

                ut.calculateMeanStdev(
                    fold_result_acc,
                    args.classifier,
                    labelled_level,
                    path,
                    dataset,
                    fold_result_f1_score
                )

ut.calculateMeanStdev(
    fold_result_acc_final,
    args.classifier,
    labelled_level,
    path,
    'FINAL-RESULTS',
    fold_result_f1_score_final
)