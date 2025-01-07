import argparse
import os
import warnings
from random import seed

from pandas import read_csv
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB as Naive

from src.ssl.ensemble import Ensemble
from src.ssl.self_flexcon import SelfFlexCon
from src.utils import calculate_mean_stdev, list_knn_full, list_knn_Prelax, \
    list_knn_seeds, list_tree, result, select_labels

crs = [0.05]
thresholds = [0.95]

warnings.simplefilter("ignore")

parser = argparse.ArgumentParser(description="Escolha um classificador para criar um cômite")
parser.add_argument(
    'classifier',
    metavar='c',
    type=int,
    help='Escolha um classificador para criar um cômite.'
        'Opções: 1 - Naive Bayes, 2 - Tree Decision, 3 - Knn, 4 - Heterogeneous'
)
parent_dir = "path_for_results"
datasets_dir = "../FlexCon/datasets"
datasets = sorted(os.listdir(datasets_dir))
init_labelled = [0.03, 0.05, 0.08, 0.10, 0.13, 0.15, 0.18, 0.20, 0.23, 0.25]

args = parser.parse_args()

fold_result_acc_final = []
fold_result_f1_score_final = []

if args.classifier >= 4 or args.classifier < 0:
    print(
        '\nOpção inválida! Escolha corretamente...\n'
        'Opções: 1 - Naive Bayes, 2 - Tree Decision, 3 - Knn, 4 - Heterogeneous\n'
        'Ex: python main.py 1\n'
    )
    exit()

for threshold in thresholds:

    comite_map = {
        1: "Comite_Naive_",
        2: "Comite_Tree_",
        3: "Comite_KNN_"
    }

    comite = comite_map.get(args.classifier, "Comite_Heterogeneo_")

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
            header = (
                '"DATASET","LABELLED-LEVEL","ACC-AVERAGE",'
                '"STANDARD-DEVIATION-ACC","F1-SCORE-AVERAGE",'
                '"STANDARD-DEVIATION-F1-SCORE"'
            )
            f.write(header)

    for cr in crs:
        for labelled_level in init_labelled:
            for dataset in datasets:
                comite = Ensemble(SelfFlexCon, cr=cr, threshold=threshold)

                fold_result_acc = []
                fold_result_f1_score = []
                df = read_csv('datasets/'+dataset, header=0)
                seed(214)
                kfold = StratifiedKFold(n_splits=10)
                fold = 1
                _instances = df.iloc[:,:-1].values #X
                _target_unlabelled = df.iloc[:,-1].values #Y
                # _target_unlabelled_copy = _target_unlabelled.copy()

                # round counter
                rounds = 0
                for train, test in kfold.split(_instances, _target_unlabelled):
                    X_train = _instances[train]
                    X_test = _instances[test]
                    y_train = _target_unlabelled[train]
                    y_test = _target_unlabelled[test]
                    labelled_instances = labelled_level

                    rounds += 1

                    if (fold == 1):
                        fold += 1

                    y = select_labels(y_train, X_train, labelled_instances)

                    if (args.classifier == 1) or (args.classifier == 4):
                        for i in range(9):
                            comite.add_classifier(Naive(var_smoothing=float(f'1e-{i}')))

                    if (args.classifier == 2) or (args.classifier == 4):
                        for i in list_tree:
                            comite.add_classifier(i)

                    if (args.classifier == 3) or (args.classifier == 4):
                        if dataset == 'Seeds.csv':
                            for i in list_knn_seeds:
                                comite.add_classifier(i)
                        elif dataset == 'PlanningRelax.csv':
                            for i in list_knn_Prelax:
                                comite.add_classifier(i)
                        else:
                            for i in list_knn_full:
                                comite.add_classifier(i)

                    comite.fit_ensemble(X_train, y)

                    y_pred = comite.predict(X_test)

                    result_acc = accuracy_score(y_test, y_pred)

                    # Adds new accuracy to fold_result_acc
                    fold_result_acc.append(result_acc)

                    # Adds new accuracy to fold_result_acc_final
                    fold_result_acc_final.append(result_acc)

                    # Save data to .csv
                    result_f1 = result(
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

                calculate_mean_stdev(
                    fold_result_acc,
                    args.classifier,
                    labelled_level,
                    path,
                    dataset,
                    fold_result_f1_score
                )

calculate_mean_stdev(
    fold_result_acc_final,
    args.classifier,
    labelled_level,
    path,
    'FINAL-RESULTS',
    fold_result_f1_score_final
)
