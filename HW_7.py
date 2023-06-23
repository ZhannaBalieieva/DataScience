import numpy as np
from surprise import accuracy, Dataset, SVD, NMF
from surprise.model_selection import train_test_split
from surprise.model_selection import cross_validate
from surprise.model_selection import KFold
from surprise.model_selection import GridSearchCV
from surprise import NormalPredictor
from surprise import KNNBasic
from surprise import KNNWithMeans
from surprise import KNNWithZScore
from surprise import KNNBaseline
from surprise import BaselineOnly
from surprise import SVDpp
from surprise import SlopeOne
from surprise import CoClustering
from surprise.accuracy import rmse
from surprise import accuracy



data = Dataset.load_builtin(name='ml-100k', prompt=False)
algo = SVD()
#algo.fit(data.build_full_trainset())
cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
trainset, testset = train_test_split(data, test_size=.25)
algo.fit(trainset)
predictions = algo.test(testset)
accuracy.rmse(predictions)

param_grid = {"n_epochs": [5, 10], "lr_all": [0.002, 0.005], "reg_all": [0.4, 0.6]}
gs = GridSearchCV(SVD, param_grid, measures=["rmse", "mae"], cv=3)
gs.fit(data)
print(gs.best_score["rmse"])
print(gs.best_params["rmse"])


algorithms = [SVD(), SVDpp(), SlopeOne(), NMF(), NormalPredictor(), KNNBaseline(), KNNBasic(), KNNWithMeans(), KNNWithZScore(), BaselineOnly(), CoClustering()]
print("Attempting: ", str(algorithms), '\n\n\n')
for algorithm in algorithms:
    print("Starting: ", str(algorithm))
    results = cross_validate(algorithm, data, measures=['RMSE'], cv=3, verbose=False)
    print("Done: ", str(algorithm), "\n\n")
    print(results)
print('\n\tDONE\n')





