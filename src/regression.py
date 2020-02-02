import pandas as pd
import numpy as np
import math, sys
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import ElasticNet, Lasso, BayesianRidge
from sklearn.kernel_ridge import KernelRidge
from xgboost import XGBRegressor
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from joblib import dump, load
from itertools import product
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

df = pd.read_csv('../data/raw/intern_data.csv', index_col=0)
num_cols = ['a', 'b', 'd', 'e', 'f', 'g']
ctg_cols = ['c', 'h']
Y_LABEL = 'y'

#column_to_nmlz = num_cols + ['eh']
column_to_nmlz = list(set(df.columns) - set([Y_LABEL]) - set(ctg_cols))


def preprocess(df_input, ):
    df_dummy = pd.get_dummies(df_input, columns=ctg_cols)
    # append new features
    df_dummy['ehc'] = df_dummy['e'] * df_dummy['h_white'] * df_dummy['c_blue'].map({0: 1, 1: 0})
    df_dummy['e(h+c)'] = df_dummy['e'] * (df_dummy['h_white'] + df_dummy['c_blue'].map({0: 1, 1: 0}))
    df_dummy['f+g'] = df_dummy['f'] + df_dummy['g']
    df_dummy['(f+g)*h*c'] = (df_dummy['f'] + df_dummy['g']) * df_dummy['h_white'] * df_dummy['c_blue'].map({0: 1, 1: 0})
    df_dummy['(f+g)*(h+c)'] = (df_dummy['f'] + df_dummy['g']) * (df_dummy['h_white'] + df_dummy['c_blue'].map({0: 1, 1: 0}))
    df_dummy['ehc(f+g)'] = df_dummy['ehc'] * df_dummy['f+g']
    df_dummy['e(h+c)(f+g)'] = df_dummy['e(h+c)'] * df_dummy['f+g']
    df_dummy['ehc+f+g'] = df_dummy['ehc'] + df_dummy['f+g']
    df_dummy['e(h+c)+f+g'] = df_dummy['e(h+c)'] + df_dummy['f+g']
    df_dummy['eh(f+g)*(h+c)'] = df_dummy['ehc'] * df_dummy['(f+g)*(h+c)']
    df_dummy['eh+(f+g)*(h+c)'] = df_dummy['ehc'] + df_dummy['(f+g)*(h+c)']
    df_dummy['e(h+c)(f+g)*(h+c)'] = df_dummy['e(h+c)'] * df_dummy['(f+g)*(h+c)']
    df_dummy['e(h+c)+(f+g)*(h+c)'] = df_dummy['e(h+c)'] + df_dummy['(f+g)*(h+c)']
    return df_dummy


def normalize_data(df_input, scaler='robust'):
    result = df_input.copy(deep=True)
    if scaler == 'robust':
        num_pipeline = Pipeline([('robust_scaler', RobustScaler())])
    else:
        num_pipeline = Pipeline([('std_scaler', StandardScaler())])
    #column_to_nmlz = list(set(df_input.columns) - set([y_label]))
    result[column_to_nmlz] = num_pipeline.fit_transform(df_input[column_to_nmlz])
    return result


df_dummy_normed = normalize_data(df_dummy, scaler='std')


def get_train_test_sets(test_size=0.2, dummy=False, normed=False):
    if dummy is True and normed is True:
        df_input = df_dummy_normed
    elif dummy is True and normed is False:
        df_input = df_dummy
    else:
        df_input = df

    train_set, test_set = train_test_split(df_input, test_size=test_size, random_state=42)
    x_train = train_set.drop(columns=[Y_LABEL], axis=1)
    x_test = test_set.drop(columns=[Y_LABEL], axis=1)
    # y_train, y_test = num_pipeline.fit_transform(train_set[[Y_LABEL]]), num_pipeline.fit_transform(test_set[[Y_LABEL]])
    y_train, y_test = train_set[Y_LABEL], test_set[Y_LABEL]
    y_all = df_input[Y_LABEL]
    x_all = df_input.drop(columns=[Y_LABEL])
    return x_train, x_test, y_train, y_test, x_all, y_all


x_train, x_test, y_train, y_test, x_all, y_all = get_train_test_sets(dummy=True, normed=False)


def mape(y_preds, y_test, **kwargs):
    return np.absolute((y_preds - y_test) / y_test).mean()

def train_n_evaluate(model, model_name, cv=3, scoring=mape, save_model=True):
    print("training {} regressor...".format(model_name))
    #model.fit(x_train, y_train)
    if save_model is True:
        print("{} regressor trained, saving model...".format(model_name))
        dump(model, '../models/{}.joblib'.format(model_name))
        print("saving model finished")
    print("getting validation scores...")
    scores = -cross_val_score(model, x_train, y_train, cv=cv, scoring=make_scorer(mape, greater_is_better=False))
    print("cross val scores for score:{}, avg:{}, std:{}".format(scores, scores.mean(), scores.std()))


def dict_product(d):
    keys = d.keys()
    for element in product(*d.values()):
        yield dict(zip(keys, element))


def evaluate_on_testset(model, verbose=1):
    y_preds_test = model.predict(x_test)
    mae_test = np.absolute((y_preds_test.reshape(1, -1) - y_test.values)).mean()
    if verbose > 0:
        print("MAE on test set: {}".format(mae_test))
    return mae_test


def GridSearchWithVal(model_class, param_grid, metrics='mae', cv=0, x_input=None, y_input=None):
    combinations = list(dict_product(param_grid))
    min_metrics = math.inf
    best_comb = None
    X = x_input if x_input is not None else x_train
    Y = y_input if y_input is not None else y_train
    print("{} combinations in total. Metric:{}".format(len(combinations), metrics))
    for idx, comb in enumerate(combinations):
        model = model_class(**comb)
        # model.fit(x_train, y_train)
        # y_preds = model.predict(x_test)
        # print("y_preds:{}".format(y_preds))
        # print("y_test:{}".format(y_test.values))
        error = 0;
        if metrics == 'mape':
            if cv == 0:
                model.fit(X, Y)
                y_preds = model.predict(x_test)
                metrics_num = np.absolute((y_preds - y_test) / y_test).mean()
            else:
                scores = -cross_val_score(model, X, Y, cv=cv, scoring=make_scorer(mape, greater_is_better=False))
                metrics_num = scores.mean()
        else:
            metrics_num = np.absolute(y_preds - y_test).mean()
        if metrics_num < min_metrics:
            min_metrics = metrics_num
            best_comb = comb
        progress_str = "{} / {}, best comb: {}, best score: {}".format(idx + 1, len(combinations), best_comb, min_metrics)
        sys.stdout.write('\r' + progress_str)

    print("best params:{}".format(best_comb))
    print("min {}: {}".format(metrics, min_metrics))
    return best_comb


rf_param_grid = {"n_estimators" : [200, 230, 250, 260],
                  "criterion" : ["mae"],
                  "max_depth": [None],
                  "max_features": [0.1, 0.2, 0.5, 0.8, None],
                  "min_samples_split": [2, 3, 4, 5],
                  "min_samples_leaf": [1, 3, 5, 10],
                  "bootstrap": [True, False]
                 }
rf_best = GridSearchWithVal(RandomForestRegressor, rf_param_grid, metrics='mape')
rf_best_model = RandomForestRegressor(**rf_best)
train_n_evaluate(rf_best_model, 'random_forest', cv=10)

krr_param_grid = {"alpha": [8, 9, 10, 11, 12],
                  "kernel": ['polynomial', 'rbf'],
                  "degree": [1, 2, 3],
                  "coef0": [4e7, 4.5e7, 5e7, 5.5e7, 6e7]
                 }
krr_best = GridSearchWithVal(KernelRidge, krr_param_grid, metrics='mape', cv=10)
krr_best_model = KernelRidge(**krr_best)
train_n_evaluate(krr_best_model, 'Kernel_Ridge', cv=10)

lasso_param_grid = {"alpha" : [3e-5, 5e-5, 7e-5, 1e-4],
                    "fit_intercept": [True, False],
                    "normalize": [True, False],
                    "precompute": [True, False],
                    "tol": [5e-3, 0.015, 0.02, 0.025, 0.03],
                    "positive": [True, False],
                    "selection": ["cyclic", "random"]
                 }
lasso_best_param = GridSearchWithVal(Lasso, lasso_param_grid, metrics='mape', cv=10)
lasso_best_model = Lasso(**lasso_best_param)
train_n_evaluate(lasso_best_model, 'Lasso', cv=10)

ElasticNet_param_grid = {"alpha" : [3e-7, 5e-7, 1e-6, 5e-6],
                    "l1_ratio": [0.1, 0.5, 1, 1.3],
                    "fit_intercept": [True, False],
                    "precompute": [True, False],
                    "normalize": [True, False],
                    "tol": [1e-4, 2e-4, 4e-4],
                    "positive": [True, False],
                    "selection": ["cyclic", "random"]
                 }
ElasticNet_best_param = GridSearchWithVal(ElasticNet, ElasticNet_param_grid, metrics='mape', cv=10)
ElasticNet_best_model = ElasticNet(**ElasticNet_best_param)
train_n_evaluate(ElasticNet_best_model, 'ElasticNet', cv=10)

BayesianRidge_param_grid = {"alpha_1" : [8e-3, 1e-2, 3e-2],
                    "alpha_2" : [0.03, 0.04, 0.05],
                    "lambda_1": [1, 1.1, 1.2],
                    "lambda_2": [2.4, 2.5, 2.6],
                    "compute_score": [True, False],
                    "fit_intercept": [True, False],
                    "normalize": [True, False]
                 }
BayesianRidge_best_param = GridSearchWithVal(BayesianRidge, BayesianRidge_param_grid, metrics='mape', cv=10)
BayesianRidge_best_model = BayesianRidge(**BayesianRidge_best_param)
train_n_evaluate(BayesianRidge_best_model, 'BayesianRidge', cv=10)

xgb_param_grid = {"objective" : ['reg:squarederror'],
                  "n_estimators": [20, 50, 100, 500, 1000, 1500, 2000, 2500, 3000],
                  #"base_score" : [0.05, 0.06, 0.07, 0.08, 1],
                  "base_score" : [0.01, 0.07],
                  "max_depth": [5, 10],
                  "gamma": [1e-4, 0.003],
                  "min_child_weight": range(3, 9, 2),
                  "learning_rate": [0.05, 0.2]
                 }
xgb_best = GridSearchWithVal(XGBRegressor, xgb_param_grid, metrics='mape')

xbg_best_model = XGBRegressor(**xgb_best)
xbg_best_model.fit(x_train, y_train)
train_n_evaluate(xbg_best_model, 'xgboost', cv=10)

gb_param_grid = {"n_estimators" : [3500, 4000, 4500],
                   "loss" : ["ls"],
                   "learning_rate": [0.1, 0.2, 0.25],
                   "max_depth": [3, 5, 7],
                   "min_samples_leaf": [1, 5, 10, 30],
                 }
gb_best = GridSearchWithVal(GradientBoostingRegressor, gb_param_grid, metrics='mape')


class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models

    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]
        for model in self.models_:
            model.fit(X, y)
        return self

    def predict(self, X):
        predictions = np.column_stack([model.predict(X) for model in self.models_])
        return np.mean(predictions, axis=1)


averaged_models = AveragingModels(models=(lasso_best_model, krr_best_model, ElasticNet_best_model))
train_n_evaluate(averaged_models, 'averaged', cv=10)


class EnsembleModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models, weights):
        self.models = models
        self.weights = weights

    def fit(self, x, y):
        self.models_ = [clone(m) for m in self.models]
        for model in self.models_:
            model.fit(x, y)
        return self

    def predict(self, x):
        results = []
        for model, weight in zip(self.models_, self.weights):
            results.append(model.predict(x) * weight)
        return


class StackedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds

    def fit(self, x, y):
        self.base_models_ = [[] for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=157)

        out_of_fold_predictions = np.zeros((x.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(x, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(x.iloc[train_index], y.iloc[train_index])
                y_pred = instance.predict(x.iloc[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred

        #         meta_search_space = {"alpha" : [1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3],
        #                     "fit_intercept": [True, False],
        #                     "normalize": [True, False],
        #                     "precompute": [True, False],
        #                     "tol": [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 0.05, 0.1, 0.5],
        #                     "positive": [True, False],
        #                     "selection": ["cyclic", "random"]
        #                  }
        #         print("begin grid search for meta model...")
        #         meta_best_param = GridSearchWithVal(self.meta_model, meta_search_space, metrics='mape', cv=10,
        #                                             x_input=out_of_fold_predictions, y_input=y)
        #         print("grid search for meta model finished")
        #         self.meta_model_ = self.meta_model(**meta_best_param)
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self

    def predict(self, x):
        meta_features = np.column_stack([
            np.column_stack([model.predict(x) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_
        ])
        return self.meta_model_.predict(meta_features)


stacked_avg_models = StackedModels(base_models=(lasso_best_model, krr_best_model, BayesianRidge_best_model, ElasticNet_best_model),
                                   meta_model=lasso_best_model)
train_n_evaluate(stacked_avg_models, 'stacked_avg', cv=10)

df_test = pd.read_csv("../data/raw/intern_test.csv", index_col=0)
df_test_dummy = preprocess(df_test)
averaged_models.fit(x_train, y_train)
predictions = averaged_models.predict(df_test_dummy)

df_output = pd.DataFrame(data={'i': df_test_dummy.index, 'y': predictions})
df_output.to_csv("../data/output/intern_predicted.csv", index=False)
