import xgboost as xgb
import pandas as pd
from json import dump, load
from datetime import datetime

from pathlib import Path
from autograd import grad, hessian
from autograd import numpy as anp
from numpy import diag


def approx_huber(y_hat, y, delta=5.0):
    """
    Implement the continuous approximation to the Huber loss.
    https://alexisalulema.com/2017/12/07/loss-functions-part-1/
    """

    y = y.get_label()

    def huber_loss(y_hat):
        loss = 1.0 + anp.square((y - y_hat) / delta)
        loss = anp.sum(delta * delta * (anp.sqrt(loss) - 1))
        return loss

    h_grad = grad(huber_loss)(y_hat)
    h_hess = hessian(huber_loss)(y_hat)

    # Take the diagonal of the Hessian and make a copy
    # Because xgboost expects a contiguous array
    h_hess = diag(h_hess).copy()
    return h_grad, h_hess


def fit_model(dtrain, valset, best_hyperparams, n_rounds=100, use_huber=False):
    print('Fitting model')
    evals_result = {}
    bst = xgb.train(best_hyperparams, dtrain,
                    obj=approx_huber if use_huber else None,
                    evals=valset,
                    early_stopping_rounds=10,
                    num_boost_round=n_rounds,
                    evals_result=evals_result,
                    verbose_eval=25
                    )
    feature_map = pd.DataFrame.from_dict(bst.get_fscore(),
                                         orient='index',
                                         columns=['weight']
                                         ).sort_values('weight', ascending=False)
    return bst, feature_map, evals_result


def make_predictions(model, dtest, test_df, target_col):
    prediction = pd.DataFrame(model.predict(dtest, iteration_range=(0, model.best_iteration)),
                              columns=['prediction_XGBR'],
                              index=test_df.index
                              )
    result_df = prediction.join(test_df[target_col])
    return result_df


def tune_hyperparameters(train_matrix, val_matrix, eval_metric, file_name, file_extenstion=None, hyperopt=None):
    from hyperopt import hp, fmin, tpe, Trials, STATUS_OK, space_eval

    def objective(space):
        """
        The objective function of the hyperoptimization, by using a cross-validation to optimize the hyperparameters
        """
        bst = xgb.train(space, train_matrix,
                        evals=val_matrix,
                        early_stopping_rounds=10,
                        num_boost_round=50,
                        evals_result={},
                        verbose_eval=25
                        )

        return {'loss': bst.best_score, 'status': STATUS_OK}

    space = {'max_depth': hp.choice("max_depth", [3, 5, 7, 8, 9, 12]),
             'eta': hp.choice('eta', [0.01, 0.015, 0.025, 0.05, 0.1, 0.3]),
             'gamma': hp.choice('gamma', [0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 1]),
             'reg_alpha': hp.choice('reg_alpha', [0, 0.1, 0.5, 1]),
             'reg_lambda': hp.choice('reg_lambda', [0.01, 0.1, 1]),
             'colsample_bytree': hp.choice('colsample_bytree', [0.6, 0.7, 0.8, 0.9, 1]),
             'subsample': hp.choice('subsample', [0.6, 0.7, 0.8, 0.9, 1]),
             'min_child_weight': hp.choice('min_child_weight', [1, 3, 5, 7]),
             'rate_drop': hp.choice('rate_drop', [0.1, 0.25, 0.5]),
             'skip_drop': hp.choice('skip_drop', [0.25, 0.5, 0.75]),
             'booster': 'dart',
             'objective': 'reg:squarederror',
             'seed': 0,
             'tree_method': 'hist',
             'gpu_id': -1
             }
    xgb_hyperparams_folder = 'models/xgb/' if file_extenstion is None else f'models/xgb/{file_extenstion}'

    if Path(xgb_hyperparams_folder + f'/{file_name}.json').is_file():
        with open(xgb_hyperparams_folder + f'/{file_name}.json', 'r') as fp:
            hyperparam = load(fp)
    else:
        trials = Trials()
        hyperparam = fmin(fn=objective,
                          space=space,
                          algo=tpe.suggest,
                          max_evals=150,
                          trials=trials)
        hyperparam = space_eval(space, hyperparam)
        print(hyperparam)
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        Path(xgb_hyperparams_folder).mkdir(parents=True, exist_ok=True)
        with open(f'{xgb_hyperparams_folder}/{file_name}.json', 'w') as fp:
            dump(hyperparam, fp)

    return hyperparam