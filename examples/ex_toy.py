import numpy as np
from argparse import ArgumentParser
from sklearn.datasets import load_breast_cancer, load_iris, load_boston, load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, roc_auc_score
from glm_lib.distns import Bernoulli, Gaussian, Categorical
from glm_lib.models import GLM



if __name__ == "__main__":

    np.random.seed(123)

    argparser = ArgumentParser()
    argparser.add_argument("--lr", type=float, default=0.1)
    argparser.add_argument("--verbose", action="store_true")
    args = argparser.parse_args()

    x, y = load_breast_cancer(True)
    x = np.c_[x, np.ones(len(x))]
    x_tr, x_te, y_tr, y_te = train_test_split(x, y)

    glm = GLM(Bernoulli, args.lr, verbose=args.verbose)
    glm.fit(x_tr, y_tr)

    print(f"ROC: {roc_auc_score(y_te, glm.predict(x_te).mean())}")

    x, y = load_boston(True)
    x = np.c_[x, np.ones(len(x))]
    x_tr, x_te, y_tr, y_te = train_test_split(x, y)

    glm = GLM(Gaussian, args.lr, verbose=args.verbose)
    glm.fit(x_tr, y_tr)

    print(f"R2: {r2_score(y_te, glm.predict(x_te).mean())}")

    x, y = load_iris(True)
    x = np.c_[x, np.ones(len(x))]
    x_tr, x_te, y_tr, y_te = train_test_split(x, y)

    glm = GLM(Categorical, args.lr, verbose=args.verbose)
    glm.fit(x_tr, y_tr)

    print(f"Acc: {np.mean(np.argmax(glm.predict(x_te).mean(), axis=1) == y_te)}")

