import numpy as np
from argparse import ArgumentParser
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from glm_lib.distns import Bernoulli, Gaussian, Categorical
from glm_lib.models import GLM



if __name__ == "__main__":

    np.random.seed(123)

    argparser = ArgumentParser()
    argparser.add_argument("--lr", type=float, default=0.1)
    argparser.add_argument("--sigma", type=float, default=0.5)
    argparser.add_argument("--m", type=int, default=1000)
    argparser.add_argument("--n", type=int, default=100)
    argparser.add_argument("--verbose", action="store_true")
    args = argparser.parse_args()

    x = np.random.randn(args.m, args.n)
    x = np.c_[x, np.ones(args.m)]
    theta = np.random.randn(args.n + 1)
    y = x @ theta / args.n ** 0.5 + args.sigma * np.random.randn(args.m)
    x_tr, x_te, y_tr, y_te = train_test_split(x, y)

    glm = GLM(Gaussian, args.lr, verbose=args.verbose)
    glm.fit(x_tr, y_tr)

    print(f"R2: {r2_score(y_te, glm.predict(x_te).mean())}")
