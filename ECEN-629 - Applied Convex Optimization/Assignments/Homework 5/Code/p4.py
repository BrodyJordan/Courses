import numpy as np


def dist(mu1x, mu1y, sigma1x2, sigma1y2, pi1, mu2x, mu2y, sigma2x2, sigma2y2, pi2):
    coef1 = (pi1) / (2 * np.pi * np.sqrt(sigma1x2) * np.sqrt(sigma1y2))
    exp1 = np.exp(
        -((x - mu1x) ** 2) / (2 * sigma1x2) - (y - mu1y) ** 2 / (2 * sigma1y2)
    )
    p1 = coef1 * exp1

    coef2 = (pi2) / (2 * np.pi * np.sqrt(sigma2x2) * np.sqrt(sigma2y2))
    exp2 = np.exp(
        -((x - mu2x) ** 2) / (2 * sigma2x2) - (y - mu2y) ** 2 / (2 * sigma2y2)
    )
    p2 = coef2 * exp2

    return p1 + p2


p4a = dist(0, 0, 1, 0.5, 0.4, 2, 1, np.sqrt(2), 2, 0.6)
