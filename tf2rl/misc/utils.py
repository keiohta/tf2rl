from scipy.signal import lfilter


def discount_cumsum(x, discount):
    """
    Forked from rllab for computing discounted cumulative sums of vectors.
    input: 
        vector x, 
        [x0, 
         x1, 
         x2]
    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return lfilter(
        b=[1],
        a=[1, float(-discount)],
        x=x[::-1],
        axis=0)[::-1]
