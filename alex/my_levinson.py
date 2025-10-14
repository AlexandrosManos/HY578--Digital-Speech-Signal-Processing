import numpy as np
def my_levinson(r, OrderLPC):
    """
    Lecture 4, Slide 33

    INPUT:
        r: autocorrelation sequence (numpy array), length >= order+1
           r[0] is the zero-lag autocorrelation
        OrderLPC: order of the LPC filter (integer)
    
    OUTPUT:
        aLPC: LPC coefficients (numpy array of length order+1)
           aLPC[0] = 1.0 (by convention for the all-pole filter)
           aLPC[1:] are the predictor coefficients

    Alex Manos, 2025
    """
    # Step 1: k[i] = [ r[i] - np.sum(I[:i] * r[i:0:-1]) ] / E[i-1]
    # Step 2: I[i,i] = k[i], I[i,j] = I[i-1,j] - k[i] * I[i-1, i-j], 1 <= j <= i -1
    # Step 3: E[i] = (1 - k[i] ** 2) * E[i-1]
    # Step 4: repeat
    # Final Step: I[LPC - 1, :], np.concatenate(([1.0], -I))


    # Initial Step
    I = np.zeros((OrderLPC + 1, OrderLPC))
    E = np.zeros(OrderLPC)
    k = np.zeros(OrderLPC)

    E[0] = r[0]
    k[0] = r[1] / E[0]
    I[1, 0] = k[0]
    E[0] *= (1 - k[0] ** 2)


    # Step 1-4
    for i in range(1, OrderLPC):

        # sum = 0
        # for j in range(i):  # j from 0 to i-1
        #     sum += I[i, j] * r[i - j]
        # k[i] = (r[i + 1] - sum) / E[i - 1]

        # divided = r[i+1] - np.sum(I[:i] * r[i:0:-1])
        divided = r[i+1] - np.sum(I[i, :i] * r[i:0:-1])
        k[i] = divided / E[i-1]

        I[i+1, i] = k[i]

        for j in range(i):
            I[i+1, j] = I[i, j] - k[i] * I[i, i - j - 1]

        # I[i, :i] = I[i - 1, :i] - k[i] * I[i - 1, :i][::-1]

        E[i] = (1 - k[i]**2)*E[i-1]

    a = I[OrderLPC, :OrderLPC]
    aLPC = np.concatenate(([1.0], -a))

    return aLPC