import numpy as np
from mygrad.engine import Variable

to_var = np.vectorize(lambda x: Variable(x))
to_val = np.vectorize(lambda variable: variable.value)
