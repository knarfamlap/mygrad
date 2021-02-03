from mygrad.engine import Variable
import mygrad.F as F

def LogLoss(y, p):
    return - (y * F.log(p) + (1 - y) * F.log(1 - p))


