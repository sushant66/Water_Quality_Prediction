from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
import random

#xs = np.array([1,2,3,4,5,6], dtype=np.float64)
#ys = np.array([5,4,6,5,6,7], dtype=np.float64)

# plt.scatter(xs, ys)
# plt.show()

def create_dataset(hm, variance, step = 2, correlation = 'pos'):
    #hm is how many
    val = 1
    ys = []
    for _ in range(hm):
        y = val + random.randrange(-variance, variance)
        ys.append(y)
        if correlation and correlation == 'pos':
            val+=step
        elif correlation and correlation == 'neg':
            val-=step
    xs = [i for i in range(len(ys))]
    return (np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64))


def best_fit_line(xs, ys):
    m = ((mean(xs)*mean(ys)) - mean(xs*ys))/((mean(xs)**2)-mean(xs**2))
    y = mean(ys) - (m*mean(xs))
    return (m,y)

def squared_error(ys_orig, ys_line):
    return sum((ys_line-ys_orig)**2)

def coeff_of_deter(ys_orig, ys_line):
    y_mean_line = [mean(ys_orig) for y in ys_orig]
    squared_error_regr = squared_error(ys_orig, ys_line)
    squared_error_y_mean = squared_error(ys_orig, y_mean_line)
    return 1 - (squared_error_regr/squared_error_y_mean)

xs, ys = create_dataset(40, 40, 2, correlation='pos') #if variance = 10 then r coeff increases

m,y = best_fit_line(xs, ys)
print(m,y)
regr_line = [(m*x)+y for x in xs]

predict_x = 8
predict_y = (m*predict_x)+y

r_squared = coeff_of_deter(ys, regr_line)
print(r_squared)

plt.scatter(xs, ys)
plt.scatter(predict_x, predict_y, color = 'g')
plt.plot(xs, regr_line)
plt.show()