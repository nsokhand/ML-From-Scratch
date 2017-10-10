from pandas import Series, DataFrame, concat
import numpy as np
import math


def next_value(samples, params, sigma):
    value = np.sum(np.array(samples)*np.array(params))
    noise = np.random.normal(loc=0.0, scale=sigma, size=1)
    final_value = value + noise
    return final_value

def generate_data(ar_params=None, order=5, noise_sigma=0.01, length=1000):
    if not ar_params:
        ar_params = [0.4,-0.2,0,-0.07,0.05]
    previous = list(np.random.normal(size=order))
    for k in range(length):
        next_val = next_value(previous[-order:], ar_params, noise_sigma)
        previous.append(next_val)
    series = np.zeros((length+order, 2))
    series[:, 0] = range(length+order)
    series[:, 1] = previous
    np.savetxt('train_dataset.csv', series[:int(length*0.8), :], delimiter=',')
    np.savetxt('test_dataset.csv', series[int(length*0.8):, :], delimiter=',')

def mean_squared_error(x):
    return math.sqrt(np.mean(x*x))

class AR():
    """Linear autoregression method trains a linear model to extrat the temporal correlation in time series.

    Parameters:
    -----------
    order: int
        The order of the Autoregressive model.
    params: numpy array
        The parameters of the Autoregressive model.
    """
    def __init__(self, order=3):
        self.order = order
        self.params = np.zeros((order, 1))

    def train(self, path1='train_dataset.csv', path2='ar_params.txt', p=3):
        series = Series.from_csv(path1, header=0)
        AC_matrix = self.autocorrelation(series, p)
        AC_matrix_inv= np.linalg.inv(AC_matrix[:p, :p])
        self.params = AC_matrix_inv.dot(AC_matrix[1:,0])
        self.order = p
        output_seq, training_errors = self.run_model_(series.values)
        rmse = mean_squared_error(training_errors)
        print("The training rmse: ", rmse)
        np.savetxt(path2, self.params)
        return self.params, training_errors

    def run_model_(self, values):
        output = []
        error_seq = []
        p = self.order
        for k in range(len(values)-p):
            predict = values[k:p+k].dot(self.params)
            output.append(predict)
            error_seq.append(predict-values[p+k])
        return np.array(output), np.array(error_seq)

    def test(self, path1='test_dataset.csv', path2='test_results.txt'):
        series = Series.from_csv(path1, header=0)
        results, test_errors = self.run_model_(series.values)
        rmse = mean_squared_error(test_errors)
        print("The test rmse: ", rmse)
        np.savetxt(path2, results)
        return results, test_errors

    def autocorrelation(self, time_series, order):
        values = DataFrame(time_series.values)
        data_frame = values
        for k in range(order):
            data_frame = concat([values.shift(k+1), data_frame], axis=1)
        autocorrelation_coefficients = data_frame.corr().values[:, :]
        return autocorrelation_coefficients

def main():
    generate_data()
    ar = AR(10)
    ar.train()
    ar.test()

if __name__ == "__main__":
    main()
