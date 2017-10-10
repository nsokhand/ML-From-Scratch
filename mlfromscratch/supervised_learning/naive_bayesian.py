import math
import numpy as np
import json
from random import randint

class NaiveBayes():
    def __init__(self):
        self.classes = [1, -1]
        self.X1 = None
        self.X2 = None
        self.y = None
        self.model_parameters = {}

    def train(self, path1='data.txt', path2= 'model.json'):
        data = self.load(path1)
        X1 = []
        X2 = []
        y = []
        for d in data:
            X1.append(d[0])
            X2.append(d[1])
            y.append(d[2])
        self.X1 = np.array(X1)
        self.X2 = np.array(X2)
        self.y = np.array(y)
        num_samples = len(y)
        self.X1_categories = np.unique(self.X1)
        self.X2_categories = np.unique(self.X2)
        positive_indices = np.where(self.y == 1)[0]
        num_pos = len(positive_indices)
        negative_indices = np.where(self.y == -1)[0]
        num_neg = len(negative_indices)
        priors = {}
        priors["positive"] = float(num_pos)/num_samples
        priors["negative"] = float(num_neg)/num_samples
        X1_positve = self.X1[positive_indices]
        X1_dist_negative = self.X1[negative_indices]
        X2_positve = self.X2[positive_indices]
        X2_dist_negative = self.X2[negative_indices]
        dist_X1_dist_positive = {}
        dist_X1_dist_negative = {}
        dist_X2_dist_positive = {}
        dist_X2_dist_negative = {}
        for category in self.X1_categories:
            num_pos_feature = len(np.where(X1_positve==category)[0])
            num_neg_feature = len(np.where(X1_dist_negative==category)[0])
            dist_X1_dist_positive[category] = float(num_pos_feature)/num_pos
            dist_X1_dist_negative[category] = float(num_neg_feature)/num_neg

        for category in self.X2_categories:
            num_pos_feature = len(np.where(X2_positve==category)[0])
            num_neg_feature = len(np.where(X2_dist_negative==category)[0])
            dist_X2_dist_positive[category] = float(num_pos_feature)/num_pos
            dist_X2_dist_negative[category] = float(num_neg_feature)/num_neg

        self.model_parameters['X1_dist_positive'] = dist_X1_dist_positive
        self.model_parameters['X2_dist_positive'] = dist_X2_dist_positive
        self.model_parameters['X1_dist_negative'] = dist_X1_dist_negative
        self.model_parameters['X2_dist_negative'] = dist_X2_dist_negative
        self.model_parameters['prior_probabilties'] = priors

        self.save(self.model_parameters, path2)

        train_y_preds, train_accuracy = self.test(path1)
        errors = train_y_preds - self.y

        return errors, train_accuracy

    # Classify using Bayes Rule, P(Y|X) = P(X|Y)*P(Y)/P(X)
    # P(X|Y) - Probability.
    # P(Y) - Prior
    # P(X) - Scales the posterior to the range 0 - 1 (ignored)
    # Classify the sample as the class that results in the largest P(Y|X)
    # (posterior)

    def save(self, data, file_path):
        with open(file_path, 'w') as fp:
            json.dump(data, fp, sort_keys=True, indent=4)

    def load(self, file_path):
        with open(file_path, 'r') as fp:
            model = json.load(fp)
        return model


    def classify_(self, x1, x2):
        # p(x1|Y=1)
        p1_positive = self.model_parameters['X1_dist_positive'][x1]
        # p(x2|Y=1)
        p2_positive = self.model_parameters['X2_dist_positive'][x2]
        # Assuming independence between features(Naive Assumption):
        # P(X|Y) = P(x1|Y)*P(x2|Y)*...*P(xN|Y)
        p_positive = p1_positive * p2_positive
        posterior_positive = p_positive * self.model_parameters['prior_probabilties']['positive']

        p1_negative = self.model_parameters['X1_dist_negative'][x1]
        p2_negative = self.model_parameters['X2_dist_negative'][x2]
        p_negative = p1_negative * p2_negative
        posterior_negative = p_negative * self.model_parameters['prior_probabilties']['negative']

        predict = 1

        if posterior_negative > posterior_positive:
            predict = -1

        return predict


    def test(self, path1, path2=None):
        data = self.load(path1)
        X1 = []
        X2 = []
        y = []
        for d in data:
            X1.append(d[0])
            X2.append(d[1])
            y.append(d[2])
        X1 = np.array(X1)
        X2 = np.array(X2)
        y = np.array(y)
        y_pred = []
        for k in range(len(y)):
            pred = self.classify_(X1[k], X2[k])
            y_pred.append(pred)
        if path2:
            self.save(y_pred, path2)

        accuracy = self.accuracy_(y, y_pred)
        return y_pred, accuracy

    def accuracy_(self, y, y_pred):
        accuracy = np.sum(y == y_pred) / len(y)
        return accuracy

    def generate_data(self, length, file_path_train, file_path_test):
        colors = ['red', 'blue']
        shapes = ['circle', 'square']
        X = []
        for k in range(length):
            color = colors[randint(0, 1)]
            shape = shapes[randint(0, 1)]
            if color=='blue' and shape=='circle':
                label = 1
            else:
                label = -1

            X.append([color, shape, label])
        ind = int(0.8 * length)
        self.save(X[:ind], file_path_train)
        self.save(X[ind:], file_path_test)

    def demo(self):
        train_data_path = "train_data.json"
        trained_model_path = "model.json"
        test_data_path = "test_data.json"
        test_results_path = "test_results.json"
        self.generate_data(2000, train_data_path, test_data_path)
        errors, train_accuracy = self.train(train_data_path, trained_model_path)
        test_results, test_accuracy = self.test(test_data_path, test_results_path)

        print("Training accuracy: ", train_accuracy)
        print("Test accuracy: ", test_accuracy)




def main():
    nb = NaiveBayes()
    nb.demo()

if __name__ == "__main__":
    main()
