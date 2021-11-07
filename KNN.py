from collections import Counter 

class KNN:
    def __init__(self):
        self.x = None
        self.y = None

    def distance(self,p1,p2):
        """Euclidean Distance"""
        if len(p1) != len(p2):
            raise ValueError('Datapoints are unequal in length')
        sum_squares = 0
        for i in range(len(p1)):
            sum_squares += (p1[i] - p2[i]) ** 2
        return  sum_squares ** 0.5

    def train(self,x,y):
        self.x = x
        self.y = y

    def predict(self,x,k,c):
        """
            x is a two-dimensional array, where the number of rows represent the number of datapoints and the number of columns 
            represent the number of features
            y is a one-dimensional array for KNN regression, it is an array of floats, for classification it is an array of integers
            calculates the distance between a new datapoint x and every point in the training data
            stores the distance as a tuple with the label
            Takes mean of neighbor labels in regression
            Select most common neighbor label in classification
        """
        preds = []
        for test_point in x:
            distance_label = [
                (self.distance(test_point, train_point), train_label)
                for train_point, train_label in zip(self.x, self.y)]
            distance_label.sort()
            neighbors = distance_label[:k]
            if c:
                neighbors_labels = [label for dist, label in neighbors]
                preds.append( Counter(neighbors_labels).most_common()[0][0])
            else:
                preds.append(sum(label for _, label in neighbors) / k)
        return preds

"""
Code the scaler, train_test_split yourself
Try with another dataset
What are your time and space complexities?
How can you optimise them?
"""