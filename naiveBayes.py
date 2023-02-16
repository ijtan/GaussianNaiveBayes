import math

class CustomNaiveBayes:
    def __init__(self, laplace_smoothing=True, smoothing_factor=1):
        self._classes = None
        self._class_priors = None
        self._mean = None
        self._var = None
        self._laplace_smoothing = laplace_smoothing
        self._smoothing_factor = smoothing_factor
        self._epsilon = 1e-9
        
    def _calculate_classes(self, y):
        self._classes = list(set(y))
        
    def _calculate_prior(self, y):
        
        occurances = [0] * len(self._classes)
        for cl in self._classes:
            for i in y:
                if i == cl:
                    occurances[cl] += 1
                    
        no_of_samples = len(y)
        
        if self._laplace_smoothing:
            self._class_priors = [(occurances[i] + self._smoothing_factor) / (no_of_samples + (self._smoothing_factor * len(self._classes))) for i in range(len(self._classes))]
        else:
            self._class_priors = [occurances[i] / no_of_samples for i in range(len(self._classes))]            

    def _calculate_mean(self, X, y):
        self._mean = []
        occurances = [0] * len(self._classes)
        for cl in self._classes:
            mean = [0] * len(X[0])
            for i in range(len(X)):
                if y[i] == cl:
                    for j in range(len(X[0])):
                        mean[j] += X[i][j]
                    occurances[cl] += 1
            for i in range(len(mean)):
                mean[i] /= occurances[cl]
            self._mean.append(mean)
    
    def _calculate_variance(self, X, y):
        self._var = []
        occurances = [0] * len(self._classes)
        for cl in self._classes:
            var = [0] * len(X[0])
            for i in range(len(X)):
                if y[i] == cl:
                    for j in range(len(X[0])):
                        var[j] += (X[i][j] - self._mean[cl][j]) ** 2
                    occurances[cl] += 1
            for i in range(len(var)):
                var[i] /= occurances[cl]
            self._var.append(var)
            
        
        
        
    def fit(self, X, y):
        self._calculate_classes(y)
        self._calculate_prior(y)
        self._calculate_mean(X, y)
        self._calculate_variance(X, y)
        

        
        

    def predict(self, X):
        posteriors = []
        for x in X:
            posterior = []
            for idx, c in enumerate(self._classes):
                logged_prior = []
                for i in self._class_priors:
                    logged_prior.append(math.log(i))
                prior = logged_prior[idx]
                
                conditional = 0
                pdf = self.gaussian(idx, x)
                logged_pdf = []
                for i in pdf:
                    logged_pdf.append(math.log(i + self._epsilon))
                for i in logged_pdf:
                    conditional += i
                
                posterior.append(prior + conditional)
                
            posteriors.append(posterior)
      
        preds = []
        # now, we need to find the class with the highest posterior probability
        # this is done by finding the index of the maximum value in each posterior list
        # and then using that index to find the corresponding class in self._classes
        for i in posteriors:
            preds.append(self._classes[i.index(max(i))])
        return preds

    def gaussian(self, idx, x):
        ans = []
        for i in range(len(x)):
            coeff = 1 / (math.sqrt(2 * math.pi * self._var[idx][i] + self._epsilon))
            
            exp_num = -1 * ((x[i] - self._mean[idx][i]) ** 2)
            exp_den = (2 * self._var[idx][i]) + self._epsilon
            
            ans.append(coeff * math.exp(exp_num / exp_den))
                        
            
        return ans
    
    def accuracy(self, y, preds):
        correct = 0
        for i in range(len(y)):
            if y[i] == preds[i]:
                correct += 1
        return correct / len(y)
    


if __name__ == '__main__':
# from sklearn.datasets import load_iris as load_data
    from sklearn.datasets import load_iris as load_data
    dataset = load_data()
    
    X = dataset.data
    y = dataset.target
    model = CustomNaiveBayes(laplace_smoothing=True)
    model.fit(X, y)
    preds = model.predict(X)
    accuracy = model.accuracy(y, preds)
    print(f'Accuracy: {accuracy} or {accuracy * 100}%')