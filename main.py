import numpy as np 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import pandas as pd
import os
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


class PCA(object): 

    def __init__(self, n_components: int = None) -> None:
        self.n_components = n_components
    #We need to split the data into an X (features) and Y labeled data set for training
  
    def findCovMatrix(self, X) -> np.array:
        covMatrix = np.cov(X.T)
        return covMatrix

    def eigenValueVec(self, covMatrix)-> tuple: 
        eigen_values, eigen_vectors = np.linalg.eig(covMatrix)
        return (eigen_values.real, eigen_vectors.real)
    def explained_varience_ratio(self, X)->None: 
        
        covarienceMat = self.findCovMatrix(X)
        eigen_values, eigen_vectors = self.eigenValueVec(covarienceMat)

        total_egnvalues = sum(eigen_values)
        explained_variance_ratio = [(i/total_egnvalues) for i in eigen_values]
       
        explained_variance_ratio = explained_variance_ratio[0:100]
        #we can use 50 components here

        #plot the explainedVarience
        plt.plot(explained_variance_ratio,
             label='Explained Variance Ratio')

        plt.xlabel('Principal Component')
        plt.ylabel('Explained Variance Ratio')
        plt.legend()
        plt.show() 

    
    
    def fit(self, X) -> None: 
        #get out matrices
        covarienceMat = self.findCovMatrix(X)
        eigen_values, eigen_vectors = self.eigenValueVec(covarienceMat)
        #sort
        indexes = eigen_values.argsort()[::-1]
        #get the right eigen values
        eigen_values = eigen_values[indexes]
        #sort eigen vectors based of the eigen values
        eigen_vectors = eigen_vectors[:,indexes]
        self.components = eigen_vectors[:, :self.n_components]
        

    def transform(self, X: np.array) -> np.array: 
        #normalize
        X=X-np.mean(X)
        X_pca = np.dot(X, self.components)
        return X_pca

def load_mnist_data(path='mnist_train_new.csv') -> tuple:
    df = pd.read_csv(path)

    target = 'label'
    X = df.drop(columns=target).values
    y = df[target].values

    print(
        f'Loaded data from {path}:\n\r X dimension: {X.shape}, y dimension: {y.shape}')

    return np.array(X), np.array(y)
def split(data) -> tuple: 
    
        X_train, X_val, y_train, y_val = train_test_split(
        X, y, shuffle=True, test_size=0.3, random_state= 1)
        
        return (X_train, X_val, y_train, y_val)
def performKNN(X_train_pca, X_val_pca, y_train, y_val) -> None: 
        
        knn = KNeighborsClassifier()

        knn.fit(X_train_pca, y_train)

        p_train = knn.predict(X_train_pca)
        
        train_acc = sum(y_train == p_train) / len(y_train)

        p_val = knn.predict(X_val_pca)
        val_acc = sum(y_val == p_val) / len(y_val)

        print("TRAIN ACC using KNN and PCA:  " + str(train_acc))
        print("VAL ACC using KNN and PCA:  " + str(val_acc))


data = '/Users/lucasmazza/Desktop/digit-recognizer/mnist_train_sub.csv'


if __name__ == '__main__':
    os.system('cls' if os.name == 'nt' else 'clear')
    X, y = load_mnist_data(os.path.join(
        os.path.dirname(__file__), 'mnist_train_sub.csv'))

    X_train, X_val, y_train, y_val = split(data)
    pca = PCA(n_components=50)
    pca.fit(X)

    X_train_pca = pca.transform(X_train)
    X_val_pca = pca.transform(X_val)

    performKNN(X_train_pca, X_val_pca, y_train,y_val)
    
