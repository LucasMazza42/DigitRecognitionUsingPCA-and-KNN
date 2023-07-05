# DigitRecognitionUsingPCA-and-KNN

For this small project, I used sklearn's KNN and PCA done with numpy to create a model for digit recognition. The decision to use PCA in addition to KNN was to improve computation time, and it also improved accuracy over just the use of KNN. Another note here is that I coded PCA from scratch. 

# Here is a quick summary: 
The choice to use 50 components was used due to a drop off in explained variance after 50 as shown here: <img width="629" alt="Screenshot 2023-07-05 at 2 42 21 PM" src="https://github.com/LucasMazza42/DigitRecognitionUsingPCA-and-KNN/assets/47802441/f7beb466-d676-4014-b17e-fd2c27339e23">


The model was able to achieve about 97% accuracy in training and a 95 percent accuracy in validation, which increases by about 1-2% over purely using KNN and cuts the computation time by about one second. 

# The main goal of PCA
- The main goal of PCA is to reduce the dimensionality of the dataset, while still keeping the information that is important.
- This can improve runtime and accuracy as it did in my example. 

# Understanding our dataset: 

- Each row represents a training example or a number.
- The label column is the number that the row represents.
- All column values after the label value represent the intensity of the pixel.
- Each image is 28x28 pixels
        - This makes the dimensionality of our dataset large increasing computation cost and time! 

# Detailed summary of programming:
- This is a detailed explanation of how I programmed PCA from scratch.

        `def __init__(self, n_components: int = None) -> None:
                self.n_components = n_components`
- A super important part before we begin is to normalize the data which can be done by scaling the matrix by the mean of the dataset: 
        `def transform(self, X: np.array) -> np.array: 
                #normalize
                X=X-np.mean(X)
                X_pca = np.dot(X, self.components)
                return X_pca`
- Now we can ensure that the data will not have scaling issues i.e. the values don't really matter 1 means the same as 5. 
- This part of the code initializes how many components we are going to be working with
- In PCA, this determines the dimensions of the dataset that we are working with.


`def findCovMatrix(self, X) -> np.array:
        covMatrix = np.cov(X.T)
        return covMatrix`

- So in order to do PCA, we first need to find the covariance matrix.
- This matrix will tell us more about the variance between variables (columns) and their relationship to each other.
- A good way to show this is with a heatmap: 
<img width="883" alt="Screenshot 2023-07-04 at 5 27 02 PM" src="https://github.com/LucasMazza42/DigitRecognitionUsingPCA-and-KNN/assets/47802441/0200cc7f-c437-44b8-8717-6b9ab2b80697">

- Next, we have the computation of the eigenvectors and eigenvalues:
          - The eigenvalues represent the amount of variance captured by each component.
          - Thus, higher eigenvalues mean that that specific component captures more variance in the data.
          - The eigenvectors with the highest eigenvalue represent our principal components.
- This process can be represented by this code:
  `def eigenValueVec(self, covMatrix)-> tuple: 
        eigen_values, eigen_vectors = np.linalg.eig(covMatrix)
        return (eigen_values.real, eigen_vectors.real)`
- Next, we have the explained variance ratio:
  - the explained variance ratio shows how much of the variance is explained by each principal.
  - This helps us how many components we want to use!
  - In our code, we will find this by taking each eigen_value for each vector and dividing it by the total variance in the dataset
  - This is where the diagram comes from before, and shows why I choose to use n_components = 50

        `def explained_varience_ratio(self, X)->None: 
                
                covarienceMat = self.findCovMatrix(X)
                eigen_values, eigen_vectors = self.eigenValueVec(covarienceMat)
        
                total_egnvalues = sum(eigen_values)
                explained_variance_ratio = [(i/total_egnvalues) for i in eigen_values]
               
                explained_variance_ratio = explained_variance_ratio[0:50]
                #we can use 50 components here
        
                #plot the explained variance
                plt.plot(explained_variance_ratio,
                     label='Explained Variance Ratio')
        
                plt.xlabel('Principal Component')
                plt.ylabel('Explained Variance Ratio')
                plt.legend()
                plt.show() ` 
- Now we need to fit the data with 50 components
        - NOTE: using more components might improve accuracy, but we are looking to balance the trade-off of accuracy vs speed. 
- We are going to repeat the process from before, but now we are going to sort the eigen_values list in order to find the eigen_values that explain the most amount of variance in the data.
- Now we are finally ready to use the K-means classifier to group similar data points! 



