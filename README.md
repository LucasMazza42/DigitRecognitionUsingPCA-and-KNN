# DigitRecognitionUsingPCA-and-KNN

For this small project, I used sklearn's KNN and PCA done with numpy to create a model for digit recognition. The decision to use PCA in addition to KNN was to improve computation time, and it also improved accuracy over just the use of KNN. Another note here is the I coded PCA from scratch. 

# Here is a quick summary: 
The choice to use 50 components was used due to a drop off in explained variance after 50 as shown here: <img width="688" alt="Screen Shot 2022-12-03 at 9 43 15 AM" src="https://user-images.githubusercontent.com/47802441/210004567-ccdd9aeb-7dcf-4eeb-9576-004ee7de3856.png">

The model was able to achieve about 97% accuracy in training and a 95 percent accuracy in validation, which increases by about 1-2% over purely using KNN and cuts the computation time by about one second. 
