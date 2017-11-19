#==================================Loading Libraries==========================================#
library(kernlab)
library(readr)
library(caret)
library(dplyr)
library(ggplot2)
library(gridExtra)
library(plotly)

#===============================Exploring & Preparing the Data===============================#
#Loading Data Train Data#
mnist_trainData <- read_csv("C:/Users/Mpk1/Desktop/SVM Original/SVM/mnist_train.csv")

colnames(mnist_trainData)[1]<-"PatternRec"
for(i in seq(2,ncol(mnist_trainData),by=1)){colnames(mnist_trainData)[i]<-paste("Colmnist",as.character(i-1),sep = "")}

View(mnist_trainData)

#Loading Data Test Data#
mnist_testData <- read_csv("C:/Users/Mpk1/Desktop/SVM Original/SVM/mnist_test.csv")

colnames(mnist_testData)[1]<-"PatternRec"
for(i in seq(2,ncol(mnist_testData),by=1)){colnames(mnist_testData)[i]<-paste("Colmnist",as.character(i-1),sep = "")}

View(mnist_testData)

#Checking Dimensions of Train data, Structure of dataset
dim(mnist_trainData)
str(mnist_trainData)
head(mnist_trainData)
nrow(mnist_trainData)
names(mnist_trainData)

#Checking Dimensions of Test data, Structure of dataset
dim(mnist_testData)
str(mnist_testData)
head(mnist_testData)
nrow(mnist_testData)
names(mnist_testData)

#summarry of Test and Train Data
summary(mnist_trainData)
summary(mnist_testData)

#Checking NA or Missing values from mnist_testData and mnist_trainData
sapply(mnist_trainData, function(x) sum(is.na(x)))
sapply(mnist_testData, function(x) sum(is.na(x)))

# Splitting the data between mnist_trainData and mnist_testData
mnist_trainData$PatternRec<-factor(mnist_trainData$PatternRec)
mnist_testData$PatternRec<-factor(mnist_testData$PatternRec)
set.seed(100)
train.indic = sample(1:nrow(mnist_trainData), 0.5*nrow(mnist_trainData))
test.indic= sample(1:nrow(mnist_testData), 1.0*nrow(mnist_testData))

train_mnist = mnist_trainData[train.indic, ]
test_mnist = mnist_testData[test.indic, ]

#===========================Model Training for Data=========================================#

#==================================Model_linear=============================================#
Model_linear <- ksvm(PatternRec~ ., data = train_mnist, scale = FALSE, kernel = "vanilladot")
Eval_linear<- predict(Model_linear, test_mnist)
confusionMatrix(Eval_linear,test_mnist$PatternRec)

# Accuracy    : 0.917


# ================Hyperparameter tuning and Cross Validation  - Linear - SVM====================


#______________We use train function from caret package to perform crossvalidation______________#

trainControl <- trainControl(method="cv", number=5)
metric <- "Accuracy"
set.seed(100)
Model_linear

# making a grid of C values. 
grid <- expand.grid(C=seq(1, 5, by=1))

# Performing 5-fold cross validation
fit.svm <- train(PatternRec~., data=train_mnist, method="svmLinear", metric=metric, 
                 tuneGrid=grid, trControl=trainControl)

# Printing cross validation result
print(fit.svm)
# Best tune at C=1, 
# Accuracy = 0.9101596

# Plotting "fit.svm" results
plot(fit.svm)

#**************# Valdiating the model after cross validation on test data*********************#

evaluate_linear_test<- predict(fit.svm, mnist_testData)
plot_ly(x = ~evaluate_linear_test, type = "histogram")
plot_ly(x = ~evaluate_linear_test, type = "box")
confusionMatrix(evaluate_linear_test, mnist_testData$PatternRec)

# Accuracy    - 0.917

#==================================Model_RBF================================================#
Model_RBF <- ksvm(PatternRec~ ., data = train_mnist, kernel = "rbfdot")
Eval_RBF<- predict(Model_RBF, test_mnist)
confusionMatrix(Eval_RBF,test_mnist$PatternRec)

#Accuracy : 0.9616 

trainControl_rbf <- trainControl(method="cv", number=5)
metric <- "Accuracy"
set.seed(100)
Model_RBF

#cost C = 1
#Hyperparameter : sigma =  1.65169854128169e-07 
#Training error :  0.020336 

# Making grid of "sigma" and C values. 
grid_rbf <- expand.grid(.sigma=c(0.025, 0.05), .C=c(0.1,0.5,1,2) )

# Performing 5-fold cross validation
fit.svm <- train(PatternRec~., data=train_mnist, method="svmRadial", metric=metric_rbf, 
                        tuneGrid=grid_rbf, trControl=trainControl_rbf)

# Printing cross validation result
print(fit.svm_rbf)
# Best tune at sigma =  & C=, Accuracy - 

# Plotting model results
plot(fit.svm_rbf)

#================================================================================================
# Checking overfitting - Non-Linear - SVM
#================================================================================================

# Validating the model results on test data

evaluate_rbf<- predict(fit.svm_rbf, mnist_testData)
plot_ly(x = ~evaluate_rbf, type = "histogram")
plot_ly(x = ~evaluate_rbf, type = "box")

confusionMatrix(evaluate_rbf, mnist_testData$PatternRec)
#Accuracy : 0.9616 

#================================================================================================