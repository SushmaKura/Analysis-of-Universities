# Load required libraries
library(caret)
library(pROC)
library(glmnet)
library(Metrics)

######################################################################################################################################
######################################################################################################################################

#Load Data
Data <- file("College.csv", "rt")
College_Data <- read.csv(Data)

######################################################################################################################################
######################################################################################################################################

# Split the data into training and test sets
set.seed(123)
trainIndex <- sample(x = nrow(College_Data), size = nrow(College_Data) * 0.7)
dataTrain <- College_Data[trainIndex,]
dataTest <- College_Data[-trainIndex,]

dataTrain_x <- model.matrix(Grad.Rate ~., dataTrain)[,-1]
dataTest_x <- model.matrix(Grad.Rate ~., dataTest)[,-1]

dataTrain_y <- dataTrain$Grad.Rate
dataTest_y <- dataTest$Grad.Rate

######################################################################################################################################
######################################################################################################################################

#Finding the value for Lambda using Ridge Regression 
set.seed(123)
cvfit <- cv.glmnet(dataTrain_x, dataTrain_y, alpha = 0, nfolds = 10)
plot(cvfit)

# Display the lambda.min value
best_lambda_Train_R <- cvfit$lambda.min
cat("lambda.min for Ridge Regression for Training Model:", best_lambda_Train_R, "\n")

# Fit the final model on the training data
final_model_Train_R <- glmnet(dataTrain_x, dataTrain_y, alpha = 0, lambda = best_lambda_Train_R)

# Display coefficients
options(scipen = 999)
cat("Coefficients for Ridge Regression:\n")
print(coef(final_model_Train_R))

# Display the lambda.1se value for Test set
best_lambda_Test_R <- cvfit$lambda.1se
cat("lambda.1se for Ridge Regression for Testing Model:", best_lambda_Test_R, "\n")

# Fit the final model on the training data
final_model_Test_R <- glmnet(dataTest_x, dataTest_y, alpha = 0, lambda = best_lambda_Test_R)

# Display coefficients
options(scipen = 999)
cat("Coefficients for Ridge Regression:\n")
print(coef(final_model_Test_R))

######################################################################################################################################
######################################################################################################################################

#Finding the value for Lambda using Lasso Regression 
set.seed(123)
cvlasso <- cv.glmnet(dataTrain_x, dataTrain_y, alpha = 1, nfolds = 10)
plot(cvlasso)

# Display the lambda.min value for Training set 
best_lambda_Train_L <- cvlasso$lambda.min
cat("lambda.min for Lasso Regression for Training Model:", best_lambda_Train_L, "\n")

# Fit the final model on the training data
final_model_Train_L <- glmnet(dataTrain_x, dataTrain_y, alpha = 1, lambda = best_lambda_Train_L)

# Display coefficients
options(scipen = 999)
cat("Coefficients for Lasso Regression:\n")
print(coef(final_model_Train_L))

# Display the lambda.1se value for Test set
best_lambda_Test_L <- cvlasso$lambda.1se
cat("lambda.1se for Lasso Regression for Testing Model:", best_lambda_Test_L, "\n")

# Fit the final model on the training data
final_model_Test_L <- glmnet(dataTest_x, dataTest_y, alpha = 1, lambda = best_lambda_Test_L)

# Display coefficients
options(scipen = 999)
cat("Coefficients for lasso Regression:\n")
print(coef(final_model_Test_L))

######################################################################################################################################
######################################################################################################################################

#Coefficient of ols model with no regularization
ols <- lm(Grad.Rate ~., data = dataTrain)
coef(ols)

######################################################################################################################################
######################################################################################################################################

# Make predictions on the test data for Ridge
predictions <- predict(final_model_Test_R, newx = dataTest_x)

# Compute the RMSE
rmse <- rmse(dataTest_y, predictions)
cat("Root Mean Square Error (RMSE) on test data for RIDGE Regression:", rmse, "\n")

# Make predictions on the train data for Ridge
predictions <- predict(final_model_Train_R, newx = dataTrain_x)

# Compute the RMSE
rmse <- rmse(dataTrain_y, predictions)
cat("Root Mean Square Error (RMSE) on train data RIDGE Regression:", rmse, "\n")

# Make predictions on the test data for Lasso
predictions <- predict(final_model_Test_L, newx = dataTest_x)

# Compute the RMSE
rmse <- rmse(dataTest_y, predictions)
cat("Root Mean Square Error (RMSE) on test data for LASSO:", rmse, "\n")

# Make predictions on the train data for lasso
predictions <- predict(final_model_Train_L, newx = dataTrain_x)

# Compute the RMSE
rmse <- rmse(dataTrain_y, predictions)
cat("Root Mean Square Error (RMSE) on train data for LASSO:", rmse, "\n")


######################################################################################################################################
######################################################################################################################################