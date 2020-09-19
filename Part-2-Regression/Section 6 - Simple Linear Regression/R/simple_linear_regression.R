# importing the dataset
dataset = read.csv('Salary_Data.csv')
# dataset = dataset[,2:3]


# Splitting the dataset into training and test sets
library('caTools')
set.seed(123)
split = sample.split(dataset$Salary , SplitRatio = 2/3)
training_set = subset(dataset , split == TRUE)
test_set = subset(dataset , split == FALSE)

# we don't need feature scaling because the lm function will take care of the scaling
# fitting the simple linear regression on the training set
regressor = lm(formula = Salary ~ YearsExperience,
               data = training_set)
# we use summary(regressor) in the console to know some information about the regressor
# *** means that the years of Experience has a great effect on our regression
# the less value of P means that the years of Experience has a great effect on our regression
# predicting the Test set results
y_pred = predict(regressor , newdata = test_set)
# visualizing the training data
# ggplot2 is used to visualize the data
#install.packages('ggplot2')
library(ggplot2)
ggplot() + 
  geom_point(aes(x = training_set$YearsExperience , y = training_set$Salary),
   colour = 'red') +
  geom_line(aes(x = training_set$YearsExperience , y = predict(regressor , newdata = training_set)),
            colour = 'blue') + 
  ggtitle('Salary vs years of experience (training set)') +
  xlab('Years of experience') +
  ylab('Salary')

# visualizing the test data
# here we change the points from the training set to the points of the test set 
# but the regression line we didn't change anything in it because we want the same line that the model is trained on
library(ggplot2)
ggplot() + 
  geom_point(aes(x = test_set$YearsExperience , y = test_set$Salary),
             colour = 'red') +
  geom_line(aes(x = training_set$YearsExperience , y = predict(regressor , newdata = training_set)),
            colour = 'blue') + 
  ggtitle('Salary vs years of experience (test set)') +
  xlab('Years of experience') +
  ylab('Salary')