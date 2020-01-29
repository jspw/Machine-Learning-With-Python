#importing data
dataset = read.csv("Data.csv")

#deal with the missing data
dataset$Age = ifelse(is.na(dataset$Age),
                     ave(dataset$Age,FUN = function(x) mean(x,na.rm = TRUE)),
                     dataset$Age)
#deal with the missing data
dataset$Salary = ifelse(is.na(dataset$Salary),
                     ave(dataset$Salary,FUN = function(x) mean(x,na.rm = TRUE)),
                     dataset$Salary)
 
#encode catagorical data
dataset$Country = factor(dataset$Country,levels = c('France','Spain','Germany'),
                         labels = c(1,2,3))
dataset$Purchased = factor(dataset$Purchased,
                            levels= c('No','Yes'),
                            labels = c(0,1))
#spliting the dataset into training set and test set
#install.packages('caTools')   #install 'catools' library
#library(caTools) #active the toos

set.seed(123)
split = sample.split(dataset$Purchased,SplitRatio = 0.8) 
#ratio set for dependent variable purchase
#this will return true and false



training_set = subset(dataset,split==TRUE) #if true then set as trainning set
test_set = subset(dataset,split==FALSE) #if false then test set


#scalling data
training_set[,2:3]=scale(training_set[,2:3])
test_set[,2:3]=scale(test_set[,2:3])








