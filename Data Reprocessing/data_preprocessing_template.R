#importing data
dataset = read.csv("Data.csv")
dataset=dataset[,2:3]

#spliting the dataset into training set and test set
#install.packages('caTools')   #install 'catools' library
#library(caTools) #active the toos

set.seed(123)
split = sample.split(dataset$Purchased,SplitRatio = 0.8) 
#ratio set for dependent variable purchase
#this will return true and false



training_set = subset(dataset,split==TRUE) #if true then set as trainning set
test_set = subset(dataset,split==FALSE) #if false then test set


# #scalling data
# training_set[,2:3]=scale(training_set[,2:3])
# test_set[,2:3]=scale(test_set[,2:3])








