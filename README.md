# Practical Machine Learning: Prediction Assignment Writeup

April 21, 2018 by IoriOikawa





The version of R and the platform is as follows:

```
R version 3.4.4 (2018-03-15) -- "Someone to Lean On"
Copyright (C) 2018 The R Foundation for Statistical Computing
Platform: x86_64-apple-darwin15.6.0 (64-bit)
[R.app GUI 1.70 (7507) x86_64-apple-darwin15.6.0]
```

## Step 1 - Briefing and Cleaning the Data

First, take a quick look at the data,

```R
> training <- read.csv("pml-training.csv", na.strings = c("NA", "#DIV/0!", ""))
> testing  <- read.csv("pml-testing.csv",  na.strings = c("NA", "#DIV/0!", ""))
> str(training, list.len=15)
```

and after glanced the data frame as follows,

```R
'data.frame':	19622 obs. of  160 variables:
 $ X                       : int  1 2 3 4 5 6 7 8 9 10 ...
 $ user_name               : Factor w/ 6 levels "adelmo","carlitos",..: 2 2 2 2 2 2 2 2 2 2 ...
 $ raw_timestamp_part_1    : int  1323084231 1323084231 1323084231 1323084232 1323084232 1323084232 1323084232 1323084232 1323084232 1323084232 ...
 $ raw_timestamp_part_2    : int  788290 808298 820366 120339 196328 304277 368296 440390 484323 484434 ...
 $ cvtd_timestamp          : Factor w/ 20 levels "02/12/2011 13:32",..: 9 9 9 9 9 9 9 9 9 9 ...
 $ new_window              : Factor w/ 2 levels "no","yes": 1 1 1 1 1 1 1 1 1 1 ...
 $ num_window              : int  11 11 11 12 12 12 12 12 12 12 ...
 $ roll_belt               : num  1.41 1.41 1.42 1.48 1.48 1.45 1.42 1.42 1.43 1.45 ...
 $ pitch_belt              : num  8.07 8.07 8.07 8.05 8.07 8.06 8.09 8.13 8.16 8.17 ...
 $ yaw_belt                : num  -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 ...
 $ total_accel_belt        : int  3 3 3 3 3 3 3 3 3 3 ...
 $ kurtosis_roll_belt      : num  NA NA NA NA NA NA NA NA NA NA ...
 $ kurtosis_picth_belt     : num  NA NA NA NA NA NA NA NA NA NA ...
 $ kurtosis_yaw_belt       : logi  NA NA NA NA NA NA ...
 $ skewness_roll_belt      : num  NA NA NA NA NA NA NA NA NA NA ...
  [list output truncated]
```

take a particularly look at *classe* which is the variables need to predict:

```R
> table(training$classe)
```

```R
   A    B    C    D    E 
5580 3797 3422 3216 3607 
```

```R
> prop.table(table(training$user_name, training$class), 1)
```

```R
                   A         B         C         D         E
  adelmo   0.2993320 0.1993834 0.1927030 0.1323227 0.1762590
  carlitos 0.2679949 0.2217224 0.1584190 0.1561697 0.1956941
  charles  0.2542421 0.2106900 0.1524321 0.1815611 0.2010747
  eurico   0.2817590 0.1928339 0.1592834 0.1895765 0.1765472
  jeremy   0.3459730 0.1437390 0.1916520 0.1534392 0.1651969
  pedro    0.2452107 0.1934866 0.1911877 0.1796935 0.1904215
```



```R
> prop.table(table(training$classe))
```

```R
        A         B         C         D         E 
0.2843747 0.1935073 0.1743961 0.1638977 0.1838243 
```
Based on the above information, a brief data clean-up could be applied to the raw data frame.

First, remove columns 1 to 6, which are there just for information for reference purposes...

```R
> training <- training[, 7:160]
> testing <- testing[, 7:160]
> is_data <- apply(!is.na(training), 2, sum) > 19621
> training <- training[, is_data]
> testing <- testing[, is_data]
```

Then, remove the columns with too many NAs.

```R
> is_data  <- apply(!is.na(training), 2, sum) > 19621  # which is the number of observations
> training <- training[, is_data]
> testing  <- testing[, is_data]
```

Before moving forward with data analysis,
the training set should be split into two parts for cross validation,
anyway, caret should be on stage:

```R
> library(caret)
Loading required package: lattice
Loading required package: ggplot2
```

This time, I got randomly subsampled 60% of the set for training purposes / model building,
while the left 40% will be used only for testing, evaluation and accuracy measurements.

```R
> set.seed(19950222)
> inTrain <- createDataPartition(y = training$classe, p = 0.60, list = FALSE)
> train1 <- training[inTrain,]
> train2 <- training[-inTrain,]
```

At this stage, *train1* is the training data set (with 11776 observations, about 60% of the training data set), 
and *train2* is the testing data set (with 7846 observations, about 40% of the training data set). 

```R
> dim(train1)
[1] 11776    54
> dim(train2)
[1] 7846   54
```

The dataset *train2* will never be looked at, and will be used only for accuracy measurements.



## Step 2 - Manipulating and Digging 

To find out the representive variable out of 53 covariates, their relative importance could be observed by  using randomForest() rather than the caret package.

```R
> library(randomForest)
randomForest 4.6-14
Type rfNews() to see new features/changes/bug fixes.

Attaching package: ‘randomForest’

The following object is masked from ‘package:ggplot2’:

    margin
```

This procedure is conducted mostly for speed purposes, because at this time, the number of trees to use in caret can't be specified.

```R
> set.seed(233333)
> fitModel <- randomForest(classe~., data=train1, importance=TRUE, ntree=100)
```

And, plotting data importance can be displayed using varImpPlot():

```R
> varImpPlot(fitModel)
```

![螢幕快照 2018-04-20 14.18.08](/Users/atorsirakawa/Desktop/Screen Shots/螢幕快照 2018-04-20 14.18.08.png)

Using the Accuracy and Gini graphs above, the top 10 variables can be listed for model building now.
If the accuracy of the resulting model is acceptable, limiting the number of variables is a good idea to ensure readability and interpretability of the model.
As it in here, a model with 10 parameters is certainly much friendly than 53 parameters.

Our 10 covariates are: *yaw_belt*, *roll_belt*, *num_window*, *pitch_belt*, *magnet_dumbbell_y*, *magnet_dumbbell_z*, *pitch_forearm*, *accel_dumbbell_y*, *roll_arm*, and *roll_forearm*.

Let’s analyze the correlations between these 10 variables. The following code calculates the correlation matrix, replaces the 1s in the diagonal with 0s, 

```R
> correl = cor(train1[,c("yaw_belt","roll_belt","num_window","pitch_belt","magnet_dumbbell_z","magnet_dumbbell_y","pitch_forearm","accel_dumbbell_y","roll_arm","roll_forearm")])
> diag(correl) <- 0
```

and outputs which variables have an absolute value correlation above 75%:

```R
> which(abs(correl)>0.75, arr.ind=TRUE)
          row col
roll_belt   2   1
yaw_belt    1   2
> cor(train1$roll_belt, train1$yaw_belt)
[1] 0.815411
```

So, I think we may have some problems with *roll_belt* and *yaw_belt*,
because they have a high correlation (above 75%) with each other.

These two variables are on top of the Accuracy and Gini graphs, and it may seem scary to eliminate one of them. Let’s be bold and without doing any PCA analysis, we eliminate *yaw_belt* from the list of 10 variables and concentrate only on the remaining 9 variables.

By re-running the correlation script above (eliminating *yaw_belt*) and outputting max(correl), we find that the maximum correlation among these 9 variables is 50.57% so we are satisfied with this choice of relatively independent set of covariates.

We can identify an interesting relationship between *roll_belt* and *magnet_dumbbell_y*:

```R
> qplot(roll_belt, magnet_dumbbell_y, colour = classe, data = train1)
```

![螢幕快照 2018-04-20 17.00.31](/Users/atorsirakawa/Desktop/Screen Shots/螢幕快照 2018-04-20 17.00.31.png)

and another peek...

```R
> qplot(roll_belt, magnet_dumbbell_y, colour = classe, data = train2)
```

![螢幕快照 2018-04-20 17.00.48](/Users/atorsirakawa/Desktop/Screen Shots/螢幕快照 2018-04-20 17.00.48.png)

This graph suggests that we could probably categorize the data into groups based on *roll_belt* values.

Incidentally, a quick tree classifier selects *roll_belt* as the first discriminant among all 53 covariates (which explains why we have eliminated *yaw_belt* instead of *roll_belt*, and not the opposite: it is a “more important” covariate):

```R
> library(rpart.plot)
Loading required package: rpart
> fitModel <- rpart(classe~., data=train1, method="class")
> prp(fitModel)
```

And here is the decision tree:

![螢幕快照 2018-04-20 17.08.03](/Users/atorsirakawa/Desktop/Screen Shots/螢幕快照 2018-04-20 17.08.03.png)

Useful it does, yet we will not investigate by tree classifiers from no on,
for the Random Forest algorithm will prove the very satisfactory.



## Step 3 - Modeling

It's time to create the model with RandomForest package, and the train() function from the caret package.
We are using 9 variables out of the 53 as model parameters. These variables were among the most significant variables generated by an initial Random Forest algorithm, and are *roll_belt*, *num_window*, *pitch_belt*, *magnet_dumbbell_y*, *magnet_dumbbell_z*, *pitch_forearm*, *accel_dumbbell_y*, *roll_arm*, and *roll_forearm*. These variable are relatively independent as the maximum correlation among them is 50.57%.
We are using a 2-fold cross-validation control. This is the simplest k-fold cross-validation possible and it will give a reduced computation time. Because the data set is large, using a small number of folds is justified.

```R
> set.seed(233333)
> fitModel <- train(classe~roll_belt+num_window+pitch_belt+magnet_dumbbell_y+magnet_dumbbell_z+pitch_forearm+accel_dumbbell_y+roll_arm+roll_forearm,
+  data=train1,
+  method="rf",
+  trControl=trainControl(method="cv",number=2),
+  prox=TRUE,
+  verbose=TRUE,
+  allowParallel=TRUE)
```

The above line of code required more than 4 minutes,
although the result may be the same, thinking of time,
different trees was tried after the first try.

But in this write-up, we go as straight, by just saving this model and go on:

```R
> saveRDS(fitModel, "modelRF.Rds")
> fitModel <- readRDS("modelRF.Rds")
# We can later use this tree, by allocating it directly to a variable using the command above.
```

We can use caret’s confusionMatrix() function applied on *train2* (the test set) to get an idea of the accuracy:

```R
> predictions <- predict(fitModel, newdata = train2)
> confusionMat <- confusionMatrix(predictions, train2$classe)
> confusionMat
Confusion Matrix and Statistics

          Reference
Prediction    A    B    C    D    E
         A 2230    4    0    0    0
         B    2 1510    2    0    1
         C    0    4 1366    7    0
         D    0    0    0 1279    3
         E    0    0    0    0 1438

Overall Statistics
                                          
               Accuracy : 0.9971          
                 95% CI : (0.9956, 0.9981)
    No Information Rate : 0.2845          
    P-Value [Acc > NIR] : < 2.2e-16       
                                          
                  Kappa : 0.9963          
 Mcnemar's Test P-Value : NA              

Statistics by Class:

                     Class: A Class: B Class: C Class: D Class: E
Sensitivity            0.9991   0.9947   0.9985   0.9946   0.9972
Specificity            0.9993   0.9992   0.9983   0.9995   1.0000
Pos Pred Value         0.9982   0.9967   0.9920   0.9977   1.0000
Neg Pred Value         0.9996   0.9987   0.9997   0.9989   0.9994
Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
Detection Rate         0.2842   0.1925   0.1741   0.1630   0.1833
Detection Prevalence   0.2847   0.1931   0.1755   0.1634   0.1833
Balanced Accuracy      0.9992   0.9970   0.9984   0.9970   0.9986
```

After examined the overall statistics, I think this 99.71% accuracy is good enough.
So this model with 9 covariates is very promising, if it passed the test of out-of-sample error rate.

As for the above, the *train2* test set was picked out and left over from variable selection, training and optimizing by the Random Forest algorithm. Therefore I can expect that this testing subset could give an unbiased estimate of the Random Forest algorithm’s prediction accuracy (99.71% as calculated above). 

The Random Forest’s out-of-sample error rate is derived by 100% - Accuracy = 0.293%, 
which can be calculated directly by the following code:

```R
missClass = function(values, predicted) {
+   sum(predicted != values) / length(values)
+ }
> OOS_errRate = missClass(train2$classe, predictions)
> OOS_errRate
[1] 0.00293143
```

With averything has been set, it's time to answer Coursera’s challenge, by predicting the 20 observations in *testing* (recall that *testing* corresponds to the data set pml-testing.csv) to the quiz.



## Step 4 - Coursera Quiz

We predict the classification of the 20 observations of the *testing* data set for Coursera’s “Course Project: Submission” challenge page:

```R
> predictions <- predict(fitModel, newdata=testing)
> testing$classe <- predictions
```

A .CSV file is created with all the results, presented in two columns (named *problem_id* and *classe*) and 20 rows of data. And another group of .TXT files were uploaded one by one to the Coursera website (the 20 files created are called problem_1.txt to problem_20.txt, and you can find them in the folder under this repo):

```R
submit <- data.frame(problem_id = testing$problem_id, classe = predictions)
> write.csv(submit, file = "coursera-submission.csv", row.names = FALSE)
> answers = testing$classe
> write_files = function(x){
+   n = length(x)
+   for(i in 1:n){
+     filename = paste0("problem_",i,".txt")
+     write.table(x[i], file=filename, quote=FALSE, row.names=FALSE, col.names=FALSE)
+   }
+ }
> write_files(answers)
```

And the result, with no surprise, is a solid 20/20:

![螢幕快照 2018-04-21 12.38.57](/Users/atorsirakawa/Desktop/Screen Shots/螢幕快照 2018-04-21 12.38.57.png)

