library(magrittr)
library(car)

# Exploratory analyses on the pokemon dataset 
# Exploring this dataset because I used to watch pokemon and play the video games when I was younger
# Data retrieved from: https://www.kaggle.com/rounakbanik/pokemon 

# Some possible analyses: 
  # Is it possible to build a classifier to identify legendary Pokemon? - done 
  # How does height and weight of a Pokemon correlate with its various base stats?
  # Which pokemon is the strongest overall? Which is the weakest? Fastest? Slowest? - done 
  # Which type is the most likely to be a legendary Pokemon? - done 
  # Create distribution plots to see distribution of all stats  

### Load Dataset

setwd("C:/Users/Parabhjot/Dropbox/R Studio/r_pokemon_exploratory_analyses")

pokemon <- read.csv("data/pokemon.csv")

### Exploratory analyses 
# Can open and look at dataset anytime since it is quite small 

str(pokemon)

# See if there are any columns with nulls 

na_count <-sapply(pokemon, function(y) sum(length(which(is.na(y)))))
na_count <- data.frame(na_count)

# columns with nulls values are: 
  # 1) percentage_male - 98 observations
  # 2) height_m - 20 observations
  # 3) weight_kg - 20 observations

### Find pokemon with highest/lowest 

# Use colnames function to find variables of interest 
colnames(pokemon)

  # 1) attack 
pokemon[which.max(pokemon$attack),]$name
pokemon[which.min(pokemon$attack),]$name

  # 2) base_egg_steps
pokemon[which.max(pokemon$base_egg_steps),]$name
pokemon[which.min(pokemon$base_egg_steps),]$name

  # 3) base_total
pokemon[which.max(pokemon$base_total),]$name
pokemon[which.min(pokemon$base_total),]$name

  # 4) defence 
pokemon[which.max(pokemon$defense),]$name
pokemon[which.min(pokemon$defense),]$name

  # 5) sp_attack 
pokemon[which.max(pokemon$sp_attack),]$name
pokemon[which.min(pokemon$sp_attack),]$name

  # 6) sp_defence 
pokemon[which.max(pokemon$sp_defense),]$name
pokemon[which.min(pokemon$sp_defense),]$name

  # 7) speed 
pokemon[which.max(pokemon$speed),]$name
pokemon[which.min(pokemon$speed),]$name

  # 8) weight_kg
pokemon[which.max(pokemon$weight_kg),]$name
pokemon[which.min(pokemon$weight_kg),]$name

### Find all legendary pokemon 
unique(sort(pokemon$is_legendary))
legendary_pokemon <- pokemon[(pokemon$is_legendary==1),]

### Logistic Regression to predict whether a pokemon is legendary or not 

# Used this article for a reference on step-wise logistic regression: 
  # http://www.sthda.com/english/articles/36-classification-methods-essentials/150-stepwise-logistic-regression-essentials-in-r/   

# Load libraries 

pacman::p_load(Mass, Caret, psych, DT, Rcmdr)

# Create reduced dataset dropping variables that aren't useful/have nulls
  # 1) japanese_name
  # 2) name
  # 3) percentage_male 
  # 4) pokedex_number 
  # 5) generation - irrelevant because every generation corressponds with a new arc in the show and each generation has it's own legendary pokemon 
# Also convert response variable to factor 

pokemon <- within(pokemon, {is_legendary <- factor(is_legendary)})

pokemon_reduced <- pokemon[ , !(names(pokemon) %in% "japanese_name")]
pokemon_reduced <- pokemon_reduced[ , !(names(pokemon_reduced) %in% "percentage_male")]
pokemon_reduced <- pokemon_reduced[ , !(names(pokemon_reduced) %in% "pokedex_number")]
pokemon_reduced <- pokemon_reduced[ , !(names(pokemon_reduced) %in% "generation")]
pokemon_reduced <- pokemon_reduced[ , !(names(pokemon_reduced) %in% "height_m")]
pokemon_reduced <- pokemon_reduced[ , !(names(pokemon_reduced) %in% "weight_kg")]

# Creating training/test split 

set.seed(123)
training.samples <- pokemon_reduced$is_legendary %>% createDataPartition(p = 0.7, list = FALSE)
df_train <- pokemon_reduced[training.samples, ]
df_test <- pokemon_reduced[-training.samples, ]

# Fit the model
step.model <- glm(is_legendary ~ 
                    attack + 
                    base_egg_steps + 
                    base_total + 
                    capture_rate + 
                    defense + 
                    experience_growth + 
                    sp_attack + 
                    sp_defense + 
                    speed
                  ,
                  data = df_train, family = binomial, maxit = 100) %>%
  stepAIC(trace = FALSE)

# Summarize the final selected model
summary(step.model)

# Make predictions
probabilities <- predict(step.model, df_test, type = "response")
predicted.classes <- ifelse(probabilities > 0.5, 1, 0)

# Prediction accuracy
observed.classes <- df_test$is_legendary
mean(predicted.classes == observed.classes)

# Add predictions to df
df_test$predicted <- predicted.classes

# Odds ratio 
exp(cbind(OR = coef(step.model), confint(step.model)))

# Confusion Matrix & Pearson's Chi-Squared Test
local({
  .Table <- xtabs(~is_legendary+predicted, data=df_test)
  cat("\nFrequency table:\n")
  print(.Table)
  cat("\nRow percentages:\n")
  print(rowPercents(.Table))
  .Test<- chisq.test(.Table, correct=FALSE)
  print(.Test)
  print(fisher.test(.Table))
})

### Random Forest Classifier 
# Reference article: https://www.r-bloggers.com/how-to-implement-random-forests-in-r/  

# Load libraries 
pacman::p_load(randomForest, caret, e1071)

# Create a Random Forest model with default parameters

model1 <- randomForest(is_legendary ~ 
                         attack + 
                         base_egg_steps + 
                         base_total + 
                         capture_rate + 
                         defense + 
                         experience_growth + 
                         sp_attack + 
                         sp_defense + 
                         speed
                       ,
                       data = df_train, importance = TRUE)
model1

# We can fine tune parameters of Random Forest model by:  
# 1. Changing the number of trees (ntree)   
# 2. Changing the number of variables randomly sampled at each stage (mtry)  

# Ntree: Number of trees to grow. This should not be set to too small a number, to ensure that every input row gets predicted at least a few times.  
# Mtry: Number of variables randomly sampled as candidates at each split. The default values are different for classification (sqrt(p) where p is number of variables in x) and regression (p/3)  

model2 <- randomForest(is_legendary ~ 
                         attack + 
                         base_egg_steps + 
                         base_total + 
                         capture_rate + 
                         defense + 
                         experience_growth + 
                         sp_attack + 
                         sp_defense + 
                         speed
                       ,
                       data = df_train, ntree = 100, mtry = 6, importance = TRUE)
model2

# Test on the the train set 
predTrain <- predict(model2, df_train, type = "class")
mean(predTrain == df_train$is_legendary)  
table(predTrain, df_train$is_legendary)

# Test on the testing set  
predValid <- predict(model2, df_test, type = "class")
mean(predValid == df_test$is_legendary)   
table(predValid,df_test$is_legendary)

# Variable importance of predictor variables 

varImpPlot(model2)

# Base egg steps are the most important predictor variable. This aligns with the pokemon show as well 
# Capture_rate and base_total are also important predictor variables. These make sense as well 

# Using a loop to identify the right mtry for model and input into the model above.   
a=c()
for (i in 1:20) {
  model3 <- randomForest(is_legendary ~ 
                           attack + 
                           base_egg_steps + 
                           base_total + 
                           capture_rate + 
                           defense + 
                           experience_growth + 
                           sp_attack + 
                           sp_defense + 
                           speed
                         ,
                         data = df_train, ntree = 1500, 
                         mtry = i, importance = TRUE)
  predValid <- predict(model3, df_test, type = "class")
  a[i] = mean(predValid == df_test$is_legendary)
}
a

plot(1:20,a)
