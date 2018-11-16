options(java.parameters = "-XX:+UseG1GC") # Should avoid java gc overhead
options(java.parameters = "-Xmx16000m")

#install.packages(c("ggplot2","OpenML","lm","xlConnect","farff", "RWeka","rJava","glmnet","ranger","xgboost","mlr","liquidSVM"))

library(plyr)
library(farff)
library(OpenML)
library(rJava)
library(ggplot2)
library(mlr)

library(liquidSVM)
library(RWeka)
library(glmnet)
library(ranger)

###############################################################################################################
## INFORMATION: Der Code darf im ersten Schritt nur bis Zeile 785 ausgefuehrt werden (Stelle ist markiert). 
## Dann muss die R-Session beendet und die librarys neu geladen werden. Da sonst der heap zu voll ist. 

##################################################
##      insert path to the cloned repository 
###############################################

path <- "C:/Users/ru76beb/Documents/openml-bench-regr"
saveOMLConfig(apikey = "f825394339f0ff5facce111e26bf0933", arff.reader = "RWeka", overwrite=TRUE)

#################################################################################################################################################
#----------------------------------------------- Alle Regressiondatensaetze aus OpenML importieren -----------------------------------------------
#################################################################################################################################################

# Nur Tasks vom Typ Supervised Regression

# Nur Tasks, bei denen eine 10-fold Crossvalidation durchfuehrbar ist
# Nur Tasks ohne fehlende Werte

reg = listOMLTasks(task.type = "Supervised Regression", estimation.procedure = "10-fold Crossvalidation", number.of.missing.values = 0)

# 377 Tasks
dim(reg)

###################################################################################################################################
#------------------------------------------------  Nur Tasks mit eineindeutiger data.id aufnehmen ---------------------------------
###################################################################################################################################

# ordne die Tasks nach der data.id
reg = reg[order(reg$data.id),]

# erstelle Vektor aus Wahrheitswerten fuer jeden Datensatz
logic = rep(TRUE, nrow(reg))

# Wenn die Data.id zum Vorgaenger identisch ist setze FALSE in Vektor
for(i in 2:nrow(reg))
  if(reg$data.id[i] == reg$data.id[i-1]) logic[i] = FALSE

# Nur Tasks mit eineindeutiger Data.id bleiben bestehen 
reg = reg[logic,]

# 348 Tasks
dim(reg)

#########################################################################################################################################
#-------------------------------------------------- Nur Tasks mit eineindeutigem Namen --------------------------------------------------
#########################################################################################################################################

# Schreibe doppelte Namen in Vektor 
doppelt = names(sort(table(reg$name)[table(reg$name) > 1]))

#erstelle Dataframe reg reduziert auf Namen die doppelt vorkommen 
doppelt = reg[reg$name %in% doppelt, ]

#Ordne die Tasks nach dem Namen 
doppelt = doppelt[order(doppelt$name), ]

#Zeige sortierten dateframe an 
doppelt[, c(1,3,4,5,7,9,10,11,14,15)]

## Wenn der Name schon vorkommt, dann wird er aus den Tasks geloescht
# iterator fuer Tasks die bleiben 
stay <- 1

#Position an der der naechste Task angehaengt wird, wenn er bleiben soll 
stayPosition <- 2

for(i in 1:nrow(doppelt)){
  if(i >1){
    #Ersteneinmal bleibt der Task, bis ueberprueft ist ob der Name schon auftaucht
    bleiben <- TRUE
    y <- i-1
    #checke ob der Name schon vorkommt:
    # vergleiche der Reihe nach alle einzeln mit den vorherigen, 
    # taucht der Name auf, dann nimm den 2. nicht mit auf 
    for(a in 1:y){
      #wenn Name schon vorkommt nicht mitaufnehmen
      if(doppelt[a,4]==doppelt[i,4]){
        bleiben <- FALSE
      }  
    }
    #Wenn er nicht schon vorkommt, nimm seine Position in doppelt in den Vektor mit auf 
    if(bleiben == TRUE){
      stay[stayPosition] <- i
      # Der Vektor wurde erweitert, verschiebe Position des naechsten, der hinzukommt
      stayPosition <- stayPosition +1
    }
    print(stay)
    print(i)
  }
}

# Ids der Tasks, die nicht bleiben sollen 
raus <- doppelt[-stay,]$task.id
raus

# reduziere die Tasks auf die, die keine doppelten Namen haben 
reg = reg[!(reg$task.id %in% raus),]
reg = reg[order(reg$name), ]

## 342 Tasks
dim(reg)

###############################################################################################################################
###---------------------------------------  Entferne Tasks, die weniger als 4 Features haben
###############################################################################################################################

reg <- reg[reg$number.of.features > 4, ]
#309 Tasks 
dim(reg)

############################################################################################################################################
#-------------------------------------------------   Nur Datensaetze mit weniger als 100000 Beobachtungen und mehr als 150
###########################################################################################################################################

reg <- reg[reg$number.of.instances > 150,]
# 185 Tasks
dim(reg)

##########################################################################################################################################
#############################             manuelle Selektion der Datensaetze                         ######################################
##########################################################################################################################################
#### -------------------------------------------------------------------------------------------------------------------------############

######################################################################################################################################
##############################            Datenaetze aussortieren, die nicht mehr als 20 Auspraegungen im Targetfeature haben ###########
##############################################################################################################################

### Datensatz mnist_rotation hat nur 10 Auspraegungen im Targetfeature 
reg <- reg[!(reg$task.id == 167218),]

### datensatz chscase_foot hat nur 3 Auspraegungen im Targetfeature
reg <- reg[!(reg$task.id == 5012),]

### a3a a4a a5a a6a a7a a8a a9a, haben nur 2 Auspraegungen -> Kriterium von mindestens 20 Auspraegungen verletzt
reg <- reg[!(reg$task.id %in% c(7564,7565,7566,7567,7568,7569,7570)),]

###   analcatdata_supreme  hat nur 10 Auspraegungen  
reg <- reg[!(reg$task.id %in% c(4832)),]

## german.numer hat nur 2 Auspraegungen 
reg <- reg[!(reg$task.id %in% c(7575)),]

### ozone-level hat nur 2 Auspraegungen im Targetfeature 
reg <- reg[!(reg$task.id %in% c(4771)),]

####   quake hat nur 12 Auspraegungen im Targetfeature 
reg <- reg[!(reg$task.id %in% c(2300)),]

### real-sim hat nur 2 Auspraegungen im Targetfeature 
reg <- reg[!(reg$task.id %in% c(12720)),]

### satellite-image hat nur 6 Auspraegungen im Targetfeature 
reg <- reg[!(reg$task.id %in% c(4708)),]

#### SensIT-Vehicle-Combined   hat nur 3 Auspraegungen im Targetfeature 
reg <- reg[!(reg$task.id %in% c(12733)),]

##### sensory hat nur 11 Auspraegungen im targetfeature
reg <- reg[!(reg$task.id %in% c(4871)),]

####  slashdot hat nur 2 Auspraegungen im Targetfeature 
reg <- reg[!(reg$task.id %in% c(146815)),]

### socmob zwar nur numerisch kodiert und eigentlich kategorial / ordinal 
reg <- reg[!(reg$task.id %in% c(4866)),]

### splice hat nur 3 nominelle Auspraegungen um Targetfeature
reg <- reg[!(reg$task.id %in% c(12721)),]

## svmguide1 und svmguide3 haben jeweils nur 2 Auspraegungen 
reg <- reg[!(reg$task.id %in% c(7572, 12730)),]

### ijcnn, IMDB.drama, poker,rcv1.binary, COMET_MC haben nur 2 Auspraegungen im targetfeature 
reg <- reg[!(reg$task.id %in% c(12718,4707,10102,12719,14949)),]

### ICU hat nur 3 Auspraegungen im Targetfeature
reg <- reg[!(reg$task.id %in% c(5038)),]

### news20 hat nur 20 Auspraegungen im targetfeature 
reg <- reg[!(reg$task.id %in% c(12734)),]

### pol hat nur 11 Auspraegungen im Targetfeature 
reg <- reg[!(reg$task.id %in% c(2292)),]

### ESL hat nur 9 Auspraegungen im targefeature 
reg <- reg[!(reg$task.id %in% c(5021)),]

### heart hat nur 2 Auspraegungen 
reg <- reg[!(reg$task.id %in% c(12717)),]

## libras_move hat nur 15 Auspraegungen im targetfeature 
reg <- reg[!(reg$task.id %in% c(4709)),]

### liver-disorders hat nur 16 Auspraegungen im Targetfeature 
reg <- reg[!(reg$task.id %in% c(52948)),]

### TurkiyeStudentEvaluation hat nur 3 Auspraegungen im Targetfeature
reg <- reg[!(reg$task.id %in% c(14950)),]

## wap.wc hat nur 20 Auspraegungen im targetfeature 
reg <- reg[!(reg$task.id %in% c(4723)),]

### SWD hat nur 4 Auspraegungen 
reg <- reg[!(reg$task.id %in% c(5022)),]

### tr11.wc, tr12.wc, tr21.wc, tr23.wc, tr31.wc, tr41.wc und tr.45.wc haben jeweils zu wenige Auspraegungen
reg <- reg[!(reg$task.id %in% c(4714,4722,4711,4715, 4712,4725,4710)),]

#### w1a, w2a, w3a, w4a, w5a, w6a, w7a, w8a haben alle nur 2 Auspraegungen im Targetfeature 
reg <-  reg[!(reg$task.id %in% c(12722,12723, 12724,12725,12726,12727,12728,12729)),]

## arsenic-female-bladder hat nur 14 Auspraegungen im Targetfeature
## arsenic-female-lung nur 18 Auspraegungen
## arsenic-male-bladder nur 14 Auspraegungen
##Arsnic-male-lung nur 18 Auspraegungen 
reg <- reg[!(reg$task.id %in%  c(4858,4841,4818,4861)),]

## alle .wc datensaetze raus da Kategoriales Targetfeature
reg <- reg[!(reg$task.id %in% c(4716, 4721, 4718,4716, 4717,4726,4713, 4719,4724,4720  )),]

## connect-4 hat nur 3 Auspraegungen im Targetfeature
reg <- reg[!(reg$task.id == 12731),]

##der Datensatz coil2000 hat nur 2 Auspraegungen im Targetfeature
reg <- reg[!(reg$task.id == 4770),]

## credit-g ist kategorial im Targetfeature 
reg <- reg[!(reg$task.id ==  146813),]

## ERA hat nur 9 Auspraegungen im Targetfeature
reg <- reg[!(reg$task.id %in% c(5024 )  ),]

### balloon hat nur 2 Features neben dem targetfeature
reg <- reg[!(reg$task.id %in%  c(4840)),]
 
### analcatdata_wildcat hat nur 17 verschhiedene Auspraegungen im Taragetfeature 
reg <- reg[!(reg$task.id %in% c(4846 )  ),]

## wine-quality hat nur 7 Auspraegungen im targetfeature 
reg <- reg[!(reg$task.id %in% c(4768)),]

## mu284 hat Cluster im Targeetfeatrue 
reg <- reg[!(reg$task.id %in% c(4865)),]

## fourclass und fourclass_scale haben nur 2 Auspraegungen im Targetfeature 
reg <- reg[!(reg$task.id %in% c(7574,12714 )  ),]

### LEV hat nur 5 Auspraegungen im Targetfeature
reg <- reg[!(reg$task.id %in% c(5023)),]

# blood-transfusion-service-center ist kategorial
reg <- reg[!(reg$task.id %in%  c(168295)),]

# 118 Tasks
dim(reg)

####################################################################################################################################
########################################     Datensaetze die keine Beschreibung auf OpenML haben  #############################
#################################################################################################################################

# BNG Datensaetze haben beide keine beschreibung 
reg <- reg[!(reg$task.id %in% c(7323,7320,7319,7321,7324,7318,7322,7325,7327)),]

### nicht boston_corrected da dieser datensatz keine beschreibung hat und nur einer von beiden   
reg <- reg[!(reg$task.id == 4868),]

# 115 Tasks 
dim(reg)

###################################################################################################################
##############################     Datensaetze mit gleichem Ursprung oder gleichem Themengebiet entfernen  #########
###################################################################################################################

## alle Friedmann Datensaetze raus da alle Variablen sehr aehnlich 
reg <- reg[!(substr(reg$name,1,3)=="fri"),]

## nur cpu_act da cpu_small aehnlicher Datensatz ist wird er nicht mit aufgenommen 
reg <- reg[!(reg$task.id == 2315),]

## elevators und delta_elevators kommen aus der gleichen Domain wie Ailerons, deswegen werden sie nicht mitaufgenommen 
reg <- reg[!(reg$task.id %in% c(2289,2307)),]

### house_16H und house_8L Census Daten, solche sind schon in der Datensatzsammlung. Deswegen werden sie nicht mitaufgenommen 
reg <- reg[!(reg$task.id %in% c(4893,2309 )  ),]

###Nur ein Datensatz aus den drug-design Datensaetzen mtp, mtp2, benzo32,yprop, keine beschriftung der Variablen, Au?erdem mit adriana codesoftware hergestellt und als daf?r Repr?sentativer datensatz ist schon 
### topo_2_1 in die Sammlung mitaufgenommen worden 
reg <- reg[!(reg$task.id %in% c(4779,4804,2286,4808,4790)),]

## chscase_census2, chscase_census3, chscase_census4, chscase_census5, chscase_census6 werden nicht mitaufgenommen, da schon so Census datensaetze in der Datensatzsammlung sind 
reg <-  reg[!(reg$task.id %in% c(4990,4989,4988,4987,4982)),]

### cpu.with.vendor raus, da cpu_act schon in der Sammlung ist 
reg <- reg[!(reg$task.id %in% c(7561)),]

#### loesche BNG datensaetze bis auf 

### cpu raus, da cpu_act schon in der Sammlung ist 
reg <- reg[!(reg$task.id %in% c(4728)),]

### no2 und pm10 basieren auf dem gleichen Problem, pm10 raus 
reg <- reg[!(reg$task.id %in% c(4848)),]

### autoPrice ist doppelt vorhanden,ein Datensatz muss raus 
reg <- reg[!(reg$task.id %in% c(2298)),]

## nur ein visualizing Task, visualizing_galaxy raus 
reg <- reg[!(reg$task.id %in% c(5001)),]

## bank8FM aus der Sammlung raus weil bank32NH aehnlicher Datensatz ist 
reg <-  reg[!(reg$task.id %in% c(4891)),]
## puma8NH aus der Sammlung weil Puma32NH aehnlicher Datensatz ist 
reg <-  reg[!(reg$task.id %in% c(2313)),]

# 32 Tasks 
dim(reg)
###############################################################################################################################
######################################   kuenstliche Datensaetze isolieren    #################################################
###############################################################################################################################

### 2dplanes, Taskid = 2306 ist ein Task mit kuenstlicher Datensatz
reg_syn <- reg[(reg$task.id == 2306),]
reg <- reg[!(reg$task.id == 2306),]

#bank32nh und bank8FM sind syntethisch
#bank32NH in syntethische Datensaetze mitaufnehmen 
reg_syn <- rbind(reg_syn,reg[reg$task.id == 4881,])
reg <- reg[!(reg$task.id %in%  c(4881)),]

### BNG(satelite_image) ist kuenstlich 
reg_syn <- rbind(reg_syn,reg[reg$task.id %in% c(7326),])
reg <- reg[!(reg$task.id %in% c(7326)),]

### mv kuenstlicher Datensatz 
reg_syn <- rbind(reg_syn,reg[(reg$task.id %in% c(4774)),] )
reg <- reg[!(reg$task.id %in% c(4774)),]

### pollen ist synthetisch 
reg_syn <- rbind(reg_syn,reg[(reg$task.id %in% c(855)),] )
reg <-  reg[!(reg$task.id %in% c(4855)),]

## puma32H in synthetische Datensaetze mitaufnehmen 
reg_syn <- rbind(reg_syn,reg[reg$task.id == 4772,])
reg <-  reg[!(reg$task.id %in% c(4772)),]

##pwLinear
reg_syn <- rbind(reg_syn,reg[reg$task.id == 2317,])
reg <- reg[!(reg$task.id %in% c(2317)),]

dim(reg_syn)
dim(reg)

####################################################################################################
#### Ueberpruefe Bestimmtheitsmass auf nicht synthetischen Datensaetzen    #############################
####################################################################################################

## Erstelle Dataframe in dem Task.id, DatensatzID, Anzahl an Beobachtungen und das Bestimmtheitsmass gespeichert werden 
all_R_2 <- data.frame(task.id = character(), data.id = character(), data.name = character(), N_observations = integer(), R_2 = double())

for(i in 1:nrow(reg)){
  ## Versuche den Task aus OpenML herunterzuladen 
  if((reg$task.id[i])){
    try(newtask <-  getOMLTask(reg[i,1]))
  }
  if( exists("newtask") == TRUE){
    task = newtask 
    #print(summary(task$input$data.set$data))
    PosOfTargetFeature <- which(names(task$input$data.set$data) == task$input$target.features)
    #print(PosOfTargetFeature)
    print(paste("Task ID: ",task$task.id))
    print(paste("Data ID: ", reg$data.id[reg$task.id==task$task.id]))
    
    ## fuehre lineare Regression mit allen Kovariablen als Praediktoren fuer das  Targetfeature durch 
    lmTask <- lm(as.formula(paste(task$input$target.features,"~",paste(names(task$input$data.set$data[-PosOfTargetFeature]), collapse = "+") )), 
      data = task$input$data.set$data)
    
    R_2 <- summary(lmTask)[8]
    Task_R_2 <- data.frame(task.id = task$task.id, data.id= reg$data.id[reg$task.id==task$task.id],data.name = task$input$data.set$desc$name, N_observations = nrow(task$input$data.set$data), R_2 = R_2)
    all_R_2 <- rbind(all_R_2, Task_R_2)
    print(task$input$data.set$colnames.old)
    print(all_R_2)
    print(colnames(task$input$data.set$data))
    ### loesche Task, damit beim Fehlschlagen im naechsten durchlauf nicht der vorherige wiederbenutzt wird
    remove(newtask)
  }
}

R_2_is_1 <- all_R_2[all_R_2$r.squared == 1, 1]
R_2_is_1

#### sortiere Tasks aus bei denen R-Squared 1 ergab 
reg <- reg[!(reg$task.id %in% R_2_is_1),]

# 24 Tasks 
dim(reg)

#########################################################################################################################
########## Ueberpruefe Bestimmtheitsmass der Linearen Regression auf kuenstlichen Datensaetzen ##########################
#########################################################################################################################

# gleiches Vorgehen wie bei der Berechnung des Bestimmtheitsmasses fuer nicht synthetische Datensaetze 
all_R_2_artificial <- data.frame(task.id = character(),data.id = character(),data.name = character(),  N_observations = integer(), R_2=double())

for(i in 1:nrow(reg_syn)){
  if((reg_syn$task.id[i])){
    try(newtask <-  getOMLTask(reg_syn[i,1]))
  }
  if( exists("newtask") == TRUE){
    task = newtask 
    
    #print(summary(task$input$data.set$data))
    PosOfTargetFeature <- which(names(task$input$data.set$data) == task$input$target.features)
    #print(PosOfTargetFeature)
    print(paste("Task ID: ",task$task.id))
    print(paste("Data ID: ", reg_syn$data.id[reg_syn$task.id==task$task.id]))
    
    lmTask <- lm(as.formula(paste(task$input$target.features,"~",paste(names(task$input$data.set$data[-PosOfTargetFeature]), collapse = "+") )), 
      data = task$input$data.set$data)
    
    R_2 <- summary(lmTask)[8]
    Task_R_2 <- data.frame(task.id = task$task.id, data.id= reg_syn$data.id[reg_syn$task.id==task$task.id],data.name = task$input$data.set$desc$name, N_observations = nrow(task$input$data.set$data), R_2 = R_2)
    all_R_2_artificial <- rbind(all_R_2_artificial, Task_R_2)
    print(task$input$data.set$colnames.old)
    print(all_R_2_artificial)
    print(colnames(task$input$data.set$data))
    remove(newtask)
  }
}

### Kein synthetischer Datensatz erreich Bestimmtheitsmass von 1
all_R_2_artificial

# 6 Tasks 
dim(reg_syn)

##################################################################################################################################
#############################################      benchmarken der nicht synthetischen Datensaetze       #########################
##################################################################################################################################

####### importiere Tasks die noch in reg enthalten sind und konvertiere in MLRTask
selectedTasks <- list()
for (y in 1:length(reg$task.id)){
  ### konvertiere alle OML Tasks zu MLR Tasks um die funktion BEnchmark auf sie anwenden zu k?nnen
  task <- convertOMLTaskToMlr(getOMLTask(task.id = reg$task.id[y]))$mlr.task 
  ## fuege mlr task in die liste ein 
  selectedTasks[[y]] <- task
}

### Manuelles einfuegen der nicht-OpenML Tasks als MLR Tasks

##################################################################################################
##### Titanic Datensatz  #########################################################################
##################################################################################################

Titanic_data <- read.csv("./data/train_and_test2_titanic.csv", header = T)

### delete rows with missing values 
Titanic_data <- na.omit(Titanic_data)

### Delete the ID from the Dataset 
Titanic_data <- Titanic_data[,-1]
head(Titanic_data)

####### remove Columns that only contain the Value '0'
empty_cols <- vector()
par(mfrow=c(4,4))
for(i in 1:ncol(Titanic_data)-4){
  print(4+i)
  print(range(Titanic_data[,4+i]))
  boxplot(Titanic_data[,4+i])
  if(range(Titanic_data[,4+i])[1]== 0 && range(Titanic_data[,4+i])[2]== 0){
    empty_cols <- rbind(empty_cols, 4+i)
  }
}
empty_cols
Titanic_data <- Titanic_data[,-empty_cols]

#### Pclass as factor 
Titanic_data$Pclass <- factor(Titanic_data$Pclass)
revalue(Titanic_data$Pclass, c("1"="First Class","2"="Second Class","3"= "Third Class" ))

#### Emarked as factor 
Titanic_data$Embarked <- factor(Titanic_data$Embarked)
revalue(Titanic_data$Embarked, c("0" ="Cherbourg", "1"="Queenstown", "2" = "Southampton" ))

### Create MLR TASK
Titanic <- makeRegrTask(id="Titanic", data = Titanic_data, target = "Fare")
### Add Task to selectedTasks 
selectedTasks[[y+1]] <- Titanic

##################################################################################################
#########  Monthly Data Rainfall Bangladesh ######################################################
##################################################################################################
rainfallBangladesh_data <- read.csv(".\data\data_monthly_rainfall.csv", header = T)

### delete rows with missing values
rainfallBangladesh_data <- na.omit(rainfallBangladesh_data)
rainfallBangladesh_data

### Delete Row with the Station Index
rainfallBangladesh_data <- rainfallBangladesh_data[,-5]
rainfallBangladesh_data

### Months as words 
months <- c("January","February","March","April","May","June","July","August","September","October","November","December")

for (i in 1:nrow(rainfallBangladesh_data)){
  for(a in 1:length(months)){
    if (rainfallBangladesh_data[i,3] == a){
      rainfallBangladesh_data[i,3] <- months[a]
    }
  }
}

rainfallBangladesh_data$Month <- factor(rainfallBangladesh_data$Month) 
levels(rainfallBangladesh_data$Month)

## make MLR Regression Task
rainBangladesh <- makeRegrTask(id="Rainfall_Bangladesh", data = rainfallBangladesh_data, target = "Rainfall" )

## Add Task to the seleceted Tasks list
selectedTasks[[y+2]] <- rainBangladesh

###################################################################################################################################################
#################################################   Regression von Askinth    ##################################################################################################
###################################################################################################################################################

regrAkshith_data <- read.csv(".\data\Regression_akshith.csv", header = T) 

##### Zeilen mit fehlenden Werten entfernen
regrAkshith_data <- na.omit(regrAkshith_data)
nrow(regrAkshith_data)

##### RegrAkshith als MLR Task 

regrAkshith <- makeRegrTask(id="Regression Akshith", data = regrAkshith_data, target = "Purchase" )
selectedTasks[[y+3]] <- regrAkshith

###############################################################################################################################################################################################################
##################################################################### AIRBNB #####################################################################
###############################################################################################################################################################################################################

AIRBNB_data <- read.csv(".\data\AIRBNB_modified.csv", header = T)

#### delete dates for now
AIRBNB_data <- AIRBNB_data[,-c(1,15,17)]

##### Create Regression Task for MLR 
AIRBNB <- makeRegrTask(id="AIRBNB logPrice", data=AIRBNB_data, target = "log_price")
selectedTasks[[y+4]] <- AIRBNB

###################################################################################################################################################
#### Definiere Resampling Methode 10-fold Kreuzvalidierung 
###################################################################################################################################################

rdesc <- makeResampleDesc("CV", iters= 10)

#### Definiere Learner regr.IBK (K-nearest Neighbors), regr.ranger (Random Forest), regr.rpart (Decision Tree), regr.glmnet (Elastic Net)
lrns <- list(makeLearner("regr.IBk"),makeLearner("regr.ranger"),makeLearner("regr.rpart"),makeLearner("regr.glmnet"), makeLearner("regr.liquidSVM"))

####### Speicher alle bmrs in einer Liste (Guetemasse Kendallstau und Bestimmtheitsmass)
# dafuer erst alle benchmarks einzeln erstellen und dann an Liste anh?ngen 

### leere liste fuer die benchmark objekte 
bmrs_neu <- list()

#### Fuege nacheinander die benchmark objekte in die liste ein 
for(i in seq_along(selectedTasks)) {
  bmrs_neu[[i]] <- benchmark(lrns, selectedTasks[[i]], rdesc, keep.pred = FALSE, measures = list(kendalltau,mse, rsq))
}

save(bmrs_neu, file = paste(path,"bmrs_neu.RData",sep = ""))
save(reg, file = paste(path,"Reg.RData",sep = ""))
save(reg_syn, file = paste(path,"RegSyn.RData",sep = ""))

