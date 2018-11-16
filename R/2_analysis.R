##################################################################################################################
############################    Hier R-Studio schliessen damit heap wieder frei wird      ##############################
#  Nach schliessen die librarys wieder laden, java.options setzen, OML KOnfiguration wieder setzen und path neu zuweisen 
##################################################################################################################
options(java.parameters = "-XX:+UseG1GC") # Should avoid java gc overhead
options(java.parameters = "-Xmx16000m")
library(plyr)
library(farff)
library(OpenML)
library(rJava)
library(RWeka)
library(mlr)
library(glmnet)
library(ranger)
library(ggplot2)

### Speicherort erneut angeben 
path <- "C:/Users/ru76beb/Documents/openml-bench-regr"

### Konfigutration von OML    (wird in Datei gespeichert beim ersten ausf?hren, aber zur Sicherheit noch einmal)
saveOMLConfig(apikey = "f825394339f0ff5facce111e26bf0933", arff.reader = "RWeka", overwrite=TRUE)

load( file = paste(path,"bmrs_neu.RData",sep = ""))
load(file=paste(path,"Reg.RData",sep = ""))
load( file = paste(path,"RegSyn.RData",sep = ""))

#############################################################################################################
#####################    Auswertung der Ergebnisse von benchmark()                ###########################################
#############################################################################################################
#######      Erstelle data.frame zum Festhalten der Ergebnisse der learner und der Masse Kendalls Tau und Bestimmtheitsmass
all_measures_neu <- data.frame("task.id" = character(), "learner.id" = character(), "kendalltau.test.mean" = double(), "mse.test.mean" = double(), "rsq.test.mean" = double())

for (  i in 1:length(bmrs_neu)){
  all_measures_neu <- rbind(all_measures_neu, getBMRAggrPerformances(bmrs_neu[[i]], as.df = TRUE))
}

### bestimmtheitsma? gleich 0 setzen wenn wert kleiner 0 
all_measures_neu[all_measures_neu[,5]<=0,5] <- 0 

### Kaggle_bike_sharing_demand_challange   abk?rzen, da sonst zu gro?er Name in der Grafik
levels(all_measures_neu$task.id) <- sub("^Kaggle_bike_sharing_demand_challange$","kbsdc*",levels(all_measures_neu$task.id))

### GeographicalOriginalofMusic  abk?rzen, da sonst zu gro?er Name in der Grafik
levels(all_measures_neu$task.id) <- sub("^GeographicalOriginalofMusic$","GOM*",levels(all_measures_neu$task.id))

### Alle Namen der Learner fuer ggplot in einem Faktor speichern 
all_measures_neu[,2] <- as.factor(all_measures_neu[,2])

### Setze Null als wert bei NAs in Kendalls Tau
#all_measures_neu[is.na(all_measures_neu[,3]) == TRUE,3] <- -1

### learner.id in Learner umbenennen
names(all_measures_neu)[2] <- "Learner"

a <-ggplot(data = rbind(all_measures_neu) , aes(y = task.id, x = rsq.test.mean, group = Learner, color= Learner)) +
  geom_point() +
  lims(x=c(0,1))+
  ylab( "Datensatz") +
  xlab("Bestimmtheitsma?")+
  labs(caption="* kbscd steht als Abk?rzung f?r 'Kaggle_bike_sharing_demand_challange' \n * GOM steht als Abk?rzung f?r 'GeographicalOriginalofMusic'")


png("bs_neu.png", width = 15, height = 15, units="cm", res = 1024)
a
dev.off()

##### Kendalls tau 
b <- ggplot(data = all_measures_neu , aes(y = task.id, x = kendalltau.test.mean, group = Learner, color= Learner)) +
  geom_point() +
  lims(x=c(0,1))+
  ylab( "Datensatz") +
  xlab("Kendalls Tau")+
  labs(caption="* kbscd steht als Abk?rzung f?r 'Kaggle_bike_sharing_demand_challange' \n * GOM steht als Abk?rzung f?r 'GeographicalOriginalofMusic'")

png("kt_neu.png", width = 15, height = 15, units="cm", res = 1024)
b
dev.off()

###############################################################################################################################
##################################    histogramme der Targetfeatures                ##########################################
###############################################################################################################################

####### Ailerons 
###ailerons <- getOMLTask(reg[reg$name=="Ailerons",1])
###hist(ailerons$input$data.set$data$goal, main =paste(ailerons$input$target.features, " - Targetfeature von Ailerons"),xlab = ailerons$input$target.features,ylab = "absolute H?ufigkeit", col= "deepskyblue")
###
######## bodyfat   
###bodyfat <- getOMLTask(reg[reg$name=="bodyfat",1])
###hist(bodyfat$input$data.set$data$class,main =paste(bodyfat$input$target.features, " - Targetfeature von bodyfat"), xlab = bodyfat$input$target.features,ylab = "absolute H?ufigkeit", col= "deepskyblue")
###
######## boston 
###boston <- getOMLTask(reg[reg$name=="boston",1])
###hist(boston$input$data.set$data$MEDV, xlab = boston$input$target.features, main =paste(boston$input$target.features, " - Targetfeature von boston"),ylab = "absolute H?ufigkeit", col= "deepskyblue")
###
####### chatfield_4
###chatfield_4 <- getOMLTask(reg[reg$name=="chatfield_4",1])
###hist(chatfield_4$input$data.set$data$col_13, main =paste(chatfield_4$input$target.features, " - Targetfeature von chatfield_4"), xlab = chatfield_4$input$target.features , ylab = "absolute H?ufigkeit", col= "deepskyblue")
###
####### cps_85_wages
###cps_85_wages <- getOMLTask(reg[reg$name=="cps_85_wages",1])
###hist(cps_85_wages$input$data.set$data$WAGE,main =paste(cps_85_wages$input$target.features, " - Targetfeature von cps_85_wages") , xlab = cps_85_wages$input$target.features , ylab = "absolute H?ufigkeit", col= "deepskyblue")
###
#######   cpu_act
###cpu_act <- getOMLTask(reg[reg$name=="cpu_act",1])
###hist(cpu_act$input$data.set$data$usr, main =paste(cpu_act$input$target.features, " - Targetfeature von cpu_act"), xlab = cpu_act$input$target.features , ylab = "absolute H?ufigkeit", col= "deepskyblue")
###
####### GeographicalOriginOfMusic
###GeographicalOriginOfMusic <- getOMLTask(reg[reg$name=="GeographicalOriginalofMusic",1])
###hist(GeographicalOriginOfMusic$input$data.set$data$V100,  main =paste(GeographicalOriginOfMusic$input$target.features, " - Targetfeature von GeographicalOriginOfMusic"), xlab = GeographicalOriginOfMusic$input$target.features , ylab = "absolute H?ufigkeit", col= "deepskyblue")
###
###
######## houses
###houses  <- getOMLTask(reg[reg$name=="houses",1])
###hist(houses$input$data.set$data$median_house_value, main =paste(houses$input$target.features, " - Targetfeature von houses") , xlab = houses$input$target.features , ylab = "absolute H?ufigkeit", col= "deepskyblue")
### 
###### Kaggle_bike_sharing_demand_challange
###Kaggle_bike_sharing_demand_challange   <-  getOMLTask(reg[reg$name=="Kaggle_bike_sharing_demand_challange",1])
###hist(Kaggle_bike_sharing_demand_challange$input$data.set$data$count, main =paste(Kaggle_bike_sharing_demand_challange$input$target.features, " - Targetfeature von Kaggle_bike_sharing_demand_challange") , xlab = Kaggle_bike_sharing_demand_challange$input$target.features , ylab = "absolute H?ufigkeit", col= "deepskyblue")
###
###
###### kin8nm
###kin8nm <-  getOMLTask(reg[reg$name=="kin8nm",1])
###hist(kin8nm$input$data.set$data$y, main =paste(kin8nm$input$target.features, " - Targetfeature von kin8nm") , xlab = kin8nm$input$target.features , ylab = "absolute H?ufigkeit", col= "deepskyblue")
### 
####### lowbwt 
###lowbwt <- getOMLTask(reg[reg$name=="lowbwt",1])
###hist(lowbwt$input$data.set$data$class, main =paste(lowbwt$input$target.features, " - Targetfeature von lowbwt") , xlab = lowbwt$input$target.features , ylab = "absolute H?ufigkeit", col= "deepskyblue")
###
###### machine_cpu
###machine_cpu <- getOMLTask(reg[reg$name=="machine_cpu",1])
###hist(machine_cpu$input$data.set$data$class, main =paste(machine_cpu$input$target.features, " - Targetfeature von machine_cpu") , xlab = machine_cpu$input$target.features , ylab = "absolute H?ufigkeit", col= "deepskyblue")
###
###### no2
###no2 <- getOMLTask(reg[reg$name=="no2",1])
###hist(no2$input$data.set$data$no2_concentration, main =paste(no2$input$target.features, " - Targetfeature von no2") , xlab = no2$input$target.features , ylab = "absolute H?ufigkeit", col= "deepskyblue")
###range(no2$input$data.set$data$no2_concentration)
###### plasma_retinol
###
###plasma_retinol <- getOMLTask(reg[reg$name=="plasma_retinol",1])
###hist(plasma_retinol$input$data.set$data$RETPLASMA, main =paste(plasma_retinol$input$target.features, " - Targetfeature von plasma_retinol") , xlab = plasma_retinol$input$target.features , ylab = "absolute H?ufigkeit", col= "deepskyblue")
###
###### rmftsa_ladata
###rmftsa_ladata <- getOMLTask(reg[reg$name=="rmftsa_ladata",1])
###hist(rmftsa_ladata$input$data.set$data$Respiratory_Mortality, main =paste(rmftsa_ladata$input$target.features, " - Targetfeature von rmftsa_ladata") , xlab = rmftsa_ladata$input$target.features , ylab = "absolute H?ufigkeit", col= "deepskyblue")
###
###### space_ga
###space_ga <- getOMLTask(reg[reg$name=="space_ga",1])
###hist(space_ga$input$data.set$data$ln.VOTES.POP., main =paste(space_ga$input$target.features, " - Targetfeature von space_ga"), xlab = space_ga$input$target.features , ylab = "absolute H?ufigkeit", col= "deepskyblue")
###range(space_ga$input$data.set$data$ln.VOTES.POP)
###
###### stock
###stock <- getOMLTask(reg[reg$name=="stock",1])
###hist(stock$input$data.set$data$company10, main =paste(stock$input$target.features, " - Targetfeature von stock"), xlab = stock$input$target.features , ylab = "absolute H?ufigkeit", col= "deepskyblue")
###
####### strikes
###strikes <- getOMLTask(reg[reg$name=="strikes",1])
###hist(strikes$input$data.set$data$strike_volume, main = paste(strikes$input$target.features, "- Targetfeature von strikes "), xlab = strikes$input$target.features , ylab = "absolute H?ufigkeit", col= "deepskyblue")
###
####### tecator
###tecator <- getOMLTask(reg[reg$name=="tecator",1])
###hist(tecator$input$data.set$data$fat, main = paste(tecator$input$target.features, "- Targetfeature von tecator "), xlab = tecator$input$target.features , ylab = "absolute H?ufigkeit", col= "deepskyblue")
###
####### topo_2_1
###topo_2_1 <- getOMLTask(reg[reg$name=="topo_2_1",1])
###hist(topo_2_1$input$data.set$data$oz267, main = paste(topo_2_1$input$target.features, "- Targetfeature von topo_2_1 "), xlab = topo_2_1$input$target.features , ylab = "absolute H?ufigkeit", col= "deepskyblue")
###
###### triazines
###triazines  <- getOMLTask(reg[reg$name=="triazines",1])
###hist(triazines$input$data.set$data$activity, main = paste(triazines$input$target.features, "- Targetfeature von triazines "), xlab = triazines$input$target.features , ylab = "absolute H?ufigkeit", col= "deepskyblue")
###
###
####### visualizing_soil
###visualizing_soil  <- getOMLTask(reg[reg$name=="visualizing_soil",1])
###hist(visualizing_soil$input$data.set$data$track, main = paste(visualizing_soil$input$target.features, "- Targetfeature von visualizing_soil "), xlab = visualizing_soil$input$target.features , ylab = "absolute H?ufigkeit", col= "deepskyblue")
### 
####### wind
###wind <- getOMLTask(reg[reg$name=="wind",1])
###hist(wind$input$data.set$data$MAL, main = paste(wind$input$target.features, "- Targetfeature von wind "), xlab = wind$input$target.features , ylab = "absolute H?ufigkeit", col= "deepskyblue")
###
###### wisconsin
###wisconsin <- getOMLTask(reg[reg$name=="wisconsin",1])
###hist(wisconsin$input$data.set$data$time, main = paste(wisconsin$input$target.features, "- Targetfeature von wisconsin ") , xlab = wisconsin$input$target.features , ylab = "absolute H?ufigkeit", col= "deepskyblue")
###

##################################################################################################################################
###########################################           Kuenstliche Datensaetze                #############################################
###############################################################################################################################


#### Definiere Resampling Methode 10-fold Kreuzvalidierung 
rdesc <- makeResampleDesc("CV", iters= 10)

#### Definiere Learner regr.IBK (K-nearest Neighbors), regr.ranger (Random Forest), regr.rpart (Decision Tree), regr.glmnet (Elastic Net)
lrns <- list(makeLearner("regr.IBk"),makeLearner("regr.ranger"),makeLearner("regr.rpart"),makeLearner("regr.glmnet"), makeLearner("regr.liquidSVM"))

### erstelle Leere liste fuer die Tasks 
selectedArtificialTasks <- list() 

for (y in 1:length(reg_syn$task.id)){
  ##konvertiere  jeden OML Task in MLR Task um mlr funktionen auf ihm anwenden zu k?nnen  
  task <- convertOMLTaskToMlr(getOMLTask(task.id = reg_syn$task.id[y]))$mlr.task  
  selectedArtificialTasks[[y]] <- task
}

## erstelle leere liste, an die alle benchmark objekte nacheinander angeh?ngt werden 
bmrs_neu_artificial <- list()

bmr_2dplanes <- benchmark(lrns,selectedArtificialTasks[[1]],keep.pred = FALSE, rdesc,  measures = list(kendalltau,mse, rsq))

bmrs_neu_artificial[[1]] <- bmr_2dplanes 

bmr_bank32nh <-  benchmark(lrns,selectedArtificialTasks[[2]],keep.pred = FALSE, rdesc,  measures = list(kendalltau,mse, rsq))
bmrs_neu_artificial[[2]] <- bmr_bank32nh

bmr_mv <- benchmark(lrns,createDummyFeatures(selectedArtificialTasks[[3]]),keep.pred = FALSE, rdesc,  measures = list(kendalltau,mse, rsq))
bmrs_neu_artificial[[3]] <- bmr_mv

bmr_pollen <- benchmark(lrns,selectedArtificialTasks[[4]],keep.pred = FALSE, rdesc,  measures = list(kendalltau,mse, rsq))
bmrs_neu_artificial[[4]] <- bmr_pollen

bmr_puma32H   <-  benchmark(lrns,selectedArtificialTasks[[5]],keep.pred = FALSE, rdesc,  measures = list(kendalltau,mse, rsq))

bmrs_neu_artificial[[5]] <- bmr_puma32H 

bmr_pwLinear <- benchmark(lrns,selectedArtificialTasks[[6]],keep.pred = FALSE, rdesc,  measures = list(kendalltau,mse, rsq))

bmrs_neu_artificial[[6]] <- bmr_pwLinear

##################################################################################################################
##############################      Auswertung der Ergebnisse von benchmark()    #################################
##################################################################################################################

#### Erstelle data.frame zum Festhalen der ergebnisse der learner und der Ma?e Kendalls Tau und Bestimmtheitsmass

all_measures_neu_artificial <- data.frame("task.id" = character(), "learner.id" = character(), "kendalltau.test.mean" = double(), "mse.test.mean" = double(), "rsq.test.mean" = double())

for (  i in 1:length(bmrs_neu_artificial)){
  all_measures_neu_artificial <- rbind(all_measures_neu_artificial, getBMRAggrPerformances(bmrs_neu_artificial[[i]], as.df = TRUE))
}

### bestimmtheitsma? gleich 0 setzen wenn wert kleiner 0 
all_measures_neu_artificial[all_measures_neu_artificial[,5]<=0,5] <- 0 

### Alle Messungen in einem Faktor speichern 
all_measures_neu_artificial[,2] <- as.factor(all_measures_neu_artificial[,2])

### Setze Null als wert bei NAs in Kendalls Tau
all_measures_neu_artificial[is.na(all_measures_neu_artificial[,3]) == TRUE,3] <- 0

names(all_measures_neu_artificial)[2] <- "Learner"

c <- ggplot(data = all_measures_neu_artificial , aes(y = task.id, x = rsq.test.mean, group = Learner, color= Learner)) +
  geom_point() +
  lims(x=c(0,1))+
  ylab( "Datensatz") +
  xlab("Bestimmtheitsma?")+
  theme(plot.margin = unit(c(5,0,5,0),"cm"))

png("bm_neu_art.png", width = 15, height = 15, units="cm", res = 1024)
c
dev.off()

##### Kendalls tau 
d <- ggplot(data = all_measures_neu_artificial , aes(y = task.id, x = kendalltau.test.mean, group =Learner, color= Learner)) +
  geom_point() +
  lims(x=c(0,1))+
  ylab( "Datensatz") +
  xlab("Kendalls Tau")+
  theme(plot.margin = unit(c(5,0,5,0),"cm"))

png("kt_neu_art.png", width = 15, height = 15, units="cm", res = 1024)
d
dev.off()

######################################################################################################################################
########################## Histogramme der Targetfeatures synthetische Datensaetze     ################################################
############################################################################################################################

####### 2dplanes
###Twodplanes <- getOMLTask(reg_syn[reg_syn$name=="2dplanes",1])
###hist(Twodplanes$input$data.set$data$y, main = paste(Twodplanes$input$target.features, "- Targetfeature von Twodplanes "), col="deepskyblue" , xlab = Twodplanes$input$target.features , ylab = "absolute H?ufigkeit")
###
####### bank32nh
###bank32nh <- getOMLTask(reg_syn[reg_syn$name=="bank32nh",1])
###hist(bank32nh$input$data.set$data$rej, main = paste(bank32nh$input$target.features, "- Targetfeature von bank32nh ") , xlab = bank32nh$input$target.features , ylab = "absolute H?ufigkeit", col="deepskyblue")
###
#######  mv 
###mv <- getOMLTask(reg_syn[reg_syn$name=="mv",1])
###hist(mv$input$data.set$data$y, main = paste(mv$input$target.features, "- Targetfeature von mv ") , xlab = mv$input$target.features , ylab = "absolute H?ufigkeit", col="deepskyblue")
###
###### pollen
###pollen  <- getOMLTask(reg_syn[reg_syn$name=="pollen",1])
###hist(pollen$input$data.set$data$DENSITY,main = paste(pollen$input$target.features, "- Targetfeature von pollen ")  , xlab = pollen$input$target.features , ylab = "absolute H?ufigkeit", col="deepskyblue")
###
######  puma32H 
###puma32H <- getOMLTask(reg_syn[reg_syn$name=="puma32H",1])
###hist(puma32H$input$data.set$data$thetadd6,main = paste(puma32H$input$target.features, "- Targetfeature von puma32H ")  , xlab = puma32H$input$target.features , ylab = "absolute H?ufigkeit", col="deepskyblue")
###
###### pwLinear
###pwLinear <- getOMLTask(reg_syn[reg_syn$name=="pwLinear",1])
###hist(pwLinear$input$data.set$data$class, main = paste(pwLinear$input$target.features, "- Targetfeature von pwLinear ")  , xlab = pwLinear$input$target.features , ylab = "absolute H?ufigkeit", col= "deepskyblue")
###