### Hierarchical model including environmental variables in INLA ### 

library(INLA)
library(sp)
library(rgdal)
library(ROCR)

source("C:/R_files/Scripts/Tutorial/penalised_priors_INLA.R")

#### Format data ####

## Load predictor variables
df <- read.csv("C:/R_files/XSS/Environment/selected_ind_env.csv")

# Load XSS data points
hh <- read.csv("C:/R_files/XSS/Environment/xss_all_env.csv")
hh <- hh[c("houseID", "x", "y")]

## Convert to UTM
coordinates(hh) <- c("x", "y")
proj4string(hh) <- CRS("+proj=longlat +datum=WGS84")
hh <- spTransform(hh, CRS("+proj=utm +zone=50 +datum=WGS84 +units=m +no_defs +ellps=WGS84 +towgs84=0,0,0"))
hh <- data.frame(hh)

## Merge coordinates with data
df <- merge(df, hh, by="houseID")

## Create unique ID for each place sampled (household)
df$xy <- paste(df$x, df$y)
df$sp <- match(df$xy, unique(df$xy))

## Set up data frame for INLA
sp <- df$sp ## Space
hid <- df$houseID ## House ID
age <- factor(df$ageCat) ## Age category
gen <- factor(df$gender) ## Gender
fr <- factor(df$goForest) ## Go to forest
ins <- factor(df$insect) ## Insecticide use
hgt <- factor(df$hgt) ## Height of house
dem <- df$dem ## Elevation (1000m)
ofpa <- df$of5000pa 
irfd <- df$ir300fd
ac <- df$ac3000
popa <- df$po3000pa
Y <- df$pk ## outcome (seropositivity)
inla.df <- data.frame(sp, hid, age, gen, fr, ins, hgt, dem, ofpa, irfd, ac, popa, Y)
covars <- inla.df[,c(1:12)]

##### Model 0: no spatial or random effects #### 
form0 <- Y ~ 0 + b0 + age + gen + fr + ins + hgt + dem + ofpa + irfd + ac + popa
res0 <- inla(form0, family="binomial", 
             data=data.frame(Y=Y, b0=1, covars), num.threads=1,
             control.compute=list(dic=TRUE, cpo=TRUE), 
             control.predictor=list(compute=TRUE,
                                    quantiles=c(0.025,0.5,0.975)),verbose = TRUE)
summary(res0)
res0$dic$dic

# cross-validatory measures
fitted.values <- res0$summary.fitted.values$mean
observed.values <- df$pk
ROC_auc <- performance(prediction(fitted.values,observed.values),"auc")
ROC_auc@y.values[[1]] # AUC

#### Model 1: no spatial effects and household as random effect #####
form1 <- Y ~ 0 + b0 + age + gen + fr + ins + hgt + dem + ofpa + irfd + ac + popa + f(hid, model="iid")
res1 <- inla(form1, family="binomial", 
             data=data.frame(Y=Y, b0=1, covars), num.threads=1,
             control.compute=list(dic=TRUE, cpo=TRUE), 
             control.predictor=list(compute=TRUE,
                                    quantiles=c(0.025,0.5,0.975)),verbose = TRUE)
summary(res1)
res1$dic$dic

# cross-validatory measures
fitted.values <- res1$summary.fitted.values$mean
ROC_auc <- performance(prediction(fitted.values,observed.values),"auc")
ROC_auc@y.values[[1]] # AUC

#### Model 2: no spatial effects and household as random effect with covariates #####
form2 <- Y ~ 0 + b0 + age + gen + fr + ins + f(hid, irfd, ac, popa, ofpa, dem, hgt, model="iid")
res2 <- inla(form1, family="binomial", 
             data=data.frame(Y=Y, b0=1, covars), num.threads=1,
             control.compute=list(dic=TRUE, cpo=TRUE), 
             control.predictor=list(compute=TRUE,
                                    quantiles=c(0.025,0.5,0.975)),verbose = TRUE)
summary(res2)
res2$dic$dic

# cross-validatory measures
fitted.values <- res2$summary.fitted.values$mean
ROC_auc <- performance(prediction(fitted.values,observed.values),"auc")
ROC_auc@y.values[[1]] # AUC

#### Model 3: individual included as random effect and spatial correlation #####

## Scale coordinates
xmin <- min(df$x)
ymin <- min(df$y)
df$x <- df$x - xmin
df$y <- df$y - ymin
df$x <- df$x/100
df$y <- df$y/100

## Set domain
xmin <- min(df$x)
ymin <- min(df$y)
xmax <- max(df$x)
ymax <- max(df$y)
x <- c(xmin, xmin, xmax, xmax)
y <- c(ymin, ymax, ymin, ymax)
boundary <- cbind(x,y)

## Create mesh
pts <- cbind(df$x, df$y)
mesh1 <- inla.mesh.create.helper(points=pts, max.edge=c(15,30), cut=6)
plot(mesh1)
points(pts[,1], pts[,2], pch=19, cex=.5, col="red")

## Create observation matrix
A <- inla.spde.make.A(mesh1, loc=pts)
spde <- inla.spde2.matern(mesh1)
ind.2 <- inla.spde.make.index('s', mesh1$n)

## Create data stack 
stk2 <- inla.stack(data=list(Y=Y), A=list(A,1), tag="sdata",
                   effects=list(ind.2, list(data.frame(b0=1, covars))))

## Fit model with spatial correlation
form3 <- Y ~ 0 + b0 + age + gen + fr + ins + hgt + dem + ofpa + irfd + ac + popa + f(hid, model="iid") + f(s, model=spde)
res3 <- inla(form3, family = "binomial", data=inla.stack.data(stk2),
             control.predictor = list(A=inla.stack.A(stk2), compute=TRUE),
             control.compute=list(dic=TRUE, cpo=TRUE))
summary(res3)
res3$dic$dic

# cross-validatory measures
str(sdat <- inla.stack.index(stk2, 'sdata')$data)
fitted.values <- res3$summary.fitted.values$mean[sdat]
ROC_auc <- performance(prediction(fitted.values,observed.values),"auc")
ROC_auc@y.values[[1]] # AUC

#### Model 4: individual included as random effect, spatial correlation and priors #####

# Priors: precision = 1/ sigma ^2 - weakly informative priors
fixed.priors <- list(mean.intercept = 0, prec.intercept=1/100, mean = list(age=0, gen=0, fr=0, ins= 0, hgt=0, dem=0, ofpa=0, ac=0, popa=0), 
                     prec=list(age=1/100, gen=1/100, fr=1/100, ins= 1/100, hgt=1/100, dem=1/100, ofpa=1/100, ac=1/100, popa=1/100))

## Use penalised priors for spatial component
spde2 = local.inla.spde2.matern.new(mesh1, prior.pc.sig = c(0.1, 0.1), prior.pc.rho = c(50, 0.1))

## Use penalised priors for spatial component
rho0 = 50
sig0 = 1
spde3 = local.inla.spde2.matern.new(mesh2, prior.pc.rho = c(rho0, 0.5), prior.pc.sig = c(sig0, 0.5))

## Fit model with spatial correlation
form4 <- Y ~ 0 + b0 + age + gen + fr + ins + hgt + dem + ofpa + irfd + ac + popa + f(hid, model="iid") + f(s, model=spde3)
res4 <- inla(form4, family = "binomial", data=inla.stack.data(stk2),
             control.predictor = list(A=inla.stack.A(stk2), compute=TRUE),
             control.fixed=fixed.priors,
             control.compute=list(dic=TRUE, cpo=TRUE))
summary(res4)
res4$dic$dic

# cross-validatory measures
str(sdat <- inla.stack.index(stk2, 'sdata')$data)
fitted.values <- res4$summary.fitted.values$mean[sdat]
ROC_auc <- performance(prediction(fitted.values,observed.values),"auc")
ROC_auc@y.values[[1]] # AUC
