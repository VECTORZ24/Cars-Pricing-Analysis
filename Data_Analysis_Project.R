########################################################
# Data Analysis & Regression + KMeans Clustering in R  #
########################################################

# Load libraries
library(readr)    # for read_csv
library(dplyr)    # for data manipulation
library(ggplot2)  # for plotting
library(stats)    # for lm, kmeans

#-----------------------------------------------
# 1. Importing the data
#-----------------------------------------------

# Read the CSV (same as: raw_data = pd.read_csv("raw_data.csv"))
raw_data <- read_csv("raw_data.csv", show_col_types = FALSE)

# Show first 10 rows (like raw_data.head(10))
head(raw_data, 10)

# Summary (similar to .describe())
summary(raw_data)

#-----------------------------------------------
# 2. Cleaning and preprocessing dataset
#-----------------------------------------------

# Checking null values (raw_data.isnull().sum())
colSums(is.na(raw_data))

# Copy data and drop rows with any NA
# data = raw_data.copy(); data.dropna(axis=0, inplace=True)
data <- raw_data
data <- na.omit(data)

# Check again for missing values (data.isnull().sum())
colSums(is.na(data))

#-----------------------------------------------
# 3. Checking outliers (Price, EngineV, Mileage, Year)
# Using the same quantile logic as in Python
#-----------------------------------------------

# Hist of Price (sns.displot(data["Price"]))
ggplot(data, aes(x = Price)) +
  geom_histogram(bins = 30) +
  ggtitle("Price distribution (raw)")

# ---- Price outliers: q = 0.99 quantile ----
# q = data[["Price"]].quantile(0.99)[0]
q_price <- quantile(data$Price, 0.99, na.rm = TRUE)

# data2 = data[data["Price"] < q]
data2 <- subset(data, Price < q_price)

ggplot(data2, aes(x = Price)) +
  geom_histogram(bins = 30) +
  ggtitle("Price distribution (Price < 0.99 quantile)")

# ---- EngineV outliers ----
# In Python: from the graph we take q = 6.5 and do:
# q = 6.5
# data3 = data2[data['EngineV'] < q]

ggplot(data2, aes(x = EngineV)) +
  geom_histogram(bins = 30) +
  ggtitle("EngineV distribution (before filtering)")

q_enginev <- 6.5
data3 <- subset(data2, EngineV < q_enginev)

ggplot(data3, aes(x = EngineV)) +
  geom_histogram(bins = 30) +
  ggtitle("EngineV distribution (EngineV < 6.5)")

# ---- Mileage outliers ----
# q = data3[["Mileage"]].quantile(0.99)[0]
q_mileage <- quantile(data3$Mileage, 0.99, na.rm = TRUE)

# data4 = data3[data3["Mileage"] < q]
data4 <- subset(data3, Mileage < q_mileage)

ggplot(data4, aes(x = Mileage)) +
  geom_histogram(bins = 30) +
  ggtitle("Mileage distribution (Mileage < 0.99 quantile)")

# ---- Year outliers ----
# sns.displot(data4["Year"])
ggplot(data4, aes(x = Year)) +
  geom_histogram(bins = 30) +
  ggtitle("Year distribution (before filtering)")

# q = data4[["Year"]].quantile(0.01)[0]
q_year <- quantile(data4$Year, 0.01, na.rm = TRUE)

# data5 = data4[data4["Year"] > q]
data5 <- subset(data4, Year > q_year)

ggplot(data5, aes(x = Year)) +
  geom_histogram(bins = 30) +
  ggtitle("Year distribution (Year > 0.01 quantile)")

#-----------------------------------------------
# 4. Final cleaned dataset
#-----------------------------------------------

# cleaned_data = data5.dropna(axis=0)
cleaned_data <- na.omit(data5)

# Check for missing values (cleaned_data.isnull().sum())
colSums(is.na(cleaned_data))

# Reset index (cleaned_data = cleaned_data.reset_index())
# In R we just ensure there is a simple row index:
cleaned_data <- cleaned_data %>% mutate(index = row_number()) %>%
  select(index, everything())

# Show cleaned_data
head(cleaned_data)
nrow(cleaned_data)

# Summary (similar to .describe())
summary(cleaned_data)

#-----------------------------------------------
# 5. Linear Regression model (like sklearn)
#    Features: EngineV, Mileage, Year
#    Target: Price
#-----------------------------------------------

# X = cleaned_data[["EngineV", "Mileage", "Year"]]
# y = cleaned_data["Price"]

# Train / test split (test_size = 0.2, random_state = 42)
set.seed(42)
n <- nrow(cleaned_data)
test_size <- floor(0.2 * n)
test_indices <- sample(seq_len(n), size = test_size)
train_indices <- setdiff(seq_len(n), test_indices)

train_data <- cleaned_data[train_indices, ]
test_data  <- cleaned_data[test_indices, ]

# Create and train the model (LinearRegression)
# model.fit(X_train, y_train)
model <- lm(Price ~ EngineV + Mileage + Year, data = train_data)

# Model coefficients and intercept (like model.coef_ and model.intercept_)
coef(model)    # Intercept + slopes
intercept <- coef(model)[["(Intercept)"]]
coefficients <- coef(model)[c("EngineV", "Mileage", "Year")]

# Make predictions on test set
# y_pred = model.predict(X_test)
y_pred <- predict(model, newdata = test_data)

# Evaluate
# mse = mean_squared_error(y_test, y_pred)
mse <- mean((test_data$Price - y_pred)^2)

# r2 = r2_score(y_test, y_pred)
SST <- sum((test_data$Price - mean(test_data$Price))^2)
SSE <- sum((test_data$Price - y_pred)^2)
r2  <- 1 - SSE / SST

cat("Model Coefficients (EngineV, Mileage, Year):\n")
print(coefficients)
cat("Intercept:\n")
print(intercept)
cat("Mean Squared Error:\n")
print(mse)
cat("R² Score:\n")
print(r2)

#-----------------------------------------------
# 6. Predicting multiple car prices (same as Python)
#-----------------------------------------------

# cars = pd.DataFrame({...})
cars <- data.frame(
  EngineV = c(2.0, 4.4, 1.6),
  Mileage = c(250, 350, 400),
  Year    = c(2018, 2016, 2015)
)

# predictions = model.predict(cars)
cars$Predicted_Price <- predict(model, newdata = cars)

print(cars)

#-----------------------------------------------
# 7. KMeans Clustering on numerical features
#-----------------------------------------------

# X_cluster = cleaned_data[["EngineV", "Mileage", "Year"]]
X_cluster <- cleaned_data %>% select(EngineV, Mileage, Year)

# kmeans = KMeans(n_clusters=3, random_state=42)
set.seed(42)
kmeans_result <- kmeans(X_cluster, centers = 3, nstart = 25)

# cleaned_data["Cluster"] = kmeans.fit_predict(X_cluster)
cleaned_data$Cluster <- kmeans_result$cluster

# Show the first 20 rows with clusters
head(cleaned_data, 20)

# Optional: simple scatterplot of clusters (e.g., EngineV vs Mileage)
ggplot(cleaned_data, aes(x = EngineV, y = Mileage, color = factor(Cluster))) +
  geom_point() +
  labs(color = "Cluster") +
  ggtitle("KMeans Clusters (EngineV vs Mileage)")
