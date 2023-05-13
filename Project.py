import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn.inspection import permutation_importance
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
warnings.simplefilter(action='ignore', category=Warning)


# Function to print horizontal line
def print_hl():
    print('---------------------------------'
          '--------------------------------'
          '---------------------------------------')

# Divide the features to categorical and numeric features
def col_names(df):
    # Categorical Variables
    cat_cols = [col for col in df.columns if df[col].dtypes not in ["int64", "float64"]]
    # Numerical Variables
    num_cols = [col for col in df.columns if df[col].dtypes in ["int64", "float64"]]
    # Numerical but Categorical Variables
    num_but_cat = [col for col in num_cols if df[col].nunique() < 12]
    # Adding num_but_cat to cat_cols
    cat_cols = num_but_cat + cat_cols
    # num_but_cat removing from num_cols
    num_cols = [col for col in num_cols if col not in num_but_cat]
    # Categorical but Cardinal Variables
    cat_but_car = [col for col in cat_cols if df[col].nunique() > 12]
    print(
        f"Numerical Cols: {num_cols} \nCategorical Cols: {cat_cols} \nNumerical but Categorical: {num_but_cat} \nCategorical but Cardinal: {cat_but_car}")
    return num_cols, cat_cols


# Removing outliers using IQR
def remove_outliers_iqr(data_with_outliers, col):
    Q3 = np.quantile(data_with_outliers[col], 0.75)
    Q1 = np.quantile(data_with_outliers[col], 0.25)
    IQR = Q3 - Q1

    # print("IQR value for column %s is: %s" % (col, IQR))

    lower_range = Q1 - 1.5 * IQR
    upper_range = Q3 + 1.5 * IQR
    outlier_free_list = [x for x in data_with_outliers[col] if (
            (x > lower_range) & (x < upper_range))]
    return outlier_free_list


# Plotting Data features function definition
def plot_feature_importance(importance, names, model_type):
    # Create arrays from feature importance and feature names
    feature_importance = np.array(importance)
    feature_names = np.array(names)

    # Create a DataFrame using a Dictionary
    data = {'feature_names': feature_names, 'feature_importance': feature_importance}
    fi_df = pd.DataFrame(data)

    # Sort the DataFrame in order decreasing feature importance
    fi_df.sort_values(by=['feature_importance'], ascending=False, inplace=True)

    # Define size of bar plot
    plt.figure(figsize=(10, 8))
    # Plot Searborn bar chart
    sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
    # Add chart labels
    plt.title(model_type + 'FEATURE IMPORTANCE')
    plt.xlabel('FEATURE IMPORTANCE')
    plt.ylabel('FEATURE NAMES')


# Importing dataset
data_raw = pd.read_csv('healthcare-dataset-stroke-data.csv')

# Checking if there is duplicated id
no_duplicates = True
is_duplicated = data_raw.duplicated()
for x in is_duplicated:
    if x:
        no_duplicates = False
if no_duplicates:
    print('No duplicate id in the Dataset')

# Creating dataframe
data_frame = pd.DataFrame(data_raw)
raw_data_frame = data_frame.copy()
data_frame.head()
data_frame.tail()
data_frame.describe()

# Finding missing values
print('Findig Null values in data')
print_hl()
print(data_frame.isnull().sum())
print_hl()

# Bar plot for the null and not null values in BMI feature
print('Null Values in bmi:')
print_hl()
sns.barplot(data=data_frame, x=["Null", "Not Null"],
            y=[data_frame["bmi"].isnull().sum(), data_frame["bmi"].notnull().sum()])
plt.show()

# Plotting numeric data before Cleaning
num_cols, cat_cols = col_names(data_frame)
for col in num_cols:
    sns.histplot(x=data_frame[col], data=data_frame, color="lightblue")
    plt.show(block=True)
    sns.boxplot(x=data_frame[col], data=data_frame, color="lightblue")
    plt.show(block=True)
    data_frame.plot.scatter(col, 'stroke')
    plt.show()

# Plotting categorical data before cleaning
for col in cat_cols:
    sns.displot(data_frame[col])
    plt.show()

# Check attributes correlation
print(sns.heatmap(data_frame[num_cols].corr(), annot=True, linewidths=0.5, cmap="pink"))

# -------- Data Cleaning --------
# Filling missing and unknown values with mean (numeric) and most frequent (categorical)

# Removing nan values to calculate mean in bmi
clean_bmi_list = [x for x in data_frame['bmi'] if str(x) != 'nan']
print("> Ignoring N/A tuples in bmi list to work on it")
outliers_remove = [clean_bmi_list]
# Removing nan values to calculate mean value accurately
bmi_no_outliers = remove_outliers_iqr(outliers_remove, 0)
print("> removing outliers to calculate mean value accurately")
# Calculating mean of bmi without nan values
data_bmi_mean = np.mean(bmi_no_outliers)
print("> replacing unknown values with mean")
print("bmi mean value:", data_bmi_mean)
# Filling bmi N/A values with mean data cleaning
data_frame['bmi'] = data_frame['bmi'].replace(np.NaN, data_bmi_mean)

# Replace unknown categorical data in smoking status to most frequent
# finding mean status of smoking
smk_st = ['never smoked', 'formerly smoked', 'smokes']
smk_st_cout = [0, 0, 0]
for x in data_frame['smoking_status']:
    if str(x) == 'Unknown':
        pass
    else:
        smk_st_cout[smk_st.index(str(x))] += 1
smk_most_freq = smk_st[smk_st_cout.index(max(smk_st_cout))]
# filling unknown smoking data with smoking mean value
data_frame['smoking_status'] = data_frame['smoking_status'].replace('Unknown', smk_most_freq)
print("Data after replacing any unknown values:")
data_frame

# Removing Outliers using IQR technique
# no age outliers
# filtering outliers in bmi
bmi_clean = remove_outliers_iqr(data_frame, 'bmi')
filtered_bmi_data = data_frame.loc[data_frame['bmi'].isin(bmi_clean)]
# filtering outliers in average glucose level
avg_glc_clean = remove_outliers_iqr(data_frame, 'avg_glucose_level')
filtered_avg_glc_data = data_frame.loc[data_frame['avg_glucose_level'].isin(avg_glc_clean)]
# filtering outliers in both bmi and average glucose level
filtered_data = filtered_bmi_data.loc[data_frame['avg_glucose_level'].isin(avg_glc_clean)]
f_data_frame = pd.DataFrame(filtered_data)
print('**** Data Set after excluding outliers bmi , average glucose level ****')
data_frame = filtered_data
data_frame

# Replacing any spaces in string in values in smoking_status to _
data_frame['smoking_status'] = data_frame['smoking_status'].str.replace('\s+', '_', regex=True)
data_frame

# Finding Binary valued categorical data features
binary_cols = [col for col in data_frame.columns if
               data_frame[col].dtype not in ["int64", "float64"] and data_frame[col].nunique() == 2]
binary_cols

# Convert Binary Categorical Data to Numeric Values (0,1)
le = LabelEncoder()
for col in binary_cols:
    data_frame[col] = le.fit_transform(data_frame[col])
data_frame.head()

# Making copy of data frame before turning all its categorical features to binary features and discretize it
data_frame_I = data_frame.copy()

# Encode any categorical feature to numerical value
Label_Encoder = LabelEncoder()
for i in f_data_frame.select_dtypes(include=['object']).columns.tolist():
    f_data_frame[i] = Label_Encoder.fit_transform(f_data_frame[i])

# Convert discrete data values to binary data values
ohe_cols = [col for col in data_frame.columns if 2 < data_frame[col].nunique() < 10]
ohe_cols
data_frame = pd.get_dummies(data_frame, columns=ohe_cols, drop_first=True)
data_frame.head()

# Normalize continuous numeric value features to range ( 0 > 1 )
sc = MinMaxScaler()
data_frame["age"] = sc.fit_transform(data_frame[["age"]])
data_frame["bmi"] = sc.fit_transform(data_frame[["bmi"]])
data_frame["avg_glucose_level"] = sc.fit_transform(data_frame[["avg_glucose_level"]])
data_frame.head()

# Visualize data after cleaning
for col in data_frame.columns:
    sns.histplot(x=data_frame[col], data=data_frame, color="lightblue")
    plt.show(block=True)
    sns.boxplot(x=data_frame[col], data=data_frame, color="lightblue")
    plt.show(block=True)
    data_frame.plot.scatter(col, 'stroke')
    plt.show()


# -------- Building Classification Models --------

# Splitting data into 2 Sets first one x to train the model and second one y to test trained model
x = data_frame.drop(["stroke", "id"], axis=1)
y = data_frame["stroke"]

x_data = data_frame.loc[:, data_frame.columns != 'stroke']
x_data = x_data.loc[:, x_data.columns != 'id']
y_data = data_frame['stroke']
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, stratify=y_data, test_size=0.2)
print("> Train Data:", y_train.shape[0])
print(y_train.value_counts())
y_train.value_counts().plot(kind='bar')
plt.show()
print_hl()
print("> Test Data:", y_test.shape[0])
print(y_test.value_counts())
y_test.value_counts().plot(kind='bar')
plt.show()
print_hl()

# Classification using decision tree classifier
clf = DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(x_train, y_train)
res = clf.predict(x_test)
accuracy = accuracy_score(y_test, res)
print("Decision Tree Accuracy = ", accuracy)
res_df = pd.DataFrame(res, columns=['stroke_prediction'])
print("> Prediction result:")
print(res_df)
print(res_df.value_counts())
res_df['stroke_prediction'].value_counts().plot(kind='bar')
plt.show()

# Finding vital role attributes for decision tree
print("Features sorted from least important according to Decision Tree:")
print_hl()
c = 1
data_features = x_train.columns
sort = clf.feature_importances_.argsort()
sort
for i in sort:
    print("{:<5} {:<20} {:<5} {:<20} ".format(c, data_features[i], ' | ', clf.feature_importances_[i]))
    print_hl()
    c = c + 1
# Plot important features according to decision tree
plot_feature_importance(clf.feature_importances_, x_train.columns, 'Decision Tree ')
plt.show()
print_hl()

# Classification using KNN classifier
neighbors = np.arange(1, 9)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

# Loop over K values
for i, k in enumerate(neighbors):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train)

    # Compute training and test data accuracy
    train_accuracy[i] = knn.score(x_train, y_train)
    test_accuracy[i] = knn.score(x_test, y_test)

plt.plot(neighbors, test_accuracy, label='Testing dataset Accuracy')
plt.plot(neighbors, train_accuracy, label='Training dataset Accuracy')

plt.legend()
plt.xlabel('n_neighbors')
plt.ylabel('Accuracy')
plt.show()

# Prediction using best k value for KNN classifier
knn = KNeighborsClassifier(n_neighbors=4)
knn.fit(x_train, y_train)
knn_res = knn.predict(x_test)
knn_accuracy = accuracy_score(y_test, knn_res)
print("KNN Accuracy = ", knn_accuracy)
print_hl()
knn_res_df = pd.DataFrame(knn_res, columns=['stroke_prediction'])
print("> Prediction result:")
print(knn_res_df)
print(knn_res_df.value_counts())
print_hl()
knn_res_df['stroke_prediction'].value_counts().plot(kind='bar')
plt.show()
print_hl()

# Classification using naive bayes classifier
# Use the dataframe without converting its categorical features into binary values
x_data_I = data_frame_I.loc[:, data_frame_I.columns != 'stroke']
x_data_I = x_data_I.loc[:, x_data_I.columns != 'id']
y_data_I = data_frame_I['stroke']
x_train_I, x_test_I, y_train_I, y_test_I = train_test_split(x_data_I, y_data_I, stratify=y_data_I, test_size=0.2)

# Applying naive bayes
nb = GaussianNB()
nb.fit(x_train, y_train)
res_nb = nb.predict(x_test)
print("Naive bayes predicted = ", res_nb)
accuracy_nb = accuracy_score(y_test, res_nb)
print("Accuracy = ", accuracy_nb)
res_df_nb = pd.DataFrame(res_nb, columns=['stroke_prediction'])
print("> prediction result:", res_df_nb.shape[0])
print(res_df_nb.value_counts())
res_df_nb['stroke_prediction'].value_counts().plot(kind='bar')
plt.show()

# Finding vital role attributes for naive bayes
print("Features sorted from least important according to Naive Bayes:")
print_hl()
c = 1
data_features = x_train.columns
imps = permutation_importance(nb, x_test, y_test)
sort = imps.importances_mean.argsort()
sort
for i in sort:
    print("{:<5} {:<20} {:<5} {:<20} ".format(c, data_features[i], ' | ', imps.importances_mean[i]))
    print_hl()
    c = c + 1
# Plot important features according to naive bayes
plot_feature_importance(imps.importances_mean, x_train.columns, 'NAIVE BAYES ')
plt.show()
print_hl()

# -------- Classification using Cross Validation --------
models = [("LR", LogisticRegression()),
          ("KNN", KNeighborsClassifier()),
          ("DecisionTree", DecisionTreeClassifier()),
          ("RF", RandomForestClassifier()),
          ("SVC", SVC()),
          ("GBM", GradientBoostingClassifier()),
          ("XGBoost", XGBClassifier()),
          ("LightGBM", LGBMClassifier())]

for name, regressor in models:
    cv = cross_validate(regressor, x, y, cv=5, scoring=["accuracy", "f1", "recall", "precision"])
    print(f'{name} \n {"Accuracy"}:{cv["test_accuracy"].mean()} \n {"Recall"}:{cv["test_recall"].mean()} '
          f'\n {"Precision"}:{cv["test_precision"].mean()} \n {"F-Score"}:{cv["test_f1"].mean()}')
