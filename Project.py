import pandas as pd
import numpy as np
from numpy import mean
from numpy import std
import seaborn as sns
import matplotlib.pyplot as plt

### Functions Definition:

def print_hl():
    print('---------------------------------'
          '--------------------------------'
          '---------------------------------------')


# Setting removing ouliers techniques
# IQR (Used as default)
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


# Ploting Data features function defintion

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


###  Project Steps

# importing dataset
# read data , check duplicate ids , create dataframe

data_raw = pd.read_csv('healthcare-dataset-stroke-data.csv')

# checking if there duplicated id
no_duplicates = True
is_duplicated = data_raw.duplicated()
for x in is_duplicated:
    if x:
        no_duplicates = False
if no_duplicates:
    print('No duplicate id in the Dataset')

# creating dataframe
data_frame = pd.DataFrame(data_raw)
raw_data_frame = data_frame.copy()
data_frame.describe()

# Plotting Data before Cleaning

print('bmi plot')
sns.displot(data_frame['bmi'])
plt.show()
print_hl()

print('average glucose level plot')
sns.displot(data_frame['avg_glucose_level'])
plt.show()
print_hl()

# **Data Cleaning:**
# Filling missing and unknown values with mean (numiric) and most frequent (categorical)

# Getting actual mean value of **bmi** after removing outliers and replace unknown values with it

# Removing outliers to calculate mean in bmi
clean_bmi_list = [x for x in data_frame['bmi'] if str(x) != 'nan']
print("> Ignoring N/A tuples in bmi list to work on it")
outliers_remove = [clean_bmi_list]
# removing outliers to calculate mean value accuratly
bmi_no_outliers = remove_outliers_iqr(outliers_remove, 0)
print("> removing outliers to calculate mean value accuratly")
# calculating mean of bmi without outliers
data_bmi_mean = np.mean(bmi_no_outliers)
print("> replacing unknown values with mean")
print("bmi mean value:", data_bmi_mean)
# Filling bmi N/A values with mean Data Cleaning
data_frame['bmi'] = data_frame['bmi'].replace(np.NaN, data_bmi_mean)

# Replace unknown categorical data in smoking status to most frequent

# ---------- Filling unknown smoking data --------------
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

# Another technique to replace unknown data using Simple imputer (to try not used)


from sklearn.impute import SimpleImputer

data_frame_SI = raw_data_frame.copy()
# only include numerical attributes
numerical_features = data_frame_SI.select_dtypes(exclude=['object']).columns.tolist()
# dataframe of only numerical attributes
num_data_frame = data_frame_SI[numerical_features]
# data cleaning filling N/A using simpleimputer
imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
imp_mean.fit(num_data_frame)
num_data_frame = imp_mean.transform(num_data_frame)
si_bmi_mean = num_data_frame[1][5]
# data cleaning of categorical features
cat_features = data_frame.select_dtypes(include='object').columns.tolist()
cat_data_frame = data_frame[cat_features]
# data cleaning filling unknown using simpleimputer most frequent element
unk_imp = SimpleImputer(missing_values='Unknown', strategy='most_frequent')
unk_imp.fit(cat_data_frame)
cat_data_frame = unk_imp.transform(cat_data_frame)
# merge two data frames
simpleimputed_data = np.hstack((num_data_frame, cat_data_frame))
print('> by comparing two methods mean bmi using simpleImputer = ', si_bmi_mean)
print('while by calculating with excluding outliers = ', data_bmi_mean)
print('> Replacing Unknown Smoking status with most common:', smk_most_freq)

# Removing Outliers using **IQR** technique:

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
filtered_data

# Visualizing data after cleaning

print('bmi plot after replacing missing values and removing outliers')
sns.displot(filtered_data['bmi'])
plt.show()
print_hl()

print('average glucose level plot after removing outliers')
sns.displot(filtered_data['avg_glucose_level'])
plt.show()
print_hl()

# Replacing any spaces in data to _

f_data_frame['smoking_status'] = f_data_frame['smoking_status'].str.replace('\s+', '_', regex=True)
f_data_frame

# Data Encoding: converting string data to numiric values using label encoder

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import KBinsDiscretizer

Label_Encoder = LabelEncoder()
for i in f_data_frame.select_dtypes(include=['object']).columns.tolist():
    f_data_frame[i] = Label_Encoder.fit_transform(f_data_frame[i])

# Splitting Data to Train and Test Data
#  Avoid overfitting and underfitting


from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree

x = f_data_frame.loc[:, f_data_frame.columns != 'stroke']
x = x.loc[:, x.columns != 'id']
y = f_data_frame['stroke']
x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.2)

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

# Split data using cross validation

# cross validation code here


# Using Decision Tree Classifier to create the model

clf = DecisionTreeClassifier()
clf = clf.fit(x_train, y_train)

# predict stroke
res = clf.predict(x_test)

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, res)
print("Accuracy = ", accuracy)

res_df = pd.DataFrame(res, columns=['stroke_prediction'])
print("> prediction result:", res_df.shape[0])
print(res_df.value_counts())
res_df['stroke_prediction'].value_counts().plot(kind='bar')
plt.show()

# Finding vital role attributes

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

plot_feature_importance(clf.feature_importances_, x_train.columns, 'RANDOM FOREST')
plt.show()
print_hl()

# Classification Using Random forrest regressioin:

from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance

rf = RandomForestRegressor(n_estimators=150)
rf.fit(x_train, y_train)
sort = rf.feature_importances_.argsort()
plot_feature_importance(rf.feature_importances_, x_train.columns, 'RANDOM FOREST')
plt.show()
# Visualize Data
