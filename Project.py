import pandas as pd
import numpy as np
from numpy import mean
from numpy import std
import seaborn as sns
import matplotlib.pyplot as plt

def print_hl():
    print('---------------------------------'
          '--------------------------------'
          '---------------------------------------')


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


# **importing dataset**
# (read data , check duplicate ids , create dataframe):

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
print(data_frame.describe())

# **Plotting Data before Cleaning**

print('bmi plot')
sns.displot(data_frame['bmi'])

print('average glucose level plot')
sns.displot(data_frame['avg_glucose_level'])


# **Data Cleaning:**
# Filling missing and unkown values with mean (numiric) and most frequent (categorical)

# Getting actual mean value of **bmi** after removing outliers and replace unkown values with it

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

# Replace unkown categorical data in smoking status to most frequent

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

# Another technique to replace unknown data using **Simple imputer** (to try not used)

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

# **Visualizing data after cleaning:**

print('bmi plot after replacing missing values and removing outliers')
sns.displot(filtered_data['bmi'])

print('average glucose level plot after removing outliers')
sns.displot(filtered_data['avg_glucose_level'])

# Replacing any spaces in data to _

f_data_frame['smoking_status'] = f_data_frame['smoking_status'].str.replace('\s+', '_', regex=True)
f_data_frame

# **Splitting Data to Train and Test Data:**
#  Avoid overfitting and underfitting

from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split

y_true = f_data_frame['stroke'].values
x_train, data_test, y_train, y_test = train_test_split(f_data_frame, y_true, stratify=y_true, test_size=0.2)
data_train, data_cv, y_train, y_cv = train_test_split(x_train, y_train, stratify=y_train, test_size=0.2)

print("> Train Data:", data_train.shape[0])
print(data_train['stroke'].value_counts())
data_train['stroke'].value_counts().plot(kind='bar')
plt.show()
print_hl()
print("> Test Data:", data_test.shape[0])
print(data_test['stroke'].value_counts())
data_test['stroke'].value_counts().plot(kind='bar')
plt.show()
print_hl()
print("> Cross validation Data:", data_cv.shape[0])
print(data_cv['stroke'].value_counts())
data_cv['stroke'].value_counts().plot(kind='bar')
plt.show()
print_hl()
