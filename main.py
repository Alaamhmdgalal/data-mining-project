import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import zscore
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from numpy import mean
from numpy import std

plt.style.use("seaborn-v0_8")


def print_hl():
    print('---------------------------------'
          '--------------------------------'
          '---------------------------------------'
          '--------------------------------------')


# ---------- removing outliers algorithms --------------

# remove outliers using std
def remove_outliers_std(data_with_outliers, col):
    data_mean, data_std = mean(data_with_outliers[col]), std(data_with_outliers[col])
    cut_off = data_std * 3
    lower, upper = data_mean - cut_off, data_mean + cut_off
    outliers = [x for x in data_with_outliers[col] if x < lower or x > upper]
    # print('Identified outliers: %d' % len(outliers))
    outliers_removed = [x for x in data_with_outliers[col] if x >= lower and x <= upper]
    # print('Non-outlier observations: %d' % len(outliers_removed))
    return outliers_removed

# remove outliers using IQR
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

# ---------- importing dataset from .csv --------------
data = pd.read_csv('healthcare-dataset-stroke-data.csv')
data_raw = pd.read_csv('healthcare-dataset-stroke-data.csv')

# checking if there duplicated id
print(data)
no_duplicates = True
is_duplicated = data.duplicated()
for x in is_duplicated:
    if x:
        no_duplicates = False
if no_duplicates:
    print('No duplicate id in the Dataset')
print_hl()

# ---------- Filling N/A with mean value --------------
# Removing outliers to calculate mean in bmi
clean_bmi_list = [x for x in data['bmi'] if str(x) != 'nan']
print("Ignoring N/A tuples in bmi list: \n", clean_bmi_list)
outliers_remove = [clean_bmi_list]
bmi_no_outliers = remove_outliers_iqr(outliers_remove, 0)
# calculating mean of bmi without outliers
data_bmi_mean = np.mean(bmi_no_outliers)
# Filling bmi N/A values with mean Data Cleaning
data['bmi'] = data['bmi'].replace(np.NaN, data_bmi_mean)
print("Filling bmi N/A values with mean values: \n", data['bmi'])
print_hl()


# ---------- Filling unknown smoking data --------------
# finding mean status of smoking
smk_st = ['never smoked', 'formerly smoked', 'smokes']
smk_st_cout = [0, 0, 0]
for x in data['smoking_status']:
    if str(x) == 'Unknown':
        pass
    else:
        smk_st_cout[smk_st.index(str(x))] += 1
smk_most_freq = smk_st[smk_st_cout.index(max(smk_st_cout))]
# filling unknown smoking data with smoking mean value
data['smoking_status'] = data['smoking_status'].replace('Unknown', smk_most_freq)
print("Filling smoking status unknown values with most frequent values: \n", data['smoking_status'])
print_hl()


# --------- Data cleaning using simpleimputer ----------
data_frame = pd.DataFrame(data_raw)
# only include numerical attributes
numerical_features = data_frame.select_dtypes(exclude=['object']).columns.tolist()
# dataframe of only numerical attributes
num_data_frame = data_frame[numerical_features]
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
# print(simpleimputed_data)
print('by comparing two methods mean bmi using simpleImputer = ', si_bmi_mean,
      'while by calculating with excluding outliers = ', data_bmi_mean)
print('Replacing Unknown Smoking status with most common:', smk_most_freq)
print_hl()
print("Data cleaning using simple imputer (alternative method): \n", simpleimputed_data)
print_hl()


# --------------- filtering outliers --------------------

# ---------------------- std method -----------------------------
unfiltered_data_frame_001 = data.copy()
# print('unfiltered Data')
# print(unfiltered_data_frame_001)
# trying std method
print('> removing outliers using std')
# bmi filtering
std_filtered_bmi = remove_outliers_std(unfiltered_data_frame_001,'bmi')
std_filtered_data = unfiltered_data_frame_001.loc[data['bmi'].isin(std_filtered_bmi)]
# avg glucose level filtering
std_filtered_avg_glc = remove_outliers_std(std_filtered_data,'avg_glucose_level')
std_filtered_data = std_filtered_data.loc[data['avg_glucose_level'].isin(std_filtered_avg_glc)]
print('> data without outliers using std method')
print(std_filtered_data)

# ---------------------- iqr method -----------------------------

# no age outliers
# filtering outliers in bmi
bmi_clean = remove_outliers_iqr(data, 'bmi')
filtered_bmi_data = data.loc[data['bmi'].isin(bmi_clean)]
# filtering outliers in average glucose level
avg_glc_clean = remove_outliers_iqr(data, 'avg_glucose_level')
filtered_avg_glc_data = data.loc[data['avg_glucose_level'].isin(avg_glc_clean)]
# filtering outliers in both bmi and average glucose level
filtered_data = filtered_bmi_data.loc[data['avg_glucose_level'].isin(avg_glc_clean)]
f_data_frame = pd.DataFrame(filtered_data)
print('**** Data Set after excluding outliers and filling missing data ****')
print(filtered_data)
print_hl()

# -------------------------- Data Transformation --------------------------------

# ---------- Convert all categorical data into numeric --------------
Label_Encoder = LabelEncoder()
f_data_frame_004 = f_data_frame.copy()
for i in f_data_frame_004.select_dtypes(include=['object']).columns.tolist():
    f_data_frame_004[i] = Label_Encoder.fit_transform(f_data_frame_004[i])
print("Data after encoding categorical data: \n", f_data_frame_004)
print_hl()


# ---------- Discretization --------------
discretization = KBinsDiscretizer(n_bins=3, strategy='uniform', encode='ordinal')
c = discretization.fit(f_data_frame_004)
print("Discretization bin edges: ")
print(c.bin_edges_)

dataframe = discretization.transform(f_data_frame_004)
print("Data after discretization: \n", dataframe)
print_hl()

# ------------------------- Normalization ---------------------------
normalize_features = ['age', 'avg_glucose_level', 'bmi']
# --------------- MinMax Normalization ------------------
f_data_frame_001 = f_data_frame.copy()
norm_filtered_data_frame = f_data_frame_001
filtered_num_data_frame = norm_filtered_data_frame[normalize_features]
normalizer = MinMaxScaler(feature_range=(0, 1))
norm_data_frame = normalizer.fit_transform(filtered_num_data_frame)
norm_filtered_data_frame[normalize_features] = norm_data_frame
print('> Min Max Normalized range 0 -> 1 filtered data ')
print(norm_filtered_data_frame)
print_hl()

# --------------- z-score Normalization ------------------
f_data_frame_002 = f_data_frame.copy()
z_filtered_data_frame = f_data_frame_002

for i in normalize_features:
    z_filtered_data_frame[i] = zscore(z_filtered_data_frame[i])
print('> z-score Normalized filtered data ')
print(z_filtered_data_frame)
print_hl()

# --------------- Correlation Matrix ---------------------
matrix = f_data_frame_004.corr(method='pearson', min_periods=1)
print("Correlation matrix: \n", matrix)
print_hl()

# --------------------- train test split ------------------------
print('> Test Train Split')
f_data_frame_003 = f_data_frame.copy()
print(f_data_frame_003['stroke'].value_counts())
input_features = ['age', 'bmi', 'avg_glucose_level']
X = f_data_frame_003[input_features]
Y = f_data_frame_003['stroke']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.9, random_state=2)
print("Input train set: ", X_train.shape)
print("Input test set: ", X_test.shape)
print("Output train set: ", y_train.shape)
print("Output test set: ", y_test.shape)
print_hl()

# --------------- linear regression ---------------------
print("Linear regression using correlation rules")
age_mean = filtered_data.age.mean()
avg_glc_mean = filtered_data.avg_glucose_level.mean()
bmi_mean = filtered_data.bmi.mean()
stroke_mean = filtered_data.stroke.mean()
# print("Stroke mean: ", stroke_mean)
# calculating standard deviation
age_sum = sum((data.age - age_mean)**2)
avg_glc_sum = sum((data.avg_glucose_level - avg_glc_mean)**2)
bmi_sum = sum((data.bmi - bmi_mean)**2)
age_sum_s = sum((data.age - age_mean)*(data.stroke - stroke_mean))
avg_glc_sum_s = sum((data.avg_glucose_level - avg_glc_mean)*(data.stroke - stroke_mean))
bmi_sum_s = sum((data.bmi - bmi_mean)*(data.stroke - stroke_mean))

b1 = age_sum_s / age_sum
b2 = avg_glc_sum_s / avg_glc_sum
b3 = bmi_sum_s / bmi_sum
b0 = stroke_mean - b1*age_mean - b2*avg_glc_mean - b3*bmi_mean

predicted_stroke = b0 + b1*(X_test.age) + b2*(X_test.avg_glucose_level) + b3*(X_test.bmi)
print("Predicted data: \n", predicted_stroke)
print_hl()
print("Accurate data: \n", y_test)
print_hl()


# --------------- Important features ---------------------
decision_tree = DecisionTreeClassifier()
drop_features = ['id', 'stroke']
decision_tree.fit(f_data_frame_004.drop(drop_features, axis=1), f_data_frame_004['stroke'])
plt.style.use('ggplot')
plt.bar(decision_tree.feature_names_in_, decision_tree.feature_importances_)
plt.show()

# --------------- Predictions ---------------------
# model = LinearRegression()
# model.fit(X_train, y_train)
# Y_predict = model.predict(X_test)
# print("Y predict: \n", Y_predict)
# results = pd.DataFrame(Y_predict)
# print(results)

print("** Prediction using logistic regression **")
model_logistic = LogisticRegression()
model_logistic.fit(X_train, y_train)
Y_predict_logistic = model_logistic.predict(X_test)
print("Y predict logistic: \n", Y_predict_logistic)
count_p = np.count_nonzero(Y_predict_logistic == 1)
print("Number of predicted ones: \n", count_p)
print('Accuracy of logistic regression classifier on test set:'' {:.2f}'.format(model_logistic.score(X_test, y_test)))
print_hl()


# --------------------- plotting ------------------------

# ------------------ Scatter plotting -------------------

# sns.regplot(x="bmi", y="avg_glucose_level", data=data)
# plt.title("BMI / Average glucose level plot with outliers")
# plt.show()
# sns.regplot(x="age", y="bmi", data=data)
# plt.title("Age / BMI plot with outliers")
# plt.show()
# sns.regplot(x="age", y="avg_glucose_level", data=data)
# plt.title("Age / Average glucose level plot with outliers")
# plt.show()
# # plotting without outliers
# sns.regplot(x="bmi", y="avg_glucose_level", data=filtered_data)
# plt.title("BMI / Average glucose level plot without outliers")
# plt.show()
# sns.regplot(x="age", y="bmi", data=filtered_data)
# plt.title("Age / BMI plot without outliers")
# plt.show()
# sns.regplot(x="age", y="avg_glucose_level", data=filtered_data)
# plt.title("Age / Average glucose level plot without outliers")
# plt.show()
# # plotting MinMax normalized filtered data
# sns.regplot(x="bmi", y="avg_glucose_level", data=norm_filtered_data_frame)
# plt.title("BMI / Average glucose level plot MinMax normalized filtered data")
# plt.show()
# sns.regplot(x="age", y="bmi", data=norm_filtered_data_frame)
# plt.title("Age / BMI plot MinMax normalized filtered data")
# plt.show()
# sns.regplot(x="age", y="avg_glucose_level", data=norm_filtered_data_frame)
# plt.title("Age / Average glucose level plot MinMax normalized filtered data")
# plt.show()
# # plotting z-score normalized filtered data
# sns.regplot(x="bmi", y="avg_glucose_level", data=z_filtered_data_frame)
# plt.title("BMI / Average glucose level plot z-score normalized filtered data")
# plt.show()
# sns.regplot(x="age", y="bmi", data=z_filtered_data_frame)
# plt.title("Age / BMI plot z-score normalized filtered data")
# plt.show()
# sns.regplot(x="age", y="avg_glucose_level", data=z_filtered_data_frame)
# plt.title("Age / Average glucose level plot z-score normalized filtered data")
# plt.show()
#
#
# # --------------------- Boxplot ------------------------
# # Boxplot visualization for age
# sns.boxplot(data_raw['age'])
# plt.title("Age before removing outliers")
# plt.show()
#
# sns.boxplot(data['age'])
# plt.title("Age after removing outliers")
# plt.show()
#
# # Boxplot visualization for avg_glucose
# sns.boxplot(data_raw['avg_glucose_level'])
# plt.title("Average glucose level before removing outliers")
# plt.show()
#
# sns.boxplot(data['avg_glucose_level'])
# plt.title("Average glucose level after removing outliers")
# plt.show()
#
# # Boxplot visualization for bmi
# sns.boxplot(clean_bmi_list)
# plt.title("Bmi before removing outliers")
# plt.show()
#
# sns.boxplot(bmi_no_outliers)
# plt.title("Bmi after removing outliers")
# plt.show()
#
# sns.boxplot(data['bmi'])
# plt.title("Bmi after filling nan values with mean")
# plt.show()

