import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import KBinsDiscretizer

plt.style.use("seaborn-v0_8")


def print_hl():
    print('---------------------------------'
          '--------------------------------'
          '---------------------------------------'
          '--------------------------------------')


# ---------- removing outliers algorithms --------------

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
print_hl()
# ---------- Filling N/A with mean value --------------
# Removing outliers to calculate mean in bmi
clean_bmi_list = [x for x in data['bmi'] if str(x) != 'nan']
outliers_remove = [clean_bmi_list]
bmi_no_outliers = remove_outliers_iqr(outliers_remove, 0)
# calculating mean of bmi without outliers
data_bmi_mean = np.mean(bmi_no_outliers)
# Filling bmi N/A values with mean Data Cleaning
data['bmi'] = data['bmi'].replace(np.NaN, data_bmi_mean)

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
# print(data)


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

# --------------- filtering outliers --------------------

# no age outliers
# filtering outliers in bmi
bmi_clean = remove_outliers_iqr(data, 'bmi')
filtered_bmi_data = data.loc[data['bmi'].isin(bmi_clean)]
# filtering outliers in average glucose level
avg_glc_clean = remove_outliers_iqr(data, 'avg_glucose_level')
filtered_avg_glc_data = data.loc[data['avg_glucose_level'].isin(avg_glc_clean)]
# filtering outliers in both bmi and average glucose level
filtered_data = filtered_bmi_data.loc[data['avg_glucose_level'].isin(avg_glc_clean)]
print('**** Data Set after excluding outliers an filling missing data ****')
print(filtered_data)
print_hl()

# -------------------------- Data Transformation --------------------------------

# ---------- Convert all categorical data into numeric --------------
Label_Encoder = LabelEncoder()
for i in data.select_dtypes(include=['object']).columns.tolist():
    data[i] = Label_Encoder.fit_transform(data[i])

print(data)

# ---------- Discretization --------------
discretization = KBinsDiscretizer(n_bins=3, strategy='uniform', encode='ordinal')
c = discretization.fit(data)
print("Discretization bin edges: ")
print(c.bin_edges_)

dataframe = discretization.transform(data)
print("Data after discretization: \n", dataframe)

# ------------------------- Normalization ---------------------------
normalize_features = ['age', 'avg_glucose_level', 'bmi']
# --------------- MinMax Normalization ------------------
norm_filtered_data_frame = pd.DataFrame(filtered_data)
filtered_num_data_frame = norm_filtered_data_frame[normalize_features]
normalizer = MinMaxScaler(feature_range=(0, 1))
norm_data_frame = normalizer.fit_transform(filtered_num_data_frame)
norm_filtered_data_frame[normalize_features] = norm_data_frame
print('> Min Max Normalized range 0 -> 1 filtered data ')
print(norm_filtered_data_frame)
print_hl()

# --------------- z-score Normalization ------------------
z_filtered_data_frame = pd.DataFrame(filtered_data)

for i in normalize_features:
    z_filtered_data_frame[i] = zscore(z_filtered_data_frame[i])
print('> z-score Normalized filtered data ')
print(z_filtered_data_frame)
print_hl()


# --------------- linear regression ---------------------
age_mean = filtered_data.age.mean()
avg_glc_mean = filtered_data.avg_glucose_level.mean()
bmi_mean = filtered_data.bmi.mean()
# calculating standard deviation
age_sum = sum((data.age - age_mean)**2)
avg_glc_sum = sum((data.avg_glucose_level - avg_glc_mean)**2)
bmi_sum = sum((data.bmi - bmi_mean)**2)

bmi_av_glc = sum((data.bmi - bmi_mean)*(data.avg_glucose_level - avg_glc_mean))

b1 = bmi_av_glc / bmi_sum
b0 = avg_glc_mean - (b1 * bmi_mean)


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
# plotting MinMax normalized filtered data
# sns.regplot(x="bmi", y="avg_glucose_level", data=norm_filtered_data_frame)
# plt.title("BMI / Average glucose level plot MinMax normalized filtered data")
# plt.show()
# sns.regplot(x="age", y="bmi", data=norm_filtered_data_frame)
# plt.title("Age / BMI plot MinMax normalized filtered data")
# plt.show()
# sns.regplot(x="age", y="avg_glucose_level", data=norm_filtered_data_frame)
# plt.title("Age / Average glucose level plot MinMax normalized filtered data")
# plt.show()
# plotting z-score normalized filtered data
# sns.regplot(x="bmi", y="avg_glucose_level", data=z_filtered_data_frame)
# plt.title("BMI / Average glucose level plot z-score normalized filtered data")
# plt.show()
# sns.regplot(x="age", y="bmi", data=z_filtered_data_frame)
# plt.title("Age / BMI plot z-score normalized filtered data")
# plt.show()
# sns.regplot(x="age", y="avg_glucose_level", data=z_filtered_data_frame)
# plt.title("Age / Average glucose level plot z-score normalized filtered data")
# plt.show()


# --------------------- Boxplot ------------------------
# # Boxplot visualization for age
# sns.boxplot(extracting_age_list)
# plt.title("Age before removing outliers")
# plt.show()
#
# sns.boxplot(data['age'])
# plt.title("Age after removing outliers")
# plt.show()
#
# # Boxplot visualization for avg_glucose
# sns.boxplot(extracting_avg_glucose_list)
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

