import pandas as pd
import sys
import math
import datetime
import numpy as np
import json
# function for getting the previous quarterend given a date
def previous_quarter(ref):
    ref = datetime.datetime.strptime(ref, '%Y-%m-%d')
    if ref.month < 4:
        return datetime.date(ref.year - 1, 12, 31)
    elif ref.month < 7:
        return datetime.date(ref.year, 3, 31)
    elif ref.month < 10:
        return datetime.date(ref.year, 6, 30)
    return datetime.date(ref.year, 9, 30)

# function for getting the previous monthend given a date
def previous_month(ref):
    print("Ref : ", ref)
    ref = datetime.datetime.strptime(ref, '%Y-%m-%d').replace(day=1)
    return ref - datetime.timedelta(days=1)

def oneFactorTransform_data(file, sheet_name, transform_type, date_column="Date"):
    with open('../mas-ds/mas-ardl/config.json') as f:
        configjson = json.load(f)
    df = pd.read_excel(file, sheet_name = sheet_name)

    # df[date_column] = df[date_column].dt.strftime('%Y-%m-%d')
    try:
        df.index = pd.to_datetime(df[date_column], format='%Y-%m-%d') # make date column as the index of the dataframe
        df = df.drop(date_column, axis=1)
    except:
        print("Your data does not contain Date column")

    for i in df.columns:
        if 'MEV-MSA' in i:
            df = df.drop(i, axis=1)

    transformation = ""
    tdf = pd.DataFrame()
    if transform_type == 'DIFFERENCE':
        transformation = transform_type
        tdf = difference(df)
    elif transform_type == 'LOG':
        print("LN")
        tdf=log(df)
        transformation = transform_type
    elif transform_type == 'LOG-DIFFERENCE':
        print("RAW_DIFF")
        tdf=log_diff(df)
        transformation = transform_type
    elif transform_type == 'PERCENT-DIFFERENCE':
        tdf = percent_diff(df)
        transformation = transform_type
    elif transform_type == 'LAG':
        tdf=lag(df)
        transformation = transform_type

    # Rename columns and remove MEV prefix
    cols = []
    for i in df.columns:
        if 'MEV' in i:
            cols.append(i[4:])
        else:
            cols.append(i)

    df.columns = cols
    # print("DataFrame : ", df)

    return [df, transformation, tdf]

def difference(df, l=1):
    tdf = df - df.shift(1)
    if not isinstance(df, pd.Series):
        cols = []
        for i in df.columns:
            cols.append(i + '_diff')
        tdf.columns = cols
    return tdf

def log(df):
    tdf = np.log(df)
    if not isinstance(df, pd.Series):
        cols = []
        for i in df.columns:
            cols.append(i + '_log')
        tdf.columns = cols
    return tdf

def log_diff(df, l=1):
    tdf = np.log(df) - np.log(df).shift(l)
    if not isinstance(df, pd.Series):
        cols = []
        for i in df.columns:
            cols.append(i + '_log_diff')
        tdf.columns = cols

    return tdf

def percent_diff(df, l=1):
    tdf = df / df.shift(1)-1
    if not isinstance(df, pd.Series):
        cols = []
        for i in df.columns:
            cols.append(i + '_percent_diff')
        tdf.columns = cols
    return tdf

def lag(df, l=1):
    tdf = df.shift(1)
    if not isinstance(df, pd.Series):
        cols = []
        for i in df.columns:
            cols.append(i + '_lag')
        tdf.columns = cols
    return tdf


# LN transformation func
def ln_transformation(df, keys=None):
    for (name, values) in df.items():
        #if name in keys.keys() and keys[name][0] == "Nominal":
        if 'MEV' in name:
            for i, val in values.items():
                if isinstance(val, float) or isinstance(val, int):
                    if val > 0:
                        df.at[i, name] = np.log(val)
                    else:
                        df.at[i, name] = 0
    return df
    # df.to_csv(outputfile_name + ".csv", header = True, index = False)



def raw_difference_transformation0(df, raw_df, isMonthly):
    dateSeries = pd.Series(df.index, name = 'Date')
    print("DateSeries : ", dateSeries)

    beginDate = datetime.datetime.strptime(dateSeries[0], "%Y-%m-%d")
    year = beginDate.year
    month  = beginDate.month
    day = beginDate.day
    beginDate = datetime.date(year, month, day)
    print("BeginDate : ", beginDate)
    for(name, column) in df.items():
        for i, val in column.items():
            if pd.isnull(val) == False:
                pastDate = None
                if isMonthly == True:
                    pastDate = previous_month(i)
                else:
                    pastDate = previous_quarter(i)
                if pastDate > beginDate:
                    print(" Past Date : ", pastDate)
                    pastDate = pastDate.strftime("%Y-%m-%d")
                    if(float(raw_df.at[pastDate, name]) > 0):
                        df.at[i, name] = ( float(val) / float(raw_df.at[pastDate, name])) - 1
                    #df.at[i, name] = val - raw_df.at[pastDate, name] - 1
    return df


# raw_diff func
def raw_difference_transformation(df, raw_df, isMonthly, keys=None):
    dateSeries = df.index
    print("Date Series : ", dateSeries)
    beginDate = datetime.datetime.strptime(dateSeries[0], '%Y-%m-%d')
    dateSeries1 = pd.DataFrame(dateSeries)
    print("Begin Date : ", beginDate)
    for (name, column) in df.items():
        nominal = True
        for i, val in column.items():
            if pd.isnull(val)== False:
                # Get previous date & dateIdx
                pastDate = None
                if isMonthly == True:
                    pastDate = previous_month(df.at[i, 'Date'])
                else:
                    pastDate = previous_quarter(df.at[i, 'Date'])
                if(pd.to_datetime(pastDate) > beginDate):
                    #print("dateSeries[dateSeries == pd.to_datetime(pastDate)]: ", dateSeries[dateSeries == pd.to_datetime(pastDate)])
                    #print("Past Date : ", dateSeries[pd.to_datetime(pastDate)])
                    print("Past Date : ", pastDate)
                    pastDateIdx = dateSeries[dateSeries == pastDate].index[0]
                    print("Type : ", type(dateSeries))
                    print("pastDateIdx : ", pastDateIdx)
                    # Calculate Raw Difference
                    if nominal == True:
                        df.at[i, name] = (val / raw_df.at[pastDateIdx, name]) - 1
                    else:
                        df.at[i, name] = val - raw_df.at[pastDateIdx, name]
    return df
    # df.to_csv(outputfile_name + ".csv", header = True, index = False)

# ln_diff func
def ln_diff_transformation(df, ln_df, isMonthly, keys=None):
    # set Date data series
    dateSeries = df[config.date]
    beginDate = df.at[0, config.date]

    for (name, column) in df.items():
        if name in keys.keys():
            for i, val in column.items():
                # if the value is not null
                if pd.isnull(val)== False:
                    # Get previous quarter & quarterIdx
                    pastDate = None
                    if isMonthly == True:
                        pastDate = previous_month(df.at[i, config.date])
                        # pastDate =
                    else:
                        pastDate = previous_quarter(df.at[i, config.date])
                    if(pd.to_datetime(pastDate) > beginDate):
                        pastDateIdx = dateSeries[dateSeries == pd.to_datetime(pastDate)].index[0]
                        df.at[i, name] = val - ln_df.at[pastDateIdx, name]

    return df
    # df.to_csv(outputfile_name + ".csv", header = True, index = False)


#   make a deep copy of quarterly and monthly data (note: chosen at random, could easily be transformation on user data)
# raw series
# oneQ = quarterly_data.copy()
# oneM = monthly_data.copy()
#
# twoQ = quarterly_data.copy()
# twoM = monthly_data.copy()

# LN series
# ln_transformation(twoQ, "quarterly_LN")
# ln_transformation(twoM, "monthly_LN")

# the difference in raw series
# threeQ = quarterly_data.copy()
# threeM = monthly_data.copy()
# raw_difference_transformation(threeQ, oneQ, "quarterly_RawDiff", False)
# raw_difference_transformation(threeM, oneM, "monthly_RawDiff", True)
#
# # the difference in LN series
# fourQ = twoQ.copy()
# fourM = twoM.copy()
# ln_diff_transformation(fourQ, twoQ, "quarterly_LNDiff", False)
# ln_diff_transformation(fourM, twoM, "monthly_LNDiff", True)
