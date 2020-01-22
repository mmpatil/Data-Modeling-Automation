import sys
import datetime as dt
import ecm2 as ecm2
import json
from pathlib import Path
import database as databaseConnection
import pandas as pd

# script usage : python test.py <path to config.json> <env>
# example : python3 testARDL.py config/config.json development
# make a new instance of the DbConnection class to handle all our db connections

with open('../mas-ds/mas-ardl/config.json') as f:
# with open('config.json') as f:
  configjson = json.load(f)

data_folder = Path("../mas-ds/uploads")
# data_folder = Path("../uploads")

filepath = data_folder / configjson['filename']

''' getting a database connection object to insertion and updating the database'''
connection = databaseConnection.getDatabaseConnectionObject1(sys.argv[1], sys.argv[2])

modelType = configjson['model']

df, transformations = ecm2.wrangle_model_data(filepath, configjson['sheetname'])

df = df.loc[configjson['start']:configjson['end']]  # Filtering the dataframe based on the index of the rows

df = df.dropna(axis=1)

if 'dependentCol' in configjson:
    depName = configjson['dependentCol']
else:
    depName = df.index[0]
dep = df[depName]

dep.index = df.index
dep.name = depName
dependentVariableName = str(df.columns[0])

'''inserting a row in the database for the run'''
runId = databaseConnection.createNewRunDetail(Connection=connection, StartDate=dt.datetime.now(), ModelType=modelType)
depVariableID = databaseConnection.createNewDependentVariable(Connection=connection, RunId=runId, Name=dependentVariableName)

''' saving the user input to database'''
databaseConnection.fillInUserInput(connection, configjson, RunID=runId)

# find the data generating process of the dependent variable
dgpa = ecm2.find_dgp(dep)

# find the order of integration of the dependent variables
dep_order = ecm2.integration_order(dep, alpha=float(configjson['adf_alpha']))

'''creating a Series of stats tests received from config'''
stats_tests = configjson['stats_test']
stats_Test_Series = pd.Series(stats_tests)
stats_Test_Series.index = stats_tests

'''getting exlusion dates from the database'''
exclusions = []
for val in configjson['exclusions']:
    valArr = []
    dates = val.split(',')
    if(len(dates) > 1):
        for date in dates:
            valArr.append(pd.to_datetime(date))
    else:
        valArr = [pd.to_datetime(dates[0])]
    exclusions.append(valArr)

base, dummyNames = ecm2.exclusions_to_dummies(dep, exclusions)

regs = ecm2.int_filter(df, dep_order, alpha=float(configjson['adf_alpha']))

reasons, candidates = ecm2.run_stats_tests(dummyNames, transformations, stats_Test_Series, connection, runId, base, regs, modelType, float(configjson['param_alpha']), float(configjson['bg_alpha']), float(configjson['bp_alpha']), float(configjson['white_alpha']), float(configjson['sw_alpha']),
                                      float(configjson['adf_alpha']))


'''for storing intermediate output to database generate 3 jsons'''
depJSON = dep.to_json(orient='split', date_format='iso', date_unit='s')
regsJSON = regs.to_json(orient='split', date_format='iso', date_unit='s')
baseJSON = base.to_json(orient='split', date_format='iso', date_unit='s')

''' calling function to save intermediate output'''
databaseConnection.saveIntermediateOutputForRunID(connection, baseJSON, regsJSON, depJSON, runId)

''' calling function to convert the job status to SUCCESS'''
databaseConnection.updateStatusOfARun(Connection=connection, runId=runId, EndDate=dt.datetime.now(), Status='SUCCESS')