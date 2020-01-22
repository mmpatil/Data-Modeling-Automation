import ecm3 as ecm
import sys
import json
import database as databaseConnection
import datetime
import pandas as pd
import OneFactorTransformation as oneFactorTransformations
import datetime as dt
import numpy as np
from pathlib import Path

#with open('../mas-ds/mas-ardl/config.json') as f:
with open('config_old.json') as f:
    configjson = json.load(f)

#data_folder = Path("../mas-ds/uploads")
data_folder = Path("../uploads")

filepath= data_folder / configjson['filename']

connection = databaseConnection.getDatabaseConnectionObject1(sys.argv[1], sys.argv[2])
ModelType = configjson['model']

df, transformations, tdf = oneFactorTransformations.oneFactorTransform_data(filepath, configjson['sheetname'], transform_type="DIFFERENCE")

df = df.loc[configjson['start']:configjson['end']]

df = df.dropna(axis=1)

if configjson['dependentCol']:
    depName = configjson['dependentCol']
else:
    depName = df.index[0]

dep = df[depName]
dep.name = depName
dependentVariableName = str(df.columns[0])

mev_df = df.drop(depName, axis=1)

diff_df = oneFactorTransformations.difference(df)

log_df = oneFactorTransformations.log(df)

log_diff_df = oneFactorTransformations.log_diff(df)

percent_diff_df = oneFactorTransformations.percent_diff(df)

#dep_diff = oneFactorTransformations.difference(dep)
#dep_diff = oneFactorTransformations.percent_diff(dep)
dep_diff = oneFactorTransformations.log_diff(dep)
#dep_diff = oneFactorTransformations.lag(dep)

regs = pd.concat([mev_df, diff_df, log_df, log_diff_df, percent_diff_df], axis=1)

regs = regs.replace([np.inf, -np.inf], np.nan)
regs = regs.drop(regs.index[0])
regs = regs.dropna(axis=1)

dep_diff = dep_diff.drop(dep_diff.index[0])

runId = databaseConnection.createNewRunDetail(Connection=connection, StartDate=datetime.datetime.now(), ModelType=ModelType)
depVariableID = databaseConnection.createNewDependentVariable(Connection=connection, RunId=runId, Name=dependentVariableName)

databaseConnection.fillInUserInput(connection, configjson, RunID=runId)

# find the data generating process of the dependent variable
dgpa = ecm.find_dgp(dep)

# find the order of integration of dependent variable
dep_order = ecm.integration_order(dep, alpha=float(configjson['adf_alpha']))

stats_tests = configjson['stats_test']

stats_Test_Series = pd.Series(stats_tests)

stats_Test_Series.index = stats_tests

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

base, dummyNames = ecm.exclusions_to_dummies(dep, exclusions)

base.index = pd.to_datetime(base.index, format='%Y-%m-%d')
regs.index = pd.to_datetime(regs.index, format='%Y-%m-%d')

base = base.loc[regs.index]

reasons, candidates = ecm.run_stats_tests(dummyNames, transformations, stats_Test_Series, connection, runId, base, regs, ModelType, float(configjson['param_alpha']), float(configjson['bg_alpha']), float(configjson['bp_alpha']), float(configjson['white_alpha']), float(configjson['sw_alpha']),float(configjson['adf_alpha']))

print("Candidates : ", candidates)

depJSON = dep.to_json(orient='split', date_format='iso', date_unit='s')
regsJSON = regs.to_json(orient='split', date_format='iso', date_unit='s')

baseJSON = base.to_json(orient='split', date_format='iso', date_unit='s')

databaseConnection.saveIntermediateOutputForRunID(connection, baseJSON, regsJSON, depJSON, runId)

databaseConnection.updateStatusOfARun(Connection=connection, runId=runId, EndDate=dt.datetime.now(), Status='SUCCESS')



