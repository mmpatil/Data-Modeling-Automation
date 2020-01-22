import json
import datetime as dt
from pathlib import Path
import database as databaseConnection
import sys
import pandas as pd
import ecm2 as ecm2


def createAndSaveJSON(params, ps, connection, varname, runId):
  dictParams = params.to_dict('index')
  dictPvalues = ps.to_dict('index')

  for keys, values in dictParams.items():
    name = varname + ' model ' + keys + ' param.png'
    ecm2.getJsonFromDictionary('date', 'value', values, name, connection, runId, varname, 'RegressionPval')

  for keys, values in dictPvalues.items():
    name = varname + ' model ' + keys + ' pval.png'
    ecm2.getJsonFromDictionary('date', 'value', values, name, connection, runId, varname, 'RegressionParam')

with open('../mas-ds/mas-ardl/config.json') as f:
# with open('config.json') as f:
  configjson = json.load(f)

data_folder = Path("../mas-ds/uploads")
#data_folder = Path("../uploads")
filepath = data_folder / configjson['filename']

connection = databaseConnection.getDatabaseConnectionObject1(sys.argv[1], sys.argv[2])

runId = sys.argv[3]

intermediateOutput = databaseConnection.getIntermediateOutputForRunID(connection, runId=runId)

baseJSON = intermediateOutput.BaseDataframeJSON
regsJSON = intermediateOutput.RegsDataframeJSON
depJSON = intermediateOutput.DependentJSON

base = pd.read_json(baseJSON, orient='split')
regs = pd.read_json(regsJSON, orient='split')
dep = pd.read_json(depJSON, orient='split', typ='series')
dependentVariableTransformation = databaseConnection.getDependentVariableTransformationOnRunId(Connection=connection, runId=runId)

shortlistedList = databaseConnection.getShortListedValues(connection, runId)
candidateList = databaseConnection.getCandidateModelsForRunID(connection, runId)

mape, long_mape = ecm2.backtesting(runId, connection, candidateList, base, regs, configjson['backtest_dates'], configjson['backtest_long_dates'])

databaseConnection.fillBacktestResult(mape, long_mape, runId, connection, configjson)

# try:
#     short_list = shortlistedList
# except:
short_list = candidateList

for i in short_list:
    _,X = ecm2.create_design(base, regs, i)
    newdep = dep[X.index]
    n = X.index.get_loc(configjson['rq'])
    params, ps = ecm2.recursive_reg(newdep, X, n, varname=i)
    createAndSaveJSON(params, ps, connection, i, runId)

ecm2.stress_test_plot(filepath, configjson['shtm'], configjson['shtb'], configjson['shta'], configjson['shts'], short_list, configjson['pq0'],
                     configjson['pq1'], base, regs, connection, runId, configjson['bottom'], configjson['top'])

new_model_summary, old_model_summary = ecm2.stress_test_compare(filepath, configjson['shtm'], configjson['shtb'], configjson['shta'],
                                                               configjson['shts'], configjson['shtc'], short_list, configjson['pq0'],
                                                               configjson['pq1'], base, regs, dep, connection, runId, bottom=configjson['bottom'],
                                                               top=configjson['top'])

# ecm2.create_sensitivity(filepath, configjson['shtm'], configjson['shtb'], base, regs, short_list, configjson['pq0'], configjson['pq1'], connection, runId)

pqs = ['2008-03-31', '2012-03-31', '2015-03-31', '2016-03-31']
ecm2.out_of_time(candidateList, base, regs, pd.to_datetime('2014-12-31'), pqs, connection, runId)

databaseConnection.updateEndTimeForTests(connection,RunId=runId, EndTimeForTests=dt.datetime.now())
ecm2.compile_results(short_list)

ecm2.copy_output('key_results')

folder = str(dt.datetime.now())[:16]
ecm2.version_output(folder)
