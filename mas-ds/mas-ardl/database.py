import idb as db

def listToString(s):
    '''
        converting list to string
        inputs:
        * s - list of string
        outputs
        * str1 - concatenated string
    '''

    # initialize an empty string
    str1 = ""

    # traverse in the string
    for ele in s:
        str1 += ele

        # return string
    return str1

def fillInUserInput(connection, configjson, RunID):
    '''
           storing the userInput to the User Input table
           inputs:
           * connection - database connection object
           * configjson - config file
           * RunID - RunId for the current job
       '''
    if configjson['UserId'] != None:
        userId = connection.newUserInput(RunID=RunID, UserID=int(configjson['UserId']))
    else:
        userId = connection.newUserInput(RunID=RunID, UserID=0)

    if configjson['DWLimitHigh'] != None:
        connection.updateUserInput(RunID, DWLimitHigh=int(configjson['DWLimitHigh']))

    if configjson['DWLimitLow'] != None:
        connection.updateUserInput(RunID, DWLimitLow=int(configjson['DWLimitLow']))

    if configjson['WhiteSkedasticityLimit'] != None:
        connection.updateUserInput(RunID, WhiteSkedasticityLimit=int(configjson['WhiteSkedasticityLimit']))

    if configjson['BGLimit'] != None:
        connection.updateUserInput(RunID, BGLimit=int(configjson['BGLimit']))

    if configjson['VIFLimit'] != None:
        connection.updateUserInput(RunID, VIFLimit=int(configjson['VIFLimit']))

    if configjson['ADFLimit'] != None:
        connection.updateUserInput(RunID, ADFLimit=int(configjson['ADFLimit']))

    statsString = listToString(configjson['stats_test'])
    if configjson['stats_test'] != None:
        connection.updateUserInput(RunID, StatsTest=statsString)

    if 'dep_transform' in configjson:
        connection.updateUserInput(RunID, DependentVariableTransformation=configjson['dep_transform'])

    if configjson['dependentCol'] != None:
        connection.updateUserInput(RunID, DependentVariableName=configjson['dependentCol'])

    if configjson['backtest_dates'] != None:
        i = 1
        for date in configjson['backtest_dates']:
            columnName = 'DynamicBacktestRange' + str(i)
            columnValue = date
            i += 1
            updated = connection.updateBackTestRangeForUserInput(RunID, columnName, columnValue)

    if configjson['backtest_long_dates'] != None:
        i = 1
        for date in configjson['backtest_long_dates']:
            columnName = 'DynamicBacktestLongRange' + str(i)
            columnValue = date
            i += 1
            updated = connection.updateBackTestRangeForUserInput(RunID, columnName, columnValue)

def fillBacktestResult(mapeDataframe, long_mapeDataframe, runId, connection,configjson):
    '''
              storing the backtesting result to the ModelOutput table
              inputs:
              * mapeDataframe - dataframe containing mape values
              * long_mapeDataframe - dataframe containing mape values
              * RunID - RunId for the current job
              * connection - database connection object
              * configjson - config file
    '''

    for index, row in mapeDataframe.iterrows():
        modelId = connection.getModelId(runId, IndependentVariableName=index)
        i = 1
        for date in configjson['backtest_dates']:
            columnName = 'DynamicBacktestRange' + str(i) + 'MAPE'
            columnValue = row[date]
            i += 1
            updated = connection.updateModelOutputForBacktest(ModelId=modelId, ColumnName=columnName,
                                                              RangeValue=columnValue)

    for index, row in long_mapeDataframe.iterrows():
        modelId = connection.getModelId(runId, IndependentVariableName=index)
        i = 1
        for date in configjson['backtest_long_dates']:
            columnName = 'DynamicBacktestLongRange' + str(i)+'MAPE'
            columnValue = row[date]
            i += 1
            updated = connection.updateModelOutputForBacktest(ModelId=modelId, ColumnName=columnName, RangeValue=columnValue)


def getDatabaseConnectionObject1(argument1, argument2):
    '''
        getting a connection object
        inputs:
        * argument1 - config file specifying the database connection
        * argument2 - the database name where the user wants to connect
        output:
        connection - database connection object.
       '''
    connection = db.DbConnection(argument1, argument2)
    return connection

def createNewRunDetail(Connection, StartDate, ModelType):
    '''
           creating a new entry for the run
           inputs:
           * Connection - database connection object
           * StartDate - Date and Time for starting the Run
           * ModelType - type of the Model that is run
    '''
    runId = Connection.newRunDetail(StartDate=StartDate, ModelType=ModelType)
    return runId

def createNewDependentVariable(Connection, RunId, Name):
    '''
        creating a new entry for the dependent variable
        inputs:
        * Connection - database connection object
        * RunId - RunId for the current job
        * Name - name of the dependent variable
    '''
    depVariableID = Connection.newDependentVariableResult(RunId=RunId, Name=Name)
    return depVariableID

def model_result_save_in_db(scalar_diagnostic, paramsdf, pacf_plot, name, connection, runId, modelId, modelOutputID, independentVariableId, dummies):
    '''
            saving the model output to the database
            inputs:
            * scalar_diagnostic: Dataframe containing RSquared, Adjusted, fpval, RMSE, ABS Err, MAE, MAPE,  bg_pval, bp_pval, whit_pval, sw_pval, AIC and DurbinWatson values
            * paramsdf: Dataframe containing Coefficient, P-Value, Standard Error, Newey-West p, Newey-West SE, VIF
            * pacf_plot : dataframe containing the bytearray for PACF plot
            * name - name of the independent variable
            * connection - database connection object
            * RunId - RunId for the current job
            * modelId - modelId for the independent vraiable
            * modelOutputId - the id in the ModelOutput table where the output for the Model is stored.
            * dummies - name of the dummy variables
    '''
    paramsArray = paramsdf['Coefficient'].values
    pvaluesArray = paramsdf['P-Value'].values
    #print("pvaluesArray : ", pvaluesArray)
    # print("paramsArray : ", paramsArray)
    vifArray = paramsdf['VIF'].values

    updated = connection.updateIndependentVariableResult(id=independentVariableId, coefficient=paramsArray[0], pvalue=pvaluesArray[0], vif=vifArray[0])

    if 'White p' in scalar_diagnostic:
        updated = connection.updateModelOutput(id=modelOutputID, RMSE=scalar_diagnostic['RMSE'].iloc[0], WhiteSkedacityPval=scalar_diagnostic['White p'].iloc[0], RSquared=scalar_diagnostic['R Sq'].iloc[0],MAE=scalar_diagnostic['MAE'].iloc[0], MAPE=scalar_diagnostic['MAPE'].iloc[0], AIC=scalar_diagnostic['AIC'].iloc[0])
    else:
        updated = connection.updateModelOutput(id=modelOutputID, RMSE=scalar_diagnostic['RMSE'].iloc[0], RSquared=scalar_diagnostic['R Sq'].iloc[0],MAE=scalar_diagnostic['MAE'].iloc[0], MAPE=scalar_diagnostic['MAPE'].iloc[0], AIC=scalar_diagnostic['AIC'].iloc[0])

    if 'Shapiro Wilk p' in scalar_diagnostic:
        updated = connection.updateModelOutput(id=modelOutputID, ShapiroWilk=scalar_diagnostic['Shapiro Wilk p'].iloc[0])
    if 'DurbinWatson1' in scalar_diagnostic:
        updated = connection.updateModelOutput(id=modelOutputID, DurbinWatson1=scalar_diagnostic['DurbinWatson1'].iloc[0], DurbinWatson2=scalar_diagnostic['DurbinWatson2'].iloc[0], DurbinWatson3=scalar_diagnostic['DurbinWatson3'].iloc[0], DurbinWatson4=scalar_diagnostic['DurbinWatson4'].iloc[0])

    if 'Breusch Godfrey p' in scalar_diagnostic:
        updated = connection.updateModelOutput(id=modelOutputID, BGPVal=scalar_diagnostic['Breusch Godfrey p'].iloc[0])

    if 'Breusch Pagan p' in scalar_diagnostic:
        updated = connection.updateModelOutput(id=modelOutputID, BreuschPagan=scalar_diagnostic['Breusch Pagan p'].iloc[0])

    pacfPlotName = pacf_plot['PlotName'].iloc[0]
    byteArray = pacf_plot['Plot'].iloc[0]

    savePACFGraph(connection, modelId, pacfPlotName, byteArray)

    dummiesCoefResultDict = {}
    dictionaryCoef = {}
    dictionaryPvalues = {}
    dictionaryVifs = {}
    dummiesPvalesResultDict = {}

    numberOfParameters = paramsArray.shape[0]

    for i in range(1, numberOfParameters):
        dictionaryCoef[paramsdf.index[i]] = paramsArray[i]
        dictionaryPvalues[paramsdf.index[i]] = pvaluesArray[i]
        dictionaryVifs[paramsdf.index[i]] = vifArray[i]

    for i in range(0, len(dummies)):
        if dummies[i] in dictionaryCoef.keys():
            dummiesCoefResultDict[dummies[i]] = dictionaryCoef.get(dummies[i])
        if dummies[i] in dictionaryPvalues.keys():
            dummiesPvalesResultDict[dummies[i]] = dictionaryPvalues.get(dummies[i])

    independentVariableCoef = dictionaryCoef.get(name)
    independentVariablePvalue = dictionaryPvalues.get(name)
    independentVariableVif = dictionaryVifs.get(name)

    updated = connection.updateIndependentVariableResult(id=independentVariableId, coefficient=independentVariableCoef, pvalue=independentVariablePvalue, vif=independentVariableVif)

    for key in dictionaryCoef.keys():
        if '_lag' in key:
            lagCoefValue = dictionaryCoef.get(key)
            lagPval = dictionaryPvalues.get(key)
            lagVif = dictionaryVifs.get(key)
            created = connection.newIndependentVariableResult(ModelId=modelId, Name=key, Transformations='-', RunId=runId, Coef=lagCoefValue, Pvalue=lagPval, Vif=lagVif)

    for key in dummiesCoefResultDict:
        if(dummiesPvalesResultDict.get(key) != 'nan'):
            connection.newDummyVariable(RunId=runId, Name=key, Coef=dummiesCoefResultDict.get(key),
                                    Pvalue=dummiesPvalesResultDict.get(key))

def updateModelOutputForResult(Connection, modelOutputID, result, reason):
    '''
                saving the model output to the database
                inputs:
                * connection - database connection object
                * modelOutputId - the id in the ModelOutput table where the output for the Model is stored.
                * result - if a model passed all the tests or failed
                * reason - reasons for failure if any
        '''
    Connection.updateModelOutput(id=modelOutputID, AcceptReject=result, AcceptRejectReason=str(reason))

def dbCreateModelIdIndependentVarAndOutputId(Connection, runId, name, transformations):
    '''
            creating new entries in the ModelRunDetail, IndependentVariableResult and ModelOutput table for an independent variable.
            inputs:
            * Connection - database connection object
            * runId - the runId for the run
            * name - name of the independent variable
            * transformations - transformations applied to the independent variable
    '''
    modelId = Connection.newModelRunDetail(RunId=runId)
    independentVariableId = Connection.newIndependentVariableResult(ModelId=modelId, Name=str(name),
                                                                    Transformations=transformations, RunId=runId)
    modelOutputID = Connection.newModelOutput(ModelId=modelId)

    return modelId, independentVariableId, modelOutputID

def savePACFGraph(Connection, modelId, figName, byteArray):
    Connection.newPACFPlot(modelId=modelId, name=figName, plot=byteArray)

def saveJson(Connection, modelId, figname, json, jsontype):
    Connection.newBackTestPlot(modelId=modelId, name=figname, json=json, jsonTyp=jsontype)

def getModelId(connection, runId, variableName):
    modelId = connection.getModelId(runId, IndependentVariableName=variableName)
    return modelId

def saveIntermediateOutputForRunID(Connection, baseJson, regsJson, depJson, runId):
    Connection.newIntermediateOutput(baseJson=baseJson, regsJson=regsJson, depJson=depJson, runId=runId)

def updateStatusOfARun(Connection, runId, EndDate, Status):
    Connection.updateRunDetail(id=runId, EndDate=EndDate, Status=Status)

def updateEndTimeForTests(Connection,RunId, EndTimeForTests):
    Connection.updateRunDetail(id=RunId, EndTimeForTest=EndTimeForTests)

def getShortListedValues(Connection, RunId):
    result = Connection.getShortListedNames(RunId=RunId)
    return result

def getIntermediateOutputForRunID(Connection, runId):
    result = Connection.getIntermediateOutput(RunId=runId)
    return result

def getCandidateModelsForRunID(Connection, runId):
    result = Connection.getCandidates(runId)
    return result

def getDependentVariableTransformationOnRunId(Connection, runId):
    transformation = Connection.getDependentVariableTransformation(RunId=runId)
    return transformation