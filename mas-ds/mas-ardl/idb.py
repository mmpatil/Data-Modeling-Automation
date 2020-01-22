import sqlalchemy
from sqlalchemy import create_engine
from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, Float, Boolean, Text
from sqlalchemy.dialects.mssql.base import MSVarBinary
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import sqlalchemy.pool as pool
import json
import pymssql
import datetime

from sqlalchemy.pool import NullPool

Base = declarative_base()

class RunDetail(Base):
    __tablename__ = 'RunDetail'
    id = Column(Integer, primary_key=True)
    StartDate = Column(DateTime)
    EndDate = Column(DateTime)
    EndDateTimeForTests = Column(DateTime)
    Status = Column(String)
    ModelType = Column(String)

class PACFPlots(Base):
    __tablename__ = 'PACFPlots'
    id = Column(Integer, primary_key=True)
    ModelId = Column(ForeignKey('ModelRunDetail.id'))
    Name = Column(String)
    PLOT = Column(MSVarBinary)

class BackTestPlots(Base):
    __tablename__ = 'BackTestPlots'
    id = Column(Integer, primary_key=True)
    Name = Column(String)
    JSON = Column(String)
    JSONType = Column(String)
    ModelId = Column(ForeignKey('ModelRunDetail.id'))

class ShortlistedModels(Base):
    __tablename__ = 'ShortlistedModels'
    id = Column(Integer, primary_key=True)
    RunId = Column(ForeignKey('RunDetail.id'))
    ModelId = Column(ForeignKey('ModelRunDetail.id'))

class IntermediateOutput(Base):
    __tablename__ = 'IntermediateOutput'
    id = Column(Integer, primary_key=True)
    RegsDataframeJSON = Column(Text)
    BaseDataframeJSON = Column(String)
    DependentJSON = Column(String)
    RunId = Column(ForeignKey('RunDetail.id'))

class UserInput(Base):
    __tablename__ = 'UserInput'
    id = Column(Integer, primary_key=True)
    RunId = Column(ForeignKey('RunDetail.id'))
    UserId = Column(Integer)
    DWLimitLow = Column(Float)
    DWLimitHigh = Column(Float)
    BGLimit = Column(Float)
    WhiteSkedacityLimit = Column(Float)
    VIFLimit = Column(Float)
    ADFLimit = Column(Float)
    DynamicBacktestRange1 = Column(DateTime)
    DynamicBacktestRange2 = Column(DateTime)
    DynamicBacktestRange3 = Column(DateTime)
    DynamicBacktestRange4 = Column(DateTime)
    DynamicBacktestRange5 = Column(DateTime)
    DynamicBacktestRange6 = Column(DateTime)
    DynamicBacktestRange7 = Column(DateTime)
    DynamicBacktestRange8 = Column(DateTime)
    DynamicBacktestRange9 = Column(DateTime)
    DynamicBacktestRange10 = Column(DateTime)
    DynamicBacktest1Weight = Column(Float)
    DynamicBacktest2Weight = Column(Float)
    DynamicBacktest3Weight = Column(Float)
    DynamicBacktest4Weight = Column(Float)
    DynamicBacktest5Weight = Column(Float)
    DynamicBacktest6Weight = Column(Float)
    DynamicBacktest7Weight = Column(Float)
    DynamicBacktest8Weight = Column(Float)
    DynamicBacktest9Weight = Column(Float)
    DynamicBacktest10Weight = Column(Float)
    DynamicBacktestLongRange1 = Column(DateTime)
    DynamicBacktestLongRange2 = Column(DateTime)
    DynamicBacktestLongRange3 = Column(DateTime)
    DynamicBacktestLongRange4 = Column(DateTime)
    DynamicBacktestLongRange5 = Column(DateTime)
    DynamicBacktestLongRange6 = Column(DateTime)
    DynamicBacktestLongRange7 = Column(DateTime)
    DynamicBacktestLongRange8 = Column(DateTime)
    DynamicBacktestLongRange9 = Column(DateTime)
    DynamicBacktestLongRange10 = Column(DateTime)
    StatsTest = Column(String)
    DependentTransformationType = Column(String)
    DependentVariable = Column(String)

class ModelRunDetail(Base):
    __tablename__ = 'ModelRunDetail'
    id = Column(Integer, primary_key=True)
    RunId = Column(ForeignKey('RunDetail.id'))


class ModelOutput(Base):
    __tablename__ = 'ModelOutput'
    id = Column(Integer, primary_key=True)
    ModelId = Column(ForeignKey('ModelRunDetail.id'))
    BGPVal = Column(Float)
    WhiteSkedacityPval = Column(Float)
    VIFPval = Column(Float)
    ADFResidual = Column(Float)
    RSquared = Column(Float)
    RMSE = Column(Float)
    MAE = Column(Float)
    MAPE = Column(Float)
    AIC = Column(Float)
    DynamicBacktestRange1MAPE = Column(Float)
    DynamicBacktestRange2MAPE = Column(Float)
    DynamicBacktestRange3MAPE = Column(Float)
    DynamicBacktestRange4MAPE = Column(Float)
    DynamicBacktestRange5MAPE = Column(Float)
    DynamicBacktestRange6MAPE = Column(Float)
    DynamicBacktestRange7MAPE = Column(Float)
    DynamicBacktestRange8MAPE = Column(Float)
    DynamicBacktestRange9MAPE = Column(Float)
    DynamicBacktestRange10MAPE = Column(Float)
    AcceptReject = Column(Boolean)
    AcceptRejectReason = Column(String)
    DynamicBacktestLongRange1MAPE = Column(Float)
    DynamicBacktestLongRange2MAPE = Column(Float)
    DynamicBacktestLongRange3MAPE = Column(Float)
    DynamicBacktestLongRange4MAPE = Column(Float)
    DynamicBacktestLongRange5MAPE = Column(Float)
    DynamicBacktestLongRange6MAPE = Column(Float)
    DynamicBacktestLongRange7MAPE = Column(Float)
    DynamicBacktestLongRange8MAPE = Column(Float)
    DynamicBacktestLongRange9MAPE = Column(Float)
    DynamicBacktestLongRange10MAPE = Column(Float)
    DurbinWatson1 = Column(Float)
    DurbinWatson2 = Column(Float)
    DurbinWatson3 = Column(Float)
    DurbinWatson4 = Column(Float)
    ShapiroWilk = Column(Float)
    BreuschGodfrey = Column(Float)
    BreuschPagan = Column(Float)

class DummyVariable(Base):
    __tablename__ = 'DummyVariable'
    id = Column(Integer, primary_key=True)
    RunId = Column(ForeignKey('RunDetail.id'))
    Name = Column(String)
    Coefficient = Column(Float)
    Pval = Column(Float)


class IndependentVariableResult(Base):
    __tablename__ = 'IndependentVariableResult'
    id = Column(Integer, primary_key=True)
    ModelId = Column(ForeignKey('ModelRunDetail.id'))
    Name = Column(String)
    Coefficient = Column(Float)
    Pval = Column(Float)
    Transformations = Column(String)
    VIF = Column(String)
    RunId = Column(ForeignKey('RunDetail.id'))


class DependentVariableResult(Base):
    __tablename__ = 'DependentVariableResult'
    id = Column(Integer, primary_key=True)
    RunId = Column(ForeignKey('RunDetail.id'))
    Name = Column(String)
    Coefficient = Column(Float)
    Pval = Column(Float)
    Transformations = Column(String)
    UnitRoot = Column(String)


class DbConnection:
    engine = ""
    Session = ""

    def __init__(self, configFile, env):
        self.env = env
        with open(configFile) as json_file:
            data = json.load(json_file)
            config = data["development"]
            if env in data:
                config = data[env]
            host = config["host"]
            user = config["username"]
            password = config["password"]
            database = config["database"]
            self.engine = create_engine("mssql+pymssql://" + user + ":" + password + "@" + host + ":1433/" + database, poolclass=NullPool)
            Base.metadata.bind = self.engine
            self.Session = sessionmaker(bind=self.engine)

    def testQuery(self):
        session = self.Session()
        date = datetime.datetime.now()
        run = RunDetail(StartDate=date)
        session.add(run)
        session.commit()
        session.close()
        return run.id

    def newUserInput(self, RunID=None, UserID=None):
        session = self.Session()
        userInput = UserInput(RunId=RunID, UserId=UserID)
        session.add(userInput)
        session.commit()
        id = userInput.id
        session.close()
        return id

    # creates a new row in RunDetail and returns the id of the row
    def newRunDetail(self, StartDate=None, ModelType=None):
        session = self.Session()
        if StartDate == None:
            date = datetime.datetime.now()
            run = RunDetail(StartDate=date)
        else:
            run = RunDetail(StartDate=StartDate)
        if ModelType != None:
            run.ModelType = ModelType
        session.add(run)
        session.commit()
        id = run.id
        session.close()
        return id

    # creates a
    def newDependentVariableResult(self, RunId, Name):
        session = self.Session()
        dep = DependentVariableResult(RunId=RunId, Name=Name)
        session.add(dep)
        session.commit()
        id = dep.id
        session.close()
        return id

    def newIndependentVariableResult(self, ModelId, Name, Transformations, RunId, Coef=None, Pvalue=None, Vif=None):
        session = self.Session()
        independent = IndependentVariableResult(ModelId=ModelId, Name=Name, Transformations=Transformations,
                                                RunId=RunId)
        if Coef != None:
            independent.Coefficient = Coef
        if Pvalue != None:
            independent.Pval = Pvalue
        if Vif != None:
            independent.VIF = Vif

        session.add(independent)
        session.commit()
        id = independent.id
        session.close()
        return id

    def newDummyVariable(self, RunId, Name, Coef, Pvalue):
        session = self.Session()
        dummuy = DummyVariable(RunId=RunId, Name=Name, Coefficient=Coef, Pval=Pvalue)
        session.add(dummuy)
        session.commit()
        id = dummuy.id
        session.close()
        return id

    # creates a new row in ModelRunDetail and returns the id of the row
    def newModelRunDetail(self, RunId):
        session = self.Session()
        modelRun = ModelRunDetail(RunId=RunId)
        session.add(modelRun)
        session.commit()
        id = modelRun.id
        session.close()
        return id

    def newPACFPlot(self, modelId, name, plot):
        session = self.Session()
        pacfPlot = PACFPlots(ModelId=modelId, Name=name, PLOT=plot)
        #plot = Plots(RunId=runId, Name=name, PLOT=plot)
        session.add(pacfPlot)
        session.commit()
        id = pacfPlot.id
        session.close()
        return id

    # creates a new row in ModelOutput and returns the id of the row
    def newModelOutput(self, ModelId):
        session = self.Session()
        modelOut = ModelOutput(ModelId=ModelId)
        session.add(modelOut)
        session.commit()
        id = modelOut.id
        session.close()
        return id

    def newBackTestPlot(self, modelId, name, json, jsonTyp):
        session = self.Session()
        backTestPlot = BackTestPlots(Name=name, ModelId=modelId, JSON=json, JSONType=jsonTyp)
        session.add(backTestPlot)
        session.commit()
        id = backTestPlot.id
        session.close()
        return id

    def newIntermediateOutput(self, regsJson, baseJson, depJson, runId):
        session = self.Session()
        intermediateoutput = IntermediateOutput(RegsDataframeJSON=regsJson, BaseDataframeJSON=baseJson, DependentJSON=depJson, RunId=runId)
        session.add(intermediateoutput)
        session.commit()
        id = intermediateoutput.id
        session.close()
        return id

    # updates a previously created row in RunDetail.  StartDate is not editable
    def updateRunDetail(self, id, EndDate=None, Status=None, EndTimeForTest=None):
        session = self.Session()
        # get modelOutput instance
        run = session.query(RunDetail).filter_by(id=id).first()
        if run == None:
            return False
        if EndDate != None:
            run.EndDate = EndDate
        if Status != None:
            run.Status = Status
        if EndTimeForTest!=None:
            run.EndDateTimeForTests = EndTimeForTest
        session.commit()
        session.close()
        return True

    def updateUserInput(self, RunId, DWLimitHigh=None, DWLimitLow=None, BGLimit=None, WhiteSkedasticityLimit=None,
                        VIFLimit=None, ADFLimit=None, StatsTest=None, DependentVariableTransformation=None, DependentVariableName=None):
        session = self.Session()
        userInput = session.query(UserInput).filter_by(RunId=RunId).first()
        if userInput == None:
            return False
        if DWLimitHigh != None:
            userInput.DWLimitHigh = DWLimitHigh

        if DWLimitLow != None:
            userInput.DWLimitLow = DWLimitLow

        if BGLimit != None:
            userInput.BGLimit = BGLimit

        if WhiteSkedasticityLimit != None:
            userInput.WhiteSkedacityLimit = WhiteSkedasticityLimit

        if VIFLimit != None:
            userInput.VIFLimit = VIFLimit

        if ADFLimit != None:
            userInput.ADFLimit = ADFLimit

        if StatsTest != None:
            userInput.StatsTest = StatsTest

        if DependentVariableTransformation != None:
            userInput.DependentTransformationType = DependentVariableTransformation

        if DependentVariableName != None:
            userInput.DependentVariable = DependentVariableName

        session.commit()
        session.close()
        return True

    def updateBackTestRangeForUserInput(self, id, columnName, rangeValue):
        session = self.Session()
        userInput = session.query(UserInput).filter_by(RunId=id).first()
        updated = False
        if hasattr(UserInput, columnName):
            updated = True
            setattr(userInput, columnName, rangeValue)

        session.commit()
        session.close()
        return updated

    def updateIndependentVariableResult(self, id, coefficient=None, pvalue=None, vif=None):
        session = self.Session()
        independentVariableResults = session.query(IndependentVariableResult).filter_by(id=id).first()
        if independentVariableResults == None:
            return False
        if coefficient != None:
            independentVariableResults.Coefficient = coefficient
        if pvalue != None:
            independentVariableResults.Pval = pvalue
        if vif != None:
            independentVariableResults.VIF = vif

        session.commit()
        session.close()
        return True


    def getModelId(self, RunId, IndependentVariableName):
        session = self.Session()
        independentVariableDetail = session.query(IndependentVariableResult).filter(
            IndependentVariableResult.RunId == RunId,
            IndependentVariableResult.Name.like(IndependentVariableName)).all()
        if independentVariableDetail == None:
            return False
        for row in independentVariableDetail:
            return row.ModelId

    # updates a previously created ModelOutput row
    def updateModelOutput(self, id, BGPVal=None, WhiteSkedacityPval=None, VIFPval=None, ADFResidual=None, RSquared=None,
                          RMSE=None, MAE=None, MAPE=None, AIC=None, DynamicBacktestRange1MAPE=None,
                          DynamicBacktestRange2MAPE=None, DynamicBacktestRange3MAPE=None,
                          DynamicBacktestRange4MAPE=None, DynamicBacktestRange5MAPE=None,
                          DynamicBacktestRange6MAPE=None, DynamicBacktestRange7MAPE=None,
                          DynamicBacktestRange8MAPE=None, DynamicBacktestRange9MAPE=None,
                          DynamicBacktestRange10MAPE=None, DynamicBacktestLongRange1MAPE = None,
                          DynamicBacktestLongRange2MAPE=None, DynamicBacktestLongRange3MAPE=None,
                          DynamicBacktestLongRange4MAPE=None, DynamicBacktestLongRange5MAPE=None,
                          DynamicBacktestLongRange6MAPE=None, DynamicBacktestLongRange7MAPE=None,
                          DynamicBacktestLongRange8MAPE=None, DynamicBacktestLongRange9MAPE=None,
                          DynamicBacktestLongRange10MAPE=None, DurbinWatson1=None,
                          DurbinWatson2=None, DurbinWatson3=None, DurbinWatson4=None,
                          AcceptReject=None, AcceptRejectReason=None, ShapiroWilk=None, BreuschGodfrey=None, BreuschPagan=None):

        session = self.Session()
        # get modelOutput instance
        modelOutput = session.query(ModelOutput).filter_by(id=id).first()
        if modelOutput == None:
            return False
        if BGPVal != None:
            modelOutput.BGPVal = BGPVal
        if WhiteSkedacityPval != None:
            modelOutput.WhiteSkedacityPval = WhiteSkedacityPval
        if VIFPval != None:
            modelOutput.VIFPval = VIFPval
        if ADFResidual != None:
            modelOutput.ADFResidual = ADFResidual
        if RSquared != None:
            modelOutput.RSquared = RSquared
        if RMSE != None:
            modelOutput.RMSE = RMSE
        if MAE != None:
            modelOutput.MAE = MAE
        if MAPE != None:
            modelOutput.MAPE = MAPE
        if AIC != None:
            modelOutput.AIC = AIC
        if DynamicBacktestRange1MAPE != None:
            modelOutput.DynamicBacktestRange1MAPE = DynamicBacktestRange1MAPE
        if DynamicBacktestRange2MAPE != None:
            modelOutput.DynamicBacktestRange2MAPE = DynamicBacktestRange2MAPE
        if DynamicBacktestRange3MAPE != None:
            modelOutput.DynamicBacktestRange3MAPE = DynamicBacktestRange3MAPE
        if DynamicBacktestRange4MAPE != None:
            modelOutput.DynamicBacktestRange4MAPE = DynamicBacktestRange4MAPE
        if DynamicBacktestRange5MAPE != None:
            modelOutput.DynamicBacktestRange5MAPE = DynamicBacktestRange5MAPE
        if DynamicBacktestRange6MAPE != None:
            modelOutput.DynamicBacktestRange6MAPE = DynamicBacktestRange6MAPE
        if DynamicBacktestRange7MAPE != None:
            modelOutput.DynamicBacktestRange7MAPE = DynamicBacktestRange7MAPE
        if DynamicBacktestRange8MAPE != None:
            modelOutput.DynamicBacktestRange8MAPE = DynamicBacktestRange8MAPE
        if DynamicBacktestRange9MAPE != None:
            modelOutput.DynamicBacktestRange9MAPE = DynamicBacktestRange9MAPE
        if DynamicBacktestRange10MAPE != None:
            modelOutput.DynamicBacktestRange10MAPE = DynamicBacktestRange10MAPE
        if DynamicBacktestLongRange1MAPE != None:
            modelOutput.DynamicBacktestLongRange1MAPE = DynamicBacktestLongRange1MAPE
        if DynamicBacktestLongRange2MAPE != None:
            modelOutput.DynamicBacktestLongRange2MAPE = DynamicBacktestLongRange2MAPE
        if DynamicBacktestLongRange3MAPE != None:
            modelOutput.DynamicBacktestLongRange3MAPE = DynamicBacktestLongRange3MAPE
        if DynamicBacktestLongRange4MAPE != None:
            modelOutput.DynamicBacktestLongRange4MAPE = DynamicBacktestLongRange4MAPE
        if DynamicBacktestLongRange5MAPE != None:
            modelOutput.DynamicBacktestLongRange5MAPE = DynamicBacktestLongRange5MAPE
        if DynamicBacktestLongRange6MAPE != None:
            modelOutput.DynamicBacktestLongRange6MAPE = DynamicBacktestLongRange6MAPE
        if DynamicBacktestLongRange7MAPE != None:
            modelOutput.DynamicBacktestLongRange7MAPE = DynamicBacktestLongRange7MAPE
        if DynamicBacktestLongRange8MAPE != None:
            modelOutput.DynamicBacktestLongRange8MAPE = DynamicBacktestLongRange8MAPE
        if DynamicBacktestLongRange9MAPE != None:
            modelOutput.DynamicBacktestLongRange9MAPE = DynamicBacktestLongRange9MAPE
        if DynamicBacktestLongRange10MAPE != None:
            modelOutput.DynamicBacktestLongRange10MAPE = DynamicBacktestLongRange10MAPE
        if DurbinWatson1 != None:
            modelOutput.DurbinWatson1 = DurbinWatson1
        if DurbinWatson2 != None:
            modelOutput.DurbinWatson2 = DurbinWatson2
        if DurbinWatson3 != None:
            modelOutput.DurbinWatson3 = DurbinWatson3
        if DurbinWatson4 != None:
            modelOutput.DurbinWatson4 = DurbinWatson4
        if AcceptReject != None:
            modelOutput.AcceptReject = AcceptReject
        if AcceptRejectReason != None:
            modelOutput.AcceptRejectReason = AcceptRejectReason
        if ShapiroWilk !=None:
            modelOutput.ShapiroWilk = ShapiroWilk
        if BreuschGodfrey != None:
            modelOutput.BreuschGodfrey = BreuschGodfrey
        if BreuschPagan != None:
            modelOutput.BreuschPagan = BreuschPagan
        session.commit()
        session.close()
        return True

    def updateModelOutputForBacktest(self, ModelId, ColumnName, RangeValue):
        session = self.Session()
        userInput = session.query(ModelOutput).filter_by(ModelId=ModelId).first()
        updated = False
        if hasattr(ModelOutput, ColumnName):
            updated = True
            setattr(userInput, ColumnName, RangeValue)

        session.commit()
        session.close()
        return updated

    def getShortListedNames(self, RunId):
        candidateList = []
        session = self.Session()
        result = session.query(ShortlistedModels).filter_by(RunId=RunId).all()

        for row in result:
            independentVariableResult = session.query(IndependentVariableResult).filter_by(ModelId=row.ModelId, RunId=RunId).first()
            candidateList.append(independentVariableResult.Name)
        return candidateList

    def getIntermediateOutput(self, RunId):
        session = self.Session()
        result = session.query(IntermediateOutput).filter_by(RunId=RunId).first()
        print("Intermediate Result JSON: ", result)
        return result

    def getCandidates(self, RunId):
        cadidateList = []
        session = self.Session()
        result = session.query(IndependentVariableResult).join(ModelOutput, IndependentVariableResult.ModelId==ModelOutput.ModelId).filter(IndependentVariableResult.RunId==RunId).filter(ModelOutput.AcceptReject==1).all()
        for row in result:
            if '_lag' not in row.Name:
                cadidateList.append(row.Name)
        return cadidateList

    def getDependentVariableTransformation(self, RunId):
        session = self.Session()
        result = session.query(UserInput).filter_by(RunId=RunId).first()
        return result.DependentVariable
