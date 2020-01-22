import sys
import db
import datetime

# script usage : python test.py <path to config.json> <env>
# example : python3 test.py ../mas-webapp/config/config.json development

# make a new instance of the DbConnection class to handle all our db connections
connection = db.DbConnection(sys.argv[1], sys.argv[2])

# insert a new row in RunDetail, returns the id of the row
runId = connection.newRunDetail(StartDate=datetime.datetime.now())
print("{} : {}".format("inserted runId", runId))

# insert a new row in ModelRunDetail, returns the id of the row
modelId = connection.newModelRunDetail(RunId=runId)
print("{} : {}".format("inserted modelId", modelId))

# insert a new row in ModelOutput, returns the id of the row
modelOutputId = connection.newModelOutput(ModelId=modelId)
print("{} : {}".format("inserted modelOutputId", modelOutputId))

# update the previously created row in ModelOutput
# id is a required field, but we're using named fields to selectively update some columns without touching others
modelOutputUpdate = connection.updateModelOutput(id=modelOutputId, BGPVal=0.5, AcceptReject=False, AcceptRejectReason="Who Knows")
print("{} : {}, was successful? {}".format("update to modelOutputId", modelOutputId, modelOutputUpdate))

# update the previously created row in RunDetail
runUpdate = connection.updateRunDetail(id=runId, EndDate=datetime.datetime.now(), Status="Success")
print("{} : {}, was successful? {}".format("update to runDetail id", runId, runUpdate))
