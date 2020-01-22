const express = require('express');
const moment = require('moment');
const router  = express.Router();
const models  = require('../models/index.js');


function formatResults(results) {
	var formattedData = {
		detail:[]
	}
	for(var i = 0; i < results.length; i++) {
		var detail = results[i].dataValues
		//up to 5 decimal place!

		for(var key in detail) {
			if (typeof detail[key] == "number" && !key.toLowerCase().includes("id")) {
				detail[key] = detail[key].toFixed(5)
			}
		}
		formattedData.detail.push(detail)
	}
	return formattedData
}

function formatGraphs(results) {
	var graphs = [];
	var backtestingplots = {
		type: "backtesting",
		id: "BACKTESTING",
		values: []
	}

	var ootplots = {
		type: "out of time",
		id: "OOT",
		values: []
	}

	var regressionparamplots = {
		type: "regression param",
		id: "REGRESSIONPARAM",
		values: []
	}

	var regressionpvalplots = {
		type: "regression pval",
		id: "REGRESSIONPVAL",
		values: []
	}

	var sensitivityplots = {
		type: "sensitivity",
		id: "SENSITIVITY",
		values: []
	}
	var stressplots = {
		type: "stress",
		id: "STRESS",
		values: []
	}
	var forecastplots = {
		type: "forecast",
		id: "FORECAST",
		values: []
	}

	for (var i = results.length - 1; i >= 0; i--) {
		results[i] = results[i].dataValues
		try {
			results[i].values = JSON.parse(results[i].JSON).values
			delete results[i].JSON
			if (results[i].JSONType === "RegressionParam") {
				regressionparamplots.values.push(results[i])
			} else if (results[i].JSONType === "RegressionPval") {
				regressionpvalplots.values.push(results[i])
			} else if (results[i].JSONType === "BackTestJson") {
				backtestingplots.values.push(results[i])
			} else if (results[i].JSONType === "StressTestForecast") {
				forecastplots.values.push(results[i])
			} else if (results[i].JSONType === "StressTestCompare") {
				stressplots.values.push(results[i])
			} else if (results[i].JSONType === "SensitivityTest") {
				sensitivityplots.values.push(results[i])
			} else if (results[i].JSONType === "OutOfTime") {
				ootplots.values.push(results[i])
			}
		} catch (err) {
			console.log(err)
		}

	};

	if (backtestingplots.values.length > 0) {
		graphs.push(backtestingplots)
	}

	if (regressionpvalplots.values.length > 0) {
		graphs.push(regressionpvalplots)
	}

	if (regressionparamplots.values.length > 0) {
		graphs.push(regressionparamplots)
	}

	if (forecastplots.values.length > 0) {
		graphs.push(forecastplots)
	}
	if (stressplots.values.length > 0) {
		graphs.push(stressplots)
	}

	if (sensitivityplots.values.length > 0) {
		graphs.push(sensitivityplots)
	}

	if (ootplots.values.length > 0) {
		graphs.push(ootplots)
	}
	return graphs;
}

function formatRange(userInput) {
	var userInputVal = userInput[0].dataValues
	for(var key in userInputVal) {
		if(key.includes("Range")) {
			var d = new Date(userInputVal[key]),
					month = '' + (d.getUTCMonth()+1),
					day = ''+d.getUTCDate(),
					year = d.getUTCFullYear();

			userInputVal[key]= [year, month, day].join('-');
		}
	}
	return userInputVal
}

/* GET a single model output. */
router.get('/:id', function(req, res, next) {

	var modelId = req.params.id
	var modelDetail = []
	var indDetail = []
	var modelDetailId = ""
	var graphs = []
	var range = []
	var runId;

	models.ModelRunDetail.findOne({
		where: {
		    id: modelId
		},
		include: [
					models.ModelOutput,
					models.IndependentVariableResult,
					models.PACFPlots,
					models.BackTestPlots
				   ]
	}).then(function (results) {
		modelDetail = formatResults(results.dataValues.ModelOutputs)

		var plotDetail = results.dataValues.PACFPlots[0].dataValues
		var plot = plotDetail['PLOT'].toString('base64')

		graphs = formatGraphs(results.dataValues.BackTestPlots)


		indDetail = formatResults(results.dataValues.IndependentVariableResults)

		runId = indDetail.detail[0]['RunId']
		//getting independent variable name
		for(var i = 0; i < indDetail.detail.length; i++) {
			var eachOne = indDetail.detail[i]
			if(!eachOne['Name'].includes("_lag")) {
				modelDetailId = eachOne['Name']
				break;
			}
		}

		modelDetail.modelId = modelDetailId

		//query range for backtest table
		models.UserInput.findAll({
			where: {
				RunId:runId
			}
		}).then(function(userInputs) {
			range = formatRange(userInputs)

			res.render('model', {
				modelDetail: modelDetail,
				graphs: graphs,
				indDetail: indDetail,
				plot: plot,
				range: range
			});
		})

	})

	// //find modelOutput with modelId
	// models.ModelOutput.findAll({
	//   where: {
	//     ModelId: modelId
	//   }
	// })
	// .then(function (results) {
	// 	modelDetail = formatResults(results)
	// 	// graphs = getGraphs()

	// 	//query independent variable results with ModelId
	// 	models.IndependentVariableResult.findAll({
	// 		where: {
	// 			ModelId: modelId
	// 		}
	// 		 //TODO: this is a placeholder
	// 	})
	// 	.then(function(indResults) {
	// 				indDetail = formatResults(indResults)
	// 				runId = indDetail.detail[0]['RunId']
	// 				//getting independent variable name
	// 				for(var i = 0; i < indDetail.detail.length; i++) {
	// 					var eachOne = indDetail.detail[i]
	// 					if(!eachOne['Name'].includes("_lag")) {
	// 						modelDetailId = eachOne['Name']
	// 						break;
	// 					}
	// 				}
	// 				//query range for backtest table
	// 				models.UserInput.findAll({
	// 					where: {
	// 						RunId:runId
	// 					}
	// 				}).then(function(userInputs) {

	// 					range = formatRange(userInputs)
	// 				})

	// 				//query pacf png files
	// 				models.PACFPlots.findAll({
	// 					where: {
	// 						ModelId: modelId
	// 					}
	// 				}).then (function(results){
	// 					var plotDetail = results[0].dataValues
	// 					var plot = plotDetail['PLOT'].toString('base64')

	// 					modelDetail.modelId = modelDetailId

	// 					//query plots data
	// 					models.BackTestPlots.findAll({
	// 					  where: {
	// 					    ModelId: modelId
	// 					  }
	// 					})
	// 					.then(function (results) {
	//

	// 						console.log({
	// 							modelDetail: modelDetail,
	// 							graphs: graphs,
	// 							indDetail: indDetail,
	// 							plot: plot,
	// 							range: range
	// 						})


	// 						res.render('model', {
	// 							modelDetail: modelDetail,
	// 							graphs: graphs,
	// 							indDetail: indDetail,
	// 							plot: plot,
	// 							range: range
	// 						});
	// 					});
	// 				});
	// 		});
	// })

});


module.exports = router;
