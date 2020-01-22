const express = require('express');
const moment = require('moment');
const router  = express.Router();
const multer  = require('multer');
const models  = require('../models/index.js');
const fs = require('fs');
const Sequelize  = require('Sequelize');
const sequelize  = require('sequelize');
const Op = Sequelize.Op

const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    if(!fs.existsSync('../mas-ds/uploads')) {
      fs.mkdirSync('../mas-ds/uploads')
    }
    cb(null, '../mas-ds/uploads')
  },
  filename: function (req, file, cb) {
  	var parts = file.originalname.split(".");
  	var ending = parts[parts.length - 1];
    cb(null, file.fieldname + '-' + Date.now() + '.' + ending);
  }
})

function getModelsWithShortlist(results, shortlist) {
	var newResults = {
		rejected: [],
		shortlist: []
	};
	var ids = []
	shortlist.forEach(function(item) {
		ids.push(item.dataValues.ModelId)
	})

	results.forEach(function (result) {
		var modeloutput = result.dataValues.ModelOutputs[0].dataValues
		modeloutput.Name = result.dataValues.IndependentVariableResults[0].dataValues.Name
		for(var key in modeloutput) {
			if (typeof modeloutput[key] == "number" && !key.toLowerCase().includes("id")) {
				modeloutput[key] = modeloutput[key].toFixed(3)
			}
		}
		modeloutput.href="/models/" + modeloutput.ModelId
		if (ids.indexOf(result.dataValues.id) > -1){
			newResults.shortlist.push(modeloutput)
		} else {
			newResults.rejected.push(modeloutput)
		}
	})
	return newResults
}

function getShortlistedModels(results) {
	var formattedData = []
	for (var i = 0; i < results.length; i++) {
		var result = results[i].dataValues
    for(var key in result) {
			if (typeof result[key] == "number" && !key.toLowerCase().includes("id")) {
				result[key] = result[key].toFixed(3)
			}
		}
		result.href="/models/" + result.ModelId
		formattedData.push(result)
	}
	return formattedData
}

function getCandidateModels(results) {
	var formattedData = {
		accepted: [],
		rejected: []
	}
  results.forEach(function (result) {
    var modeloutput = result.dataValues.ModelOutputs[0].dataValues
    modeloutput.Name = result.dataValues.IndependentVariableResults[0].dataValues.Name

		if(modeloutput.AcceptReject === true) {
      for(var key in modeloutput) {
  			if (typeof modeloutput[key] == "number" && !key.toLowerCase().includes("id")) {
  				modeloutput[key] = modeloutput[key].toFixed(3)
  			}
  		}
      modeloutput.href="/models/" + modeloutput.ModelId
			formattedData.accepted.push(modeloutput)
		} else if (modeloutput.AcceptReject === false) {
      for(var key in modeloutput) {
  			if (typeof modeloutput[key] == "number" && !key.toLowerCase().includes("id")) {
  				modeloutput[key] = modeloutput[key].toFixed(3)
  			}
  		}
			formattedData.rejected.push(modeloutput)
		}
  })
	return formattedData
}

function getJobData(results) {
	var formattedData = {
		successful: [],
		pending: [],
		failed: []
	}
	var formatted = [];
	for (var i = 0; i < results.length; i++) {
		var result = results[i].dataValues

		result.StartDate = moment(result.StartDate).utc().format('MMMM Do YYYY, h:mm:ss a');
		if (result.EndDate) {
			result.EndDate = moment(result.EndDate).utc().format('MMMM Do YYYY, h:mm:ss a');
		}

		result.href = "/jobs/" + result.id
		if (result.Status === "SUCCESS") {
			formattedData.successful.push(result)
		} else if (result.Status === "FAIL") {
			formattedData.failed.push(result)
		} else {
			formattedData.pending.push(result)
		}
	};
	return formattedData
}

const upload = multer({ storage: storage })

/* GET all jobs. */
router.get('/', function(req, res, next) {
	models.RunDetail.findAll().then(function (results) {
		res.render('jobs', {jobs: getJobData(results)});
	})
});

/* POST new job. */
router.post('/', upload.single('file'), function(req, res, next) {
	const file = req.file
	if (!file) {
		const error = new Error('Please upload a file')
		error.httpStatusCode = 400
		return next(error)
	}
	console.log(file.filename)

  var configjson = JSON.parse(JSON.stringify(req.body))
  configjson["filename"] = file.filename
  configjson["shtm"] = req.body.sheetname

  // create json from the body
  fs.writeFile("../mas-ds/mas-ardl/config.json", JSON.stringify(configjson), function(err, result) {
      if(err) console.log('error', err);
  });



  // //spawn the child process - python script
  const {spawn} = require('child_process')
  const path = require('path')

  //TODO: modify ARDL code config
  function runScript() {
    if(configjson['model'] === 'OneVar') {
      console.log('1FACTOR MODEL Started')
      return spawn('python3',["../mas-ds/mas-ardl/testOneVar.py",
                '../mas-webapp/config/config.json',
                'development'])
    } else if(configjson['model'] ==='ARDL') {
      console.log('ARDL MODEL Started')
      return spawn('python3',["../mas-ds/mas-ardl/testARDL.py",
                '../mas-webapp/config/config.json',
                'development'])
    }
  }
  const subprocess = runScript();
  // print output of script
  subprocess.stdout.on('data', (data) => {
          console.log(`data:${data}`);
  });
  subprocess.stderr.on('data', (data) => {
         console.log(`error:${data}`);
  });
  subprocess.stderr.on('close', () => {
             console.log("Ending stat script");
  });


	res.redirect("/jobs")
});

/* GET a single job's candidate models. */
router.get('/:id', function(req, res, next) {
	var id = req.params.id

	models.Shortlist.findAll({
		where: { RunId: id}
	}).then(function (shortlist) {
		if (shortlist.length === 0) {

			models.ModelRunDetail.findAll({
				where:{
					RunId: id
					 },
				include: [
					models.ModelOutput,
					models.IndependentVariableResult
				   ]
			}).then(function (results) {
				var job = getCandidateModels(results)
				job.id = id
				job.hasShortlist = false
				res.render('job', { job: job });
			})
		} else {
			//show shortlist and rejected

			models.ModelRunDetail.findAll({
				where:{
					RunId: id
					 },
				include: [
					models.ModelOutput,
					models.IndependentVariableResult
				   ]
			}).then(function (results) {
				var job = getModelsWithShortlist(results, shortlist)
				job.id = id
				job.hasShortlist = true
				res.render('job', { job: job });

			})


			// var allResults = []
			// var shortlistedModelIds = []
			// sequelize.Promise.each(shortlist, function(item) {
			// 	return models.ModelOutput.findOne({
			// 			where: { ModelId: item.dataValues.ModelId }
			// 	}).then(function (r) {
			// 		shortlistedModelIds.push(r.dataValues.id)
			// 		allResults.push(r)
			// 	})
			// }).then(function () {
			// 	models.ModelOutput.findAll({
			// 		where: {
			// 			Id: {[Op.notIn]:shortlistedModelIds}
			// 		},
			// 		include: [
			// 		   {
			// 			model: models.ModelRunDetail,
			// 			where: { RunId: id,
			// 			 }
			// 		   }
			// 		]
			// 	}).then(function (results) {
			// 		var job = {}
			// 		job.rejected = getShortlistedModels(results)

			// 		job.id = id
			// 		job.shortlist = getShortlistedModels(allResults)
			// 		job.hasShortlist = true
			// 		res.render('job', { job: job });
			// 	})
			// });
		}

	});
});

module.exports = router;
