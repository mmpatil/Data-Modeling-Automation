const express = require('express');
const moment = require('moment');
const router  = express.Router();
const models  = require('../models/index.js');
const Sequelize  = require('Sequelize');
const sequelize  = require('sequelize');
const Op = Sequelize.Op

/* POST to create a shortlist. */
router.post('/', function(req, res, next) {
	var values = []
	var keys = []
	Object.keys(req.body).forEach(function (key) {
		if (key.includes("model")) {
			key = key.replace("model", "")
			keys.push(parseInt(key))
		}
	})
	models.ModelOutput.update(
		{ AcceptReject: true },
		{
		  where: {
			ModelId: {[Op.in]: keys}
		}
	}).then(function(results) {
		for (var i = 0; i < keys.length; i++) {
			var id = keys[i]
			var shortItem = {
				ModelId: id,
				RunId: req.body.runId
			}
			values.push(shortItem)
		};

		models.Shortlist.bulkCreate(values, {returning: true});

		// //spawn the child process - python script
	  const {spawn} = require('child_process')
	  const path = require('path')

	  //TODO: modify ARDL code config
	  function runScript() {
      console.log('After shortlist script started')
      return spawn('python3',["../mas-ds/mas-ardl/afterShortlist.py",
										'../mas-webapp/config/config.json',
										'development', req.body.runId])
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

		res.redirect("/jobs/" + req.body.runId)

	})

});
module.exports = router;
