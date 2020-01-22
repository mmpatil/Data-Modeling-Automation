let config = {
	backtesting: "backtesting",
	oot: "out of time",
	pval: "regression pval",
	param: "regression param",
	sensitivity: "sensitivity",
	stress: "stress",
	forecast: "forecast"
}

let dimensions = {
	width: window.innerWidth / 3 > 375 ? window.innerWidth / 3 : 375,
	height: window.innerWidth / 3 / 1.6 > 250 ? window.innerWidth /  3 / 1.6 : 250
}

function getData(graphs, val) {
	for (var i = 0; i < graphs.length; i++) {
		if (graphs[i].id === parseInt(val)) {
			return graphs[i].values
		}
	};
}
//				 2006-03-31T00:00:00+00:00
// format date: "2006-03-31T00:00:00"
let parseTime = d3.timeParse("%Y-%m-%dT%H:%M:%S")
let parseTimeZone = d3.timeParse("%Y-%m-%dT%H:%M:%S+Z")

graphs.forEach(function (type, j) {
	type.values.forEach(function (graph, k) {
		graph.values.forEach(function(d) {
			d.values.forEach(function (value) {
				value.date = value.date.split("+")[0]
		    	value.date = parseTime(value.date)

		    })

		})
	})

	let graph = type.values[0]

	//only call chart once intially
	//graph has name, id, values
	let chartData = linechart()
			.width(dimensions.width)
			.height(dimensions.height)
	if (type.type === config.backtesting) {
		chartData = linechart()
			.width(dimensions.width)
			.height(dimensions.height)
	} else if (type.type === config.oot) {
		chartData = linechart()
			.width(dimensions.width)
			.height(dimensions.height)
			.labels(true)
	} else if (type.type === config.pval) {
		chartData = linechart()
			.width(dimensions.width)
			.height(dimensions.height)
			.labels(true)
	} else if (type.type === config.param) {
		chartData = linechart()
			.width(dimensions.width)
			.height(dimensions.height)
			.labels(true)
			.precision(6)
	} else if (type.type === config.sensitivity) {
		chartData = splitlinechart()
		  	.width(dimensions.width)
		  	.height(dimensions.height)
		    .labels(false)
		    .type("sensitivity")
		    .actualName("Dependent")
		    .forecastName("Rand")
	} else if (type.type === config.stress) {
		chartData = splitlinechart()
		  	.width(dimensions.width)
		  	.height(dimensions.height)
		    .labels(true)
		    .type("stress")
		    .actualName("Actual")
		    .baseName("Model Base")
		    .severeName("Model_Severe")
		    .adverseName("Model_Adverse")
		    .baseForecastName("Alternative_Assets_Base")
		    .severeForecastName("Alternative_Assets_Severe")
		    .adverseForecastName("Alternative_Assets_Adverse")
		    .labels(false)
	} else if (type.type === config.forecast) {
		chartData = splitlinechart()
		  	.width(dimensions.width)
		  	.height(dimensions.height)
		    .type("stress")
		    .actualName("Dependent")
		    .baseName("Base")
		    .severeName("Severe")
		    .adverseName("Adverse")
		    .labels(false)
	}

	d3.select("#graph-" + type.id + "-a")
		.datum(graph.values)
		.call(chartData);

	d3.select("#graph-" + type.id + "-b")
		.datum(graph.values)
		.call(chartData);

	$("#graph-select-" + type.id + "-a").change(function(e) {
	    $("#graph-" + type.id + "-a").html("");
	    let val = $(this).val();
	    let data = getData(type.values, val)
	   	d3.select("#graph-" + type.id + "-a")
		    .datum(data)
		    .call(chartData);
	  })

	$("#graph-select-" + type.id + "-b").change(function(e) {
	    $("#graph-" + type.id + "-b").html("");
	    let val = $(this).val();
	    let data = getData(type.values, val)
	   	d3.select("#graph-" + type.id + "-b")
		    .datum(data)
		    .call(chartData);
	  })

})