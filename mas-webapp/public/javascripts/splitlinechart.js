function splitlinechart() {
    let xDomain = "date"
    let yDomain = "value"
    let margin = {top: 20, right: 80, bottom: 100, left: 50}
    let svgWidth = 960
    let svgHeight = 500
    let parseTime = d3.timeParse("%Y-%m");
    let labels = true;
    let min = null;
    let unit = null;
    let type = "stress";
  	let schemes = {
      sensitivity: {
        actual: "#000000",
        forecast: "#e41a1c"
      },
      stress: {
        actual: "#000000",
        base: "#4daf4a",
        adverse: "#377eb8",
        severe: "#e41a1c",
        baseF: "#4daf4a",
        adverseF: "#377eb8",
        severeF: "#e41a1c"
      }
    }
    let actualName = "actual"
    let forecastName = "Rand"
    let baseName = "base"
    let adverseName = "adverse"
    let severeName = "severe"
    let severeForecastName = "severeForecast"
    let baseForecastName = "baseForecast"
    let adverseForecastName = "adverseForecast"
    let precision = 2

    function my(selection) {
      selection.each(function(data) {

        let skeys = {
          actual: actualName,
          base: baseName,
          adverse: adverseName,
          severe: severeName,
          baseF: baseForecastName,
          adverseF: adverseForecastName,
          severeF: severeForecastName,
          forecast: forecastName
        }

        if (!labels) {
          margin = {top: 20, right: 80, bottom: 100, left: 50}
        }
        let width = svgWidth - margin.left - margin.right
    		let height = svgHeight - margin.top - margin.bottom
        let bisectDate = d3.bisector(function(d) { return d[xDomain]; }).left;
        let actualIdx = 0;
        data.forEach(function(d,i) { if (d.id ===actualName) { actualIdx = i } })

        let z = function (id) {
          for(var key of Object.keys(schemes[type])) {
            if (id.startsWith(skeys[key])) {
              return schemes[type][key]
            }
          }
        }

        let isDashed = function (d) {
          return d.id === skeys.baseF || d.id === skeys.adverseF || d.id === skeys.severeF
        }

        let svg = d3.select(this)
        	.append("svg")
        	.attr("width", svgWidth)
        	.attr("height", svgHeight)
        let g = svg.append("g").attr("transform", "translate(" + margin.left + "," + margin.top + ")");

        let x = d3.scaleTime().range([0, width])
        let y = d3.scaleLinear().range([height, 0])

        let xAxis = d3.axisBottom(x)
        	.tickFormat(d3.timeFormat("%Y-%m"))

        let yAxis = d3.axisLeft(y);

    		let line = d3.line()
        	.y(function(d) { return y(d[yDomain]); })
        	.x(function(d) { return x(d[xDomain]); });

       	let xGroup = g.append("g")
          .attr("class", "axis axis--x")
          .attr("transform", "translate(0," + height + ")");

       	let yGroup = g.append("g")
            .attr("class", "axis axis--y");

        x.domain([
          d3.min(data, function(c) { return d3.min(c.values, function(d) { return d[xDomain]; }); }),
          d3.max(data, function(c) { return d3.max(c.values, function(d) { return d[xDomain]; }); })
        ]);

        if (min === null) {
          var yMin = d3.min(data, function(c) { return d3.min(c.values, function(d) { return d[yDomain]; }); })
          var yMax = d3.max(data, function(c) { return d3.max(c.values, function(d) { return d[yDomain]; }); })
          if (Math.abs(yMax - yMin) > 1) {
            yMin = 0.75*yMin
          } else {
            yMin = yMin - 0.10*(Math.abs(yMax - yMin))
          }
          y.domain([yMin, yMax])
        } else {
          y.domain([min,
          d3.max(data, function(c) { return d3.max(c.values, function(d) { return d[yDomain]; }); })
        ]);
        }

        xGroup.call(xAxis)
          .selectAll("text")
              .style("text-anchor", "end")
              .attr("dx", "-.8em")
              .attr("dy", ".15em")
              .attr("transform", "rotate(-65)");

        yGroup.call(yAxis)


        let focus = g.append("g")
              .attr("class", "focus")
              .style("display", "none");


        let singleLine = g.selectAll(".singleLine")
          .data(data)
          .enter().append("g")
            .attr("class", "singleLine")

        let paths = singleLine.append("path")
            .attr("class", "line")
            .attr("d", function(d) { return line(d.values); })
            .style("stroke", function(d) { return z(d.id); })
            .attr("id", function(d) {
              return d.id.substring(0, 3).toUpperCase();
            })
		        .style("stroke-dasharray", function(d) {
              if (isDashed(d)) {
                return ("3, 3")
              } else {
                return null
              }
            })

				if (labels) {
          singleLine.append("text")
            .datum(function(d) {
          return {id: d.id, value: d.values[d.values.length - 1]}; })
            .attr("transform", function(d) {
          return "translate(" + x(d.value[xDomain]) + "," + y(d.value[yDomain]) + ")"; })
            .attr("x", 3)
            .attr("dy", "0.35em")
            .style("font", "13px sans-serif")
            .style("fill", function(d) { return z(d.id) })
            .text(function(d) { return d.id; })
        }


        let hover = singleLine.append("g")
          .attr("class", function(d) { return "hover hover-" + d.id; })
          .style("display", "none")

        hover.append("line")
          .attr("class", "x-hover-line hover-line")
          .attr("y1", 0)
          .attr("y2", height);

        hover.append("circle")
          .attr("r", 5)
          .style("stroke", function(d) { return z(d.id) })
          .style("fill", "white")

        hover.append("text")
          .style("font", "13px sans-serif")
          .style("fill",  function(d) { return z(d.id) })
          .style("stroke", "white")
          .style("paint-order", "stroke")
          .style("stroke-width", "5")

        svg.append("rect")
            .attr("transform", "translate(" + margin.left + "," + margin.top + ")")
            .attr("class", "overlay")
            .attr("width", width)
            .attr("height", height)
            .on("mouseover", function() { hover.style("display", null); })
            .on("mouseout", function() { hover.style("display", "none"); })
            .on("mousemove", mousemove);

        function mousemove() {
          var x0 = x.invert(d3.mouse(this)[0])
          var line = data[actualIdx]
          if (x0 <= line.values[line.values.length - 1].date) {
            var  i = bisectDate(line.values, x0, 1),
                d0 = line.values[i - 1],
                d1 = line.values[i],
                d = x0 - d0.date > d1.date - x0 ? d1 : d0;
            hover.style("display", "none");
            var tooltip = svg.selectAll(".hover-" + line.id)
            tooltip.style("display", null)
            tooltip.attr("transform", "translate(" + x(d.date) + "," + y(d.value) + ")");
            if (unit !== null) {
              tooltip.selectAll("text").text(unit + d.value)
            } else {
              tooltip.selectAll("text").text(d.value)
            }
            tooltip.selectAll(".x-hover-line").attr("y2", height - y(d.value));
          } else {
            data.forEach(function (line) {
             if (line.id !== actualName) {
              var  i = bisectDate(line.values, x0, 1),
                    d0 = line.values[i - 1],
                    d1 = line.values[i],
                    d = x0 - d0.date > d1.date - x0 ? d1 : d0;
                var tooltip = svg.selectAll(".hover-" + line.id)
                tooltip.style("display", null)
                tooltip.attr("transform", "translate(" + x(d.date) + "," + y(d.value) + ")");
                if (unit !== null) {
                  tooltip.selectAll("text").text(unit + d.value.toFixed(precision))
                } else {
                  tooltip.selectAll("text").text(d.value.toFixed(precision))
                }
                tooltip.selectAll(".x-hover-line").attr("y2", height - y(d.value));
              } else {
                var tooltip = svg.selectAll(".hover-" + line.id)
                tooltip.style("display", "none")
              }
             })
          }
        }
      })
    }

    my.xDomain = function(value) {
      if (!arguments.length) return xDomain;
      xDomain = value;
      return my;
    }

    my.yDomain = function(value) {
      if (!arguments.length) return yDomain;
      yDomain = value;
      return my;
    }


    my.width = function(value) {
      if (!arguments.length) return svgWidth;
      svgWidth = value;
      return my;
    }

    my.height = function(value) {
      if (!arguments.length) return svgHeight;
      svgHeight = value;
      return my;
    }

    my.labels = function(value) {
      if (!arguments.length) return labels;
      labels = value;
      return my;
    }

    my.min = function(value) {
      if (!arguments.length) return min;
      min = value;
      return my;
    }

    my.unit = function(value) {
      if (!arguments.length) return unit;
      unit = value;
      return my;
    }

    my.type = function(value) {
      if (!arguments.length) return type;
      type = value;
      return my;
    }

    my.precision = function(value) {
      if (!arguments.length) return precision;
      precision = value;
      return my;
    }

    my.actualName = function(value) {
      if (!arguments.length) return actualName;
      actualName = value;
      return my;
    }
    my.baseName = function(value) {
      if (!arguments.length) return baseName;
      baseName = value;
      return my;
    }
    my.baseForecastName = function(value) {
      if (!arguments.length) return baseForecastName;
      baseForecastName = value;
      return my;
    }
    my.severeName = function(value) {
      if (!arguments.length) return severeName;
      severeName = value;
      return my;
    }
    my.severeForecastName = function(value) {
      if (!arguments.length) return severeForecastName;
      severeForecastName = value;
      return my;
    }
    my.adverseName = function(value) {
      if (!arguments.length) return adverseName;
      adverseName = value;
      return my;
    }
    my.adverseForecastName = function(value) {
      if (!arguments.length) return adverseForecastName;
      adverseForecastName = value;
      return my;
    }
    my.forecastName = function(value) {
      if (!arguments.length) return forecastName;
      forecastName = value;
      return my;
    }

    return my
  }