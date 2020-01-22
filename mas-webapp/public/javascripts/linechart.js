function linechart() {
    let xDomain = "date"
    let yDomain = "value"
    let margin = {top: 20, right: 80, bottom: 100, left: 50}
    let svgWidth = 960
    let svgHeight = 500
    let parseTime = d3.timeParse("%Y-%m");
    let labels = true;
    let min = null;
    let unit = null;
    let colorScheme = 0;
    let precision = 2;
    let schemes = [
      d3.schemeCategory10,
      ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#ffff33"]
    ];

    function my(selection) {
      selection.each(function(data) {
        if (!labels) {
          margin = {top: 20, right: 20, bottom: 100, left: 50}
        }
        let width = svgWidth - margin.left - margin.right
        let height = svgHeight - margin.top - margin.bottom
        let bisectDate = d3.bisector(function(d) { return d[xDomain]; }).left;

        let svg = d3.select(this)
          .append("svg")
          .attr("width", svgWidth)
          .attr("height", svgHeight)
        let g = svg.append("g").attr("transform", "translate(" + margin.left + "," + margin.top + ")");

        let x = d3.scaleTime().range([0, width])
        let y = d3.scaleLinear().range([height, 0])
        let z = d3.scaleOrdinal(schemes[colorScheme])

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

        z.domain(data.map(function(c) { return c.id; }));

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
          data.forEach(function (line) {
           var  i = bisectDate(line.values, x0, 1),
                d0 = line.values[i - 1],
                d1 = line.values[i],
                d = x0 - d0.date > d1.date - x0 ? d1 : d0;
            var tooltip = svg.selectAll(".hover-" + line.id)
            tooltip.attr("transform", "translate(" + x(d.date) + "," + y(d.value) + ")");
            if (unit !== null) {
              tooltip.selectAll("text").text(unit + d.value.toFixed(precision))
            } else {
              tooltip.selectAll("text").text(d.value.toFixed(precision))
            }
            tooltip.selectAll(".x-hover-line").attr("y2", height - y(d.value));
          })


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

    my.colorScheme = function(value) {
      if (!arguments.length) return colorScheme;
      colorScheme = value;
      return my;
    }


    my.precision = function(value) {
      if (!arguments.length) return precision;
      precision = value;
      return my;
    }
    return my
  }