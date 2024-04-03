// create chart
document.addEventListener("DOMContentLoaded", function () {
  Highcharts.setOptions({
    colors: [
      "#7cb5ec",
      "#434348",
      "#90ed7d",
      "#f7a35c",
      "#8085e9",
      "#f15c80",
      "#e4d354",
      "#2b908f",
      "#f45b5b",
      "#91e8e1",
    ],
  });

  // Get the available years for plotting
  var years = eval(document.getElementById("model_years").innerHTML).map(function (x) { return parseInt(x); });
  // Sort the years
  years.sort(function (a, b) { return a - b; });
  // Get the year ticks and interval
  var year_ticks = years.length == 2 ? years : null;



  // Chart:water_1_percent_to_limit
  Highcharts.chart("water_1_percent_to_limit", {
    chart: {
      type: "spline",
      marginRight: 200,
    },

    title: {
      text: "Water Use as Percentage of Sustainable Limit",
    },

    credits: {
      enabled: false,
    },

    data: {
      csv: document.getElementById("water_1_percent_to_limit_csv").innerHTML,
    },
    xAxis: {
      tickPositions: year_ticks,
    },
    yAxis: {
      title: {
        text: "Percentage to Limit (%)",
      },
    },

    legend: {
      itemStyle: {
        fontSize: "11px",
        textOverflow: "ellipsis",
      },
      align: "right",
      verticalAlign: "top",
      layout: "vertical",
      x: 5,
      y: 180,
    },

    tooltip: {
      formatter: function () {
        return `<b>Year:</b> ${this.x}<br><b>${this.series.name
          }:</b>${this.y.toFixed(2)}<br/>`;
      },
    },

    exporting: {
      sourceWidth: 1200,
      sourceHeight: 600,
    },
  });

  // Chart:water_2_volum_to_limit
  Highcharts.chart("water_2_volum_to_limit", {
    chart: {
      type: "spline",
      marginRight: 200,
    },

    title: {
      text: "Water Use by Drainage Division/River Region",
    },

    credits: {
      enabled: false,
    },

    data: {
      csv: document.getElementById("water_2_volum_to_limit_csv").innerHTML,
    },

    xAxis: {
      tickPositions: year_ticks,
    },

    yAxis: {
      title: {
        text: "Water Use (ML)",
      },
    },

    legend: {
      itemStyle: {
        fontSize: "10px",
        textOverflow: "ellipsis",
      },
      align: "right",
      verticalAlign: "top",
      layout: "vertical",
      x: -10,
      y: 180,
    },

    tooltip: {
      formatter: function () {
        return `<b>Year:</b> ${this.x}<br><b>${this.series.name
          }:</b>${this.y.toFixed(2)}<br/>`;
      },
    },

    exporting: {
      sourceWidth: 1200,
      sourceHeight: 600,
    },
  });

  // Chart:water_3_volum_by_sector
  Highcharts.chart("water_3_volum_by_sector", {
    chart: {
      type: "column",
      marginRight: 200,
    },

    title: {
      text: "Water Use by Broad Land-use and Management Type",
    },

    credits: {
      enabled: false,
    },

    series: JSON.parse(
      document.getElementById("water_3_volum_by_sector_csv").innerHTML
    ),

    xAxis: {
      tickPositions: year_ticks,
    },

    yAxis: {
      title: {
        text: "Water Use (ML)",
      },
    },

    legend: {
      align: "right",
      verticalAlign: "top",
      layout: "vertical",
      x: 10,
      y: 250,
    },

    tooltip: {
      formatter: function () {
        return `<b>Year:</b> ${this.x}<br><b>${this.series.name
          }:</b>${this.y.toFixed(2)}<br/>`;
      },
    },

    plotOptions: {
      column: {
        stacking: "normal",
        dataLabels: {
          enabled: false,
        },
      },
    },

    exporting: {
      sourceWidth: 1200,
      sourceHeight: 600,
    },
  });

  // Chart:water_4_volum_by_landuse
  Highcharts.chart("water_4_volum_by_landuse", {
    chart: {
      type: "column",
      marginRight: 200,
    },

    title: {
      text: "Water Use by Land-use and Agricultural Commodity",
    },

    credits: {
      enabled: false,
    },

    data: {
      csv: document.getElementById("water_4_volum_by_landuse_csv").innerHTML,
    },
    xAxis: {
      tickPositions: year_ticks,
    },
    yAxis: {
      title: {
        text: "Water Use (ML)",
      },
    },

    legend: {
      itemStyle: {
        fontSize: "11px",
        textOverflow: "ellipsis",
      },
      align: "right",
      verticalAlign: "top",
      layout: "vertical",
      x: 5,
      y: -10,
    },

    tooltip: {
      formatter: function () {
        return `<b>Year:</b> ${this.x}<br><b>${this.series.name
          }:</b>${this.y.toFixed(2)}<br/>`;
      },
    },

    plotOptions: {
      column: {
        stacking: "normal",
        dataLabels: {
          enabled: false,
        },
      },
    },

    exporting: {
      sourceWidth: 1200,
      sourceHeight: 600,
    },
  });

  // Chart:water_5_volum_by_irrigation
  Highcharts.chart("water_5_volum_by_irrigation", {
    chart: {
      type: "column",
      marginRight: 200,
    },

    title: {
      text: "Water Use by Irrigation Type",
    },

    credits: {
      enabled: false,
    },

    data: {
      csv: document.getElementById("water_5_volum_by_irrigation_csv")
        .innerHTML,
    },
    xAxis: {
      tickPositions: year_ticks,
    },
    yAxis: {
      title: {
        text: "Water Use (ML)",
      },
    },

    legend: {
      align: "right",
      verticalAlign: "top",
      layout: "vertical",
      x: -50,
      y: 280,
    },

    tooltip: {
      formatter: function () {
        return `<b>Year:</b> ${this.x}<br><b>${this.series.name
          }:</b>${this.y.toFixed(2)}<br/>`;
      },
    },

    plotOptions: {
      column: {
        stacking: "normal",
        dataLabels: {
          enabled: false,
        },
      },
    },

    exporting: {
      sourceWidth: 1200,
      sourceHeight: 600,
    },
  });
});
