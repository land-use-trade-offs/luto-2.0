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
      marginRight: 380,
    },

    title: {
      text: "Water Use as Percentage to Avaliable Water Resources",
    },

    credits: {
      enabled: false,
    },

    series: JSON.parse(
      document.getElementById("water_1_percent_to_limit_csv").innerHTML
    ),
    xAxis: {
      tickPositions: year_ticks,
    },
    yAxis: {
      title: {
        text: "Water stress (water use as % of yield)",
      },
    },

    legend: {
      align: "right",
      layout: "vertical",
      x: -80,
      verticalAlign: "middle",
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

  // Chart:water_2_yield_to_limit
  Highcharts.chart("water_2_yield_to_limit", {
    chart: {
      type: "spline",
      marginRight: 380,
    },

    title: {
      text: "Water Net Yield by Drainage Division/River Region",
    },

    credits: {
      enabled: false,
    },

    series: JSON.parse(
      document.getElementById("water_2_yield_to_limit_csv").innerHTML
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
      layout: "vertical",
      x: -80,
      verticalAlign: "middle",
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

  // Chart:water_3_net_yield_by_sector
  Highcharts.chart("water_3_net_yield_by_sector", {
    chart: {
      type: "column",
      marginRight: 380,
    },

    title: {
      text: "Water Net Yield by Broad Land-use and Management Type",
    },

    credits: {
      enabled: false,
    },

    series: JSON.parse(
      document.getElementById("water_3_net_yield_by_sector_csv").innerHTML
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
      layout: "vertical",
      x: -150,
      verticalAlign: "middle",
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

  // Chart:water_4_net_yield_by_landuse
  Highcharts.chart("water_4_net_yield_by_landuse", {
    chart: {
      type: "column",
      marginRight: 380,
    },

    title: {
      text: "Water Net Yield by Land-use and Agricultural Commodity",
    },

    credits: {
      enabled: false,
    },

    series: JSON.parse(
      document.getElementById("water_4_net_yield_by_landuse_csv").innerHTML
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
      itemStyle: {
        fontSize: "11px",
      },
      align: "right",
      layout: "vertical",
      x: 0,
      y: -10,
      verticalAlign: "middle",
      itemMarginTop: 0,
      itemMarginBottom: 1,
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

  // Chart:water_5_net_yield_by_Water_supply
  Highcharts.chart("water_5_net_yield_by_Water_supply", {
    chart: {
      type: "column",
      marginRight: 380,
    },

    title: {
      text: "Water Net Yield by Irrigation Type",
    },

    credits: {
      enabled: false,
    },

    series: JSON.parse(
      document.getElementById("water_5_net_yield_by_Water_supply_csv").innerHTML
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
      layout: "vertical",
      x: -180,
      verticalAlign: "middle",
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
