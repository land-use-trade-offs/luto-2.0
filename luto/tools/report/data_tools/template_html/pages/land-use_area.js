document.addEventListener("DOMContentLoaded", function () {

  Highcharts.setOptions({
    colors: [
      "#8085e9",
      "#f15c80",
      "#e4d354",
      "#2b908f",
      "#f45b5b",
      "#7cb5ec",
      "#434348",
      "#90ed7d",
      "#f7a35c",
      "#91e8e1",
    ],
  });



  // Get the available years for plotting
  var years = eval(document.getElementById("model_years").innerHTML).map(function (x) {return parseInt(x);});
  // Sort the years
  years.sort(function (a, b) {return a - b;});
  // Get the year ticks and interval
  var year_ticks = years.length == 2 ? years : null;

  // Set the title alignment to left
  Highcharts.setOptions({
    title: {
        align: 'left'
    }
  });

  // Chart:area_0_grouped_lu_area_wide
  Highcharts.chart("area_0_grouped_lu_area_wide", {
    chart: {
      type: "column",
      marginRight: 380,
    },
    title: {
      text: "Total Area in Each Land-use Group",
    },
    series: JSON.parse(
      document.getElementById("area_0_grouped_lu_area_wide_csv").innerHTML
    ),
    credits: {
      enabled: false,
    },
    xAxis: {
      tickPositions: year_ticks,
    },
    yAxis: {
      title: {
        text: "Area (million km2)",
      },
    },

    legend: {
      align: "right",
      layout: "vertical",
      x: -120,
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
      },
    },

    exporting: {
      sourceWidth: 1200,
      sourceHeight: 600,
    },
  });

  
  // Chart:area_1_total_area_wide
  Highcharts.chart("area_1_total_area_wide", {
    chart: {
      type: "column",
      marginRight: 380,
    },
    title: {
      text: "Total Area by Land-use and Agricultural Commodity",
    },
    series: JSON.parse(
      document.getElementById("area_1_total_area_wide_csv").innerHTML
    ),
    credits: {
      enabled: false,
    },
    xAxis: {
      tickPositions: year_ticks,
    },
    yAxis: {
      title: {
        text: "Area (million km2)",
      },
    },

    legend: {
      itemStyle: {
        fontSize: "11px",
      },
      align: "right",
      layout: "vertical",
      x: -30,
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
      },
    },

    exporting: {
      sourceWidth: 1200,
      sourceHeight: 600,
    },
  });

  // Chart:area_2_Water_supply_area_wide
  Highcharts.chart("area_2_Water_supply_area_wide", {
    chart: {
      type: "column",
      marginRight: 380,
    },
    title: {
      text: "Total Area by Irrigation Type",
    },
    series: JSON.parse(
      document.getElementById("area_2_Water_supply_area_wide_csv").innerHTML
    ),
    credits: {
      enabled: false,
    },
    xAxis: {
      tickPositions: year_ticks,
    },
    yAxis: {
      title: {
        text: "Area (million km2)",
      },
    },

    legend: {
      align: "right",
      layout: "vertical",
      x: -250,
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
      },
    },

    exporting: {
      sourceWidth: 1200,
      sourceHeight: 600,
    },
  });

  // area_3_non_ag_lu_area_wide
  Highcharts.chart("area_3_non_ag_lu_area_wide", {
    chart: {
      type: "column",
      marginRight: 380,
    },
    title: {
      text: "Non-Agricultural Land-Use Area",
    },
    series: JSON.parse(
      document.getElementById("area_3_non_ag_lu_area_wide_csv").innerHTML,
    ),
    credits: {
      enabled: false,
    },
    xAxis: {
      tickPositions: year_ticks,
    },
    yAxis: {
      title: {
        text: "Area (million km2)",
      },
    },

    legend: {
      align: "right",
      layout: "vertical",
      x: 0,
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
      },
    },

    exporting: {
      sourceWidth: 1200,
      sourceHeight: 600,
    },
  });

  // area_4_am_total_area_wide
  Highcharts.chart("area_4_am_total_area_wide", {
    chart: {
      type: "column",
      marginRight: 380,
    },
    title: {
      text: "Agricultural Management Area by Type",
    },
    series: JSON.parse(
      document.getElementById("area_4_am_total_area_wide_csv").innerHTML
    ),
    credits: {
      enabled: false,
    },
    xAxis: {
      tickPositions: year_ticks,
    },
    yAxis: {
      title: {
        text: "Area (million km2)",
      },
    },

    legend: {
      align: "right",
      layout: "vertical",
      x: -100,
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
      },
    },

    exporting: {
      sourceWidth: 1200,
      sourceHeight: 600,
    },
  });

  // area_5_am_lu_area_wide
  Highcharts.chart("area_5_am_lu_area_wide", {
    chart: {
      type: "column",
      marginRight: 380,
    },
    title: {
      text: "Agricultural Management Area by Land-use Type",
    },
    series:
      JSON.parse(document.getElementById("area_5_am_lu_area_wide_csv").innerHTML)
    ,
    credits: {
      enabled: false,
    },
    xAxis: {
      tickPositions: year_ticks,
    },
    yAxis: {
      title: {
        text: "Area (million km2)",
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
      },
    },

    exporting: {
      sourceWidth: 1200,
      sourceHeight: 600,
    },
  });

  // area_6_begin_end_area
  document.getElementById("area_6_begin_end_area").innerHTML = document.getElementById(
    "area_6_begin_end_area_csv"
  ).innerText;

  // area_7_begin_end_pct
  document.getElementById("area_7_begin_end_pct").innerHTML = document.getElementById(
    "area_7_begin_end_pct_csv"
  ).innerText;
});

