document.addEventListener("DOMContentLoaded", function () {


  const support_info = JSON.parse(document.getElementById('Supporting_info').innerText);
  const colors = support_info.colors;
  const model_years = support_info.years;
  

  // Get the available years for plotting
  var years = model_years.map(function (x) {return parseInt(x);});
  years.sort(function (a, b) {return a - b;});
  var year_ticks = years.length == 2 ? years : null;

  
  // Set the title alignment to left
  Highcharts.setOptions({
    colors: colors,
    title: {
        align: 'left'
    }
  });


  // Chart:Area_overview_2_Category
  Highcharts.chart("Area_overview_2_Category_chart", {
    chart: {
      type: "column",
      marginRight: 380,
    },
    title: {
      text: "Total Area in Each Land-use Group",
    },
    series: JSON.parse(
      document.getElementById("Area_overview_2_Category").innerHTML
    ).AUSTRALIA,
    credits: {
      enabled: false,
    },
    xAxis: {
      tickPositions: year_ticks,
    },
    yAxis: {
      title: {
        text: "Area (ha)",
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

  // Chart:Area_overview_1_Land-use
  Highcharts.chart("Area_overview_1_Land-use_chart", {
    chart: {
      type: "column",
      marginRight: 380,
    },
    title: {
      text: "Total Area by Specific Land-use Type",
    },
    series: JSON.parse(
      document.getElementById("Area_overview_1_Land-use").innerHTML
    ).AUSTRALIA,
    credits: {
      enabled: false,
    },
    xAxis: {
      tickPositions: year_ticks,
    },
    yAxis: {
      title: {
        text: "Area (ha)",
      },
    },

    legend: {
      itemStyle: {
        fontSize: "10.5px",
      },
      align: "right",
      layout: "vertical",
      x: -40,
      y: -20,
      verticalAlign: "middle",
      itemMarginTop: 0,
      itemMarginBottom: 0.75,
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

  // Chart:Area_overview_3_Source-use
  Highcharts.chart("Area_overview_3_Source_chart", {
    chart: {
      type: "column",
      marginRight: 380,
    },
    title: {
      text: "Total Area by Broad Land-use Type",
    },
    series: JSON.parse(
      document.getElementById("Area_overview_3_Source").innerHTML
    ).AUSTRALIA,
    credits: {
      enabled: false,
    },
    xAxis: {
      tickPositions: year_ticks,
    },
    yAxis: {
      title: {
        text: "Area (ha)",
      },
    },

    legend: {
      align: "right",
      layout: "vertical",
      x: -10,
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

  // Area_Ag_1_Land-use
  Highcharts.chart("Area_Ag_1_Land-use_chart", {
    chart: {
      type: "column",
      marginRight: 380,
    },
    title: {
      text: "Agricultural Land-Use Area",
    },
    series: JSON.parse(
      document.getElementById("Area_Ag_1_Land-use").innerHTML,
    ).AUSTRALIA,
    credits: {
      enabled: false,
    },
    xAxis: {
      tickPositions: year_ticks,
    },
    yAxis: {
      title: {
        text: "Area (ha)",
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

  // Area_NonAg_1_Land-use
  Highcharts.chart("Area_NonAg_1_Land-use_chart", {
    chart: {
      type: "column",
      marginRight: 380,
    },
    title: {
      text: "Non-Agricultural Land-use Area",
    },
    series: JSON.parse(
      document.getElementById("Area_NonAg_1_Land-use").innerHTML
    ).AUSTRALIA,
    credits: {
      enabled: false,
    },
    xAxis: {
      tickPositions: year_ticks,
    },
    yAxis: {
      title: {
        text: "Area (ha)",
      },
    },

    legend: {
      align: "right",
      layout: "vertical",
      x: -50,
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

  // Area_Am_1_Type
  Highcharts.chart("Area_Am_1_Type_chart", {
    chart: {
      type: "column",
      marginRight: 380,
    },
    title: {
      text: "Agricultural Management Area by BroadLand-use Type",
    },
    series:
      JSON.parse(document.getElementById("Area_Am_1_Type").innerHTML).AUSTRALIA,
    credits: {
      enabled: false,
    },
    xAxis: {
      tickPositions: year_ticks,
    },
    yAxis: {
      title: {
        text: "Area (ha)",
      },
    },

    legend: {
      align: "right",
      layout: "vertical",
      x: -50,
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

  // Area_Am_3_Land-use
  Highcharts.chart("Area_Am_3_Land-use_chart", {
    chart: {
      type: "column",
      marginRight: 380,
    },
    title: {
      text: "Agricultural Management Area by BroadLand-use Type",
    },
    series:
      JSON.parse(document.getElementById("Area_Am_3_Land-use").innerHTML).AUSTRALIA,
    credits: {
      enabled: false,
    },
    xAxis: {
      tickPositions: year_ticks,
    },
    yAxis: {
      title: {
        text: "Area (ha)",
      },
    },

    legend: {
      layout: 'vertical',
      align: 'right',
      verticalAlign: 'middle',
      x: -150, 
      floating: true
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

  // area_begin_end_area
  document.getElementById("area_begin_end_area").innerHTML = JSON.parse(document.getElementById(
    "Area_transition_start_end"
  ).innerText).AUSTRALIA.area;

  // area_begin_end_pct
  document.getElementById("area_begin_end_pct").innerHTML = JSON.parse(document.getElementById(
    "Area_transition_start_end"
  ).innerText).AUSTRALIA.pct;
});

