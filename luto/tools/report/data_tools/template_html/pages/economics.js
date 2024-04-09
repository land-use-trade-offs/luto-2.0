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

  // Chart:economics_0_rev_cost_all_wide.json
  Highcharts.chart("economics_0_rev_cost_all_wide", {
    chart: {
      type: "column",
      marginRight: 200,
    },

    title: {
      text: "Revenue and Cost in General",
    },

    credits: {
      enabled: false,
    },

    series: JSON.parse(
      document.getElementById("economics_0_rev_cost_all_wide_csv").innerHTML
    ),

    xAxis: {
      tickPositions: year_ticks,
    },

    yAxis: {
      title: {
        text: "Value (billion AU$)",
      },
    },

    tooltip: {
      formatter: function () {
        return (
          "<b>" +
          this.series.name +
          "</b>: " +
          Highcharts.numberFormat(this.y, 2) +
          " (billion AU$)"
        );
      },
    },

    plotOptions: {
      column: {
        stacking: "normal",
      },
    },

    legend: {
      align: "right",
      verticalAlign: "left",
      layout: "vertical",
      x: 19,
      y: 200,
      itemStyle: {
        fontSize: '11px' 
      }
    },

    exporting: {
      sourceWidth: 1200,
      sourceHeight: 600,
    },
  });
    

  // Chart:economics_1_ag_revenue_1_Irrigation_wide
  Highcharts.chart("economics_1_ag_revenue_1_Irrigation_wide", {
    chart: {
      type: "column",
      marginRight: 200,
    },

    title: {
      text: "Agricultural Revenue by Irrigation Status",
    },

    credits: {
      enabled: false,
    },

    series: JSON.parse(
      document.getElementById("economics_1_ag_revenue_1_Irrigation_wide_csv").innerHTML
    ),

    xAxis: {
      tickPositions: year_ticks,
    },

    yAxis: {
      title: {
        text: "Revenue (billion AU$)",
      },
    },

    legend: {
      align: "right",
      verticalAlign: "left",
      layout: "vertical",
      x: -50,
      y: 300,
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

  // Chart:economics_1_ag_revenue_2_Source_wide
  // Highcharts.chart("economics_1_ag_revenue_2_Source_wide", {
  //   chart: {
  //     type: "column",
  //     marginRight: 200,
  //   },

  //   title: {
  //     text: "Revenue by Agricultural Product",
  //   },

  //   credits: {
  //     enabled: false,
  //   },

  //   data: {
  //     csv: document.getElementById("economics_1_ag_revenue_2_Source_wide_csv")
  //       .innerHTML,
  //   },

  //   yAxis: {
  //     title: {
  //       text: "Revenue (billion AU$)",
  //     },
  //   },

  //   xAxis: {
  //     tickPositions: year_ticks,
  //   },

  //   legend: {
  //     align: "right",
  //     verticalAlign: "left",
  //     layout: "vertical",
  //     x: 10,
  //     y: 50,
  //   },

  //   tooltip: {
  //     formatter: function () {
  //       return `<b>Year:</b> ${this.x}<br><b>${
  //         this.series.name
  //       }:</b>${this.y.toFixed(2)}<br/>`;
  //     },
  //   },

  //   plotOptions: {
  //     column: {
  //       stacking: "normal",
  //     },
  //   },

  //   exporting: {
  //     sourceWidth: 1200,
  //     sourceHeight: 600,
  //   },
  // });

  // Chart:economics_1_ag_revenue_3_Source_type_wide
  Highcharts.chart("economics_1_ag_revenue_3_Source_type_wide", {
    chart: {
      type: "column",
      marginRight: 200,
    },

    title: {
      text: "Agricultural Revenue by Commodity",
    },

    credits: {
      enabled: false,
    },

    series: JSON.parse(
      document.getElementById("economics_1_ag_revenue_3_Source_type_wide_csv").innerHTML
    ),

    yAxis: {
      title: {
        text: "Revenue (billion AU$)",
      },
    },

    xAxis: {
      tickPositions: year_ticks,
    },

    legend: {
      align: "right",
      verticalAlign: "left",
      layout: "vertical",
      x: 10,
      y: 50,
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


  // Chart:economics_1_ag_revenue_4_Type_wide
  Highcharts.chart("economics_1_ag_revenue_4_Type_wide", {
    chart: {
      type: "column",
      marginRight: 200,
    },

    title: {
      text: "Agricultural Revenue by Commodity Type",
    },

    credits: {
      enabled: false,
    },

    series: JSON.parse(
      document.getElementById("economics_1_ag_revenue_4_Type_wide_csv").innerHTML
    ),

    xAxis: {
      tickPositions: year_ticks,
    },

    yAxis: {
      title: {
        text: "Revenue (billion AU$)",
      },
    },

    legend: {
      align: "right",
      verticalAlign: "left",
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
      },
    },

    exporting: {
      sourceWidth: 1200,
      sourceHeight: 600,
    },
  });



  // Chart:economics_1_ag_revenue_5_crop_lvstk_wide
  Highcharts.chart("economics_1_ag_revenue_5_crop_lvstk_wide", {
    chart: {
      type: "column",
      marginRight: 200,
    },

    title: {
      text: "Agricultural Revenue by Crop/Livestock",
    },

    series: JSON.parse(
      document.getElementById("economics_1_ag_revenue_5_crop_lvstk_wide_csv").innerHTML
    ),

    credits: {
      enabled: false,
    },

    yAxis: {
      title: {
        text: "Revenue (billion AU$)",
      },
    },
    xAxis: {
      tickPositions: year_ticks,
    },

    legend: {
      align: "right",
      verticalAlign: "left",
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
      },
    },

    exporting: {
      sourceWidth: 1200,
      sourceHeight: 600,
    },
  });

  // Chart:economics_2_ag_cost_1_Irrigation_wide
  Highcharts.chart("economics_2_ag_cost_1_Irrigation_wide", {
    chart: {
      type: "column",
      marginRight: 200,
    },

    title: {
      text: "Agricultural Cost by Irrigation Status",
    },

    credits: {
      enabled: false,
    },

    series: JSON.parse(
      document.getElementById("economics_2_ag_cost_1_Irrigation_wide_csv").innerHTML
    ),

    yAxis: {
      title: {
        text: "Cost (billion AU$)",
      },
    },
    xAxis: {
      tickPositions: year_ticks,
    },

    legend: {
      align: "right",
      verticalAlign: "left",
      layout: "vertical",
      x: -50,
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
      },
    },

    exporting: {
      sourceWidth: 1200,
      sourceHeight: 600,
    },
  });

  // Chart:economics_2_ag_cost_2_Source_wide
  Highcharts.chart("economics_2_ag_cost_2_Source_wide", {
    chart: {
      type: "column",
      marginRight: 200,
    },

    title: {
      text: "Agricultural Cost by Land-use",
    },

    credits: {
      enabled: false,
    },

    series: JSON.parse(
      document.getElementById("economics_2_ag_cost_2_Source_wide_csv").innerHTML
    ),

    yAxis: {
      title: {
        text: "Cost (billion AU$)",
      },
    },
    xAxis: {
      tickPositions: year_ticks,
    },

    legend: {
      align: "right",
      verticalAlign: "left",
      layout: "vertical",
      x: 10,
      y: 80,
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

  // // Chart:economics_2_ag_cost_3_Source_type_wide
  // Highcharts.chart('economics_2_ag_cost_3_Source_type_wide', {

  //     chart: {
  //         type: 'column',
  //         marginRight: 200
  //     },

  //     title: {
  //         text: 'Cost of Production by Commodity'
  //     },

  //     credits: {
  //         enabled: false
  //     },

  //     data: {
  //         csv: document.getElementById('economics_2_ag_cost_3_Source_type_wide_csv').innerHTML,
  //     },

  //     yAxis: {
  //         title: {
  //             text: 'Cost (billion AU$)'
  //         },
  //     },xAxis: {
  //     tickPositions: tickposition
  // },

  //     legend: {
  //         align: 'right',
  //         verticalAlign: 'left',
  //         layout: 'vertical',
  //         x: 80,
  //         y: 10

  //     },

  //     tooltip: {
  //         formatter: function () {
  //             return `<b>Year:</b> ${this.x}<br><b>${this.series.name}:</b>${this.y.toFixed(2)}<br/>`;
  //         }
  //     },

  //     plotOptions: {
  //         column: {
  //             stacking: 'normal',
  //         }
  //     },

  //     exporting: {
  //         sourceWidth: 1200,
  //         sourceHeight: 600,
  //     }
  // });

  // Chart:economics_2_ag_cost_4_Type_wide
  Highcharts.chart("economics_2_ag_cost_4_Type_wide", {
    chart: {
      type: "column",
      marginRight: 200,
    },

    title: {
      text: "Agricultural Cost by Cost Type",
    },

    credits: {
      enabled: false,
    },

    series: JSON.parse(
      document.getElementById("economics_2_ag_cost_4_Type_wide_csv").innerHTML
    ),

    yAxis: {
      title: {
        text: "Cost (billion AU$)",
      },
    },
    xAxis: {
      tickPositions: year_ticks,
    },

    legend: {
      align: "right",
      verticalAlign: "left",
      layout: "vertical",
      x: -50,
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
      },
    },

    exporting: {
      sourceWidth: 1200,
      sourceHeight: 600,
    },
  });

  // Chart:economics_2_ag_cost_5_crop_lvstk_wide
  Highcharts.chart("economics_2_ag_cost_5_crop_lvstk_wide", {
    chart: {
      type: "column",
      marginRight: 200,
    },

    title: {
      text: "Agricultural Cost of by Crop/Livestock",
    },

    series: JSON.parse(
      document.getElementById("economics_2_ag_cost_5_crop_lvstk_wide_csv").innerHTML
    ),

    credits: {
      enabled: false,
    },

    yAxis: {
      title: {
        text: "Cost (billion AU$)",
      },
    },

    xAxis: {
      tickPositions: year_ticks,
    },

    legend: {
      align: "right",
      verticalAlign: "left",
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
      },
    },

    exporting: {
      sourceWidth: 1200,
      sourceHeight: 600,
    },
  });

  // Chart:economics_3_rev_cost_all

  // Highcharts.chart("economics_3_rev_cost_all", {

  //   chart: {
  //     type: "columnrange",
  //     marginRight: 200,
  //   },

  //   title: {
  //     text: "Agricultural Revenue and Cost of Production",
  //   },

  //   credits: {
  //     enabled: false,
  //   },

  //   series:
  //     JSON.parse(
  //       document.getElementById("economics_3_rev_cost_all_csv").innerText
  //     )['series']
  //   ,

  //   xAxis: {
  //     categories: JSON.parse(
  //       document.getElementById("economics_3_rev_cost_all_csv").innerText
  //     )['categories'],
  //   },

  //   yAxis: {
  //     title: {
  //       text: "Value (billion AU$)",
  //     },
  //   },

  //   tooltip: {
  //     formatter: function () {
  //       return (
  //         "<b>" +
  //         this.series.name +
  //         "</b>: " +
  //         Highcharts.numberFormat(this.point.low, 2) +
  //         " - " +
  //         Highcharts.numberFormat(this.point.high, 2) +
  //         " (billion AU$)"
  //       );
  //     },
  //   },

  //   plotOptions: {
  //     columnrange: {
  //       borderRadius: "50%",
  //     },
  //   },

  //   legend: {
  //     align: "right",
  //     verticalAlign: "left",
  //     layout: "vertical",
  //     x: -50,
  //     y: 280,
  //   },
  // });

  // Chart:economics_4_am_revenue_1_Land-use_wide
  Highcharts.chart("economics_4_am_revenue_1_Land-use_wide", {
    chart: {
      type: "column",
      marginRight: 200,
    },

    title: {
      text: "Agricultural Management Revenue by Land-use",
    },

    credits: {
      enabled: false,
    },

    series: JSON.parse(
      document.getElementById("economics_4_am_revenue_1_Land-use_wide_csv").innerHTML
    ),

    yAxis: {
      title: {
        text: "Revenue (billion AU$)",
      },
    },
    xAxis: {
      tickPositions: year_ticks,
    },

    legend: {
      align: "right",
      verticalAlign: "left",
      layout: "vertical",
      x: 0,
      y: 30,
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


  // Chart:economics_4_am_revenue_2_Management Type_wide
  Highcharts.chart("economics_4_am_revenue_2_Management Type_wide", {
    chart: {
      type: "column",
      marginRight: 200,
    },

    title: {
      text: "Agricultural Management Revenue by Management Type",
    },

    credits: {
      enabled: false,
    },

    series: JSON.parse(
      document.getElementById("economics_4_am_revenue_2_Management Type_wide_csv").innerHTML
    ),

    yAxis: {
      title: {
        text: "Revenue (billion AU$)",
      },
    },
    xAxis: {
      tickPositions: year_ticks,
    },

    legend: {
      align: "right",
      verticalAlign: "left",
      layout: "vertical",
      x: 0,
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
      },
    },

    exporting: {
      sourceWidth: 1200,
      sourceHeight: 600,
    },
  });

  // Chart:economics_4_am_revenue_3_Water_wide
  Highcharts.chart("economics_4_am_revenue_3_Water_wide", {
    chart: {
      type: "column",
      marginRight: 200,
    },

    title: {
      text: "Agricultural Management Revenue by Water Source",
    },

    credits: {
      enabled: false,
    },

    series: JSON.parse(
      document.getElementById("economics_4_am_revenue_3_Water_wide_csv").innerHTML
    ),

    yAxis: {
      title: {
        text: "Revenue (billion AU$)",
      },
    },
    xAxis: {
      tickPositions: year_ticks,
    },

    legend: {
      align: "right",
      verticalAlign: "left",
      layout: "vertical",
      x: -100,
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
      },
    },

    exporting: {
      sourceWidth: 1200,
      sourceHeight: 600,
    },
  });


  // Chart:economics_5_am_cost_1_Land-use_wide
  Highcharts.chart("economics_5_am_cost_1_Land-use_wide", {
    chart: {
      type: "column",
      marginRight: 200,
    },

    title: {
      text: "Agricultural Management Cost by Land-use",
    },

    credits: {
      enabled: false,
    },

    series: JSON.parse(
      document.getElementById("economics_5_am_cost_1_Land-use_wide_csv").innerHTML
    ),

    yAxis: {
      title: {
        text: "Cost (billion AU$)",
      },
    },
    xAxis: {
      tickPositions: year_ticks,
    },

    legend: {
      align: "right",
      verticalAlign: "left",
      layout: "vertical",
      x: 0,
      y: 30,
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


  // Chart:economics_5_am_cost_2_Management Type_wide
  Highcharts.chart("economics_5_am_cost_2_Management Type_wide", {
    chart: {
      type: "column",
      marginRight: 200,
    },

    title: {
      text: "Agricultural Management Cost by Management Type",
    },

    credits: {
      enabled: false,
    },

    series: JSON.parse(
      document.getElementById("economics_5_am_cost_2_Management Type_wide_csv").innerHTML
    ),

    yAxis: {
      title: {
        text: "Cost (billion AU$)",
      },
    },
    xAxis: {
      tickPositions: year_ticks,
    },

    legend: {
      align: "right",
      verticalAlign: "left",
      layout: "vertical",
      x: 0,
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
      },
    },

    exporting: {
      sourceWidth: 1200,
      sourceHeight: 600,
    },
  });


  // Chart:economics_5_am_cost_3_Water_wide
  Highcharts.chart("economics_5_am_cost_3_Water_wide", {
    chart: {
      type: "column",
      marginRight: 200,
    },

    title: {
      text: "Agricultural Management Cost by Water Source",
    },

    credits: {
      enabled: false,
    },

    series: JSON.parse(
      document.getElementById("economics_5_am_cost_3_Water_wide_csv").innerHTML
    ),

    yAxis: {
      title: {
        text: "Cost (billion AU$)",
      },
    },
    xAxis: {
      tickPositions: year_ticks,
    },

    legend: {
      align: "right",
      verticalAlign: "left",
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
      },
    },

    exporting: {
      sourceWidth: 1200,
      sourceHeight: 600,
    },
  });


  // Chart:economics_6_non_ag_revenue_1_Land-use_wide
  Highcharts.chart("economics_6_non_ag_revenue_1_Land-use_wide", {
    chart: {
      type: "column",
      marginRight: 200,
    },

    title: {
      text: "Non-Agricultural Revenue by Land-use",
    },

    credits: {
      enabled: false,
    },

    series: JSON.parse(
      document.getElementById("economics_6_non_ag_revenue_1_Land-use_wide_csv").innerHTML
    ),

    yAxis: {
      title: {
        text: "Revenue (billion AU$)",
      },
    },
    xAxis: {
      tickPositions: year_ticks,
    },

    legend: {
      align: "right",
      verticalAlign: "left",
      layout: "vertical",
      x: 0,
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
      },
    },
    exporting: {
      sourceWidth: 1200,
      sourceHeight: 600,
    },
  });


  // Chart:economics_7_non_ag_cost_1_Land-use_wide
  Highcharts.chart("economics_7_non_ag_cost_1_Land-use_wide", {
    chart: {
      type: "column",
      marginRight: 200,
    },

    title: {
      text: "Non-Agricultural Cost by Land-use",
    },

    credits: {
      enabled: false,
    },

    series: JSON.parse(
      document.getElementById("economics_7_non_ag_cost_1_Land-use_wide_csv").innerHTML
    ),

    yAxis: {
      title: {
        text: "Cost (billion AU$)",
      },
    },
    xAxis: {
      tickPositions: year_ticks,
    },

    legend: {
      align: "right",
      verticalAlign: "left",
      layout: "vertical",
      x: 0,
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
      },
    },
    exporting: {
      sourceWidth: 1200,
      sourceHeight: 600,
    },
  });


  // Chart:economics_8_transition_ag2ag_cost_1_Land Use_wide
  Highcharts.chart("economics_8_transition_ag2ag_cost_1_Land Use_wide", {
    chart: {
      type: "column",
      marginRight: 200,
    },

    title: {
      text: "Transition Cost (Agricultural to Agricultural) by Land-use",
    },

    credits: {
      enabled: false,
    },

    series: JSON.parse(
      document.getElementById("economics_8_transition_ag2ag_cost_1_Land Use_wide_csv").innerHTML
    ),

    yAxis: {
      title: {
        text: "Cost (billion AU$)",
      },
    },
    xAxis: {
      tickPositions: year_ticks,
    },

    legend: {
      align: "right",
      verticalAlign: "left",
      layout: "vertical",
      x: 0,
      y: 10,
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


  // Chart:economics_8_transition_ag2ag_cost_2_Type_wide
  Highcharts.chart("economics_8_transition_ag2ag_cost_2_Type_wide", {
    chart: {
      type: "column",
      marginRight: 200,
    },

    title: {
      text: "Transition Cost (Agricultural to Agricultural) by Cost Type",
    },

    credits: {
      enabled: false,
    },

    series: JSON.parse(
      document.getElementById("economics_8_transition_ag2ag_cost_2_Type_wide_csv").innerHTML
    ),

    yAxis: {
      title: {
        text: "Cost (billion AU$)",
      },
    },
    xAxis: {
      tickPositions: year_ticks,
    },

    legend: {
      align: "right",
      verticalAlign: "left",
      layout: "vertical",
      x: 20,
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
      },
    },
    exporting: {
      sourceWidth: 1200,
      sourceHeight: 600,
    },
  });

  // Chart:economics_8_transition_ag2ag_cost_3_Water Supply_wide
  Highcharts.chart("economics_8_transition_ag2ag_cost_3_Water Supply_wide", {
    chart: {
      type: "column",
      marginRight: 200,
    },

    title: {
      text: "Transition Cost (Agricultural to Agricultural) by Irrigation Status",
    },

    credits: {
      enabled: false,
    },

    series: JSON.parse(
      document.getElementById("economics_8_transition_ag2ag_cost_3_Water Supply_wide_csv").innerHTML
    ),

    yAxis: {
      title: {
        text: "Cost (billion AU$)",
      },
    },
    xAxis: {
      tickPositions: year_ticks,
    },

    legend: {
      align: "right",
      verticalAlign: "left",
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
      },
    },
    exporting: {
      sourceWidth: 1200,
      sourceHeight: 600,
    },
  });

  // Chart:economics_9_transition_ag2non_cost_1_Cost type_wide
  Highcharts.chart("economics_9_transition_ag2non_cost_1_Cost type_wide", {
    chart: {
      type: "column",
      marginRight: 200,
    },

    title: {
      text: "Transition Cost (Agricultural to Non-Agricultural) by Cost type",
    },

    credits: {
      enabled: false,
    },

    series: JSON.parse(
      document.getElementById("economics_9_transition_ag2non_cost_1_Cost type_wide_csv").innerHTML
    ),

    yAxis: {
      title: {
        text: "Cost (billion AU$)",
      },
    },
    xAxis: {
      tickPositions: year_ticks,
    },

    legend: {
      align: "right",
      verticalAlign: "left",
      layout: "vertical",
      x: 0,
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
      },
    },
    exporting: {
      sourceWidth: 1200,
      sourceHeight: 600,
    },
  });


  // Chart:economics_9_transition_ag2non_cost_2_From land-use_wide
  Highcharts.chart("economics_9_transition_ag2non_cost_2_From land-use_wide", {
    chart: {
      type: "column",
      marginRight: 200,
    },

    title: {
      text: "Transition Cost (Agricultural to Non-Agricultural) from Base-Year-Perspective",
    },

    credits: {
      enabled: false,
    },

    series: JSON.parse(
      document.getElementById("economics_9_transition_ag2non_cost_2_From land-use_wide_csv").innerHTML
    ),

    yAxis: {
      title: {
        text: "Cost (billion AU$)",
      },
    },
    xAxis: {
      tickPositions: year_ticks,
    },

    legend: {
      align: "right",
      verticalAlign: "left",
      layout: "vertical",
      x: 10,
      y: 10,
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


  // Chart:economics_9_transition_ag2non_cost_3_To land-use_wide
  Highcharts.chart("economics_9_transition_ag2non_cost_3_To land-use_wide", {
    chart: {
      type: "column",
      marginRight: 200,
    },

    title: {
      text: "Transition Cost (Agricultural to Non-Agricultural) from Target-Year-Perspective",
    },

    credits: {
      enabled: false,
    },

    series: JSON.parse(
      document.getElementById("economics_9_transition_ag2non_cost_3_To land-use_wide_csv").innerHTML
    ),

    yAxis: {
      title: {
        text: "Cost (billion AU$)",
      },
    },
    xAxis: {
      tickPositions: year_ticks,
    },

    legend: {
      align: "right",
      verticalAlign: "left",
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
      },
    },
    exporting: {
      sourceWidth: 1200,
      sourceHeight: 600,
    },
  });



  // Chart:economics_9_transition_ag2non_cost_4_Water supply_wide
  Highcharts.chart("economics_9_transition_ag2non_cost_4_Water supply_wide", {
    chart: {
      type: "column",
      marginRight: 200,
    },

    title: {
      text: "Transition Cost (Agricultural to Non-Agricultural) by Irritation Status",
    },

    credits: {
      enabled: false,
    },

    series: JSON.parse(
      document.getElementById("economics_9_transition_ag2non_cost_4_Water supply_wide_csv").innerHTML
    ),

    yAxis: {
      title: {
        text: "Cost (billion AU$)",
      },
    },
    xAxis: {
      tickPositions: year_ticks,
    },

    legend: {
      align: "right",
      verticalAlign: "left",
      layout: "vertical",
      x: -30,
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
      },
    },
    exporting: {
      sourceWidth: 1200,
      sourceHeight: 600,
    },
  });


  // // Chart:economics_10_transition_non_ag2ag_cost_1_Cost type_wide
  // Highcharts.chart("economics_10_transition_non_ag2ag_cost_1_Cost type_wide", {
  //   chart: {
  //     type: "column",
  //     marginRight: 200,
  //   },

  //   title: {
  //     text: "Cost by Cost type",
  //   },

  //   credits: {
  //     enabled: false,
  //   },

  //   series: JSON.parse(
  //     document.getElementById("economics_10_transition_non_ag2ag_cost_1_Cost type_wide_csv").innerHTML
  //   ),

  //   yAxis: {
  //     title: {
  //       text: "Cost (billion AU$)",
  //     },
  //   },
  //   xAxis: {
  //     tickPositions: year_ticks,
  //   },

  //   legend: {
  //     align: "right",
  //     verticalAlign: "left",
  //     layout: "vertical",
  //     x: -50,
  //     y: 280,
  //   },

  //   tooltip: {
  //     formatter: function () {
  //       return `<b>Year:</b> ${this.x}<br><b>${this.series.name
  //         }:</b>${this.y.toFixed(2)}<br/>`;
  //     },
  //   },
  //   plotOptions: {
  //     column: {
  //       stacking: "normal",
  //     },
  //   },
  //   exporting: {
  //     sourceWidth: 1200,
  //     sourceHeight: 600,
  //   },
  // });


  // // Chart:economics_10_transition_non_ag2ag_cost_2_From land-use_wide
  // Highcharts.chart("economics_10_transition_non_ag2ag_cost_2_From land-use_wide", {
  //   chart: {
  //     type: "column",
  //     marginRight: 200,
  //   },

  //   title: {
  //     text: "Cost by From land-use",
  //   },

  //   credits: {
  //     enabled: false,
  //   },

  //   series: JSON.parse(
  //     document.getElementById("economics_10_transition_non_ag2ag_cost_2_From land-use_wide_csv").innerHTML
  //   ),

  //   yAxis: {
  //     title: {
  //       text: "Cost (billion AU$)",
  //     },
  //   },
  //   xAxis: {
  //     tickPositions: year_ticks,
  //   },

  //   legend: {
  //     align: "right",
  //     verticalAlign: "left",
  //     layout: "vertical",
  //     x: -50,
  //     y: 280,
  //   },

  //   tooltip: {
  //     formatter: function () {
  //       return `<b>Year:</b> ${this.x}<br><b>${this.series.name
  //         }:</b>${this.y.toFixed(2)}<br/>`;
  //     },
  //   },
  //   plotOptions: {
  //     column: {
  //       stacking: "normal",
  //     },
  //   },
  //   exporting: {
  //     sourceWidth: 1200,
  //     sourceHeight: 600,
  //   },
  // });


  // // Chart:economics_10_transition_non_ag2ag_cost_3_To land-use_wide
  // Highcharts.chart("economics_10_transition_non_ag2ag_cost_3_To land-use_wide", {
  //   chart: {
  //     type: "column",
  //     marginRight: 200,
  //   },

  //   title: {
  //     text: "Cost by To land-use",
  //   },

  //   credits: {
  //     enabled: false,
  //   },

  //   series: JSON.parse(
  //     document.getElementById("economics_10_transition_non_ag2ag_cost_3_To land-use_wide_csv").innerHTML
  //   ),

  //   yAxis: {
  //     title: {
  //       text: "Cost (billion AU$)",
  //     },
  //   },
  //   xAxis: {
  //     tickPositions: year_ticks,
  //   },

  //   legend: {
  //     align: "right",
  //     verticalAlign: "left",
  //     layout: "vertical",
  //     x: -50,
  //     y: 280,
  //   },

  //   tooltip: {
  //     formatter: function () {
  //       return `<b>Year:</b> ${this.x}<br><b>${this.series.name
  //         }:</b>${this.y.toFixed(2)}<br/>`;
  //     },
  //   },
  //   plotOptions: {
  //     column: {
  //       stacking: "normal",
  //     },
  //   },
  //   exporting: {
  //     sourceWidth: 1200,
  //     sourceHeight: 600,
  //   },
  // });


  // // Chart:economics_10_transition_non_ag2ag_cost_4_Water supply_wide
  // Highcharts.chart("economics_10_transition_non_ag2ag_cost_4_Water supply_wide", {
  //   chart: {
  //     type: "column",
  //     marginRight: 200,
  //   },

  //   title: {
  //     text: "Cost by Irritation Status",
  //   },

  //   credits: {
  //     enabled: false,
  //   },

  //   series: JSON.parse(
  //     document.getElementById("economics_10_transition_non_ag2ag_cost_4_Water supply_wide_csv").innerHTML
  //   ),

  //   yAxis: {
  //     title: {
  //       text: "Cost (billion AU$)",
  //     },
  //   },
  //   xAxis: {
  //     tickPositions: year_ticks,
  //   },

  //   legend: {
  //     align: "right",
  //     verticalAlign: "left",
  //     layout: "vertical",
  //     x: -50,
  //     y: 280,
  //   },

  //   tooltip: {
  //     formatter: function () {
  //       return `<b>Year:</b> ${this.x}<br><b>${this.series.name
  //         }:</b>${this.y.toFixed(2)}<br/>`;
  //     },
  //   },
  //   plotOptions: {
  //     column: {
  //       stacking: "normal",
  //     },
  //   },
  //   exporting: {
  //     sourceWidth: 1200,
  //     sourceHeight: 600,
  //   },
  // });




});

