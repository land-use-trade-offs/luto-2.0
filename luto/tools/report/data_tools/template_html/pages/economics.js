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


  // Update the year scrolls for the transition matrix graphs
  window.onload = function () {
    let modelYears = eval(document.getElementById('model_years').innerText);

    // Sort the modelYears array in ascending order
    modelYears.sort(function (a, b) { return a - b; });

    // Initialize the first scroll bar
    let yearInput_ag2ag = document.getElementById('year_ag2ag');
    let yearOutput_ag2ag = document.getElementById('yearOutput_ag2ag');

    let yearInput_ag2non_ag = document.getElementById('year_ag2non_ag');
    let yearOutput_ag2non_ag = document.getElementById('yearOutput_ag2non_ag');

    yearInput_ag2ag.min = yearInput_ag2non_ag.min = modelYears[0];
    yearInput_ag2ag.max = yearInput_ag2non_ag.max = modelYears[modelYears.length - 1];
    yearInput_ag2ag.step = yearInput_ag2non_ag.step = modelYears[1] - modelYears[0];  
    yearInput_ag2ag.value = yearInput_ag2non_ag.value = modelYears[0];
    yearOutput_ag2ag.value = yearOutput_ag2non_ag.value = modelYears[0];

    draw_cost_ag2ag();
    draw_cost_ag2non_ag();
  }

  // Chart:economics_0_rev_cost_all_wide.json
  Highcharts.chart("economics_0_rev_cost_all_wide", {
    chart: {
      type: "column",
      marginRight: 380,
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
      verticalalign: "left",
      layout: "vertical",
      x: -100,
      verticalAlign: "middle",
    },

    exporting: {
      sourceWidth: 1200,
      sourceHeight: 600,
    },
  });


  // Chart:economics_1_ag_revenue_3_Water_supply_wide
  Highcharts.chart("economics_1_ag_revenue_3_Water_supply_wide", {
    chart: {
      type: "column",
      marginRight: 380,
    },

    title: {
      text: "Agricultural Revenue by Irrigation Status",
    },

    credits: {
      enabled: false,
    },

    series: JSON.parse(
      document.getElementById("economics_1_ag_revenue_3_Water_supply_wide_csv").innerHTML
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
      verticalalign: "left",
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


  // Chart:economics_1_ag_revenue_1_Land-use_wide
  Highcharts.chart("economics_1_ag_revenue_1_Land-use_wide", {
    chart: {
      type: "column",
      marginRight: 380,
    },

    title: {
      text: "Agricultural Revenue by Commodity",
    },

    credits: {
      enabled: false,
    },

    series: JSON.parse(
      document.getElementById("economics_1_ag_revenue_1_Land-use_wide_csv").innerHTML
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
      verticalalign: "left",
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


  // Chart:economics_1_ag_revenue_2_Type_wide
  Highcharts.chart("economics_1_ag_revenue_2_Type_wide", {
    chart: {
      type: "column",
      marginRight: 380,
    },

    title: {
      text: "Agricultural Revenue by Commodity Type",
    },

    credits: {
      enabled: false,
    },

    series: JSON.parse(
      document.getElementById("economics_1_ag_revenue_2_Type_wide_csv").innerHTML
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
      verticalalign: "left",
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


  // Chart:economics_2_ag_cost_3_Water_supply_wide
  Highcharts.chart("economics_2_ag_cost_3_Water_supply_wide", {
    chart: {
      type: "column",
      marginRight: 380,
    },

    title: {
      text: "Agricultural Cost by Irrigation Status",
    },

    credits: {
      enabled: false,
    },

    series: JSON.parse(
      document.getElementById("economics_2_ag_cost_3_Water_supply_wide_csv").innerHTML
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
      verticalalign: "left",
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

  // Chart:economics_2_ag_cost_1_Land-use_wide
  Highcharts.chart("economics_2_ag_cost_1_Land-use_wide", {
    chart: {
      type: "column",
      marginRight: 380,
    },

    title: {
      text: "Agricultural Cost by Land-use",
    },

    credits: {
      enabled: false,
    },

    series: JSON.parse(
      document.getElementById("economics_2_ag_cost_1_Land-use_wide_csv").innerHTML
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
      verticalalign: "left",
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

  
  // Chart:economics_2_ag_cost_2_Type_wide
  Highcharts.chart("economics_2_ag_cost_2_Type_wide", {
    chart: {
      type: "column",
      marginRight: 380,
    },

    title: {
      text: "Agricultural Cost by Cost Type",
    },

    credits: {
      enabled: false,
    },

    series: JSON.parse(
      document.getElementById("economics_2_ag_cost_2_Type_wide_csv").innerHTML
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
      verticalalign: "left",
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



  // Chart:economics_4_am_revenue_1_Land-use_wide
  Highcharts.chart("economics_4_am_revenue_1_Land-use_wide", {
    chart: {
      type: "column",
      marginRight: 380,
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
      verticalalign: "left",
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


  // Chart:economics_4_am_revenue_2_Management Type_wide
  Highcharts.chart("economics_4_am_revenue_2_Management Type_wide", {
    chart: {
      type: "column",
      marginRight: 380,
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
      verticalalign: "left",
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

  // Chart:economics_4_am_revenue_3_Water_supply_wide
  Highcharts.chart("economics_4_am_revenue_3_Water_supply_wide", {
    chart: {
      type: "column",
      marginRight: 380,
    },

    title: {
      text: "Agricultural Management Revenue by Water Source",
    },

    credits: {
      enabled: false,
    },

    series: JSON.parse(
      document.getElementById("economics_4_am_revenue_3_Water_supply_wide_csv").innerHTML
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
      verticalalign: "left",
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
      marginRight: 380,
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
      verticalalign: "left",
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


  // Chart:economics_5_am_cost_2_Management Type_wide
  Highcharts.chart("economics_5_am_cost_2_Management Type_wide", {
    chart: {
      type: "column",
      marginRight: 380,
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
      verticalalign: "left",
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


  // Chart:economics_5_am_cost_3_Water_supply_wide
  Highcharts.chart("economics_5_am_cost_3_Water_supply_wide", {
    chart: {
      type: "column",
      marginRight: 380,
    },

    title: {
      text: "Agricultural Management Cost by Water Source",
    },

    credits: {
      enabled: false,
    },

    series: JSON.parse(
      document.getElementById("economics_5_am_cost_3_Water_supply_wide_csv").innerHTML
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
      verticalalign: "left",
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
      marginRight: 380,
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
      verticalalign: "left",
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


  // Chart:economics_7_non_ag_cost_1_Land-use_wide
  Highcharts.chart("economics_7_non_ag_cost_1_Land-use_wide", {
    chart: {
      type: "column",
      marginRight: 380,
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
      verticalalign: "left",
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


  // Chart:economics_8_transition_ag2ag_cost_1_From land-use_wide
  Highcharts.chart("economics_8_transition_ag2ag_cost_1_From land-use_wide", {
    chart: {
      type: "column",
      marginRight: 380,
    },

    title: {
      text: "Transition Cost (Agricultural to Agricultural) from base-year-prespective",
    },

    credits: {
      enabled: false,
    },

    series: JSON.parse(
      document.getElementById("economics_8_transition_ag2ag_cost_1_From land-use_wide_csv").innerHTML
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
      verticalalign: "left",
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

  // Chart:economics_8_transition_ag2ag_cost_2_To land-use_wide
  Highcharts.chart("economics_8_transition_ag2ag_cost_2_To land-use_wide", {
    chart: {
      type: "column",
      marginRight: 380,
    },

    title: {
      text: "Transition Cost (Agricultural to Agricultural) from target-year-prespective",
    },

    credits: {
      enabled: false,
    },

    series: JSON.parse(
      document.getElementById("economics_8_transition_ag2ag_cost_2_To land-use_wide_csv").innerHTML
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
      verticalalign: "left",
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



  // economics_8_transition_ag2ag_cost_5_transition_matrix
  let data_ag2ag = JSON.parse(
    document.getElementById("economics_8_transition_ag2ag_cost_5_transition_matrix_csv").innerHTML
  );

  // Get the slider_ag2ag and the year span
  let slider_ag2ag = document.getElementById("year_ag2ag");
  let incrementButton_ag2ag = document.getElementById("increment_ag2ag");
  let decrementButton_ag2ag = document.getElementById("decrement_ag2ag");

  // Add event listeners to the buttons
  slider_ag2ag.addEventListener("input", function () {
    yearOutput_ag2ag.innerHTML = this.value;
    draw_cost_ag2ag();
  });

  incrementButton_ag2ag.addEventListener("click", function () {
    slider_ag2ag.value = parseInt(slider_ag2ag.value) + 1;
    slider_ag2ag.dispatchEvent(new Event('input'));
  });

  decrementButton_ag2ag.addEventListener("click", function () {
    slider_ag2ag.value = parseInt(slider_ag2ag.value) - 1;
    slider_ag2ag.dispatchEvent(new Event('input'));
  });

  // Function to draw the chart
  draw_cost_ag2ag = function () {

    let values = data_ag2ag['series'].find(item => item.Year == slider_ag2ag.value)['data'];
    let lastElements = values.map(sublist => sublist[sublist.length - 1]);
    let vale_min = Math.min(...lastElements.flat());
    let vale_max = Math.max(...lastElements.flat());

    Highcharts.chart("economics_8_transition_ag2ag_cost_5_transition_matrix", {
      chart: {
        type: "heatmap",
        marginRight: 380,
        inverted: true,
      },

      title: {
        text: null,
      },

      credits: {
        enabled: false,
      },

      series: [{
        data: values,
        borderWidth: 0.2,
        tooltip: {
          headerFormat: '',
          pointFormatter: function () {
            return `${data_ag2ag["categories"][this.x]} 
                    <b>==></b> ${data_ag2ag["categories"][this.y]}: 
                    <b>${this.value.toFixed(2)} (billion $)</b>`;
          }
        },
      }],

      yAxis: {
        min: 0,
        max: data_ag2ag["categories"].length - 1,
        categories: data_ag2ag["categories"],
        title: {
          text: "To Land-use",
        },
        labels: {
          rotation: -25,
        },
      },

      xAxis: {
        categories: data_ag2ag["categories"],
        title: {
          text: "From Land-use",
        },
      },

      colorAxis: {
        stops: [
          [0, '#3060cf'],
          [0.5, '#fffbbc'],
          [0.9, '#c4463a'],
          [1, '#c4463a']
        ],
        min: vale_min,
        max: vale_max,
        startOnTick: false,
        endOnTick: false,
        reversed: false,
        labels: {
          formatter: function () {
            return this.value.toFixed(0);
          }
        }
      },

      legend: {
        align: "right",
        verticalalign: "left",
        layout: "vertical",
        x: -180,
        verticalAlign: "middle",
      },

      exporting: {
        sourceWidth: 1200,
        sourceHeight: 600,
      },
    });
  };




  // Chart:economics_8_transition_ag2ag_cost_3_Type_wide
  Highcharts.chart("economics_8_transition_ag2ag_cost_3_Type_wide", {
    chart: {
      type: "column",
      marginRight: 380,
    },

    title: {
      text: "Transition Cost (Agricultural to Agricultural) by Cost Type",
    },

    credits: {
      enabled: false,
    },

    series: JSON.parse(
      document.getElementById("economics_8_transition_ag2ag_cost_3_Type_wide_csv").innerHTML
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
      verticalalign: "left",
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

  // Chart:economics_8_transition_ag2ag_cost_4_Water Supply_wide
  Highcharts.chart("economics_8_transition_ag2ag_cost_4_Water Supply_wide", {
    chart: {
      type: "column",
      marginRight: 380,
    },

    title: {
      text: "Transition Cost (Agricultural to Agricultural) by Irrigation Status",
    },

    credits: {
      enabled: false,
    },

    series: JSON.parse(
      document.getElementById("economics_8_transition_ag2ag_cost_4_Water Supply_wide_csv").innerHTML
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
      verticalalign: "left",
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
      },
    },
    exporting: {
      sourceWidth: 1200,
      sourceHeight: 600,
    },
  });


  // Chart:economics_9_transition_ag2non_cost_5_transition_matrix

  let data_ag2non_ag = JSON.parse(
    document.getElementById("economics_9_transition_ag2non_cost_5_transition_matrix_csv").innerHTML
  );

  // Get the slider_ag2ag and the year span
  let slider_ag2non_ag = document.getElementById("year_ag2non_ag");
  let incrementButton_ag2non_ag = document.getElementById("increment_ag2non_ag");
  let decrementButton_ag2non_ag = document.getElementById("decrement_ag2non_ag");

  // Add event listeners to the buttons
  slider_ag2non_ag.addEventListener("input", function () {
    yearOutput_ag2non_ag.innerHTML = this.value;
    draw_cost_ag2non_ag();
  });

  incrementButton_ag2non_ag.addEventListener("click", function () {
    slider_ag2non_ag.value = parseInt(slider_ag2non_ag.value) + 1;
    slider_ag2non_ag.dispatchEvent(new Event('input'));
  });

  decrementButton_ag2non_ag.addEventListener("click", function () {
    slider_ag2non_ag.value = parseInt(slider_ag2non_ag.value) - 1;
    slider_ag2non_ag.dispatchEvent(new Event('input'));
  });

  // Function to draw the chart
  draw_cost_ag2non_ag = function () {

    values = data_ag2non_ag['series'].find(item => item.Year == slider_ag2non_ag.value)['data'];
    lastElements = values.map(sublist => sublist[sublist.length - 1]);
    vale_min = Math.min(...lastElements.flat());
    vale_max = Math.max(...lastElements.flat());

    Highcharts.chart("economics_9_transition_ag2non_cost_5_transition_matrix", {
      chart: {
        type: "heatmap",
        marginRight: 380,
        inverted: true,
      },

      title: {
        text: null,
      },

      credits: {
        enabled: false,
      },

      series: [{
        data: values,
        borderWidth: 0.2,
        tooltip: {
          headerFormat: '',
          pointFormatter: function () {
            return `${data_ag2non_ag["categories_from"][this.x]} 
                    <b>==></b> ${data_ag2non_ag["categories_to"][this.y]}: 
                    <b>${this.value.toFixed(2)} (billion $)</b>`;
          }
        },
      }],

      yAxis: {
        min: 0,
        max: data_ag2non_ag["categories_to"].length - 1,
        categories: data_ag2non_ag["categories_to"],
        title: {
          text: "To Land-use",
        },
      },

      xAxis: {
        min: 0,
        max: data_ag2non_ag["categories_from"].length - 1,
        categories: data_ag2non_ag["categories_from"],
        title: {
          text: "From Land-use",
        },
      },

      colorAxis: {
        stops: [
          [0, '#3060cf'],
          [0.5, '#fffbbc'],
          [0.9, '#c4463a'],
          [1, '#c4463a']
        ],
        min: vale_min,
        max: vale_max,
        startOnTick: false,
        endOnTick: false,
        reversed: false,
        labels: {
          formatter: function () {
            return this.value.toFixed(2);
          }
        }
      },

      legend: {
        align: "right",
        verticalalign: "left",
        layout: "vertical",
        x: -180,
        verticalAlign: "middle",
      },

      exporting: {
        sourceWidth: 1200,
        sourceHeight: 600,
      },
    });
  };


  // Chart:economics_9_transition_ag2non_cost_1_Cost type_wide
  Highcharts.chart("economics_9_transition_ag2non_cost_1_Cost type_wide", {
    chart: {
      type: "column",
      marginRight: 380,
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
      verticalalign: "left",
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


  // Chart:economics_9_transition_ag2non_cost_2_From land-use_wide
  Highcharts.chart("economics_9_transition_ag2non_cost_2_From land-use_wide", {
    chart: {
      type: "column",
      marginRight: 380,
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
      verticalalign: "left",
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


  // Chart:economics_9_transition_ag2non_cost_3_To land-use_wide
  Highcharts.chart("economics_9_transition_ag2non_cost_3_To land-use_wide", {
    chart: {
      type: "column",
      marginRight: 380,
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
      verticalalign: "left",
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



  // Chart:economics_9_transition_ag2non_cost_4_Water supply_wide
  Highcharts.chart("economics_9_transition_ag2non_cost_4_Water supply_wide", {
    chart: {
      type: "column",
      marginRight: 380,
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
      verticalalign: "left",
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
  //     marginRight: 380,
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
  //     verticalalign: "left",
  //     layout: "vertical",
  //     x: 0,
  //     verticalAlign: "middle",
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
  //     marginRight: 380,
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
  //     verticalalign: "left",
  //     layout: "vertical",
  //     x: 0,
  //     verticalAlign: "middle",
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
  //     marginRight: 380,
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
  //     verticalalign: "left",
  //     layout: "vertical",
  //     x: 0,
  //     verticalAlign: "middle",
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
  //     marginRight: 380,
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
  //     verticalalign: "left",
  //     layout: "vertical",
  //     x: 0,
  //     verticalAlign: "middle",
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

