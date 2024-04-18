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




  // Chart:production_1_demand_type_wide
  Highcharts.chart("production_1_demand_type_wide", {
    chart: {
      type: "column",
      marginRight: 200,
    },
    title: {
      text: "Demand, Trade, and Production of Agricultural Commodities",
    },
    series: JSON.parse(
      document.getElementById("production_1_demand_type_wide_csv").innerHTML
    ),
    credits: {
      enabled: false,
    },
    xAxis: {
      tickPositions: year_ticks,
    },
    yAxis: {
      endOnTick: false,
      maxPadding: 0.1,
      title: {
        text: "Quantity (million tonnes, million kilolitres [milk])",
      },
    },

    legend: {
      align: "right",
      verticalAlign: "top",
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
        grouping: true,
        shadow: false,
      },
    },

    exporting: {
      sourceWidth: 1200,
      sourceHeight: 600,
    },
  });

  // Chart:production_2_demand_on_off_wide
  Highcharts.chart("production_2_demand_on_off_wide", {


    title: {
      text: "Demand for Agricultural Commodities",
    },

    xAxis: {
      tickWidth: 0.05,
      
      categories: JSON.parse(
        document.getElementById("production_2_demand_on_off_wide_csv").innerHTML
        )['categories'],

      labels: {
        y: 38,
        groupedOptions: [
          {
            rotation: -90, // rotate labels for a 2st-level
            align: "center",
          },
        ],
        rotation: -90, // rotate labels for a 1st-level
        align: "center",
      },
    },

    yAxis: {
      endOnTick: false,
      maxPadding: 0.1,
      title: {
        text: "Quantity (million tonnes, million kilolitres [milk])",
      },
    },

    tooltip: {
      formatter: function () {
        return `${this.x}<br><b>${this.series.name}:</b>${this.y.toFixed(
          2
        )}<br/>`;
      },
    },

    legend: {
      align: "right",
      verticalAlign: "top",
      layout: "vertical",
      x: -80,
      y: 260,
    },

    series: JSON.parse(
      document.getElementById("production_2_demand_on_off_wide_csv").innerHTML)['series'],

    credits: {
      enabled: false,
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


  // Chart:production_3_demand_commodity
  Highcharts.chart("production_3_demand_commodity", {

    title: {
      text: "Agricultural Demand by Commodity",
    },

    xAxis: {
      tickWidth: 0.05,
      categories: JSON.parse(
        document.getElementById("production_3_demand_commodity_csv").innerHTML
      )['categories'],
      labels: {
        y: 38,
        groupedOptions: [
          {
            rotation: -90, // rotate labels for a 2st-level
            align: "center",
          },
        ],
        rotation: -90, // rotate labels for a 1st-level
        align: "center",
      },
    },

    yAxis: {
      endOnTick: false,
      maxPadding: 0.1,
      title: {
        text: "Quantity (million tonnes, million kilolitres [milk])",
      },
    },

    legend: {
      align: "right",
      verticalAlign: "top",
      layout: "vertical",
      x: 0,
      y: -10,
    },

    series: JSON.parse(
      document.getElementById("production_3_demand_commodity_csv").innerHTML
    )['series'],

    credits: {
      enabled: false,
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



  // Chart:production_4_1_demand_domestic_On-land_commodity
  Highcharts.chart("production_4_1_demand_domestic_On-land_commodity", {
    chart: {
      type: "column",
      marginRight: 200,
    },
    title: {
      text: "Domestic Consumption (Food) - On-land Commodities",
    },
    series: JSON.parse(
      document.getElementById("production_4_1_demand_domestic_On-land_commodity_csv").innerHTML
    ),
    credits: {
      enabled: false,
    },
    xAxis: {
      tickPositions: year_ticks,
    },
    yAxis: {
      title: {
        text: "Quantity (million tonnes, million kilolitres [milk])",
      },
    },

    legend: {
      align: "right",
      verticalAlign: "top",
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

  // Chart:production_4_2_demand_domestic_Off-land_commodity
  Highcharts.chart("production_4_2_demand_domestic_Off-land_commodity", {
    chart: {
      type: "column",
      marginRight: 200,
    },
    title: {
      text: "Domestic Consumption (Food) - Off-land Commodities",
    },
    series: JSON.parse(
      document.getElementById("production_4_2_demand_domestic_Off-land_commodity_csv").innerHTML
    ),
    credits: {
      enabled: false,
    },
    xAxis: {
      tickPositions: year_ticks,
    },
    xAxis: {
      tickPositions: year_ticks,
    },
    yAxis: {
      title: {
        text: "Quantity (million tonnes, million kilolitres [milk])",
      },
    },

    legend: {
      align: "right",
      verticalAlign: "top",
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

  // Chart:production_5_2_demand_Exports_commodity
  Highcharts.chart("production_5_2_demand_Exports_commodity", {
    chart: {
      type: "column",
      marginRight: 200,
    },

    title: {
      text: "Exports by Agricultural Commodity",
    },

    series: JSON.parse(
      document.getElementById("production_5_2_demand_Exports_commodity_csv").innerHTML
    ),

    credits: {
      enabled: false,
    },

    xAxis: {
      tickPositions: year_ticks,
    },

    yAxis: {
      title: {
        text: "Quantity (million tonnes, million kilolitres [milk])",
      },
    },

    legend: {
      align: "right",
      verticalAlign: "top",
      layout: "vertical",
      x: 10,
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
      },
    },

    exporting: {
      sourceWidth: 1200,
      sourceHeight: 600,
    },
  });

  // Chart:production_5_3_demand_Imports_commodity
  new Highcharts.Chart("production_5_3_demand_Imports_commodity", {
    chart: {
      type: "column",
      marginRight: 200,
    },
    title: {
      text: "Imports by Agricultural Commodity",
    },
    series: JSON.parse(
      document.getElementById("production_5_3_demand_Imports_commodity_csv").innerHTML
    ),

    credits: {
      enabled: false,
    },

    xAxis: {
      tickPositions: year_ticks,
    },

    yAxis: {
      title: {
        text: "Quantity (million tonnes, million kilolitres [milk])",
      },
    },

    legend: {
      align: "right",
      verticalAlign: "top",
      layout: "vertical",
      x: 10,
      y: -15,
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


  // Chart:production_5_4_demand_Feed_commodity
  new Highcharts.Chart({
    chart: {
      renderTo: "production_5_4_demand_Feed_commodity",
      type: "column",
      marginRight: 200,
    },
    title: {
      text: "Domestic Consumption (Feed) by Agricultural Commodity",
    },

    series: JSON.parse(
      document.getElementById("production_5_4_demand_Feed_commodity_csv").innerHTML
    ),

    credits: {
      enabled: false,
    },
    xAxis: {
      tickPositions: year_ticks,
    },
    yAxis: {
      title: {
        text: "Quantity (million tonnes, million kilolitres [milk])",
      },
    },

    legend: {
      align: "right",
      verticalAlign: "top",
      layout: "vertical",
      x: -10,
      y: 0,
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


  // Chart:production_5_5_demand_Production_commodity
  Highcharts.chart("production_5_5_demand_Production_commodity", {
    chart: {
      type: "column",
      marginRight: 200,
    },
    title: {
      text: "Total Production Requirement by Agricultural Commodity (inputs into LUTO)",
    },
    series: JSON.parse(
      document.getElementById("production_5_5_demand_Production_commodity_csv").innerHTML
    ),

    credits: {
      enabled: false,
    },
    xAxis: {
      tickPositions: year_ticks,
    },
    yAxis: {
      title: {
        text: "Quantity (million tonnes, million kilolitres [milk])",
      },
    },

    legend: {
      align: "right",
      verticalAlign: "top",
      layout: "vertical",
      x: 0,
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
      },
    },

    exporting: {
      sourceWidth: 1200,
      sourceHeight: 600,
    },
  });

  // Chart:production_5_6_demand_Production_commodity_from_LUTO
  Highcharts.chart("production_5_6_demand_Production_commodity_from_LUTO", {
    chart: {
      type: "column",
      marginRight: 200,
    },
    title: {
      text: "Total Production by Agricultural Commodity (outputs from LUTO)",
    },

    series: JSON.parse( 
      document.getElementById("production_5_6_demand_Production_commodity_from_LUTO_csv").innerHTML
    ),

    credits: {
      enabled: false,
    },
    xAxis: {
      tickPositions: year_ticks,
    },
    yAxis: {
      title: {
        text: "Quantity (million tonnes, million kilolitres [milk])",
      },
    },

    legend: {
      align: "right",
      verticalAlign: "top",
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
});

