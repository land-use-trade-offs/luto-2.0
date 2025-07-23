// create chart
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


  // Chart:Production_demand_1_Type
  Highcharts.chart("Production_demand_1_Type_chart", {
    chart: {
      type: "column",
      marginRight: 380,
    },
    title: {
      text: "Agricultural Commodities Demand by Broad Type",
    },
    series: JSON.parse(
      document.getElementById("Production_demand_1_Type").innerHTML
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
        text: "Quantity (tonnes, kilolitres [milk])",
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
        grouping: true,
        shadow: false,
      },
    },

    exporting: {
      sourceWidth: 1200,
      sourceHeight: 600,
    },
  });

  // Chart:Production_demand_2_on_off_land
  Highcharts.chart("Production_demand_2_on_off_land_chart", {

    chart: {
      marginRight: 380,
    },

    title: {
      text: "Agricultural Commodities by On/Off Land",
    },

    xAxis: {
      tickPositions: year_ticks,
    },
    yAxis: {
      endOnTick: false,
      maxPadding: 0.1,
      title: {
        text: "Quantity (tonnes, kilolitres [milk])",
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
      align: 'right', // Aligns the legend to the right side of the chart container
      verticalAlign: 'middle', // Centers the legend vertically
      layout: 'vertical', // Arranges the legend items vertically
      x: -150, // Shifts the legend left by 150px, effectively positioning it 50px from the plot area
    },


    series: JSON.parse(
      document.getElementById("Production_demand_2_on_off_land").innerHTML
    ),

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


  // Chart:Production_demand_3_Commodity
  Highcharts.chart("Production_demand_3_Commodity_chart", {

    chart: {
      marginRight: 380,
    },

    title: {
      text: "Agricultural Demand by Commodity",
    },

     xAxis: {
      tickPositions: year_ticks,
    },
    yAxis: {
      endOnTick: false,
      maxPadding: 0.1,
      title: {
        text: "Quantity (tonnes, kilolitres [milk])",
      },
    },

    legend: {
      itemStyle: {
        fontSize: "11px",
      },
      align: "right",
      layout: "vertical",
      x: -150,
      y: -10,
      verticalAlign: "middle",
    },

    series: JSON.parse(
      document.getElementById("Production_demand_3_Commodity").innerHTML
    ),

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



  // Chart:Production_demand_4_Limit
  Highcharts.chart("Production_demand_4_Limit_chart", {
    chart: {
      type: "column",
      marginRight: 380,
    },
    title: {
      text: "Agricultural Production Targets",
    },
    series: JSON.parse(
      document.getElementById("Production_demand_4_Limit").innerHTML
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
        text: "Quantity (tonnes, kilolitres [milk])",
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

  // Chart:Production_LUTO_1_Agricultural
  Highcharts.chart("Production_LUTO_1_Agricultural_chart", {
    chart: {
      type: "column",
      marginRight: 380,
    },
    title: {
      text: "Production from Agricultural Land Use",
    },
    series: JSON.parse(
      document.getElementById("Production_LUTO_1_Agricultural").innerHTML
    ).AUSTRALIA,
    credits: {
      enabled: false,
    },
    xAxis: {
      tickPositions: year_ticks,
    },
    yAxis: {
      title: {
        text: "Quantity (tonnes, kilolitres [milk])",
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

  // Chart:Production_LUTO_2_Non-Agricultural
  Highcharts.chart("Production_LUTO_2_Non-Agricultural_chart", {
    chart: {
      type: "column",
      marginRight: 380,
    },
    title: {
      text: "Production from Non-Agricultural Land Use",
    },
    series: JSON.parse(
      document.getElementById("Production_LUTO_2_Non-Agricultural").innerHTML
    ).AUSTRALIA,
    credits: {
      enabled: false,
    },
    xAxis: {
      tickPositions: year_ticks,
    },
    yAxis: {
      title: {
        text: "Quantity (tonnes, kilolitres [milk])",
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

  // Chart:Production_LUTO_3_Agricultural_Management
  Highcharts.chart("Production_LUTO_3_Agricultural_Management_chart", {
    chart: {
      type: "column",
      marginRight: 380,
    },
    title: {
      text: "Production from Agricultural Management",
    },
    series: JSON.parse(
      document.getElementById("Production_LUTO_3_Agricultural_Management").innerHTML
    ).AUSTRALIA,
    credits: {
      enabled: false,
    },
    xAxis: {
      tickPositions: year_ticks,
    },
    yAxis: {
      title: {
        text: "Quantity (tonnes, kilolitres [milk])",
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
  
  // Chart:Production_sum_1_Commodity
  Highcharts.chart("Production_sum_1_Commodity_chart", {
    chart: {
      type: "column",
      marginRight: 380,
    },
    title: {
      text: "Total Production by Commodity",
    },
    series: JSON.parse(
      document.getElementById("Production_sum_1_Commodity").innerHTML
    ).AUSTRALIA,
    credits: {
      enabled: false,
    },
    xAxis: {
      tickPositions: year_ticks,
    },
    yAxis: {
      title: {
        text: "Quantity (tonnes, kilolitres [milk])",
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

  // Chart:Production_sum_2_Type
  Highcharts.chart("Production_sum_2_Type_chart", {
    chart: {
      type: "column",
      marginRight: 380,
    },
    title: {
      text: "Total Production by Broad Type",
    },
    series: JSON.parse(
      document.getElementById("Production_sum_2_Type").innerHTML
    ).AUSTRALIA,
    credits: {
      enabled: false,
    },
    xAxis: {
      tickPositions: year_ticks,
    },
    yAxis: {
      title: {
        text: "Quantity (tonnes, kilolitres [milk])",
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
  
  // Chart:Production_achive_percent
  Highcharts.chart("Production_achive_percent_chart", {
    chart: {
      type: "spline",
      marginRight: 380,
    },
    title: {
      text: "Off-target Commodities Achievement (%)",
    },
    series: JSON.parse(
      document.getElementById("Production_achive_percent").innerHTML
    ).AUSTRALIA.concat({
      name: ' ',
      data: [[2010, 0]],
      type: 'column',
      visible: false,
      showInLegend: false,
    }),
    credits: {
      enabled: false,
    },
    xAxis: {
      tickPositions: year_ticks,
    },
    yAxis: {
      title: {
        text: "Achievement (%)",
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

    exporting: {
      sourceWidth: 1200,
      sourceHeight: 600,
    },

  });


});

