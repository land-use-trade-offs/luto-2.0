// make charts
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



  // Chart:GHG_overview
  Highcharts.chart("GHG_overview_chart", {
    chart: {
      type: "column",
      marginRight: 380,
    },

    title: {
      text: "Cumulative GHG Emissions",
    },

    credits: {
      enabled: false,
    },

    series: JSON.parse(
      document.getElementById("GHG_overview").innerHTML
    ).AUSTRALIA,

    yAxis: {
      title: {
        text: "Greenhouse Gas (Mt CO2e)",
      },
    },
    xAxis: {
      tickPositions: year_ticks,
    },

    legend: {
      // enabled: false,
      align: 'right',
      verticalAlign: 'middle',
      layout: 'vertical',
      x: 0,
      y: 50
    },

    tooltip: {
      formatter: function () {
        return `<b>Year:</b> ${this.x}<br><b>${
          this.series.name
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

  // Chart:GHG_split_Ag_2_Land-use
  Highcharts.chart("GHG_split_Ag_2_Land-use_chart", {
    chart: {
      type: "column",
      marginRight: 380,
    },
    title: {
      text: "GHG Emissions by Land-use/Management Type",
    },
    xAxis: {
      tickPositions: year_ticks,
    },
    yAxis: {
      title: {
        text: "Greenhouse Gas (Mt CO2e)",
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
        return `<b>Year:</b> ${this.x}<br><b>${
          this.series.name
        }:</b>${this.y.toFixed(2)}<br/>`;
      },
    },
    series: JSON.parse(
      document.getElementById("GHG_split_Ag_2_Land-use").innerHTML
    ).AUSTRALIA,
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

  // Chart:GHG_split_Ag_3_Land-use_type
  Highcharts.chart("GHG_split_Ag_3_Land-use_type_chart", {
    chart: {
      type: "column",
      marginRight: 380,
    },

    title: {
      text: "GHG Emissions (on-land) by Crop/Livestock",
    },

    credits: {
      enabled: false,
    },

    series: JSON.parse(
      document.getElementById("GHG_split_Ag_3_Land-use_type").innerHTML
    ).AUSTRALIA,

    yAxis: {
      title: {
        text: "Greenhouse Gas (Mt CO2e)",
      },
    },
    xAxis: {
      tickPositions: year_ticks,
    },

    legend: {
      align: "right",
      layout: "vertical",
      x: -150,
      verticalAlign: "middle",
    },

    tooltip: {
      formatter: function () {
        return `<b>Year:</b> ${this.x}<br><b>${
          this.series.name
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

  // Chart:GHG_split_Ag_5_Water_supply
  Highcharts.chart("GHG_split_Ag_5_Water_supply_chart", {
    chart: {
      type: "column",
      marginRight: 380,
    },

    title: {
      text: "GHG Emissions (on-land) by Irrigation Type",
    },

    credits: {
      enabled: false,
    },

    series: JSON.parse(
      document.getElementById("GHG_split_Ag_5_Water_supply").innerHTML
    ).AUSTRALIA,

    yAxis: {
      title: {
        text: "Greenhouse Gas (Mt CO2e)",
      },
    },
    xAxis: {
      tickPositions: year_ticks,
    },

    legend: {
      align: "right",
      layout: "vertical",
      x: -180,
      verticalAlign: "middle",
    },

    tooltip: {
      formatter: function () {
        return `<b>Year:</b> ${this.x}<br><b>${
          this.series.name
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

  // Chart:GHG_split_Ag_1_GHG_Category
  Highcharts.chart("GHG_split_Ag_1_GHG_Category_chart", {
    chart: {
      type: "column",
      marginRight: 380,
    },

    title: {
      text: "GHG Emissions (on-land) by Gas",
    },

    credits: {
      enabled: false,
    },

    series: JSON.parse(
      document.getElementById("GHG_split_Ag_1_GHG_Category").innerHTML
    ).AUSTRALIA,

    yAxis: {
      title: {
        text: "Greenhouse Gas (Mt CO2e)",
      },
    },
    xAxis: {
      tickPositions: year_ticks,
    },

    legend: {
      align: "right",
      layout: "vertical",
      x: -150,
      verticalAlign: "middle",
    },

    tooltip: {
      formatter: function () {
        return `<b>Year:</b> ${this.x}<br><b>${
          this.series.name
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

  // Chart:GHG_split_Ag_4_Source
  Highcharts.chart("GHG_split_Ag_4_Source_chart", {
    chart: {
      type: "column",
      marginRight: 380,
    },

    title: {
      text: "GHG Emissions (on-land) by Source",
    },

    credits: {
      enabled: false,
    },

    series: JSON.parse(
      document.getElementById("GHG_split_Ag_4_Source").innerHTML
    ).AUSTRALIA,

    yAxis: {
      title: {
        text: "Emissions (Mt CO2e)",
      },
    },
    xAxis: {
      tickPositions: year_ticks,
    },

    legend: {
      align: "right",
      verticalAlign: "middle",
      layout: "vertical",
      x: 0,
    },

    tooltip: {
      formatter: function () {
        return `<b>Year:</b> ${this.x}<br><b>${
          this.series.name
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



  Highcharts.chart("GHG_split_off_land_3_Commodity_chart", {
    chart: {
      type: "column",
      marginRight: 380,
    },

    title: {
      text: "GHG Emissions (off-land) by Commodity",
    },

    credits: {
      enabled: false,
    },

    series: JSON.parse(
      document.getElementById("GHG_split_off_land_3_Commodity")
        .innerHTML
    ).AUSTRALIA,

    yAxis: {
      title: {
        text: "Greenhouse Gas (Mt CO2e)",
      },
    },
    xAxis: {
      tickPositions: year_ticks,
    },

    legend: {
      align: "right",
      layout: "vertical",
      x: -150,
      verticalAlign: "middle",
    },

    tooltip: {
      formatter: function () {
        return `<b>Year:</b> ${this.x}<br><b>${
          this.series.name
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

  Highcharts.chart("GHG_split_off_land_2_Emission_Source_chart", {
    chart: {
      type: "column",
      marginRight: 380,
    },

    title: {
      text: "GHG Emissions (off-land) by Source",
    },

    credits: {
      enabled: false,
    },

    series: JSON.parse(
      document.getElementById("GHG_split_off_land_2_Emission_Source")
        .innerHTML
    ).AUSTRALIA,
    xAxis: {
      tickPositions: year_ticks,
    },
    yAxis: {
      title: {
        text: "Greenhouse Gas (Mt CO2e)",
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
        return `<b>Year:</b> ${this.x}<br><b>${
          this.series.name
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

      exporting: {
        sourceWidth: 1200,
        sourceHeight: 600,
      },
    },
  });

  Highcharts.chart("GHG_split_off_land_1_Emission_Type_chart", {
    chart: {
      type: "column",
      marginRight: 380,
    },

    title: {
      text: "GHG Emissions (off-land) by Type",
    },

    credits: {
      enabled: false,
    },

    series: JSON.parse(
      document.getElementById("GHG_split_off_land_1_Emission_Type")
        .innerHTML
    ).AUSTRALIA,
    xAxis: {
      tickPositions: year_ticks,
    },
    yAxis: {
      title: {
        text: "Greenhouse Gas (Mt CO2e)",
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
        return `<b>Year:</b> ${this.x}<br><b>${
          this.series.name
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

      exporting: {
        sourceWidth: 1200,
        sourceHeight: 600,
      },
    },
  });

  // Chart:GHG_split_NonAg_1_Land-use
  Highcharts.chart("GHG_split_NonAg_1_Land-use_chart", {
    chart: {
      type: "column",
      marginRight: 380,
    },

    title: {
      text: "GHG Emissions Abatement by Non-agricultural Land-use Type",
    },

    credits: {
      enabled: false,
    },

    series: JSON.parse(
      document.getElementById("GHG_split_NonAg_1_Land-use")
        .innerHTML
    ).AUSTRALIA,

    yAxis: {
      title: {
        text: "Greenhouse Gas (Mt CO2e)",
      },
    },
    xAxis: {
      tickPositions: year_ticks,
    },

    legend: {
      align: "right",
      layout: "vertical",
      x: 0,
      verticalAlign: "middle",
    },

    tooltip: {
      formatter: function () {
        return `<b>Year:</b> ${this.x}<br><b>${
          this.series.name
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

  // Chart:GHG_split_Am_3_Agricultural_Management_Type
  Highcharts.chart("GHG_split_Am_3_Agricultural_Management_Type_chart", {
    chart: {
      type: "column",
      marginRight: 380,
    },

    title: {
      text: "GHG Emissions Abatement by Agricultural Management Type",
    },

    credits: {
      enabled: false,
    },

    series: JSON.parse(
      document.getElementById("GHG_split_Am_3_Agricultural_Management_Type").innerHTML
    ).AUSTRALIA,

    yAxis: {
      title: {
        text: "Greenhouse Gas (Mt CO2e)",
      },
    },
    xAxis: {
      tickPositions: year_ticks,
    },

    legend: {
      align: "right",
      layout: "vertical",
      x: -100,
      verticalAlign: "middle",
    },

    tooltip: {
      formatter: function () {
        return `<b>Year:</b> ${this.x}<br><b>${
          this.series.name
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

  // Chart:GHG_split_Am_1_Land-use
  Highcharts.chart("GHG_split_Am_1_Land-use_chart", {
    chart: {
      type: "column",
      marginRight: 380,
    },

    title: {
      text: "GHG Emissions Abatement by Crop/Livestock",
    },

    credits: {
      enabled: false,
    },

    series: JSON.parse(
      document.getElementById(
        "GHG_split_Am_1_Land-use"
      ).innerHTML
    ).AUSTRALIA,

    yAxis: {
      title: {
        text: "Greenhouse Gas (Mt CO2e)",
      },
    },
    xAxis: {
      tickPositions: year_ticks,
    },

    legend: {
      align: "right",
      layout: "vertical",
      x: -150,
      verticalAlign: "middle",
    },

    tooltip: {
      formatter: function () {
        return `<b>Year:</b> ${this.x}<br><b>${
          this.series.name
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

  // Chart:GHG_split_Am_4_Water_supply
  Highcharts.chart("GHG_split_Am_4_Water_supply_chart", {
    chart: {
      type: "column",
      marginRight: 380,
    },

    title: {
      text: "GHG Emissions Abatement by Irrigation Type",
    },

    credits: {
      enabled: false,
    },

    series: JSON.parse(
      document.getElementById("GHG_split_Am_4_Water_supply")
        .innerHTML
    ).AUSTRALIA,

    yAxis: {
      title: {
        text: "Greenhouse Gas (Mt CO2e)",
      },
    },
    xAxis: {
      tickPositions: year_ticks,
    },

    legend: {
      align: "right",
      layout: "vertical",
      x: -180,
      verticalAlign: "middle",
    },

    tooltip: {
      formatter: function () {
        return `<b>Year:</b> ${this.x}<br><b>${
          this.series.name
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
