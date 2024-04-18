// make charts
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


  // Chart:GHG_1_cunsum_emission_Mt
  Highcharts.chart("GHG_1_cunsum_emission_Mt", {
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
      document.getElementById("GHG_1_cunsum_emission_Mt_csv").innerHTML
    ),

    yAxis: {
      title: {
        text: "Greenhouse Gas (Mt CO2e)",
      },
    },
    xAxis: {
      tickPositions: year_ticks,
    },

    legend: {
      enabled: false,
      // align: 'right',
      // verticalAlign: 'top',
      // layout: 'vertical',
      // x: 0,
      // y: 50
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

  // Chart:GHG_2_individual_emission_Mt
  Highcharts.chart("GHG_2_individual_emission_Mt", {
    chart: {
      marginRight: 380,
    },
    title: {
      text: "GHG Emissions by Land-use/Management Type",
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
        return `<b>Year:</b> ${this.x}<br><b>${this.series.name
          }:</b>${this.y.toFixed(2)}<br/>`;
      },
    },
    series: JSON.parse(
      document.getElementById("GHG_2_individual_emission_Mt_csv").innerHTML
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



  // Chart:GHG_4_3_1_crop_lvstk_emission_Mt
  Highcharts.chart("GHG_4_3_1_crop_lvstk_emission_Mt", {
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
      document.getElementById("GHG_4_3_1_crop_lvstk_emission_Mt_csv").innerHTML
    ),

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

  // Chart:GHG_4_3_2_dry_irr_emission_Mt
  Highcharts.chart("GHG_4_3_2_dry_irr_emission_Mt", {
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
      document.getElementById("GHG_4_3_2_dry_irr_emission_Mt_csv").innerHTML
    ),

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

  // Chart:GHG_4_3_3_category_emission_Mt
  Highcharts.chart("GHG_4_3_3_category_emission_Mt", {
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
      document.getElementById("GHG_4_3_3_category_emission_Mt_csv").innerHTML
    ),

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

  // Chart:GHG_4_3_4_sources_emission_Mt
  Highcharts.chart("GHG_4_3_4_sources_emission_Mt", {
    chart: {
      type: "column",
      marginRight: 380,
      marginBottom: 200,
    },

    title: {
      text: "GHG Emissions (on-land) by Source",
    },

    credits: {
      enabled: false,
    },

    series: JSON.parse(
      document.getElementById("GHG_4_3_4_sources_emission_Mt_csv").innerHTML
    ),

    yAxis: {
      title: {
        text: "Emissions (Mt CO2e)",
      },
    },
    xAxis: {
      tickPositions: year_ticks,
    },

    legend: {
        align: 'right',
        verticalAlign: 'middle',
        layout: 'vertical',
        x: 0,
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

  // Chart:GHG_4_3_5_lu_lm_emission_Mt_wide
  Highcharts.chart("GHG_4_3_5_lu_lm_emission_Mt_wide", {
    chart: {
      renderTo: "GHG_4_3_5_lu_lm_emission_Mt_wide",
      marginRight: 380,
    },
    title: {
      text: "GHG Emissions (on-land) - Start and End Year",
    },

    xAxis: {
      tickWidth: 0.05,

      categories: JSON.parse(
        document.getElementById("GHG_4_3_5_lu_lm_emission_Mt_wide_csv").innerHTML
      )['categories'],

      labels: {
        y:10,
        rotation: -90,
        align: "right",

    },

    },

    series: JSON.parse(
      document.getElementById("GHG_4_3_5_lu_lm_emission_Mt_wide_csv").innerHTML
    )['series'],

    yAxis: {
      title: {
        text: "Greenhouse Gas (t CO2e/ha)",
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

  // Chart:GHG_4_3_6_lu_source_emission_Mt
  Highcharts.chart("GHG_4_3_6_lu_source_emission_Mt", {
    chart: {
      type: "packedbubble",
    },
    title: {
      text: "GHG Emissions (on-land) in the target year",
    },
    tooltip: {
      useHTML: true,
      pointFormat: "<b>{point.name}:</b> {point.value} Mt CO<sub>2</sub>",
    },
    plotOptions: {
      packedbubble: {
        useSimulation: true,
        splitSeries: false,
        minSize: "1%",
        maxSize: "1000%",
        dataLabels: {
          enabled: true,
          format: "{point.name}",
          filter: {
            property: "y",
            operator: ">",
            value: 1,
          },
        },
      },
    },

    series: JSON.parse($("#GHG_4_3_6_lu_source_emission_Mt_csv").html()),

    credits: {
      enabled: false,
    },
    tooltip: {
      formatter: function () {
        return `<b>${this.series.name
          }:</b>${this.y.toFixed(3)}<br/>`;
      },
    },
  
    legend: {
      align: "right",
      layout: "vertical",
      x: 0,
      verticalAlign: "middle",
    },
    exporting: {
      sourceWidth: 1200,
      sourceHeight: 600,
    },
  });

  Highcharts.chart("GHG_4_3_7_off_land_commodity_emission_Mt", {
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
      document.getElementById("GHG_4_3_7_off_land_commodity_emission_Mt_csv")
        .innerHTML
    ),

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

  Highcharts.chart("GHG_4_3_8_off_land_sources_emission_Mt", {
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
      document.getElementById("GHG_4_3_8_off_land_sources_emission_Mt_csv")
        .innerHTML
    ),

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

      exporting: {
        sourceWidth: 1200,
        sourceHeight: 600,
      },
    },
  });


  Highcharts.chart("GHG_4_3_9_off_land_type_emission_Mt", {
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
      document.getElementById("GHG_4_3_9_off_land_type_emission_Mt_csv")
        .innerHTML
    ),

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

      exporting: {
        sourceWidth: 1200,
        sourceHeight: 600,
      },
    },
  });




  // Chart:GHG_4_4_ag_reduction_source_wide_Mt
  Highcharts.chart("GHG_4_4_ag_reduction_source_wide_Mt", {
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
      document.getElementById("GHG_4_4_ag_reduction_source_wide_Mt_csv").innerHTML

    ),

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

  // Chart:GHG_4_5_1_GHG_ag_man_df_wide_Mt
  Highcharts.chart("GHG_4_5_1_GHG_ag_man_df_wide_Mt", {
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
      document.getElementById("GHG_4_5_1_GHG_ag_man_df_wide_Mt_csv").innerHTML
    ),

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

  // Chart:GHG_4_5_2_GHG_ag_man_GHG_crop_lvstk_df_wide_Mt
  Highcharts.chart("GHG_4_5_2_GHG_ag_man_GHG_crop_lvstk_df_wide_Mt", {
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
      document.getElementById("GHG_4_5_2_GHG_ag_man_GHG_crop_lvstk_df_wide_Mt_csv").innerHTML
    ),

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

  // Chart:GHG_4_5_3_GHG_ag_man_dry_irr_df_wide_Mt
  Highcharts.chart("GHG_4_5_3_GHG_ag_man_dry_irr_df_wide_Mt", {
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
      document.getElementById("GHG_4_5_3_GHG_ag_man_dry_irr_df_wide_Mt_csv").innerHTML
    ),

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
