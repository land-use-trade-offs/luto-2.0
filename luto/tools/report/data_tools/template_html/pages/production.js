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

  // Get the years from the csv
  var tickposition;
  $(document).ready(function () {
    let csv, csv_lines;
    let years_all = [];

    csv = document.getElementById(
      "production_1_demand_type_wide_csv"
    ).innerHTML;
    csv_lines = csv.split("\n");

    // if the last line is empty, remove it
    if (csv_lines[csv_lines.length - 1] == "") {
      csv_lines.pop();
    }

    $.each(csv_lines, function (lineNo, line) {
      var items = line.split(",");

      if (lineNo != 0) {
        // Skip the first line (headers)
        years_all.push(parseFloat(items[0]));
      }
    });

    // if the length of the years_all is greater than 5, then set the tickposition = bull
    if (years_all.length < 5) {
      tickposition = years_all;
    } else {
      tickposition = null;
    }

    // Chart:production_1_demand_type_wide
    Highcharts.chart("production_1_demand_type_wide", {
      chart: {
        type: "column",
        marginRight: 200,
      },
      title: {
        text: "Demand, Trade, and Production of Agricultural Commodities",
      },
      data: {
        csv: document.getElementById("production_1_demand_type_wide_csv")
          .innerHTML,
      },
      credits: {
        enabled: false,
      },
      xAxis: {
        tickPositions: tickposition,
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
    let production_2_demand_on_off_wide_option = {
      chart: {
        renderTo: "production_2_demand_on_off_wide",
        marginRight: 200,
        type: "column",
      },
      title: {
        text: "Demand for Agricultural Commodities",
      },

      xAxis: {
        tickWidth: 0.05,
        categories: [],
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

      series: [],

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
    };

    // Extract data to populate chart
    $(document).ready(function () {
      let data, lines, years;

      data = document.getElementById(
        "production_2_demand_on_off_wide_csv"
      ).innerHTML;

      // Get the years and types
      lines = data.split("\n");
      years = lines[0].split(",").slice(1);
      years = [...new Set(years)];
      types = lines[1].split(",").slice(1);
      types = [...new Set(types)];

      years.forEach((year) => {
        production_2_demand_on_off_wide_option.xAxis.categories.push({
          name: year,
          categories: types,
        });
      });

      // Populate the chart options
      $.each(lines, function (lineNo, line) {
        var items = line.split(",");

        if (lineNo <= 1) {
          // Skip the first two lines (headers)
        } else {
          // if items is not empty, add series
          if (items[0] == "") {
            // Skip empty lines
          } else {
            // Add series
            production_2_demand_on_off_wide_option.series.push({
              name: items[0],
              data: items.slice(1).map((x) => parseFloat(x)),
              type: "column",
            });
          }
        }
      });

      let chart = new Highcharts.Chart(production_2_demand_on_off_wide_option);
    });

    // Chart:production_3_demand_commodity
    let production_3_demand_commodity_option = {
      chart: {
        renderTo: "production_3_demand_commodity",
        marginRight: 200,
        type: "column",
      },
      title: {
        text: "Agricultural Demand by Commodity",
      },

      xAxis: {
        tickWidth: 0.05,
        categories: [],
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

      series: [],

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
    };

    // Extract data to populate chart
    $(document).ready(function () {
      let data, lines, years;

      data = document.getElementById(
        "production_3_demand_commodity_csv"
      ).innerHTML;

      // Get the years and types
      lines = data.split("\n");
      years = lines[0].split(",").slice(1);
      years = [...new Set(years)];
      types = lines[1].split(",").slice(1);
      types = [...new Set(types)];

      years.forEach((year) => {
        production_3_demand_commodity_option.xAxis.categories.push({
          name: year,
          categories: types,
        });
      });

      // Populate the chart options
      $.each(lines, function (lineNo, line) {
        var items = line.split(",");

        if (lineNo <= 1) {
          // Skip the first two lines (headers)
        } else {
          // if items is not empty, add series
          if (items[0] == "") {
            // Skip empty lines
          } else {
            // Add series
            production_3_demand_commodity_option.series.push({
              name: items[0],
              data: items.slice(1).map((x) => parseFloat(x)),
              type: "column",
            });
          }
        }
      });

      let chart = new Highcharts.Chart(production_3_demand_commodity_option);
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
      data: {
        csv: document.getElementById(
          "production_4_1_demand_domestic_On-land_commodity_csv"
        ).innerHTML,
      },
      credits: {
        enabled: false,
      },
      xAxis: {
        tickPositions: tickposition,
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
      data: {
        csv: document.getElementById(
          "production_4_2_demand_domestic_Off-land_commodity_csv"
        ).innerHTML,
      },
      credits: {
        enabled: false,
      },
      xAxis: {
        tickPositions: tickposition,
      },
      xAxis: {
        tickPositions: tickposition,
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
        y: 200,
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

      data: {
        csv: document.getElementById(
          "production_5_6_demand_Production_commodity_from_LUTO_csv"
        ).innerHTML,
      },

      credits: {
        enabled: false,
      },
      xAxis: {
        tickPositions: tickposition,
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
});
