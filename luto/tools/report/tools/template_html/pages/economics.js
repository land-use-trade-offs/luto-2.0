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
    let data, lines;
    let years = [];

    data = document.getElementById(
      "economics_1_revenue_1_Irrigation_wide_csv"
    ).innerHTML;
    lines = data.split("\n");

    // if the last line is empty, remove it
    if (lines[lines.length - 1] == "") {
      lines.pop();
    }

    $.each(lines, function (lineNo, line) {
      var items = line.split(",");

      if (lineNo != 0) {
        // Skip the first line (headers)
        years.push(parseFloat(items[0]));
      }
    });

    // if the length of the years is greater than 5, then set the tickposition = bull
    if (years.length < 5) {
      tickposition = years;
    } else {
      tickposition = null;
    }

    // Chart:economics_1_revenue_1_Irrigation_wide
    Highcharts.chart("economics_1_revenue_1_Irrigation_wide", {
      chart: {
        type: "column",
        marginRight: 180,
      },

      title: {
        text: "Revenue by Irrigation Status",
      },

      credits: {
        enabled: false,
      },

      data: {
        csv: document.getElementById(
          "economics_1_revenue_1_Irrigation_wide_csv"
        ).innerHTML,
      },

      xAxis: {
        tickPositions: tickposition,
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
        x: -100,
        y: 300,
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
        },
      },

      exporting: {
        sourceWidth: 1200,
        sourceHeight: 600,
      },
    });

    // Chart:economics_1_revenue_2_Source_wide
    Highcharts.chart("economics_1_revenue_2_Source_wide", {
      chart: {
        type: "column",
        marginRight: 180,
      },

      title: {
        text: "Revenue by Agricultural Product",
      },

      credits: {
        enabled: false,
      },

      data: {
        csv: document.getElementById("economics_1_revenue_2_Source_wide_csv")
          .innerHTML,
      },

      yAxis: {
        title: {
          text: "Revenue (billion AU$)",
        },
      },

      xAxis: {
        tickPositions: tickposition,
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
          return `<b>Year:</b> ${this.x}<br><b>${
            this.series.name
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

    // Chart:economics_1_revenue_3_Source_type_wide
    Highcharts.chart("economics_1_revenue_3_Source_type_wide", {
      chart: {
        type: "column",
        marginRight: 180,
      },

      title: {
        text: "Revenue by Agricultural Commodity",
      },

      credits: {
        enabled: false,
      },

      data: {
        csv: document.getElementById(
          "economics_1_revenue_3_Source_type_wide_csv"
        ).innerHTML,
      },

      yAxis: {
        title: {
          text: "Revenue (billion AU$)",
        },
      },

      xAxis: {
        tickPositions: tickposition,
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
          return `<b>Year:</b> ${this.x}<br><b>${
            this.series.name
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

    // Chart:economics_1_revenue_4_Type_wide
    Highcharts.chart("economics_1_revenue_4_Type_wide", {
      chart: {
        type: "column",
        marginRight: 180,
      },

      title: {
        text: "Revenue by Commodity Type",
      },

      credits: {
        enabled: false,
      },

      data: {
        csv: document.getElementById("economics_1_revenue_4_Type_wide_csv")
          .innerHTML,
      },
      xAxis: {
        tickPositions: tickposition,
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
          return `<b>Year:</b> ${this.x}<br><b>${
            this.series.name
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

    // Chart:economics_1_revenue_5_crop_lvstk_wide
    Highcharts.chart("economics_1_revenue_5_crop_lvstk_wide", {
      chart: {
        type: "column",
        marginRight: 180,
      },

      title: {
        text: "Revenue by Crop/Livestock",
      },

      data: {
        csv: document.getElementById(
          "economics_1_revenue_5_crop_lvstk_wide_csv"
        ).innerHTML,
      },

      credits: {
        enabled: false,
      },

      yAxis: {
        title: {
          text: "Revenue (billion AU$)",
        },
      },
      xAxis: {
        tickPositions: tickposition,
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
          return `<b>Year:</b> ${this.x}<br><b>${
            this.series.name
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

    // Chart:economics_2_cost_1_Irrigation_wide
    Highcharts.chart("economics_2_cost_1_Irrigation_wide", {
      chart: {
        type: "column",
        marginRight: 180,
      },

      title: {
        text: "Cost of Production by Irrigation Status",
      },

      credits: {
        enabled: false,
      },

      data: {
        csv: document.getElementById("economics_2_cost_1_Irrigation_wide_csv")
          .innerHTML,
      },

      yAxis: {
        title: {
          text: "Cost (billion AU$)",
        },
      },
      xAxis: {
        tickPositions: tickposition,
      },

      legend: {
        align: "right",
        verticalAlign: "left",
        layout: "vertical",
        x: -100,
        y: 300,
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
        },
      },

      exporting: {
        sourceWidth: 1200,
        sourceHeight: 600,
      },
    });

    // Chart:economics_2_cost_2_Source_wide
    Highcharts.chart("economics_2_cost_2_Source_wide", {
      chart: {
        type: "column",
        marginRight: 180,
      },

      title: {
        text: "Cost of Production by Agricultural Product",
      },

      credits: {
        enabled: false,
      },

      data: {
        csv: document.getElementById("economics_2_cost_2_Source_wide_csv")
          .innerHTML,
      },

      yAxis: {
        title: {
          text: "Cost (billion AU$)",
        },
      },
      xAxis: {
        tickPositions: tickposition,
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
          return `<b>Year:</b> ${this.x}<br><b>${
            this.series.name
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

    // // Chart:economics_2_cost_3_Source_type_wide
    // Highcharts.chart('economics_2_cost_3_Source_type_wide', {

    //     chart: {
    //         type: 'column',
    //         marginRight: 180
    //     },

    //     title: {
    //         text: 'Cost of Production by Commodity'
    //     },

    //     credits: {
    //         enabled: false
    //     },

    //     data: {
    //         csv: document.getElementById('economics_2_cost_3_Source_type_wide_csv').innerHTML,
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

    // Chart:economics_2_cost_4_Type_wide
    Highcharts.chart("economics_2_cost_4_Type_wide", {
      chart: {
        type: "column",
        marginRight: 180,
      },

      title: {
        text: "Cost of Production by Commodity Type",
      },

      credits: {
        enabled: false,
      },

      data: {
        csv: document.getElementById("economics_2_cost_4_Type_wide_csv")
          .innerHTML,
      },

      yAxis: {
        title: {
          text: "Cost (billion AU$)",
        },
      },
      xAxis: {
        tickPositions: tickposition,
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
          return `<b>Year:</b> ${this.x}<br><b>${
            this.series.name
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

    // Chart:economics_2_cost_5_crop_lvstk_wide
    Highcharts.chart("economics_2_cost_5_crop_lvstk_wide", {
      chart: {
        type: "column",
        marginRight: 180,
      },

      title: {
        text: "Cost of Production by Crop/Livestock",
      },

      data: {
        csv: document.getElementById("economics_2_cost_5_crop_lvstk_wide_csv")
          .innerHTML,
      },

      credits: {
        enabled: false,
      },

      yAxis: {
        title: {
          text: "Cost (billion AU$)",
        },
      },

      xAxis: {
        tickPositions: tickposition,
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
          return `<b>Year:</b> ${this.x}<br><b>${
            this.series.name
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

    let economics_3_rev_cost_all_option = {
      chart: {
        type: "columnrange",
        renderTo: "economics_3_rev_cost_all",
        marginRight: 180,
      },

      title: {
        text: "Agricultural Revenue and Cost of Production",
      },

      credits: {
        enabled: false,
      },

      xAxis: {
        categories: [],
        tickPositions: tickposition,
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
            Highcharts.numberFormat(this.point.low, 2) +
            " - " +
            Highcharts.numberFormat(this.point.high, 2) +
            " (billion AU$)"
          );
        },
      },

      plotOptions: {
        columnrange: {
          borderRadius: "50%",
        },
      },

      series: [
        { name: "Revenue", data: [] },
        { name: "Cost", data: [] },
      ],

      legend: {
        align: "right",
        verticalAlign: "left",
        layout: "vertical",
        x: -50,
        y: 280,
      },
    };

    $(document).ready(function () {
      let data;
      data = document.getElementById("economics_3_rev_cost_all_csv").innerHTML;

      var lines = data.split("\n");

      // if the last line is empty, remove it
      if (lines[lines.length - 1] == "") {
        lines.pop();
      }

      $.each(lines, function (lineNo, line) {
        var items = line.split(",");

        if (lineNo != 0) {
          // Skip the first line (headers)
          economics_3_rev_cost_all_option.xAxis.categories.push(
            parseFloat(items[0])
          );
          economics_3_rev_cost_all_option.series[0].data.push([
            0,
            parseFloat(items[1]),
          ]); // Revenue
          economics_3_rev_cost_all_option.series[1].data.push([
            parseFloat(items[3]),
            parseFloat(items[1]),
          ]); // Cost
        }
      });

      // Create the chart with the correct options
      let chart = new Highcharts.Chart(economics_3_rev_cost_all_option);
    });
  });
});
