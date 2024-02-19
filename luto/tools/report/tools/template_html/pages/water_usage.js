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
      "water_1_percent_to_limit_csv"
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

    // Chart:water_1_percent_to_limit
    Highcharts.chart("water_1_percent_to_limit", {
      chart: {
        type: "spline",
        marginRight: 200,
      },

      title: {
        text: "Water Use as Percentage of Sustainable Limit",
      },

      credits: {
        enabled: false,
      },

      data: {
        csv: document.getElementById("water_1_percent_to_limit_csv").innerHTML,
      },
      yAxis: {
        title: {
          text: "Percentage to Limit (%)",
        },
      },

      legend: {
        itemStyle: {
          fontSize: "11px",
          textOverflow: "ellipsis",
        },
        align: "right",
        verticalAlign: "top",
        layout: "vertical",
        x: 5,
        y: 180,
      },

      tooltip: {
        formatter: function () {
          return `<b>Year:</b> ${this.x}<br><b>${
            this.series.name
          }:</b>${this.y.toFixed(2)}<br/>`;
        },
      },

      exporting: {
        sourceWidth: 1200,
        sourceHeight: 600,
      },
    });

    // Chart:water_2_volum_to_limit
    Highcharts.chart("water_2_volum_to_limit", {
      chart: {
        type: "spline",
        marginRight: 200,
      },

      title: {
        text: "Water Use by Drainage Division/River Region",
      },

      credits: {
        enabled: false,
      },

      data: {
        csv: document.getElementById("water_2_volum_to_limit_csv").innerHTML,
      },
      yAxis: {
        title: {
          text: "Water Use (ML)",
        },
      },

      legend: {
        itemStyle: {
          fontSize: "10px",
          textOverflow: "ellipsis",
        },
        align: "right",
        verticalAlign: "top",
        layout: "vertical",
        x: -10,
        y: 180,
      },

      tooltip: {
        formatter: function () {
          return `<b>Year:</b> ${this.x}<br><b>${
            this.series.name
          }:</b>${this.y.toFixed(2)}<br/>`;
        },
      },

      exporting: {
        sourceWidth: 1200,
        sourceHeight: 600,
      },
    });

    // Chart:water_3_volum_by_sector
    Highcharts.chart("water_3_volum_by_sector", {
      chart: {
        type: "column",
        marginRight: 200,
      },

      title: {
        text: "Water Use by Broad Land-use and Management Type",
      },

      credits: {
        enabled: false,
      },

      data: {
        csv: document.getElementById("water_3_volum_by_sector_csv").innerHTML,
      },
      xAxis: {
        tickPositions: tickposition,
      },
      yAxis: {
        title: {
          text: "Water Use (ML)",
        },
      },

      legend: {
        align: "right",
        verticalAlign: "top",
        layout: "vertical",
        x: 10,
        y: 250,
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

    // Chart:water_4_volum_by_landuse
    Highcharts.chart("water_4_volum_by_landuse", {
      chart: {
        type: "column",
        marginRight: 200,
      },

      title: {
        text: "Water Use by Land-use and Agricultural Commodity",
      },

      credits: {
        enabled: false,
      },

      data: {
        csv: document.getElementById("water_4_volum_by_landuse_csv").innerHTML,
      },
      xAxis: {
        tickPositions: tickposition,
      },
      yAxis: {
        title: {
          text: "Water Use (ML)",
        },
      },

      legend: {
        itemStyle: {
          fontSize: "11px",
          textOverflow: "ellipsis",
        },
        align: "right",
        verticalAlign: "top",
        layout: "vertical",
        x: 5,
        y: -10,
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

    // Chart:water_5_volum_by_irrigation
    Highcharts.chart("water_5_volum_by_irrigation", {
      chart: {
        type: "column",
        marginRight: 200,
      },

      title: {
        text: "Water Use by Irrigation Type",
      },

      credits: {
        enabled: false,
      },

      data: {
        csv: document.getElementById("water_5_volum_by_irrigation_csv")
          .innerHTML,
      },
      xAxis: {
        tickPositions: tickposition,
      },
      yAxis: {
        title: {
          text: "Water Use (ML)",
        },
      },

      legend: {
        align: "right",
        verticalAlign: "top",
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
});
