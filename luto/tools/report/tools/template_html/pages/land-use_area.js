document.addEventListener("DOMContentLoaded", function () {
  Highcharts.setOptions({
    colors: [
      "#8085e9",
      "#f15c80",
      "#e4d354",
      "#2b908f",
      "#f45b5b",
      "#7cb5ec",
      "#434348",
      "#90ed7d",
      "#f7a35c",
      "#91e8e1",
    ],
  });
  
  // Get the years from the csv
  var tickposition;
  $(document).ready(function () {
    let csv, csv_lines;
    let years = [];

    csv = document.getElementById("area_1_total_area_wide_csv").innerHTML;

    csv_lines = csv.split("\n");

    // if the last line is empty, remove it
    if (csv_lines[csv_lines.length - 1] == "") {
      csv_lines.pop();
    }

    $.each(csv_lines, function (lineNo, line) {
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

    // Chart:area_1_total_area_wide
    Highcharts.chart("area_1_total_area_wide", {
      chart: {
        type: "column",
        marginRight: 200,
      },
      title: {
        text: "Total Area by Land-use and Agricultural Commodity",
      },
      series:JSON.parse(
        document.getElementById("area_1_total_area_wide_csv").innerHTML
      ),
      credits: {
        enabled: false,
      },
      yAxis: {
        title: {
          text: "Area (million km2)",
        },
      },

      legend: {
        align: "right",
        verticalAlign: "top",
        layout: "vertical",
        x: 5,
        y: -15,
        itemStyle: {
          fontSize: '11px'
      }
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

    // Chart:area_2_irrigation_area_wide
    Highcharts.chart("area_2_irrigation_area_wide", {
      chart: {
        type: "column",
        marginRight: 200,
      },
      title: {
        text: "Total Area by Irrigation Type",
      },
      series:JSON.parse(
        document.getElementById("area_2_irrigation_area_wide_csv").innerHTML
      ),
      credits: {
        enabled: false,
      },
      yAxis: {
        title: {
          text: "Area (million km2)",
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

    // area_3_non_ag_lu_area_wide
    Highcharts.chart("area_3_non_ag_lu_area_wide", {
      chart: {
        type: "column",
        marginRight: 200,
      },
      title: {
        text: "Non-Agricultural Land-Use Area",
      },
      series:JSON.parse(
        document.getElementById("area_3_non_ag_lu_area_wide_csv").innerHTML,
      ),
      credits: {
        enabled: false,
      },
      yAxis: {
        title: {
          text: "Area (million km2)",
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
        },
      },

      exporting: {
        sourceWidth: 1200,
        sourceHeight: 600,
      },
    });

    // area_4_am_total_area_wide
    Highcharts.chart("area_4_am_total_area_wide", {
      chart: {
        type: "column",
        marginRight: 200,
      },
      title: {
        text: "Agricultural Management Area by Type",
      },
      series:JSON.parse(
        document.getElementById("area_4_am_total_area_wide_csv").innerHTML
      ),
      credits: {
        enabled: false,
      },
      yAxis: {
        title: {
          text: "Area (million km2)",
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
        },
      },

      exporting: {
        sourceWidth: 1200,
        sourceHeight: 600,
      },
    });

    // area_5_am_lu_area_wide
    Highcharts.chart("area_5_am_lu_area_wide", {
      chart: {
        type: "column",
        marginRight: 200,
      },
      title: {
        text: "Agricultural Management Area by Land-use Type",
      },
      series: 
        JSON.parse(document.getElementById("area_5_am_lu_area_wide_csv").innerHTML)
      ,
      credits: {
        enabled: false,
      },
      yAxis: {
        title: {
          text: "Area (million km2)",
        },
      },

      legend: {
        align: "right",
        verticalAlign: "top",
        layout: "vertical",
        x: 10,
        y: 0,
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

    // area_6_begin_end_area
    document.getElementById("area_6_begin_end_area").innerHTML = document.getElementById(
      "area_6_begin_end_area_csv"
    ).innerText;

    // area_7_begin_end_pct
    document.getElementById("area_7_begin_end_pct").innerHTML = document.getElementById(
      "area_7_begin_end_pct_csv"
    ).innerText;
  });
});
