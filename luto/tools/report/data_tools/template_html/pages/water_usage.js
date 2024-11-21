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

  // Global options
  Highcharts.setOptions({
    title: {
      align: 'left' // Align the title to the left
    },
  });

  // Chart:water_1_water_net_use_by_broader_category
  Highcharts.chart("water_1_water_net_use_by_broader_category", {
    chart: {
      marginRight: 380,
    },

    plotOptions: {
      column: {
        stacking: "normal",
        dataLabels: {
          enabled: false,
        }
      },
    },

    title: {
      text: "Water Net Yield by Broader Land-use and Management Type",
    },

    credits: {
      enabled: false,
    },

    series: JSON.parse(
      document.getElementById("water_1_water_net_use_by_broader_category_csv").innerHTML
    ),
    xAxis: {
      tickPositions: year_ticks,
    },
    yAxis: {
      title: {
        text: "Water net yield (ML)",
      },
    },

    legend: {
      align: "right",
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

    exporting: {
      sourceWidth: 1200,
      sourceHeight: 600,
    },
  });


  // Chart:water_2_water_net_yield_by_specific_landuse
  Highcharts.chart("water_2_water_net_yield_by_specific_landuse", {
    chart: {
      type: "column",
      marginRight: 380,
    },

    plotOptions: {
      column: {
        stacking: "normal",
        dataLabels: {
          enabled: false,
        },
      },
      spline: {
        showInLegend: false
      },
    },

    title: {
      text: "Water Net Yield by Broader Land-use and Management Type",
    },

    credits: {
      enabled: false,
    },

    series: JSON.parse(
      document.getElementById("water_2_water_net_yield_by_specific_landuse_csv").innerHTML
    ),
    xAxis: {
      tickPositions: year_ticks,
    },
    yAxis: {
      title: {
        text: "Water net yield (ML)",
      },
    },

    legend: {
      itemStyle: {
        fontSize: "11px",
      },
      align: "right",
      layout: "vertical",
      x: -30,
      y: -10,
      verticalAlign: "middle",
      itemMarginTop: 0,
      itemMarginBottom: 1,
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

  // Chart:water_3_water_net_yield_by_region
  const chartContainer = document.getElementById('water_3_water_net_yield_by_region');
  chartData = JSON.parse(document.getElementById("water_3_water_net_yield_by_region_csv").innerHTML);

  // Create blocks and render Highcharts in each block
  chartData.forEach((chart, index) => {
    // Create a new div for each chart
    const chartBlock = document.createElement('div');
    chartBlock.classList.add('chart-block');
    chartBlock.id = `chart-${index + 1}`;
    chartContainer.appendChild(chartBlock);

    Highcharts.chart(chartBlock.id, {

      plotOptions: {
        showInLegend: false,
        column: {
          stacking: "normal",
          dataLabels: {
            enabled: false,
          },
          tooltip: {
            formatter: function () {
              return `<b>Year:</b> ${this.x}<br><b>${this.series.name
                }:</b>${this.y.toFixed(2)}<br/>`;
            },
          },
        },
        arearange: {
          lineWidth: 0,
          marker: {
            enabled: false,
          },
          tooltip: {
            enabled: false, // Disable tooltip for arearange series
          },
          allowPointSelect: false, // Disable selection for arearange points
          states: {
            inactive: {
              opacity: 1 // Prevent any dimming effect when another series is selected
            }
          },
          enableMouseTracking: false, // Disable mouse tracking events for arearange
          zIndex: 0, // Set the zIndex to 0 to prevent the arearange series from covering other series
        },
        spline: {
          marker : {
            enabled: false,
          },
        },
      },

      title: {
        text: chart.name,
        align: 'center',
      },

      credits: {
        enabled: false,
      },

      series: chart.data,

      xAxis: {
        tickPositions: year_ticks,
      },
      yAxis: {
        title: {
          text: "Water net yield (ML)",
        },
      },

      legend: {
        itemStyle: {
          fontSize: "10px",
        }
      },

      exporting: {
        sourceWidth: 1200,
        sourceHeight: 600,
      },
    });
  });

});
