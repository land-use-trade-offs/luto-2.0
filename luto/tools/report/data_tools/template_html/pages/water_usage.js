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

  // Chart:Water_overview_AUSTRALIA
  Highcharts.chart("Water_overview_AUSTRALIA_chart", {
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
      text: "Water Net Yield Overview",
    },

    credits: {
      enabled: false,
    },

    series: JSON.parse(
      document.getElementById("Water_overview_AUSTRALIA").innerHTML
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


  // Chart:Water_overview_landuse
  Highcharts.chart("Water_overview_landuse_chart", {
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
      text: "Water Net Yield by Land Use",
    },

    credits: {
      enabled: false,
    },

    series: JSON.parse(
      document.getElementById("Water_overview_landuse").innerHTML
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
        fontSize: "10px",
      },
      align: "right",
      layout: "vertical",
      x: -30,
      y: -10,
      verticalAlign: "middle",
      itemMarginTop: 0,
      itemMarginBottom: 0.5,
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

  // Chart:Water_overview_by_watershed_region
  const chartContainer = document.getElementById('Water_overview_by_watershed_region_chart');
  const chartData = JSON.parse(document.getElementById("Water_overview_by_watershed_region").innerHTML);

  // Create blocks and render Highcharts in each block
  Object.entries(chartData).forEach(([region, series], index) => {
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
          marker: {
            enabled: false,
          },
        },
      },

      title: {
        text: region,
        align: 'center',
      },

      credits: {
        enabled: false,
      },

      series: series,

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
