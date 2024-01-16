document.addEventListener('DOMContentLoaded', function() {
    Highcharts.setOptions({
        colors: [
            '#8085e9',
            '#f15c80',
            '#e4d354',
            '#2b908f',
            '#f45b5b',
            '#7cb5ec',
            '#434348',
            '#90ed7d',
            '#f7a35c',
            '#91e8e1'
        ],
    });

    // Chart:area_1_total_area_wide
    Highcharts.chart('area_1_total_area_wide', {
        chart: {
            type: 'column',
            marginRight: 180
        },
        title: {
            text: 'Land-use Total Area Change'
        },
        data: {
            csv: document.getElementById('area_1_total_area_wide_csv').innerHTML,
        },
        credits: {
            enabled: false
        },
        yAxis: {
            title: {
                text: 'Area (km2)'
            },
        },

        legend: {
            align: 'right',
            verticalAlign: 'top',
            layout: 'vertical',
            x: 20,
            y: 50
      
        },

        tooltip: {
            formatter: function () {
                return `<b>Year:</b> ${this.x}<br><b>${this.series.name}:</b>${this.y.toFixed(2)}<br/>`;
            }
        },
    
        plotOptions: {
            column: {
                stacking: 'normal',
            },
            series: {
                events: {
                  legendItemClick: function(event) {
                    var series = this.chart.series;
                              
                              series.forEach(function(item) {
                                  if (this.name == 'Show all' || this == item) {
                                      item.setVisible(true);
                                  } else {
                                      item.setVisible(false);
                                  }
                              }, this);
                    
                    return false;
                  }
                }
              }
        },
        
        exporting: {
            sourceWidth: 1200,
            sourceHeight: 600,
        }
    });

    // Chart:area_2_irrigation_area_wide
    Highcharts.chart('area_2_irrigation_area_wide', {
        chart: {
            type: 'column',
            marginRight: 180
        },
        title: {
            text: 'Total Area by Irrigation Type'
        },
        data: {
            csv: document.getElementById('area_2_irrigation_area_wide_csv').innerHTML,
        },
        credits: {
            enabled: false
        },
        yAxis: {
            title: {
                text: 'Area (km2)'
            },
        },

        legend: {
            align: 'right',
            verticalAlign: 'top',
            layout: 'vertical',
            x: -50,
            y: 250
      
        },

        tooltip: {
            formatter: function () {
                return `<b>Year:</b> ${this.x}<br><b>${this.series.name}:</b>${this.y.toFixed(2)}<br/>`;
            }
        },
    
        plotOptions: {
            column: {
                stacking: 'normal',
            }
        }, 
        
        exporting: {
            sourceWidth: 1200,
            sourceHeight: 600,
        }
    });

    // area_3_am_total_area_wide
    Highcharts.chart('area_3_am_total_area_wide', {
        chart: {
            type: 'column',
            marginRight: 180
        },
        title: {
            text: 'Total Area by Agricultural Management Type'
        },
        data: {
            csv: document.getElementById('area_3_am_total_area_wide_csv').innerHTML,
        },
        credits: {
            enabled: false
        },
        yAxis: {
            title: {
                text: 'Area (km2)'
            },
        },

        legend: {
            align: 'right',
            verticalAlign: 'top',
            layout: 'vertical',
            x: 10,
            y: 250
      
        },

        tooltip: {
            formatter: function () {
                return `<b>Year:</b> ${this.x}<br><b>${this.series.name}:</b>${this.y.toFixed(2)}<br/>`;
            }
        },
    
        plotOptions: {
            column: {
                stacking: 'normal',
            }
        },  
        
        exporting: {
            sourceWidth: 1200,
            sourceHeight: 600,
        }
    });

    // area_4_am_lu_area_wide
    Highcharts.chart('area_4_am_lu_area_wide', {
        chart: {
            type: 'column',
            marginRight: 180
        },
        title: {
            text: 'Total Area by Land-use Type'
        },
        data: {
            csv: document.getElementById('area_4_am_lu_area_wide_csv').innerHTML,
                },
        credits: {
            enabled: false
        },
        yAxis: {
            title: {
                text: 'Area (km2)'
            },
        },

        legend: {
            align: 'right',
            verticalAlign: 'top',
            layout: 'vertical',
            x: 10,
            y: 50
      
        },

        tooltip: {
            formatter: function () {
                return `<b>Year:</b> ${this.x}<br><b>${this.series.name}:</b>${this.y.toFixed(2)}<br/>`;
            }
        },
    
        plotOptions: {
            column: {
                stacking: 'normal',
            },
            series: {
                events: {
                  legendItemClick: function(event) {
                    var series = this.chart.series;
                              
                              series.forEach(function(item) {
                                  if (this.name == 'Show all' || this == item) {
                                      item.setVisible(true);
                                  } else {
                                      item.setVisible(false);
                                  }
                              }, this);
                    
                    return false;
                  }
                }
              }
        },
        
        exporting: {
            sourceWidth: 1200,
            sourceHeight: 600,
        }
        
        
    }); 

    // area_5_begin_end_area
    document.getElementById('area_5_begin_end_area').innerHTML = '<object type="text/html" data="../../data/area_5_begin_end_area.html" ></object>'

    // area_6_begin_end_pct
    document.getElementById('area_6_begin_end_pct').innerHTML = '<object type="text/html" data="../../data/area_6_begin_end_pct.html" ></object>'
});