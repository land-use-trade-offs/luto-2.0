// create chart
document.addEventListener('DOMContentLoaded', function () {

    Highcharts.setOptions({
        colors: [
            '#7cb5ec',
            '#434348',
            '#90ed7d',
            '#f7a35c',
            '#8085e9',
            '#f15c80',
            '#e4d354',
            '#2b908f',
            '#f45b5b',
            '#91e8e1'
        ],
    });

    // Chart:production_0_1_demand_borader_cat_wide
    let production_0_1_demand_borader_cat_wide_option = {
        chart: {
            renderTo: "production_0_1_demand_borader_cat_wide",
            marginRight: 180,
            type: "column",
        },
        title: {
            text: "Agricultural Demand by Category",
        },

        xAxis: {
            tickWidth:0.05,
            categories: [],
            labels: {
                y: 38,
                groupedOptions: [{
                        rotation: -90, // rotate labels for a 2st-level
                        align: 'center'
                    }],
                    rotation: -90, // rotate labels for a 1st-level
                    align: 'center'

            }
        },

        yAxis: {
            title: {
                text: "Quantity (million tonnes, kilolitres [milk])",
            }
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
            "production_0_1_demand_borader_cat_wide_csv"
        ).innerHTML;

        // Get the years and types
        lines = data.split("\n");
        years = lines[0].split(",").slice(1);
        years = [...new Set(years)];
        types = lines[1].split(",").slice(1);
        types = [...new Set(types)];

        years.forEach((year) => {
            production_0_1_demand_borader_cat_wide_option.xAxis.categories.push({
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
                }
                else {
                    // Add series
                    production_0_1_demand_borader_cat_wide_option.series.push({
                        name: items[0],
                        data: items.slice(1).map((x) => parseFloat(x)),
                        type: "column",
                    })
                };
            }
        });

        let chart = new Highcharts.Chart(
            production_0_1_demand_borader_cat_wide_option
        );
        console.log(production_0_1_demand_borader_cat_wide_option);
    });


    // Chart:production_1_quantity_df_wide
    Highcharts.chart('production_1_quantity_df_wide', {

        chart: {
            type: 'column',
            marginRight: 180
        },

        title: {
            text: 'Agricultural Production by Commodity'
        },

        credits: {
            enabled: false
        },

        data: {
            csv: document.getElementById('production_1_quantity_df_wide_csv').innerHTML,
        },
        
        yAxis: {
            title: {
                text: 'Quantity (million tonnes, kilolitres [milk])'
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
            }
        },
        
        exporting: {
            sourceWidth: 1200,
            sourceHeight: 600,
        }
    });

    // Chart:production_2_revenue_1_Irrigation_wide
    Highcharts.chart('production_2_revenue_1_Irrigation_wide', {

        chart: {
            type: 'column',
            marginRight: 180
        },

        title: {
            text: 'Revenue by Irrigation Status'
        },

        credits: {
            enabled: false
        },
    
        data: {
            csv: document.getElementById('production_2_revenue_1_Irrigation_wide_csv').innerHTML,
        },
        
        yAxis: {
            title: {
                text: 'Revenue (billion AU$)'
            },
        },

        legend: {
            align: 'right',
            verticalAlign: 'left',
            layout: 'vertical',
            x: -100,
            y: 300
      
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

    // Chart:production_2_revenue_2_Source_wide
    Highcharts.chart('production_2_revenue_2_Source_wide', {

        chart: {
            type: 'column',
            marginRight: 180
        },

        title: {
            text: 'Revenue by Agricultural Product'
        },

        credits: {
            enabled: false
        },
    
        data: {
            csv: document.getElementById('production_2_revenue_2_Source_wide_csv').innerHTML,
        },
        
        yAxis: {
            title: {
                text: 'Revenue (billion AU$)'
            },
        },

        legend: {
            align: 'right',
            verticalAlign: 'left',
            layout: 'vertical',
            x: 10,
            y: 100
      
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

    // Chart:production_2_revenue_3_Source_type_wide
    Highcharts.chart('production_2_revenue_3_Source_type_wide', {

        chart: {
            type: 'column',
            marginRight: 180
        },

        title: {
            text: 'Revenue by Agriultural Commodity'
        },

        credits: {
            enabled: false
        },
    
        data: {
            csv: document.getElementById('production_2_revenue_3_Source_type_wide_csv').innerHTML,
        },
        
        yAxis: {
            title: {
                text: 'Revenue (billion AU$)'
            },
        },

        legend: {
            align: 'right',
            verticalAlign: 'left',
            layout: 'vertical',
            x: 30,
            y: 80
      
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

     // Chart:production_2_revenue_4_Type_wide
     Highcharts.chart('production_2_revenue_4_Type_wide', {

        chart: {
            type: 'column',
            marginRight: 180
        },

        title: {
            text: 'Revenue by Commodity Type'
        },
        
        credits: {
            enabled: false
        },

        data: {
            csv: document.getElementById('production_2_revenue_4_Type_wide_csv').innerHTML,
        },
        
        yAxis: {
            title: {
                text: 'Revenue (billion AU$)'
            },
        },

        legend: {
            align: 'right',
            verticalAlign: 'left',
            layout: 'vertical',
            x: -50,
            y: 280
      
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

     // Chart:production_2_revenue_5_crop_lvstk_wide
     Highcharts.chart('production_2_revenue_5_crop_lvstk_wide', {

        chart: {
            type: 'column',
            marginRight: 180
        },

        title: {
            text: 'Revenue by Crop/Livestock'
        },
    
        data: {
            csv: document.getElementById('production_2_revenue_5_crop_lvstk_wide_csv').innerHTML,
        },
        
        credits: {
            enabled: false
        },

        yAxis: {
            title: {
                text: 'Revenue (billion AU$)'
            },
        },

        legend: {
            align: 'right',
            verticalAlign: 'left',
            layout: 'vertical',
            x: -50,
            y: 280
      
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

    // Chart:production_3_cost_1_Irrigation_wide
    Highcharts.chart('production_3_cost_1_Irrigation_wide', {

        chart: {
            type: 'column',
            marginRight: 180
        },

        title: {
            text: 'Cost of Production by Irrigation Status'
        },

        credits: {
            enabled: false
        },
    
        data: {
            csv: document.getElementById('production_3_cost_1_Irrigation_wide_csv').innerHTML,
        },
        
        yAxis: {
            title: {
                text: 'Cost (billion AU$)'
            },
        },

        legend: {
            align: 'right',
            verticalAlign: 'left',
            layout: 'vertical',
            x: -100,
            y: 300
      
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


    // Chart:production_3_cost_2_Source_wide
    Highcharts.chart('production_3_cost_2_Source_wide', {

        chart: {
            type: 'column',
            marginRight: 180
        },

        title: {
            text: 'Cost of Production by Agricultural Product'
        },

        credits: {
            enabled: false
        },
    
        data: {
            csv: document.getElementById('production_3_cost_2_Source_wide_csv').innerHTML,
        },
        
        yAxis: {
            title: {
                text: 'Cost (billion AU$)'
            },
        },

        legend: {
            align: 'right',
            verticalAlign: 'left',
            layout: 'vertical',
            x: 10,
            y: 100
      
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

    // Chart:production_3_cost_3_Source_type_wide
    Highcharts.chart('production_3_cost_3_Source_type_wide', {

        chart: {
            type: 'column',
            marginRight: 180
        },

        title: {
            text: 'Cost of Production by Commodity'
        },

        credits: {
            enabled: false
        },
    
        data: {
            csv: document.getElementById('production_3_cost_3_Source_type_wide_csv').innerHTML,
        },
        
        yAxis: {
            title: {
                text: 'Cost (billion AU$)'
            },
        },

        legend: {
            align: 'right',
            verticalAlign: 'left',
            layout: 'vertical',
            x: 30,
            y: 80
      
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

    // Chart:production_3_cost_4_Type_wide
    Highcharts.chart('production_3_cost_4_Type_wide', {

        chart: {
            type: 'column',
            marginRight: 180
        },

        title: {
            text: 'Cost of Production by Commodity Type'
        },
        
        credits: {
            enabled: false
        },

        data: {
            csv: document.getElementById('production_3_cost_4_Type_wide_csv').innerHTML,
        },
        
        yAxis: {
            title: {
                text: 'Cost (billion AU$)'
            },
        },

        legend: {
            align: 'right',
            verticalAlign: 'left',
            layout: 'vertical',
            x: -50,
            y: 280
      
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

    // Chart:production_3_cost_5_crop_lvstk_wide
    Highcharts.chart('production_3_cost_5_crop_lvstk_wide', {

        chart: {
            type: 'column',
            marginRight: 180
        },

        title: {
            text: 'Cost of Production by Crop/Livestock'
        },
    
        data: {
            csv: document.getElementById('production_3_cost_5_crop_lvstk_wide_csv').innerHTML,
        },
        
        credits: {
            enabled: false
        },

        yAxis: {
            title: {
                text: 'Cost (billion AU$)'
            },
        },

        legend: {
            align: 'right',
            verticalAlign: 'left',
            layout: 'vertical',
            x: -50,
            y: 280
      
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

    // Chart:production_4_rev_cost_all

    let production_4_rev_cost_all_option = {
        chart: {
            type: 'columnrange',
            renderTo: 'production_4_rev_cost_all',
            marginRight: 180
        },
    
        title: {
            text: 'Agricultural Revenue and Cost of Production'
        },

        credits: {
            enabled: false
        },

        xAxis: {
            categories: []
        },
    
        yAxis: {
            title: {
                text: 'Value (billion AU$)'
            }
        },

        tooltip: {
            formatter: function() {
                return '<b>' + this.series.name + '</b>: ' + 
                       Highcharts.numberFormat(this.point.low, 2) + ' - ' + 
                       Highcharts.numberFormat(this.point.high, 2) + ' (billion AU$)';
            }
        },
    
        plotOptions: {
            columnrange: {
                borderRadius: '50%',
            }
        },
        
        series: [{name:'Revenue',data:[]},
                 {name:'Cost',data:[]}],
    
        legend: {
            align: 'right',
            verticalAlign: 'left',
            layout: 'vertical',
            x: -50,
            y: 280
        },
    
    }

    $(document).ready(function() {
        let data;
        data = document.getElementById('production_4_rev_cost_all_csv').innerHTML;

        var lines = data.split('\n');
        
        $.each(lines, function(lineNo, line) {
            var items = line.split(',');
    
            if (lineNo != 0) { // Skip the first line (headers)
                production_4_rev_cost_all_option.xAxis.categories.push(items[0]);
                production_4_rev_cost_all_option.series[0].data.push([0, parseFloat(items[1])]); // Revenue
                production_4_rev_cost_all_option.series[1].data.push([parseFloat(items[3]), parseFloat(items[1])]); // Cost
            }
        });
    
        // Create the chart with the correct options
        let chart = new Highcharts.Chart(production_4_rev_cost_all_option);
        });

    
});
    


