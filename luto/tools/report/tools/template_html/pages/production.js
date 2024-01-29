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

    // Chart:production_1_demand_type_wide
    Highcharts.chart('production_1_demand_type_wide', {
        chart: {
            type: 'column',
            marginRight: 180
        },
        title: {
            text: 'Demand in Total'
        },
        data: {
            csv: document.getElementById('production_1_demand_type_wide_csv').innerHTML,
        },
        credits: {
            enabled: false
        },
        yAxis: {
            title: {
                text: "Quantity (million tonnes, kilolitres [milk])"
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
                grouping: true,
                shadow: false
            }
        },
        
        exporting: {
            sourceWidth: 1200,
            sourceHeight: 600,
        }
    });

    // Chart:production_2_demand_on_off_wide
    let production_2_demand_on_off_wide_option = {
        chart: {
            renderTo: "production_2_demand_on_off_wide",
            marginRight: 180,
            type: "column",
        },
        title: {
            text: "Agricultural Demand by On/Off Land",
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

        tooltip: {
            formatter: function () {
                return `${this.x}<br><b>${this.series.name}:</b>${this.y.toFixed(2)}<br/>`;
            }
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
                }
                else {
                    // Add series
                    production_2_demand_on_off_wide_option.series.push({
                        name: items[0],
                        data: items.slice(1).map((x) => parseFloat(x)),
                        type: "column",
                    })
                };
            }
        });

        let chart = new Highcharts.Chart(
            production_2_demand_on_off_wide_option
        );
    });

    
    // Chart:production_3_demand_commodity
    let production_3_demand_commodity_option = {
        chart: {
            renderTo: "production_3_demand_commodity",
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
                }
                else {
                    // Add series
                    production_3_demand_commodity_option.series.push({
                        name: items[0],
                        data: items.slice(1).map((x) => parseFloat(x)),
                        type: "column",
                    })
                };
            }
        });

        let chart = new Highcharts.Chart(
            production_3_demand_commodity_option
        );
    });


    // Chart:production_4_1_demand_domestic_On_land_commodity
    Highcharts.chart('production_4_1_demand_domestic_On_land_commodity', {
        chart: {
            type: 'column',
            marginRight: 180
        },
        title: {
            text: 'Domestic Demand - On Land Commodity'
        },
        data: {
            csv: document.getElementById('production_4_1_demand_domestic_On_land_commodity_csv').innerHTML,
        },
        credits: {
            enabled: false
        },
        yAxis: {
            title: {
                text: "Quantity (million tonnes, kilolitres [milk])"
            },
        },

        legend: {
            align: 'right',
            verticalAlign: 'top',
            layout: 'vertical',
            x: 10,
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

    // Chart:production_4_2_demand_domestic_Off_land_commodity
    Highcharts.chart('production_4_2_demand_domestic_Off_land_commodity', {
        chart: {
            type: 'column',
            marginRight: 180
        },
        title: {
            text: 'Domestic Demand - Off Land Commodity'
        },
        data: {
            csv: document.getElementById('production_4_2_demand_domestic_Off_land_commodity_csv').innerHTML,
        },
        credits: {
            enabled: false
        },
        yAxis: {
            title: {
                text: "Quantity (million tonnes, kilolitres [milk])"
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

    // Chart:production_5_2_demand_Exports_commodity
    Highcharts.chart('production_5_2_demand_Exports_commodity', {
        chart: {
            type: 'column',
            marginRight: 180
        },
        title: {
            text: 'Exports Commodity'
        },
        data: {
            csv: document.getElementById('production_5_2_demand_Exports_commodity_csv').innerHTML,
        },
        credits: {
            enabled: false
        },
        yAxis: {
            title: {
                text: "Quantity (million tonnes, kilolitres [milk])"
            },
        },

        legend: {
            align: 'right',
            verticalAlign: 'top',
            layout: 'vertical',
            x: 10,
            y: -10
      
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

    // Chart:production_5_3_demand_Imports_commodity
    production_5_3_demand_Imports_commodity_option = {
        chart: {
            renderTo: "production_5_3_demand_Imports_commodity",
            type: 'column',
            marginRight: 180,
        },
        title: {
            text: 'Imports Commodity'
        },
        series: [],
        credits: {
            enabled: false
        },
        xAxis: {
            categories: [],
        },

        yAxis: {
            title: {
                text: "Quantity (million tonnes, kilolitres [milk])"
            },
        },

        legend: {
            align: 'right',
            verticalAlign: 'top',
            layout: 'vertical',
            x: 10,
            y: 30
      
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
    };

    // Extract data to populate chart
    $(document).ready(function () {

        let data, lines;

        data = document.getElementById(
            "production_5_3_demand_Imports_commodity_csv"
        ).innerHTML;
        
        // If the last line is empty, remove it
        lines = data.split("\n");
        if (lines[lines.length - 1] == "") {
            lines.pop();
        }

        // Iterate through the lines and add categories or series
        $.each(lines, function (lineNo, line) {
            var items = line.split(",");
            if (lineNo == 0) {
                // Loop throught items of this line and add names to series
                $.each(items, function (itemNo, item) {
                    if (itemNo > 0) {
                        production_5_3_demand_Imports_commodity_option.series.push({
                            name: item,
                            data: [],
                        });
                    }
                });
            } 
            else {
                // Add year to categories
                production_5_3_demand_Imports_commodity_option.xAxis.categories.push(
                    parseFloat(items[0])
                );

                // Add data to series
                $.each(items, function (itemNo, item) {
                    if (itemNo > 0) {

                        // If the item is empty, add null
                        if (item == "") {
                            production_5_3_demand_Imports_commodity_option.series[itemNo - 1].data.push(
                                null
                            );
                        } else {
                            // Add the item
                            production_5_3_demand_Imports_commodity_option.series[itemNo - 1].data.push(
                                parseFloat(item));


                        }
                    }
                });

                // Loop through series, if all null, add the showInLegend to be false
                production_5_3_demand_Imports_commodity_option.series.forEach(
                    (series) => {
                        let allNull = true;
                        series.data.forEach((data) => {
                            if (data != null) {
                                allNull = false;
                            }
                        });
                        if (allNull) {
                            series.showInLegend = false;
                        }
                    }
                );

                };

        });

        // Create the chart
        let chart = new Highcharts.Chart(
            production_5_3_demand_Imports_commodity_option
        );
        console.log(production_5_3_demand_Imports_commodity_option);


    });



    // Chart:production_5_4_demand_Feed_commodity
    Highcharts.chart('production_5_4_demand_Feed_commodity', {
        chart: {
            type: 'column',
            marginRight: 180
        },
        title: {
            text: 'Feed Commodity'
        },
        data: {
            csv: document.getElementById('production_5_4_demand_Feed_commodity_csv').innerHTML,
        },
        credits: {
            enabled: false
        },
        yAxis: {
            title: {
                text: "Quantity (million tonnes, kilolitres [milk])"
            },
        },

        legend: {
            align: 'right',
            verticalAlign: 'top',
            layout: 'vertical',
            x: -10,
            y: 200
      
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

    // Chart:production_5_5_demand_Production_commodity
    Highcharts.chart('production_5_5_demand_Production_commodity', {
        chart: {
            type: 'column',
            marginRight: 180
        },
        title: {
            text: 'Production Commodity (demands for LUTO)'
        },
        data: {
            csv: document.getElementById('production_5_5_demand_Production_commodity_csv').innerHTML,
        },
        credits: {
            enabled: false
        },
        yAxis: {
            title: {
                text: "Quantity (million tonnes, kilolitres [milk])"
            },
        },

        legend: {
            align: 'right',
            verticalAlign: 'top',
            layout: 'vertical',
            x: 0,
            y: -10
      
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

    // Chart:production_5_6_demand_Production_commodity_from_LUTO
    Highcharts.chart('production_5_6_demand_Production_commodity_from_LUTO', {
        chart: {
            type: 'column',
            marginRight: 180
        },
        title: {
            text: 'Production Commodity (outputs from LUTO)'
        },
        data: {
            csv: document.getElementById('production_5_6_demand_Production_commodity_from_LUTO_csv').innerHTML,
        },
        credits: {
            enabled: false
        },
        yAxis: {
            title: {
                text: "Quantity (million tonnes, kilolitres [milk])"
            },
        },

        legend: {
            align: 'right',
            verticalAlign: 'top',
            layout: 'vertical',
            x: 0,
            y: -10
      
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




    

    
});
    


