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

    // Chart:production_1_quantity_df_wide
    Highcharts.chart('water_1_percent_to_limit', {

        chart: {
            type: 'spline',
            marginRight: 180
        },

        title: {
            text: 'Water Usage to Limit in Percentage'
        },

        credits: {
            enabled: false
        },

        data: {
            csv: document.getElementById('water_1_percent_to_limit_csv').innerHTML,
        },
        
        yAxis: {
            title: {
                text: 'Percentage to Limit (%)'
            },
        },

        legend: {
            align: 'right',
            verticalAlign: 'top',
            layout: 'vertical',
            x: 50,
            y: 180
      
        },

        tooltip: {
            formatter: function () {
                return `<b>Year:</b> ${this.x}<br><b>${this.series.name}:</b>${this.y.toFixed(2)}<br/>`;
            }
        },
        
        exporting: {
            sourceWidth: 1200,
            sourceHeight: 600,
        }
    });

    // Chart:water_2_volum_to_limit
    Highcharts.chart('water_2_volum_to_limit', {

        chart: {
            type: 'spline',
            marginRight: 180
        },

        title: {
            text: 'Water Usage to Limit in Volume'
        },

        credits: {
            enabled: false
        },

        data: {
            csv: document.getElementById('water_2_volum_to_limit_csv').innerHTML,
        },
        
        yAxis: {
            title: {
                text: 'Usage to Limit (ML)'
            },
        },

        legend: {
            align: 'right',
            verticalAlign: 'top',
            layout: 'vertical',
            x: 50,
            y: 180
      
        },

        tooltip: {
            formatter: function () {
                return `<b>Year:</b> ${this.x}<br><b>${this.series.name}:</b>${this.y.toFixed(2)}<br/>`;
            }
        },
        
        exporting: {
            sourceWidth: 1200,
            sourceHeight: 600,
        }
    });


});