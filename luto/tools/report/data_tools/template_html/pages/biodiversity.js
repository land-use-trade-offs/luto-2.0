// Create Chart
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


    // Set the title alignment to left
    Highcharts.setOptions({
        title: {
            align: 'left'
        }
    });

    // biodiversity_GBF2_1_total_score_by_type
    Highcharts.chart("biodiversity_GBF2_1_total_score_by_type", {
        chart: {
            type: "column",
            marginRight: 380,
        },
        title: {
            text: "Contribution to GBF-2 Target by Broad Land-use Type",
        },
        series: JSON.parse(
            document.getElementById(
                "biodiversity_GBF2_1_total_score_by_type_csv"
            ).innerHTML
        ),
        xAxis: {
            tickPositions: year_ticks,
        },
        yAxis: {
            title: {
                text: "Relative Contribution to 2010 Biodiversity Degradation Level (%)",
            },
        },
        plotOptions: {
            column: {
                stacking: "normal",
            },
        },
        tooltip: {
            formatter: function () {
                return `<b>Year:</b> ${this.x}<br><b>${this.series.name
                    }:</b>${this.y.toFixed(2)}<br/>`;
            },
        },
        legend: {
            align: "right",
            verticalalign: "left",
            layout: "vertical",
            x: -150,
            verticalAlign: "middle",
        },
        credits: {
            enabled: false,
        },
    });


    // biodiversity_GBF2_2_total_score_by_landuse
    Highcharts.chart("biodiversity_GBF2_2_total_score_by_landuse", {
        chart: {
            type: "column",
            marginRight: 380,
        },
        title: {
            text: "Contribution to GBF-2 Target by Specific Land-use",
        },
        series: JSON.parse(
            document.getElementById(
                "biodiversity_GBF2_2_total_score_by_landuse_csv"
            ).innerHTML
        ),
        xAxis: {
            tickPositions: year_ticks,
        },
        yAxis: {
            title: {
                text: "Relative Contribution to 2010 Biodiversity Degradation Level (%)",
            },
        },
        plotOptions: {
            column: {
                stacking: "normal",
            },
        },
        tooltip: {
            formatter: function () {
                return `<b>Year:</b> ${this.x}<br><b>${this.series.name
                    }:</b>${this.y.toFixed(2)}<br/>`;
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
        credits: {
            enabled: false,
        },
    });

    


});

