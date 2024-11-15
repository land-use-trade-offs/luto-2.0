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

    // biodiversity_1_total_score_by_category
    Highcharts.chart("biodiversity_1_total_score_by_category", {
        chart: {
            type: "column",
            marginRight: 380,
        },
        title: {
            text: "Total Biodiversity Score by Land-use/Management",
        },
        series: JSON.parse(
            document.getElementById(
                "biodiversity_1_total_score_by_category_csv"
            ).innerHTML
        ),
        xAxis: {
            tickPositions: year_ticks,
        },
        yAxis: {
            title: {
                text: "Quality-weighted Area (million ha)",
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

    // biodiversity_3_total_score_by_landuse
    Highcharts.chart("biodiversity_3_total_score_by_landuse", {
        chart: {
            type: "column",
            marginRight: 380,
        },
        title: {
            text: "Total Biodiversity Score by Land-use",
        },
        series: JSON.parse(
            document.getElementById(
                "biodiversity_3_total_score_by_landuse_csv"
            ).innerHTML
        ),
        xAxis: {
            tickPositions: year_ticks,
        },
        yAxis: {
            title: {
                text: "Quality-weighted Area (million ha)",
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
            x: 0,
            verticalAlign: "middle",
            itemStyle: {
                // "fontSize": "11.5px",
            },
        },
        credits: {
            enabled: false,
        },
    });

    // biodiversity_4_natural_land_area
    Highcharts.chart("biodiversity_4_natural_land_area", {
        chart: {
            type: "column",
            marginRight: 380,
        },
        title: {
            text: "Land-Use Area (Natural land & Non-Agricultural land)",
        },
        series: JSON.parse(
            document.getElementById("biodiversity_4_natural_land_area_csv").innerHTML
        ),
        xAxis: {
            tickPositions: year_ticks,
        },
        yAxis: {
            title: {
                text: "Area (million ha)",
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
            x: 0,
            verticalAlign: "middle",
        },
        credits: {
            enabled: false,
        },
    });

    // biodiversity_5_contribution_score_by_group
    Highcharts.chart("biodiversity_5_contribution_score_by_group", {
        chart: {
            type: "column",
            marginRight: 380,
        },
        title: {
            text: "Biodiversity Contribution by Group",
        },
        series: JSON.parse(
            document.getElementById("biodiversity_5_contribution_score_by_group_csv").innerHTML
        ),
        xAxis: {
            tickPositions: year_ticks,
        },
        yAxis: {
            title: {
                text: "Contribution Score (%)",
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
            x: -250,
            verticalAlign: "middle",
        },
        credits: {
            enabled: false,
        },
    });

    // biodiversity_6_contribution_score_by_landuse_type_broad
    Highcharts.chart("biodiversity_6_contribution_score_by_landuse_type_broad", {
        chart: {
            type: "column",
            marginRight: 380,
        },
        title: {
            text: "Biodiversity Contribution by Land-use Type (Broad)",
        },
        series: JSON.parse(
            document.getElementById("biodiversity_6_contribution_score_by_landuse_type_broad_csv").innerHTML
        ),
        yAxis: {
            title: {
                text: "Contribution Score (%)",
            },
        },
        xAxis: {
            tickPositions: year_ticks,
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


    // biodiversity_7_contribution_score_by_landuse_type_specific
    Highcharts.chart("biodiversity_7_contribution_score_by_landuse_type_specific", {
        chart: {
            type: "column",
            marginRight: 380,
        },
        title: {
            text: "Biodiversity Contribution by Land-use Type (Specific)",
        },
        series: JSON.parse(
            document.getElementById("biodiversity_7_contribution_score_by_landuse_type_specific_csv").innerHTML
        ),
        xAxis: {
            tickPositions: year_ticks,
        },
        yAxis: {
            title: {
                text: "Contribution Score (%)",
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

