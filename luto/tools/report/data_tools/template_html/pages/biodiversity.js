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
    var year_ticks = years;


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


    // biodiversity_GBF2_3_total_score_by_agri_management
    Highcharts.chart("biodiversity_GBF2_3_total_score_by_agri_management", {
        chart: {
            type: "column",
            marginRight: 380,
        },
        title: {
            text: "Contribution to GBF-2 Target by Agricultural Management",
        },
        series: JSON.parse(
            document.getElementById(
                "biodiversity_GBF2_3_total_score_by_agri_management_csv"
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
            x: -100,
            verticalAlign: "middle",
        },
        credits: {
            enabled: false,
        },
    });


    // biodiversity_GBF2_4_total_score_by_non_agri_landuse
    Highcharts.chart("biodiversity_GBF2_4_total_score_by_non_agri_landuse", {
        chart: {
            type: "column",
            marginRight: 380,
        },
        title: {
            text: "Contribution to GBF-2 Target by Non-Agricultural Land-use",
        },
        series: JSON.parse(
            document.getElementById(
                "biodiversity_GBF2_4_total_score_by_non_agri_landuse_csv"
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
            x: 0,
            verticalAlign: "middle",
        },
        credits: {
            enabled: false,
        },
    });

    // biodiversity_GBF3_1_contribution_group_score_total
    Highcharts.chart("biodiversity_GBF3_1_contribution_group_score_total", {
        chart: {
            type: "column",
            marginRight: 380,
        },
        title: {
            text: "Total Vegetation Score (GBF3) by Group",
        },
        series: JSON.parse(
            document.getElementById(
                "biodiversity_GBF3_1_contribution_group_score_total_csv"
            ).innerHTML
        ).concat({
            name: ' ',
            data: [[2010, 0]],
            visible: false,
            showInLegend: false,
        }),
        xAxis: {
            tickPositions: year_ticks,
        },
        yAxis: {
            title: {
                text: "Vegetation Score Relative to Pre-1750 Level (%)",
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
            layout: "vertical",
            itemStyle: {
                fontSize: "11px",
            },
            itemMarginBottom:0.5,
            itemMarginTop:0,
            x: -10,
            y: 0,
            verticalAlign: "middle",
        },
        credits: {
            enabled: false,
        },
        });


    // Chart:biodiversity_GBF3_2_contribution_group_score_by_type
    const chartContainer5 = document.getElementById('biodiversity_GBF3_2_contribution_group_score_by_type');
    chartData5 = JSON.parse(document.getElementById("biodiversity_GBF3_2_contribution_group_score_by_type_csv").innerHTML);

    // Create blocks and render Highcharts in each block
    chartData5.forEach((chart, index) => {
        // Create a new div for each chart
        const chartBlock5 = document.createElement('div');
        chartBlock5.classList.add('chart-block');
        chartBlock5.id = `chart5-${index + 1}`;
        chartContainer5.appendChild(chartBlock5);

        Highcharts.chart(chartBlock5.id, {
            plotOptions: {
                showInLegend: false,
                column: {
                    stacking: "normal",
                },
                },
                title: {
                    text: chart.name,
                    align: 'center'
                },
                series: chart.data,
                xAxis: {
                    tickPositions: year_ticks,
                },
                yAxis: {
                    title: {
                        text: "Vegetation Score Relative to Pre-1750 Level (%)",
                    },
                },
                tooltip: {
                    formatter: function () {
                        return `<b>Year:</b> ${this.x}<br><b>${this.series.name
                            }:</b>${this.y.toFixed(2)}<br/>`;
                    },
                },
                credits: {
                    enabled: false,
                },
                });
                });






        


    // biodiversity_GBF4_1_contribution_group_score_total
    Highcharts.chart("biodiversity_GBF4_1_contribution_group_score_total", {
        chart: {
            type: "column",
            marginRight: 380,
        },
        title: {
            text: "Total Biodiversity Suitability Score (GBF4A) by Group",
        },
        series: JSON.parse(
            document.getElementById(
                "biodiversity_GBF4_1_contribution_group_score_total_csv"
            ).innerHTML
        ).concat({
            name: ' ',
            data: [[2010, 0]],
            visible: false,
            showInLegend: false,
        }),
        xAxis: {
            tickPositions: year_ticks,
        },
        yAxis: {
            title: {
                text: "Suitability Relative to Pre-1750 Level (%)",
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
            layout: "vertical",
            x: -200,
            y: -10,
            verticalAlign: "middle",
        },
        credits: {
            enabled: false,
        },
    });



    // Chart:biodiversity_GBF4_2_contribution_group_score_by_type
    const chartContainer = document.getElementById('biodiversity_GBF4_2_contribution_group_score_by_type');
    chartData = JSON.parse(document.getElementById("biodiversity_GBF4_2_contribution_group_score_by_type_csv").innerHTML);

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
                },
            },
            title: {
                text: chart.name,
                align: 'center'
            },
            series: chart.data,
            xAxis: {
                tickPositions: year_ticks,
            },
            yAxis: {
                title: {
                    text: "Suitability Relative to Pre-1750 Level (%)",
                },
            },
            tooltip: {
                formatter: function () {
                    return `<b>Year:</b> ${this.x}<br><b>${this.series.name
                        }:</b>${this.y.toFixed(2)}<br/>`;
                },
            },
            credits: {
                enabled: false,
            },
        });

    });

    // biodiversity_GBF4_3_contribution_group_score_by_landuse
    const chartContainer2 = document.getElementById('biodiversity_GBF4_3_contribution_group_score_by_landuse');
    const chartData2 = JSON.parse(document.getElementById("biodiversity_GBF4_3_contribution_group_score_by_landuse_csv").innerHTML);

    // Create blocks and render Highcharts in each block
    chartData2.forEach((chart, index) => {
        // Create a new div for each chart
        const chartBlock2 = document.createElement('div');
        chartBlock2.classList.add('chart-block');
        chartBlock2.id = `chart2-${index + 1}`;
        chartContainer2.appendChild(chartBlock2);

        Highcharts.chart(chartBlock2.id, {
            plotOptions: {
                showInLegend: false,
                column: {
                    stacking: "normal",
                },
            },
            title: {
                text: chart.name,
                align: 'center'
            },
            series: chart.data,
            xAxis: {
                tickPositions: year_ticks,
            },
            yAxis: {
                title: {
                    text: "Suitability Relative to Pre-1750 Level (%)",
                },
            },
            tooltip: {
                formatter: function () {
                    return `<b>Year:</b> ${this.x}<br><b>${this.series.name
                        }:</b>${this.y.toFixed(2)}<br/>`;
                },
            },
            credits: {
                enabled: false,
            },
            legend: {
                enabled: false
            }
        });
    });


    // biodiversity_GBF4_4_contribution_species_score_total
    Highcharts.chart("biodiversity_GBF4_4_contribution_species_score_total", {
        chart: {
            type: "column",
            marginRight: 380,
        },
        title: {
            text: "Total Biodiversity Suitability Score (GBF4A) by Species",
        },
        series: JSON.parse(
            document.getElementById(
                "biodiversity_GBF4_4_contribution_species_score_total_csv"
            ).innerHTML
        ).concat({
            name: ' ',
            data: [[2010, 0]],
            visible: false,
            showInLegend: false,
        }),
        xAxis: {
            tickPositions: year_ticks,
        },
        yAxis: {
            title: {
                text: "Suitability Relative to Pre-1750 Level (%)",
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
            layout: "vertical",
            x: -200,
            y: -10,
            verticalAlign: "middle",
        },
        credits: {
            enabled: false,
        },
    });

    // biodiversity_GBF4_5_contribution_species_score_by_type
    const chartContainer3 = document.getElementById('biodiversity_GBF4_5_contribution_species_score_by_type');
    const chartData3 = JSON.parse(document.getElementById("biodiversity_GBF4_5_contribution_species_score_by_type_csv").innerHTML);

    // Create blocks and render Highcharts in each block
    chartData3.forEach((chart, index) => {
        // Create a new div for each chart
        const chartBlock3 = document.createElement('div');
        chartBlock3.classList.add('chart-block');
        chartBlock3.id = `chart3-${index + 1}`;
        chartContainer3.appendChild(chartBlock3);

        Highcharts.chart(chartBlock3.id, {
            plotOptions: {
                showInLegend: false,
                column: {
                    stacking: "normal",
                },
            },
            title: {
                text: chart.name,
                align: 'center'
            },
            series: chart.data,
            xAxis: {
                tickPositions: year_ticks,
            },
            yAxis: {
                title: {
                    text: "Suitability Relative to Pre-1750 Level (%)",
                },
            },
            tooltip: {
                formatter: function () {
                    return `<b>Year:</b> ${this.x}<br><b>${this.series.name
                        }:</b>${this.y.toFixed(2)}<br/>`;
                },
            },
            credits: {
                enabled: false,
            },
        });

    });

    // biodiversity_GBF4_6_contribution_species_score_by_landuse
    const chartContainer4 = document.getElementById('biodiversity_GBF4_6_contribution_species_score_by_landuse');
    const chartData4 = JSON.parse(document.getElementById("biodiversity_GBF4_6_contribution_species_score_by_landuse_csv").innerHTML);

    // Create blocks and render Highcharts in each block
    chartData4.forEach((chart, index) => {
        // Create a new div for each chart
        const chartBlock4 = document.createElement('div');
        chartBlock4.classList.add('chart-block');
        chartBlock4.id = `chart4-${index + 1}`;
        chartContainer4.appendChild(chartBlock4);

        Highcharts.chart(chartBlock4.id, {
            plotOptions: {
                showInLegend: false,
                column: {
                    stacking: "normal",
                },
            },
            title: {
                text: chart.name,
                align: 'center'
            },
            series: chart.data,
            xAxis: {
                tickPositions: year_ticks,
            },
            yAxis: {
                title: {
                    text: "Suitability Relative to Pre-1750 Level (%)",
                },
            },
            tooltip: {
                formatter: function () {
                    return `<b>Year:</b> ${this.x}<br><b>${this.series.name
                        }:</b>${this.y.toFixed(2)}<br/>`;
                },
            },
            credits: {
                enabled: false,
            },
            legend: {
                enabled: false
            }
        });

    });



});

