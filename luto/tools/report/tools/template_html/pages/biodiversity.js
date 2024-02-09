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

    // Get the years from the csv
    var tickposition;
    $(document).ready(function () {
        let data, lines;
        let years = [];

        data = document.getElementById(
            "biodiversity_1_total_score_by_category"
        ).innerHTML;
        lines = data.split("\n");

        // if the last line is empty, remove it
        if (lines[lines.length - 1] == "") {
            lines.pop();
        }

        $.each(lines, function (lineNo, line) {
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

        // biodiversity_1_total_score_by_category
        Highcharts.chart("biodiversity_1_total_score_by_category", {
            chart: {
                type: "column",
                marginRight: 180,
            },
            title: {
                text: "Total Biodiversity Score by Category",
            },
            data: {
                csv: document.getElementById(
                    "biodiversity_1_total_score_by_category_csv"
                ).innerHTML,
            },
            yAxis: {
                title: {
                    text: "Quality-weighted Area (million ha)",
                },
            },
            xAxis: {
                tickPositions: tickposition,
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
                verticalAlign: "left",
                layout: "vertical",
                x: 0,
                y: 250,
            },
            credits: {
                enabled: false,
            },
        });

        // // biodiversity_2_total_score_by_irrigation
        // Highcharts.chart("biodiversity_2_total_score_by_irrigation", {
        //     chart: {
        //         type: "column",
        //         marginRight: 180,
        //     },
        //     title: {
        //         text: "Total Biodiversity Score by Irrigation",
        //     },
        //     data: {
        //         csv: document.getElementById(
        //             "biodiversity_2_total_score_by_irrigation_csv"
        //         ).innerHTML,
        //     },
        //     yAxis: {
        //         title: {
        //             text: "Quality-weighted Area (million ha)",
        //         },
        //     },
        //     xAxis: {
        //         tickPositions: tickposition,
        //     },
        //     plotOptions: {
        //         column: {
        //             stacking: "normal",
        //         },
        //     },
        //     tooltip: {
        //         formatter: function () {
        //             return `<b>Year:</b> ${this.x}<br><b>${this.series.name
        //                 }:</b>${this.y.toFixed(2)}<br/>`;
        //         },
        //     },
        //     legend: {
        //         align: "right",
        //         verticalAlign: "left",
        //         layout: "vertical",
        //         x: 0,
        //         y: 300,
        //     },
        //     credits: {
        //         enabled: false,
        //     },
        // });

        // biodiversity_3_total_score_by_landuse
        Highcharts.chart("biodiversity_3_total_score_by_landuse", {
            chart: {
                type: "column",
                marginRight: 180,
            },
            title: {
                text: "Total Biodiversity Score by Land Use",
            },
            data: {
                csv: document.getElementById(
                    "biodiversity_3_total_score_by_landuse_csv"
                ).innerHTML,
            },
            yAxis: {
                title: {
                    text: "Quality-weighted Area (million ha)",
                },
            },
            xAxis: {
                tickPositions: tickposition,
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
                verticalAlign: "left",
                layout: "vertical",
                x: 10,
                y: 250,
                itemStyle: {
                    // "fontSize": "11.5px",
                  },
            },
            credits: {
                enabled: false,
            },
        });


    });
});
