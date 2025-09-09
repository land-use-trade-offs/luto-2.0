window.Chart_default_options = {
    chart: {
        type: "column",
        marginRight: 300,
        height: 600,
    },
    title: {
        text: ''
    },
    yAxis: {
        title: {
            text: "Area (million km2)",
        },
    },
    legend: {
        itemStyle: {
            fontSize: "10px",
        },
        align: "right",
        layout: "vertical",
        verticalAlign: "middle",
        itemMarginTop: 0,
        itemMarginBottom: 1,
        width: 230,
    },
    tooltip: {
        formatter: function () {
            return `<b>Year:</b> ${this.x}<br><b>${this.series.name
                }:</b>${this.y.toFixed(2)}<br/>`;
        },
    },
    plotOptions: {
        column: {
            stacking: "normal",
        },
    },
    credits: {
        enabled: false,
    },
    exporting: {
        sourceWidth: 1200,
        sourceHeight: 600,
    },
};