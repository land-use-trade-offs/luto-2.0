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
        headerFormat: '<b>Year:</b> {point.key}<br/>',
        pointFormat: '<b>{series.name}:</b> {point.y:.2f}<br/>',
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