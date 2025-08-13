window.chartMemLogOptions = {
    chart: {
        type: "area",
        height: 450,
    },
    title: {
        text: null,
    },
    xAxis: {
        type: 'datetime'
    },
    yAxis: {
        title: {
            text: 'Memory Use (GB)',
        }
    },
    legend: {
        enabled: false
    },
    tooltip: {},
    plotOptions: {
        area: {
            marker: {
                radius: 2
            },
            lineWidth: 1,
            color: {
                linearGradient: {
                    x1: 0,
                    y1: 0,
                    x2: 0,
                    y2: 1
                },
                stops: [
                    [0, 'rgb(199, 113, 243)'],
                    [0.7, 'rgb(76, 175, 254)']
                ]
            },
            states: {
                hover: {
                    lineWidth: 1
                }
            },
            threshold: null
        }
    },
};