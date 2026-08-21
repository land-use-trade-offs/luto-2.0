// Shared number formatter: 2 dp for tooltip, 1 dp for axis labels.
function _fmtNum(v, dec) {
    var abs = Math.abs(v);
    if (abs === 0) return '0';
    var scaled, suffix;
    if (abs >= 1e9)      { scaled = v / 1e9; suffix = 'B'; }
    else if (abs >= 1e6) { scaled = v / 1e6; suffix = 'M'; }
    else if (abs >= 1e3) { scaled = v / 1e3; suffix = 'k'; }
    else                 { return v.toFixed(dec).replace(/\.?0+$/, ''); }
    return scaled.toFixed(dec).replace(/\.?0+$/, '') + suffix;
}

// Set as Highcharts global defaults so they survive per-chart yAxis/tooltip overrides
// (views do a shallow spread of Chart_default_options, which drops nested yAxis.labels etc.)
Highcharts.setOptions({
    yAxis: {
        labels: {
            formatter: function() { return _fmtNum(this.value, 1); }
        }
    },
    tooltip: {
        pointFormatter: function() {
            return '<b>' + this.series.name + ':</b> ' + _fmtNum(this.y, 2) + '<br/>';
        }
    }
});

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