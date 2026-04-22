/**
 * heatmap_container.js — Vue wrapper for the From→To transition area heatmap.
 *
 * Mirrors the Highcharts config from heatmap_utils.py (jinzhu_inspect_code/).
 * Expects the same JSON leaf structure:
 *   { x_categories, y_categories, data, max_val }
 *
 * Props:
 *   xCats    {Array}  To-LU labels (may contain <br> for long names)
 *   yCats    {Array}  From-LU labels (plain text)
 *   data     {Array}  Highcharts point array (mix of [x,y,v] and {x,y,value,color})
 *   maxVal   {Number} colorAxis upper bound (excludes ALL/diagonal)
 *   title    {String} chart title
 *   nullColor {String} background for null cells (default: '#2a2a3e')
 */
window.HeatmapContainer = {
    name: 'HeatmapContainer',
    props: {
        xCats: { type: Array, required: true },
        yCats: { type: Array, required: true },
        data: { type: Array, required: true },
        maxVal: { type: Number, required: true },
        nullColor: { type: String, default: '#f0f0f0' },
    },

    setup(props) {
        const { ref, onMounted, onUnmounted, watch } = Vue;
        const chartEl = ref(null);
        let chartInstance = null;

        // ------------------------------------------------------------------
        // Build the Highcharts config object — mirrors make_heatmap_html()
        // ------------------------------------------------------------------
        const buildConfig = () => {
            const xCats = props.xCats || [];
            const yCats = props.yCats || [];
            const data = props.data || [];
            const maxVal = props.maxVal || 0;

            return {
                chart: {
                    type: 'heatmap',
                    backgroundColor: null,
                    marginTop: 150,
                    marginBottom: 30,
                    marginLeft: 185,
                    marginRight: 15,
                    height: Math.max(500, yCats.length * 22 + 140),
                    style: { fontFamily: 'sans-serif' },
                    animation: false,
                },
                title: { text: null },
                xAxis: {
                    categories: xCats,
                    opposite: true,
                    labels: {
                        useHTML: true,
                        allowOverlap: false,
                        style: { color: '#444444', fontSize: '9px' },
                        rotation: -60,
                        align: 'left',
                        formatter: function () {
                            const isAll = this.pos === xCats.length - 1;
                            const label = (this.value || '').replace(/<br>/g, ' ');
                            return isAll
                                ? '<b style="color:#111111; font-weight:bold">' + label + '</b>'
                                : label;
                        },
                    },
                    tickLength: 0,
                    lineWidth: 0,
                },
                yAxis: {
                    categories: yCats,
                    reversed: true,
                    title: null,
                    labels: {
                        style: { color: '#444444', fontSize: '9px' },
                        useHTML: true,
                        formatter: function () {
                            const isAll = this.pos === yCats.length - 1;
                            return isAll
                                ? '<b style="color:#111111; font-weight:bold">' + this.value + '</b>'
                                : this.value;
                        },
                    },
                    tickLength: 0,
                    gridLineWidth: 0,
                },
                colorAxis: {
                    min: 0,
                    max: maxVal,
                    stops: [
                        [0, '#ffffb2'],
                        [0.25, '#fecc5c'],
                        [0.5, '#fd8d3c'],
                        [0.75, '#f03b20'],
                        [1.0, '#bd0026'],
                    ],
                    nullColor: props.nullColor,
                    labels: { enabled: false },
                },
                legend: { enabled: false },
                tooltip: {
                    useHTML: true,
                    backgroundColor: '#ffffff',
                    borderColor: '#e0e0e0',
                    style: { color: '#333333' },
                    formatter: function () {
                        const ha = this.point.value;
                        if (ha === null || ha === undefined) return false;
                        const xLabel = (xCats[this.point.x] || '').replace(/<br>/g, ' ');
                        const yLabel = yCats[this.point.y] || '';
                        return '<b>From:</b> ' + yLabel + '<br>' +
                            '<b>To:</b> ' + xLabel + '<br>' +
                            '<b>Area:</b> ' + Highcharts.numberFormat(ha, 0) + ' ha';
                    },
                },
                series: [{
                    name: 'Transition Area (ha)',
                    borderWidth: 1,
                    borderColor: 'rgba(180,180,180,0.28)',
                    data: data,
                    dataLabels: {
                        enabled: true,
                        useHTML: true,
                        style: {
                            fontSize: '8px',
                            fontWeight: 'normal',
                            textOutline: 'none',
                            color: '#333333',
                        },
                        formatter: function () {
                            const ha = this.point.value;
                            if (ha == null || ha <= 0) return '';
                            const isAllCol = this.point.x === xCats.length - 1;
                            const isAllRow = this.point.y === yCats.length - 1;
                            const formatted = ha >= 1e6 ? (ha / 1e6).toFixed(1) + 'M'
                                : ha >= 1e3 ? (ha / 1e3).toFixed(1) + 'k'
                                    : String(ha);
                            return (isAllCol || isAllRow)
                                ? '<b>' + formatted + '</b>'
                                : formatted;
                        },
                    },
                }],
                credits: { enabled: false },
            };
        };

        // ------------------------------------------------------------------
        // Lifecycle
        // ------------------------------------------------------------------
        const createChart = () => {
            if (!chartEl.value) return;
            if (chartInstance) { chartInstance.destroy(); chartInstance = null; }
            if (!props.data || props.data.length === 0) return;
            chartInstance = Highcharts.chart(chartEl.value, buildConfig());
        };

        const handleResize = () => { createChart(); };

        onMounted(() => {
            createChart();
            window.addEventListener('resize', handleResize);
        });
        onUnmounted(() => {
            window.removeEventListener('resize', handleResize);
            if (chartInstance) { chartInstance.destroy(); chartInstance = null; }
        });

        // Re-render whenever any prop changes
        watch(
            () => [props.xCats, props.yCats, props.data, props.maxVal],
            () => { createChart(); },
            { deep: true }
        );

        return { chartEl };
    },

    template: /*html*/`<div ref="chartEl" style="width:100%;position:relative;z-index:0;"></div>`,
};
