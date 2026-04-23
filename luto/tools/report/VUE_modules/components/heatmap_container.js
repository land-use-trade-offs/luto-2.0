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
        showAxisLabels: { type: Boolean, default: true },
        showDataLabels: { type: Boolean, default: true },
        onCellClick: { type: Function, default: null },
        exportable: { type: Boolean, default: false },
        zoomable: { type: Boolean, default: false },
        draggable: { type: Boolean, default: false },
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
            const maxVal = props.maxVal || 0;

            // Fill every missing grid cell with a null point so the series
            // borderColor still draws a faint outline even for absent cells.
            const existingKeys = new Set(
                (props.data || []).map(p => Array.isArray(p) ? p[0] + ',' + p[1] : p.x + ',' + p.y)
            );
            const data = [...(props.data || [])];
            for (let yi = 0; yi < yCats.length; yi++) {
                for (let xi = 0; xi < xCats.length; xi++) {
                    if (!existingKeys.has(xi + ',' + yi)) {
                        data.push({ x: xi, y: yi, value: null });
                    }
                }
            }

            return {
                chart: {
                    type: 'heatmap',
                    backgroundColor: null,
                    marginTop: props.showAxisLabels ? 150 : 10,
                    marginBottom: props.showAxisLabels ? 30 : 5,
                    marginLeft: props.showAxisLabels ? 185 : 10,
                    marginRight: props.showAxisLabels ? 15 : 5,
                    height: props.showAxisLabels ? Math.max(500, yCats.length * 22 + 140) : '100%',
                    style: { fontFamily: 'sans-serif' },
                    animation: false,
                },

                title: { text: null },
                xAxis: {
                    categories: xCats,
                    opposite: true,
                    labels: {
                        enabled: props.showAxisLabels,
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
                        enabled: props.showAxisLabels,
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
                    backgroundColor: 'rgba(255,255,255,1)',
                    borderColor: '#e0e0e0',
                    borderWidth: 1,
                    shadow: true,
                    style: { color: '#333333', opacity: 1, zIndex: 9999 },
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
                plotOptions: {
                    series: {
                        cursor: props.onCellClick ? 'pointer' : 'default',
                        point: {
                            events: {
                                click: props.onCellClick
                                    ? function () { props.onCellClick({ xi: this.x, yi: this.y, value: this.value }); }
                                    : undefined,
                            },
                        },
                    },
                },
                series: [{
                    name: 'Transition Area (ha)',
                    borderWidth: 1,
                    borderColor: 'rgba(180,180,180,0.28)',
                    data: data,
                    dataLabels: {
                        enabled: props.showDataLabels,
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
                exporting: { enabled: props.exportable },
            };
        };

        // ------------------------------------------------------------------
        // CSS scale + drag (mirrors chart_container pattern)
        // ------------------------------------------------------------------
        const { ref: vRef } = Vue;
        const scale = vRef(1);
        const scaleStep = 0.1;
        const position = vRef({ x: 0, y: 0 });
        const isDragging = vRef(false);
        const dragStartPos = vRef({ x: 0, y: 0 });

        const startDrag = (e) => {
            if (!props.draggable) return;
            isDragging.value = true;
            dragStartPos.value = { x: e.clientX - position.value.x, y: e.clientY - position.value.y };
        };
        const onDrag = (e) => {
            if (!isDragging.value) return;
            position.value = { x: e.clientX - dragStartPos.value.x, y: e.clientY - dragStartPos.value.y };
        };
        const stopDrag = () => { isDragging.value = false; };

        const scaleUp = () => { if (props.zoomable) scale.value = Math.round((scale.value + scaleStep) * 100) / 100; };
        const scaleDown = () => { if (props.zoomable && scale.value > scaleStep) scale.value = Math.round((scale.value - scaleStep) * 100) / 100; };

        const handleWheel = (e) => {
            if (!props.zoomable) return;
            e.preventDefault();
            e.stopPropagation();
            if (e.deltaY < 0) scaleUp(); else scaleDown();
        };

        // ------------------------------------------------------------------
        // Lifecycle
        // ------------------------------------------------------------------
        const createChart = () => {
            if (!chartEl.value) return;
            if (chartInstance) { chartInstance.destroy(); chartInstance = null; }
            if (!props.data || props.data.length === 0) return;
            chartInstance = Highcharts.chart(chartEl.value, buildConfig());
            // Lower z-index of HTML axis label containers so the tooltip always renders above them
            chartEl.value.querySelectorAll('.highcharts-axis-labels').forEach(el => {
                el.style.zIndex = '1';
            });
        };

        const handleResize = () => { createChart(); };

        onMounted(() => {
            createChart();
            window.addEventListener('resize', handleResize);
            window.addEventListener('mousemove', onDrag);
            window.addEventListener('mouseup', stopDrag);
        });
        onUnmounted(() => {
            window.removeEventListener('resize', handleResize);
            window.removeEventListener('mousemove', onDrag);
            window.removeEventListener('mouseup', stopDrag);
            if (chartInstance) { chartInstance.destroy(); chartInstance = null; }
        });

        // Re-render whenever any prop changes
        watch(
            () => [props.xCats, props.yCats, props.data, props.maxVal, props.showAxisLabels, props.showDataLabels, props.zoomable, props.exportable],
            () => { createChart(); },
            { deep: true }
        );

        return { chartEl, scale, scaleUp, scaleDown, handleWheel, position, startDrag };
    },

    template: /*html*/`
        <div style="width:100%;position:relative;z-index:0;"
            @wheel="handleWheel">
            <div
                ref="chartEl"
                :style="{
                    width: '100%',
                    height: '100%',
                    transformOrigin: 'top left',
                    transform: 'translate(' + position.x + 'px, ' + position.y + 'px) scale(' + scale + ')',
                    cursor: draggable ? 'move' : 'default',
                }"
                @mousedown="startDrag">
            </div>
            <div v-if="zoomable" style="position:absolute;top:8px;right:8px;display:flex;flex-direction:column;gap:4px;">
                <button @click="scaleUp"
                    style="background:rgba(255,255,255,0.85);border:1px solid #ccc;border-radius:50%;width:28px;height:28px;font-size:16px;line-height:1;cursor:pointer;display:flex;align-items:center;justify-content:center;">+</button>
                <button @click="scaleDown"
                    style="background:rgba(255,255,255,0.85);border:1px solid #ccc;border-radius:50%;width:28px;height:28px;font-size:16px;line-height:1;cursor:pointer;display:flex;align-items:center;justify-content:center;">−</button>
            </div>
        </div>
    `,
};
