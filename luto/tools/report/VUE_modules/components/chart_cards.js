// Chart Cards Element
// The cards of a Home panel (the targets of the run, or its inputs), stacked in a column beside a map:
// a title, the setting behind it, a small chart over the years, optional rows of text and a note for the
// selected year. A card that is switched off in the run is drawn grey with its `off` message. A card that
// names a `layer` can be clicked (emits `select` with the layer); the selected one gets a ring.
//
//   panels:   [{ key, title, accent (the colour of its metric), setting, note, unit, layer, off, empty,
//                sharedTooltip (false: the tooltip names one line),
//                sections: [{ label, text }] (paragraphs of text under a small heading each),
//                table: { columns, lines: [{ cells, tone ('under' | 'above') }], empty, hint } (scrolls inside the card),
//                stats: [{ label, value, hint, tone ('under' | 'above') }] (headline counts, side by side),
//                chartType ('column'), categories, categoryTitle (a count per category instead of lines over the years),
//                legend (false: the card has no series legend; the tooltip still names the series),
//                options, option (a picker in the header: emits `option` with the card's key and the choice),
//                series: [{ name, color, data, dashStyle }], rows: [{ label, value, off, header }],
//                drawer: { label, sections: [{ label, text }], settings?: [{ name, value, what }] } (a bar at the foot of the card that opens the
//                         paragraphs under it: where the numbers of the card come from) }]
//             (a `header` row is the heading of the rows under it)
//   year:     the selected year, marked on every chart
//   selected: the layer of the selected card

window.ChartCards = {
  name: 'ChartCards',
  props: {
    panels: {
      type: Array,
      required: true,
    },
    year: {
      type: Number,
      default: null,
    },
    selected: {
      type: String,
      default: '',
    },
    // The series legend inside the plot, in a corner where it covers no line (above the plot when no corner is free)
    legendInPlot: {
      type: Boolean,
      default: false,
    },
  },
  emits: ['select', 'option'],

  setup(props) {
    const { onMounted, onUnmounted, watch, nextTick, reactive } = Vue;

    const chartElements = {};          // panel key -> element
    const charts = {};                 // panel key -> Highcharts chart
    const setChartElement = (key, element) => { if (element) chartElements[key] = element; else delete chartElements[key]; };
    // A card with rows only (no `series` key at all) has no chart area
    const hasChart = (panel) => Array.isArray(panel.series);

    const destroyCharts = () => {
      Object.keys(charts).forEach(key => { charts[key].destroy(); delete charts[key]; });
    };

    // A compact number for an axis label or a tooltip: 65.2M, -1.4M, 12.5
    const compact = (value) => {
      const size = Math.abs(value);
      if (size >= 1e9) return (value / 1e9).toPrecision(3) + 'B';
      if (size >= 1e6) return (value / 1e6).toPrecision(3) + 'M';
      if (size >= 1e3) return (value / 1e3).toPrecision(3) + 'k';
      return String(+value.toPrecision(3));
    };

    // Where the legend covers none of the lines: the corner of the plot (top / bottom, left / right) with the
    // fewest line samples under a box of the legend's size (ties go to the first of top-left, top-right,
    // bottom-left, bottom-right) -- or, when even that corner has a line under it, a strip above the plot.
    const placeLegend = (chart) => {
      const legendBox = chart.legend?.group?.getBBox?.();
      if (!legendBox || !chart.plotWidth || !chart.plotHeight) return;
      const boxWidth = Math.min(1, (legendBox.width + 8) / chart.plotWidth);
      const boxHeight = Math.min(1, (legendBox.height + 6) / chart.plotHeight);
      const SAMPLES = 12;
      const samples = [];                // [across, down] of the plot, 0-1
      chart.series.forEach(series => {
        const points = series.points.filter(point => point.plotX != null && point.plotY != null);
        points.forEach((point, position) => {
          const next = points[position + 1] || point;
          for (let step = 0; step < SAMPLES; step++) {
            samples.push([
              (point.plotX + (next.plotX - point.plotX) * step / SAMPLES) / chart.plotWidth,
              (point.plotY + (next.plotY - point.plotY) * step / SAMPLES) / chart.plotHeight,
            ]);
          }
        });
      });
      const corners = [['top', 'left'], ['top', 'right'], ['bottom', 'left'], ['bottom', 'right']];
      const covered = ([vertical, horizontal]) => samples.filter(([across, down]) =>
        (horizontal === 'left' ? across <= boxWidth : across >= 1 - boxWidth) &&
        (vertical === 'top' ? down <= boxHeight : down >= 1 - boxHeight)).length;
      const [vertical, horizontal] = corners.reduce((best, corner) => covered(corner) < covered(best) ? corner : best);
      // No corner is free of lines: the legend takes a strip of its own above the plot instead
      if (covered([vertical, horizontal]) > 0) {
        chart.legend.update({ floating: false, align: 'left', verticalAlign: 'top', x: 0, y: -4, margin: 2, backgroundColor: 'transparent' });
        return;
      }
      chart.legend.update({
        align: horizontal, verticalAlign: vertical,
        x: horizontal === 'left' ? chart.plotLeft + 4 : -6,
        y: vertical === 'top' ? chart.plotTop : -(chart.chartHeight - chart.plotTop - chart.plotHeight) - 2,
      });
    };

    const drawCharts = () => {
      destroyCharts();
      props.panels.forEach(panel => {
        const element = chartElements[panel.key];
        if (panel.off || !element || !document.body.contains(element) || !panel.series || panel.series.length === 0) return;
        charts[panel.key] = Highcharts.chart(element, {
          chart: { type: panel.chartType || 'line', backgroundColor: 'transparent', spacing: [4, 4, 2, 2], animation: false },
          title: { text: null },
          credits: { enabled: false },
          exporting: { enabled: false },
          legend: props.legendInPlot && panel.legend !== false
            ? {
              enabled: true, floating: true, align: 'left', verticalAlign: 'top', layout: 'horizontal', x: 34, y: 0,       // moved by placeLegend
              padding: 2, margin: 0, itemDistance: 10, symbolWidth: 14, symbolHeight: 6,
              itemStyle: { fontSize: '10px', fontWeight: 'normal', color: '#505051' },
              backgroundColor: 'rgba(255, 255, 255, 0.75)',
            }
            : { enabled: false },
          // a card over the years marks the selected year; a card over categories (panel.categories: a count per band) has none
          xAxis: panel.categories
            ? { categories: panel.categories, tickLength: 0, labels: { style: { fontSize: '9px' }, rotation: 0, autoRotation: false, step: 1 },
                title: { text: panel.categoryTitle || null, style: { fontSize: '10px', color: '#6b7280' }, margin: 2 } }
            : {
              tickLength: 3,
              labels: { style: { fontSize: '10px' } },
              plotLines: props.year == null ? [] : [{ value: props.year, color: '#0ea5e9', width: 1.5, dashStyle: 'ShortDash', zIndex: 3 }],
            },
          yAxis: {
            title: { text: null },
            tickAmount: 3,
            allowDecimals: !panel.categories,
            gridLineColor: '#eef0f2',
            labels: { style: { fontSize: '10px' }, formatter() { return compact(this.value); } },
          },
          tooltip: {
            shared: panel.sharedTooltip !== false,
            outside: true,
            style: { fontSize: '11px' },
            pointFormatter() { return `<span style="color:${this.color}">●</span> ${this.series.name}: <b>${compact(this.y)}</b> ${panel.unit || ''}<br/>`; },
          },
          plotOptions: {
            series: {
              animation: false, marker: { enabled: true, radius: 2 }, lineWidth: 1.8, states: { inactive: { opacity: 1 } },
            },
            column: { pointPadding: 0.05, groupPadding: 0.05, borderWidth: 0, dataLabels: { enabled: true, style: { fontSize: '9px', fontWeight: 'normal', textOutline: 'none', color: '#505051' }, formatter() { return this.y || ''; } } },
          },
          series: panel.series.map(series => ({ name: series.name, color: series.color, dashStyle: series.dashStyle || 'Solid', data: series.data })),
        });
        if (props.legendInPlot && panel.legend !== false) placeLegend(charts[panel.key]);
      });
    };

    onMounted(() => { nextTick(drawCharts); window.addEventListener('resize', drawCharts); });
    onUnmounted(() => { window.removeEventListener('resize', drawCharts); destroyCharts(); });
    watch(() => [props.panels, props.year], () => nextTick(drawCharts), { deep: true });

    // The drawers that are open, by panel key. An open drawer frees the card of its height limit, so the
    // chart is drawn again at its new size.
    const openDrawers = reactive({});
    const toggleDrawer = (key) => { openDrawers[key] = !openDrawers[key]; nextTick(drawCharts); };
    // A drawer that lists settings has two tabs: the explanation, and the settings of the model behind the card
    const DRAWER_TABS = ['Explanation', 'Settings'];
    const drawerTabs = reactive({});          // panel key -> the tab shown ('Explanation' until another is picked)
    const drawerTab = (key) => drawerTabs[key] || DRAWER_TABS[0];
    const pickDrawerTab = (key, tab) => { drawerTabs[key] = tab; if (!openDrawers[key]) toggleDrawer(key); };

    const DEFAULT_ACCENT = '#0284c7';     // a card whose panel names no colour
    return { setChartElement, hasChart, openDrawers, toggleDrawer, DRAWER_TABS, drawerTab, pickDrawerTab, DEFAULT_ACCENT };
  },

  template: /*html*/`
    <div class="flex flex-col gap-2 min-h-full">
      <div
        v-for="panel in panels" :key="panel.key"
        @click="panel.layer && !panel.off && $emit('select', panel.layer)"
        :title="panel.layer ? 'Show this input on the map' : ''"
        class="min-h-0 rounded-[10px] px-3 py-2 flex flex-col border border-l-[5px]"
        :style="panel.off ? {} : { borderLeftColor: panel.accent || DEFAULT_ACCENT }"
        :class="[
          // a live card stands off the page: a border, an edge in its metric's colour (panel.accent) and a deeper shadow; an off card stays flat
          panel.off ? 'flex-none bg-[#eceef1] text-gray-400 border-gray-200 border-l-gray-300' : 'bg-white border-gray-300 shadow-lg',
          // a chart card grows into free space but never shrinks under a readable chart: a long card beside it scrolls
          // (with its drawer open the card is as tall as what it holds, and the chart keeps a fixed height)
          !panel.off && hasChart(panel) && !openDrawers[panel.key] ? 'flex-[1_0_200px] max-h-[300px]' : 'flex-none',
          panel.layer ? 'cursor-pointer hover:shadow-lg' : '',
          panel.layer && selected === panel.layer ? 'ring-2 ring-sky-500' : '',
        ]"
      >
        <div class="flex items-baseline justify-between gap-2 pb-1 mb-1 border-b" :class="panel.off ? 'border-transparent' : 'border-gray-200'">
          <span class="text-[1rem] font-bold leading-tight whitespace-nowrap" :class="panel.off ? 'text-gray-400' : ''" :style="panel.off ? {} : { color: panel.accent || DEFAULT_ACCENT }">{{ panel.title }}</span>
          <!-- A card of several things shown one at a time (panel.options) carries their picker -->
          <el-select v-if="panel.options && panel.options.length && !panel.off"
            :model-value="panel.option" @change="value => $emit('option', panel.key, value)" @click.stop
            size="small" filterable placement="bottom-end" class="min-w-0 flex-1 !max-w-[170px]" :title="panel.option">
            <el-option v-for="option in panel.options" :key="option" :label="option" :value="option" />
          </el-select>
          <span v-else class="text-[0.7rem] leading-tight text-right truncate" :class="panel.off ? 'text-gray-400' : 'text-gray-500'" :title="panel.setting">{{ panel.setting }}</span>
        </div>

        <!-- Switched off in this run -->
        <p v-if="panel.off" class="text-[0.75rem] text-gray-400 leading-tight mt-1">{{ panel.off }}</p>

        <template v-else>

        <!-- The series of the chart, as its legend -->
        <div v-if="!legendInPlot && panel.legend !== false" class="flex flex-wrap gap-x-3 leading-tight mt-0.5">
          <span v-for="series in panel.series" :key="series.name" class="flex items-center text-[0.7rem] text-[#505051]">
            <span class="inline-block w-[10px] h-[3px] mr-1" :style="{ backgroundColor: series.color }"></span>{{ series.name }}
          </span>
        </div>

        <div v-if="hasChart(panel)" class="relative" :class="openDrawers[panel.key] ? 'flex-none h-[190px]' : 'flex-1 min-h-[70px]'">
          <div :ref="element => setChartElement(panel.key, element)" class="absolute inset-0"></div>
          <p v-if="!panel.series || panel.series.length === 0" class="absolute inset-0 flex items-center justify-center text-[0.75rem] text-gray-400">
            {{ panel.empty || 'Not in this run' }}
          </p>
        </div>

        <!-- A table that scrolls inside the card, its header fixed: one line per target, the last column in the
             colour of how the line stands (under its target, or at / above it) -->
        <div v-if="panel.table" class="mt-1">
          <div v-if="panel.table.hint" class="text-[0.68rem] text-gray-500 leading-tight mb-1">{{ panel.table.hint }}</div>
          <div v-if="panel.table.lines.length" class="max-h-[230px] overflow-y-auto rounded border border-gray-200">
            <table class="w-full text-[0.72rem] leading-tight border-collapse">
              <thead>
                <tr>
                  <th v-for="(column, position) in panel.table.columns" :key="column"
                    class="sticky top-0 bg-[#f1f3f5] text-[0.65rem] font-semibold uppercase tracking-wide text-gray-600 px-1.5 py-1"
                    :class="position >= panel.table.columns.length - 2 ? 'text-right whitespace-nowrap' : 'text-left'">{{ column }}</th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="(line, row) in panel.table.lines" :key="row" class="border-t border-gray-100">
                  <td v-for="(cell, position) in line.cells" :key="position" class="px-1.5 py-[3px] align-top"
                    :class="[
                      position >= line.cells.length - 2 ? 'text-right whitespace-nowrap' : 'text-left',
                      position === line.cells.length - 1 && line.tone === 'under' ? 'text-amber-600 font-semibold' : '',
                      position === line.cells.length - 1 && line.tone === 'above' ? 'text-green-600 font-semibold' : '',
                    ]">{{ cell }}</td>
                </tr>
              </tbody>
            </table>
          </div>
          <p v-else class="text-[0.75rem] text-gray-400">{{ panel.table.empty }}</p>
        </div>

        <!-- Headline counts: a number each, with what it counts under it -->
        <div v-if="panel.stats && panel.stats.length" class="mt-1 grid gap-2" :style="{ gridTemplateColumns: 'repeat(' + panel.stats.length + ', minmax(0, 1fr))' }">
          <div v-for="stat in panel.stats" :key="stat.label" class="rounded bg-[#f6f7f9] px-2 py-1.5 text-center" :title="stat.hint || ''">
            <div class="text-[1.35rem] font-bold leading-tight truncate"
              :class="stat.tone === 'under' ? 'text-amber-600' : stat.tone === 'above' ? 'text-green-600' : 'text-[#1f1f1f]'">{{ stat.value }}</div>
            <div class="text-[0.68rem] text-[#505051] leading-tight">{{ stat.label }}</div>
            <div v-if="stat.hint" class="text-[0.62rem] text-gray-400 leading-tight truncate">{{ stat.hint }}</div>
          </div>
        </div>

        <!-- Paragraphs of text, a small heading over each (where an input comes from, how it was made) -->
        <div v-if="panel.sections && panel.sections.length" class="mt-1 flex flex-col gap-2">
          <div v-for="section in panel.sections" :key="section.label">
            <div class="text-[0.65rem] font-semibold uppercase tracking-wide text-gray-500">{{ section.label }}</div>
            <p class="text-[0.78rem] text-[#1f1f1f] leading-snug">{{ section.text }}</p>
          </div>
        </div>

        <!-- Rows of text: the parts of a target that have no chart -->
        <div v-if="panel.rows && panel.rows.length" class="mt-1 flex flex-col gap-[2px]">
          <div v-for="row in panel.rows" :key="(row.header ? 'group ' : '') + row.label" class="flex items-baseline justify-between gap-2 text-[0.72rem] leading-tight">
            <span v-if="row.header" class="text-[0.65rem] font-semibold uppercase tracking-wide text-gray-500 mt-1.5 border-b border-gray-200 w-full">{{ row.label }}</span>
            <span v-else :class="row.off ? 'text-gray-400' : 'text-[#1f1f1f] font-semibold'" class="whitespace-nowrap">{{ row.label }}</span>
            <span :class="row.off ? 'text-gray-400' : 'text-[#505051]'" class="text-right" :title="row.value">{{ row.value }}</span>
          </div>
        </div>

        <p v-if="panel.note" class="text-[0.72rem] text-[#505051] leading-tight truncate mt-0.5" :title="panel.note">{{ panel.note }}</p>

        </template>

        <!-- The drawer: a bar that opens the paragraphs on where the card's numbers come from (also under a card that is off) -->
        <div v-if="panel.drawer" class="mt-1 border-t" :class="panel.off ? 'border-gray-300' : 'border-gray-200'">
          <!-- A drawer with settings: its bar is the two tabs and the arrow (a tab opens the drawer on it, the
               arrow opens and closes it). Any other drawer: its label and the arrow. -->
          <div v-if="panel.drawer.settings" class="flex items-center justify-between gap-2 pt-1">
            <div class="flex gap-3">
              <button v-for="tab in DRAWER_TABS" :key="tab" @click.stop="pickDrawerTab(panel.key, tab)"
                class="pb-0.5 text-[0.72rem] font-semibold border-b-2"
                :class="openDrawers[panel.key] && drawerTab(panel.key) === tab ? 'border-sky-500 text-sky-600' : 'border-transparent text-gray-600 hover:text-sky-600'"
              >{{ tab }}</button>
            </div>
            <button @click.stop="toggleDrawer(panel.key)" :aria-expanded="!!openDrawers[panel.key]" :aria-label="panel.drawer.label"
              class="px-1 text-[0.6rem] text-gray-600 hover:text-sky-600 transition-transform" :class="openDrawers[panel.key] ? 'rotate-180' : ''">▼</button>
          </div>
          <button v-else @click.stop="toggleDrawer(panel.key)" :aria-expanded="!!openDrawers[panel.key]"
            class="w-full flex items-center justify-between gap-2 pt-1 text-[0.72rem] font-semibold text-gray-600 hover:text-sky-600">
            <span>{{ panel.drawer.label }}</span>
            <span class="text-[0.6rem] transition-transform" :class="openDrawers[panel.key] ? 'rotate-180' : ''">▼</span>
          </button>
          <div v-if="openDrawers[panel.key]" class="mt-1.5 flex flex-col gap-2">
            <template v-if="!panel.drawer.settings || drawerTab(panel.key) === 'Explanation'">
              <div v-for="section in panel.drawer.sections" :key="section.label">
                <div class="text-[0.65rem] font-semibold uppercase tracking-wide text-gray-500">{{ section.label }}</div>
                <p class="text-[0.78rem] leading-snug" :class="panel.off ? 'text-gray-500' : 'text-[#1f1f1f]'">{{ section.text }}</p>
              </div>
            </template>
            <template v-else>
              <!-- The settings of the model (settings.py) behind the card: the name, its value in this run, what it does -->
              <div v-for="setting in panel.drawer.settings" :key="setting.name">
                <div class="flex flex-wrap items-baseline gap-x-2">
                  <span class="font-mono text-[0.68rem] font-semibold text-gray-600">{{ setting.name }}</span>
                  <span class="font-mono text-[0.72rem] text-sky-700 break-all">{{ setting.value }}</span>
                </div>
                <p class="text-[0.75rem] leading-snug" :class="panel.off ? 'text-gray-500' : 'text-[#1f1f1f]'">{{ setting.what }}</p>
              </div>
            </template>
          </div>
        </div>
      </div>
    </div>
  `,
};
