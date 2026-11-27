// Home: the run at a glance, a map with a tabbed panel beside it, in one of two views (the View buttons):
//   Simulation: the simulated land use; the targets of the run, national and regional.
//   Inputs:     an input of the run as a map layer, with where it comes from and how it was made.

// The year stepper laid over a map (both maps share the year)
const HomeYearControl = {
  name: 'HomeYearControl',
  props: { years: { type: Array, required: true }, index: { type: Number, required: true } },
  emits: ['update:index'],
  template: /*html*/`
    <div class="flex items-center gap-2 pointer-events-auto w-fit">
      <span class="text-[0.8rem] font-semibold uppercase tracking-wide text-gray-700 px-1 py-0.5 w-[4rem]">Year</span>
      <div class="flex items-center gap-1 bg-white/80 rounded px-1 py-1 border border-transparent">
      <button @click="$emit('update:index', Math.max(0, index - 1))" :disabled="index === 0"
        class="w-5 h-5 rounded bg-[#f1f3f5] hover:bg-sky-100 disabled:opacity-30 text-[#1f1f1f] text-[0.65rem]" title="Previous year">◀</button>
      <span class="text-[0.85rem] leading-[1.25rem] font-bold text-[#1f1f1f] w-[2.6rem] text-center">{{ years[index] }}</span>
      <button @click="$emit('update:index', Math.min(years.length - 1, index + 1))" :disabled="index === years.length - 1"
        class="w-5 h-5 rounded bg-[#f1f3f5] hover:bg-sky-100 disabled:opacity-30 text-[#1f1f1f] text-[0.65rem]" title="Next year">▶</button>
      <el-slider
        v-if="years.length > 1"
        class="w-[130px] mx-3"
        style="--el-slider-button-size: 10px; --el-slider-height: 4px; --el-slider-button-wrapper-size: 20px; --el-slider-button-wrapper-offset: -8px; --el-slider-stop-bg-color: #cbd5e1;"
        :model-value="index"
        size="small"
        :min="0"
        :max="years.length - 1"
        :step="1"
        :format-tooltip="position => years[position]"
        :show-stops="true"
        @input="(position) => $emit('update:index', position)"
      />
      </div>
    </div>
  `,
};

// The from-to cost table of the model, drawn over the map (it has no map of its own): a heatmap, the land use a
// cell leaves down the side, the one it moves to across the top. A row of buttons picks the total or one of the
// parts it is the sum of; every part reads on the colour scale of the total. A change the model does not allow is
// a grey cell with a cross, in every part.
// With `allowedOnly` it is the exclusion matrix instead: no costs and no buttons, every pair either allowed (green)
// or not (grey, with a cross).
//   matrix: { lus, n_ag, parts: ['Total', ...], cost: { part: [[from, to, AUD/ha]] }, disallowed: [[from, to]] }
const HomeTransitionMatrix = {
  name: 'HomeTransitionMatrix',
  props: { matrix: { type: Object, required: true }, allowedOnly: { type: Boolean, default: false } },
  setup(props) {
    const { ref, computed, onMounted, onUnmounted, watch, nextTick } = Vue;
    const chartElement = ref(null);
    const selectPart = ref('Total');
    let chart = null;
    const parts = computed(() => props.matrix.parts || []);
    const DISALLOWED = '#e5e7eb';
    const ALLOWED = '#cfe8d5';
    // A long land-use name over the columns goes on two lines, broken at the space nearest its middle
    const WRAP_FROM = 24;
    const wrapped = (name) => {
      if (name.length <= WRAP_FROM) return name;
      const spaces = [...name].map((letter, at) => letter === ' ' ? at : -1).filter(at => at > 0);
      if (!spaces.length) return name;
      const cut = spaces.reduce((best, at) => Math.abs(at - name.length / 2) < Math.abs(best - name.length / 2) ? at : best);
      return name.slice(0, cut) + '<br/>' + name.slice(cut + 1);
    };
    const thousands = (value) => value >= 1000 ? +(value / 1000).toFixed(value >= 10000 ? 0 : 1) + 'k' : String(value);

    const draw = () => {
      if (chart) { chart.destroy(); chart = null; }
      const lus = props.matrix.lus || [];
      if (!chartElement.value || !lus.length) return;
      const ceiling = Math.max(1, ...(props.matrix.cost['Total'] || []).map(cell => cell[2]));
      const crosses = (props.matrix.disallowed || []).map(([from, to]) => ({ x: to, y: from, value: null, color: DISALLOWED, notAllowed: true }));
      // the exclusion matrix: every pair that is not crossed out is allowed (a land use staying as it is, too)
      const crossed = new Set(crosses.map(cell => cell.x + ',' + cell.y));
      const cells = props.allowedOnly
        ? lus.flatMap((_, from) => lus.map((__, to) => ({ x: to, y: from, value: null, color: ALLOWED, isAllowed: true }))).filter(cell => !crossed.has(cell.x + ',' + cell.y))
        : (props.matrix.cost[selectPart.value] || []).map(([from, to, cost]) => ({ x: to, y: from, value: cost }));
      const edge = props.matrix.n_ag - 0.5;                 // between the agricultural and the non-agricultural land uses
      chart = Highcharts.chart(chartElement.value, {
        chart: { type: 'heatmap', backgroundColor: 'rgba(255,255,255,0.94)', animation: false, marginTop: 122, marginLeft: 262, marginRight: 8, marginBottom: 8, style: { fontFamily: 'sans-serif' } },
        title: { text: null }, credits: { enabled: false }, exporting: { enabled: false }, legend: { enabled: false },
        xAxis: {
          categories: lus, opposite: true, tickLength: 0, lineWidth: 0,
          // upright, so a name on two lines stays inside its own column
          labels: { rotation: -90, align: 'left', step: 1, y: -4, style: { fontSize: '8px', lineHeight: '8px', color: '#444444', textOverflow: 'none' }, formatter() { return wrapped(String(this.value)); } },
          plotLines: [{ value: edge, color: '#6b7280', width: 1, zIndex: 5 }],
        },
        yAxis: {
          categories: lus, reversed: true, title: null, tickLength: 0, gridLineWidth: 0,
          labels: { step: 1, style: { fontSize: '9px', color: '#444444', whiteSpace: 'nowrap', textOverflow: 'none' } },
          plotLines: [{ value: edge, color: '#6b7280', width: 1, zIndex: 5 }],
        },
        colorAxis: { min: 0, max: ceiling, stops: [[0, '#ffffff'], [0.02, '#f1f6fb'], [0.35, '#cfe0f1'], [0.7, '#9dbfe0'], [1, '#6b9bcf']] },   // a quiet blue: the numbers in the cells carry the detail
        tooltip: {
          outside: true, useHTML: true, style: { fontSize: '11px' },
          formatter() {
            const head = `<b>From:</b> ${lus[this.point.y]}<br><b>To:</b> ${lus[this.point.x]}<br>`;
            if (this.point.notAllowed) return head + 'Not allowed in the model';
            return this.point.isAllowed ? head + 'Allowed' : head + `<b>${selectPart.value}:</b> ${Highcharts.numberFormat(this.point.value, 0)} AUD/ha`;
          },
        },
        series: [{
          borderWidth: 1, borderColor: 'rgba(160,160,160,0.3)', nullColor: '#ffffff', turboThreshold: 0,
          data: [...cells, ...crosses],
          dataLabels: {
            enabled: true, style: { fontSize: '7px', fontWeight: 'normal', textOutline: 'none', color: '#333333' },
            formatter() { return this.point.notAllowed ? '<span style="color:#b6bcc6">×</span>' : this.point.value ? thousands(this.point.value) : ''; },
          },
        }],
      });
    };
    const redraw = () => nextTick(draw);
    onMounted(() => { redraw(); window.addEventListener('resize', redraw); });
    onUnmounted(() => { window.removeEventListener('resize', redraw); if (chart) chart.destroy(); });
    watch(() => [props.matrix, selectPart.value, props.allowedOnly], redraw);
    watch(parts, (names) => { if (!names.includes(selectPart.value)) selectPart.value = names[0] ?? 'Total'; });
    return { chartElement, selectPart, parts, DISALLOWED, ALLOWED };
  },
  template: /*html*/`
    <div class="h-full flex flex-col gap-1">
      <div class="flex flex-wrap items-center gap-1 bg-white/90 rounded px-2 py-1">
        <template v-if="!allowedOnly">
          <span class="text-[0.7rem] font-semibold uppercase tracking-wide text-gray-500 mr-1">Cost</span>
          <button v-for="part in parts" :key="part" @click="selectPart = part"
            class="text-[0.75rem] px-2 py-0.5 rounded"
            :class="selectPart === part ? 'bg-sky-500 text-white' : 'bg-[#f1f3f5] text-[#1f1f1f] hover:bg-sky-100'">{{ part }}</button>
        </template>
        <span v-else class="flex items-center text-[0.7rem] text-gray-600">
          <span class="text-[0.7rem] font-semibold uppercase tracking-wide text-gray-500 mr-2">Which land use may follow which</span>
          <span class="inline-block w-[12px] h-[12px] mr-1 border border-gray-300" :style="{ backgroundColor: ALLOWED }"></span>allowed
        </span>
        <span class="ml-auto flex items-center text-[0.7rem] text-gray-600">
          <span class="inline-block w-[12px] h-[12px] mr-1 text-center leading-[12px] text-[9px] text-[#9ca3af] border border-gray-300" :style="{ backgroundColor: DISALLOWED }">×</span>not allowed in the model
          <span class="ml-3">{{ allowedOnly ? '' : 'AUD per hectare · ' }}rows: from, columns: to</span>
        </span>
      </div>
      <div ref="chartElement" class="flex-1 min-h-0 rounded overflow-hidden"></div>
    </div>
  `,
};

// A table of the inputs, drawn over the map (it has no map of its own): one bar per row, the rows under the
// heading of their group. Used for the habitat contribution (HCAS) of every land use.
//   table: { unit, rows: [[row, value, group]] }
const HomeTableBars = {
  name: 'HomeTableBars',
  props: { table: { type: Object, required: true }, title: { type: String, default: '' } },
  setup(props) {
    const { computed } = Vue;
    const top = computed(() => Math.max(1e-9, ...(props.table.rows || []).map(row => Number(row[1]) || 0), ...Object.values(props.table.reference || {}).map(Number)));
    // the rows, a heading row wherever the group changes
    const lines = computed(() => (props.table.rows || []).flatMap(([label, value, group], position, rows) => [
      ...(group && group !== rows[position - 1]?.[2] ? [{ heading: group, key: 'group ' + group }] : []),
      { label, value: Number(value), width: (Number(value) / top.value * 100).toFixed(1) + '%', key: group + ' ' + label,
        // the reference value of the row (table.reference), where it has one: a short line across the bar
        mark: props.table.reference?.[label] == null ? null : Math.min(100, Number(props.table.reference[label]) / top.value * 100).toFixed(1) + '%' },
    ]));
    return { lines };
  },
  template: /*html*/`
    <div class="h-full flex flex-col bg-white/95 rounded px-4 py-3 overflow-hidden">
      <div class="flex items-baseline justify-between mb-2">
        <span class="font-bold text-[0.9rem] text-gray-700">{{ title }}</span>
        <span class="text-[0.7rem] text-gray-500 flex items-center gap-1">
          <template v-if="table.referenceName"><span class="inline-block w-[2px] h-[11px] bg-[#1f1f1f]"></span><span class="mr-3">{{ table.referenceName }}</span></template>
          {{ table.unit }}
        </span>
      </div>
      <div class="flex-1 min-h-0 overflow-y-auto pr-2 columns-2 gap-8">
        <template v-for="line in lines" :key="line.key">
          <div v-if="line.heading" class="text-[0.65rem] font-semibold uppercase tracking-wide text-gray-500 mt-2 mb-0.5 border-b border-gray-200 break-after-avoid">{{ line.heading }}</div>
          <div v-else class="flex items-center gap-2 text-[0.75rem] leading-[1.35rem] break-inside-avoid">
            <span class="w-[46%] truncate text-[#1f1f1f]" :title="line.label">{{ line.label }}</span>
            <span class="relative flex-1 h-[9px] bg-[#f1f3f5] rounded-sm">
              <span class="block h-full bg-[#6fa58a] rounded-sm" :style="{ width: line.width }"></span>
              <span v-if="line.mark" class="absolute top-[-2px] h-[13px] w-[2px] bg-[#1f1f1f]" :style="{ left: line.mark }"></span>
            </span>
            <span class="w-[34px] text-right text-[#505051] tabular-nums">{{ line.value.toFixed(2) }}</span>
          </div>
        </template>
      </div>
    </div>
  `,
};

// The yield trend, drawn over the map (it has no map of its own): the two inputs the model can run on, one chart
// each, the one this run does not use in grey.
//   trend: { BAU: { BAU: [...] }, ABARES: { history: [...], low: [...], medium: [...], high: [...], very_high: [...] } },
//          each entry { group, lm, kind, data: [[year, multiplier on 2010]] }
//   used:  settings.PRODUCTIVITY_TREND ('BAU', 'LOW', 'MEDIUM', 'HIGH', 'VERY_HIGH')
const HomeTrendCharts = {
  name: 'HomeTrendCharts',
  props: { trend: { type: Object, required: true }, used: { type: String, default: 'BAU' }, year: { type: Number, default: null } },
  setup(props) {
    const { ref, computed, onMounted, onUnmounted, watch, nextTick } = Vue;
    const LAST_YEAR = 2050;
    const GROUP_COLORS = { 'Crops': '#ca8a04', 'Beef': '#b91c1c', 'Dairy': '#2563eb', 'Sheep': '#16a34a', 'Sheep wool': '#7c3aed' };
    const UNUSED = '#b6bac2';
    const SCENARIO_LABELS = { 'low': 'low', 'medium': 'medium', 'high': 'high', 'very_high': 'very high' };
    const bauElement = ref(null), abaresElement = ref(null);
    const setElement = (source, element) => { (source === 'BAU' ? bauElement : abaresElement).value = element; };
    const charts = [];

    const usedSource = computed(() => String(props.used).toUpperCase() === 'BAU' ? 'BAU' : 'ABARES');
    const usedScenario = computed(() => String(props.used).toLowerCase());
    // the groups a source has a series for, in the order of GROUP_COLORS
    const groupsOf = (source) => Object.keys(GROUP_COLORS).filter(group =>
      Object.values(props.trend?.[source] || {}).some(entries => entries.some(entry => entry.group === group)));
    const upTo = (points) => points.filter(point => point[0] <= LAST_YEAR);

    const baseOptions = (series) => ({
      chart: { backgroundColor: 'transparent', spacing: [6, 8, 4, 4], animation: false },
      title: { text: null }, credits: { enabled: false }, exporting: { enabled: false }, legend: { enabled: false },
      xAxis: {
        tickLength: 3, labels: { style: { fontSize: '10px' } },
        plotLines: props.year == null ? [] : [{ value: props.year, color: '#0ea5e9', width: 1.5, dashStyle: 'ShortDash', zIndex: 3 }],
      },
      yAxis: { title: { text: 'multiplier on 2010', style: { fontSize: '10px', color: '#6b7280' } }, gridLineColor: '#eef0f2', labels: { style: { fontSize: '10px' } },
        plotLines: [{ value: 1, color: '#9ca3af', width: 1, zIndex: 1 }] },
      tooltip: { style: { fontSize: '11px' }, headerFormat: '<b>{point.x}</b><br/>', pointFormat: '<span style="color:{point.color}">●</span> {series.name}: <b>{point.y:.3f}</b>' },
      plotOptions: { series: { animation: false, states: { inactive: { opacity: 1 } }, stickyTracking: false } },
      series,
    });

    const draw = () => {
      charts.splice(0).forEach(chart => chart.destroy());
      const trend = props.trend || {};

      // BAU: one projection, a line per group; irrigated dashed where it differs from dryland
      if (bauElement.value && trend['BAU']) {
        const inUse = usedSource.value === 'BAU';
        charts.push(Highcharts.chart(bauElement.value, baseOptions((trend['BAU']['BAU'] || []).map(entry => ({
          type: 'line', name: entry.group + (entry.lm === 'ALL' ? '' : ' · ' + entry.lm.toLowerCase()),
          color: inUse ? GROUP_COLORS[entry.group] : UNUSED, dashStyle: entry.lm === 'Irrigated' ? 'ShortDash' : 'Solid',
          lineWidth: 1.8, marker: { enabled: false }, data: upTo(entry.data),
        })))));
      }

      // ABARES: the observed productivity as dots, then the four scenarios; the scenario in use in colour
      if (abaresElement.value && trend['ABARES']) {
        const inUse = usedSource.value === 'ABARES';
        const series = [];
        (trend['ABARES']['history'] || []).forEach(entry => series.push({
          type: 'line', name: entry.group + ' · observed', color: inUse ? GROUP_COLORS[entry.group] : UNUSED,
          lineWidth: 0, states: { hover: { lineWidthPlus: 0 } }, marker: { enabled: true, radius: 2.2, symbol: 'circle' }, data: upTo(entry.data), zIndex: 3,
        }));
        Object.keys(SCENARIO_LABELS).forEach(scenario => (trend['ABARES'][scenario] || []).forEach(entry => {
          const isRunScenario = inUse && scenario === usedScenario.value;
          series.push({
            type: 'line', name: `${entry.group} · ${SCENARIO_LABELS[scenario]}`,
            color: isRunScenario ? GROUP_COLORS[entry.group] : UNUSED, lineWidth: isRunScenario ? 2.2 : 1, opacity: isRunScenario || !inUse ? 1 : 0.7,
            marker: { enabled: false }, data: upTo(entry.data), zIndex: isRunScenario ? 4 : 2,
          });
        }));
        charts.push(Highcharts.chart(abaresElement.value, baseOptions(series)));
      }
    };

    onMounted(() => { nextTick(draw); window.addEventListener('resize', draw); });
    onUnmounted(() => { window.removeEventListener('resize', draw); charts.splice(0).forEach(chart => chart.destroy()); });
    watch(() => [props.trend, props.used, props.year], () => nextTick(draw), { deep: true });

    return { setElement, usedSource, usedScenario, groupsOf, GROUP_COLORS, UNUSED, SCENARIO_LABELS };
  },
  template: /*html*/`
    <div class="flex flex-col gap-3 h-full">
      <div v-for="source in ['BAU', 'ABARES']" :key="source"
        class="flex-1 min-h-0 flex flex-col rounded-[10px] bg-white/95 shadow-md px-3 py-2 border"
        :class="usedSource === source ? 'border-gray-300' : 'border-gray-200'">
        <div class="flex items-baseline justify-between gap-2">
          <span class="text-[0.9rem] font-bold" :class="usedSource === source ? 'text-[#1f1f1f]' : 'text-gray-400'">
            {{ source === 'BAU' ? 'BAU projection' : 'ABARES: observed, then scenarios' }}
          </span>
          <span class="text-[0.7rem] px-1.5 py-[1px] rounded whitespace-nowrap"
            :class="usedSource === source ? 'bg-sky-500 text-white' : 'bg-gray-200 text-gray-500'">
            {{ usedSource === source ? (source === 'BAU' ? 'used by this run' : 'used by this run: ' + (SCENARIO_LABELS[usedScenario] || usedScenario)) : 'not used by this run' }}
          </span>
        </div>
        <div class="flex flex-wrap gap-x-3 text-[0.7rem] text-[#505051] leading-tight mt-0.5">
          <span v-for="group in groupsOf(source)" :key="group" class="flex items-center">
            <span class="inline-block w-[10px] h-[3px] mr-1" :style="{ backgroundColor: usedSource === source ? GROUP_COLORS[group] : UNUSED }"></span>{{ group }}
          </span>
          <span class="text-gray-400">{{ source === 'BAU' ? 'solid: dryland · dashed: irrigated' : 'dots: observed · lines: low, medium, high, very high' }}</span>
        </div>
        <div class="flex-1 min-h-0 relative">
          <div :ref="element => setElement(source, element)" class="absolute inset-0"></div>
        </div>
      </div>
    </div>
  `,
};

// What an ag-management option changes on its land uses over the years: one chart, a line per land-use group, laid
// over the map (the bundles have no map). `series` is [{ name, color, data: [[year, value]] }].
const HomeBundleChart = {
  name: 'HomeBundleChart',
  props: { title: { type: String, default: '' }, variable: { type: String, default: '' }, unit: { type: String, default: '' },
           note: { type: String, default: '' }, series: { type: Array, required: true }, year: { type: Number, default: null },
           variables: { type: Array, default: () => [] } },             // what the practice changes: the buttons over the chart
  emits: ['update:variable'],
  setup(props) {
    const { ref, onMounted, onUnmounted, watch, nextTick } = Vue;
    const LAST_YEAR = 2050;                     // the years the model can run to
    const element = ref(null);
    let chart = null;
    const draw = () => {
      if (chart) { chart.destroy(); chart = null; }
      if (!element.value) return;
      chart = Highcharts.chart(element.value, {
        chart: { backgroundColor: 'transparent', spacing: [8, 10, 6, 6], animation: false },
        title: { text: null }, credits: { enabled: false }, exporting: { enabled: false }, legend: { enabled: false },
        xAxis: { tickLength: 3, labels: { style: { fontSize: '11px' } },
          plotLines: props.year == null ? [] : [{ value: props.year, color: '#0ea5e9', width: 1.5, dashStyle: 'ShortDash', zIndex: 3 }] },
        yAxis: { title: { text: props.unit, style: { fontSize: '11px', color: '#6b7280' } }, gridLineColor: '#eef0f2', labels: { style: { fontSize: '11px' } } },
        tooltip: { shared: true, style: { fontSize: '11px' }, headerFormat: '<b>{point.x}</b><br/>', pointFormat: '<span style="color:{point.color}">●</span> {series.name}: <b>{point.y:.3f}</b><br/>' },
        plotOptions: { series: { animation: false, states: { inactive: { opacity: 1 } }, marker: { enabled: false }, lineWidth: 2 } },
        series: props.series.map(line => ({ type: 'line', name: line.name, color: line.color, data: line.data.filter(point => point[0] <= LAST_YEAR) })),
      });
    };
    onMounted(() => { nextTick(draw); window.addEventListener('resize', draw); });
    onUnmounted(() => { window.removeEventListener('resize', draw); if (chart) chart.destroy(); });
    watch(() => [props.series, props.year, props.unit], () => nextTick(draw), { deep: true });
    return { element };
  },
  template: /*html*/`
    <div class="h-full flex flex-col bg-white/95 rounded-[10px] shadow-md px-4 py-3">
      <div class="flex items-baseline justify-between gap-3">
        <span class="font-bold text-[0.9rem] text-gray-700">{{ title }} · {{ variable }}</span>
        <span class="text-[0.7rem] text-gray-500">{{ note }}</span>
      </div>
      <div class="flex flex-wrap items-center gap-1 mt-1.5 mb-1">
        <span class="text-[0.7rem] font-semibold uppercase tracking-wide text-gray-500 mr-1">What it changes</span>
        <button v-for="name in variables" :key="name" @click="$emit('update:variable', name)"
          class="text-[0.75rem] px-2 py-0.5 rounded"
          :class="variable === name ? 'bg-sky-500 text-white' : 'bg-[#f1f3f5] text-[#1f1f1f] hover:bg-sky-100'"
        >{{ name }}</button>
      </div>
      <div class="flex flex-wrap gap-x-4 text-[0.72rem] text-[#505051] leading-tight mt-1">
        <span v-for="line in series" :key="line.name" class="flex items-center">
          <span class="inline-block w-[12px] h-[3px] mr-1" :style="{ backgroundColor: line.color }"></span>{{ line.name }}
        </span>
        <span v-if="!series.length" class="text-gray-400">Nothing in this bundle</span>
      </div>
      <div class="flex-1 min-h-0 relative mt-1"><div ref="element" class="absolute inset-0"></div></div>
    </div>
  `,
};

// The assumptions of the model: settings that are not input data but decide how the model reads and uses it. Laid
// over the map (they have none). `groups` is [{ name, items: [{ setting, what, detail, options: [{ value, meaning,
// ofRun }], value }] }]: a setting with named options shows them all, the one of the run with the dashed border (as
// the option buttons of the inputs); any other setting shows its value in the run.
const HomeAssumptions = {
  name: 'HomeAssumptions',
  props: { groups: { type: Array, required: true } },
  template: /*html*/`
    <div class="h-full flex flex-col bg-white/95 rounded-[10px] shadow-md px-4 py-3 overflow-hidden">
      <div class="flex items-baseline justify-between gap-3 mb-2">
        <span class="font-bold text-[0.9rem] text-gray-700">Assumptions of the model</span>
        <span class="flex items-center gap-1 text-gray-500 text-[0.65rem]">
          <span class="inline-block px-1 py-0.5 rounded border border-dashed border-sky-700 leading-none">abc</span>
          <span>= user-selected option of this run</span>
        </span>
      </div>
      <!-- One table: a row per setting (what it decides, its options with the run's marked, what the run's option
           means), under a heading row per group; the column heads stay in view while the rows scroll -->
      <div class="flex-1 min-h-0 overflow-y-auto pr-1">
        <table class="w-full border-collapse text-[0.75rem]">
          <colgroup><col class="w-[34%]"><col class="w-[30%]"><col class="w-[36%]"></colgroup>
          <thead class="sticky top-0 bg-white z-10">
            <tr class="text-left text-[0.65rem] uppercase tracking-wide text-gray-500">
              <th class="font-semibold py-1 pr-3 border-b border-gray-300">Assumption</th>
              <th class="font-semibold py-1 pr-3 border-b border-gray-300">Options</th>
              <th class="font-semibold py-1 border-b border-gray-300">In this run</th>
            </tr>
          </thead>
          <tbody>
            <template v-for="group in groups" :key="group.name">
              <tr><td colspan="3" class="pt-3 pb-1 text-[0.68rem] font-bold uppercase tracking-wide text-sky-700 bg-sky-50/60 px-2">{{ group.name }}</td></tr>
              <!-- the rows alternate white and a faint grey, to follow a row across the three columns -->
              <tr v-for="(item, position) in group.items" :key="item.setting"
                class="group align-top border-b border-gray-100 transition-colors duration-150 hover:!bg-sky-100"
                :class="position % 2 ? 'bg-[#f6f7f9]' : 'bg-white'">
                <!-- on hover the row turns light blue, with a bar at its left edge -->
                <td class="py-1.5 pr-3 pl-2 border-l-[3px] border-transparent group-hover:border-sky-500">
                  <div class="font-semibold text-[#1f1f1f] leading-snug group-hover:text-sky-800">{{ item.what }}</div>
                  <div class="font-mono text-[0.62rem] text-gray-400">{{ item.setting }}</div>
                </td>
                <td class="py-1.5 pr-3">
                  <div class="flex flex-wrap gap-1">
                    <template v-if="item.options.length">
                      <span v-for="option in item.options" :key="option.value" :title="option.meaning"
                        class="px-1.5 py-0.5 rounded border bg-[#e9ecef] text-[#1f1f1f]"
                        :class="option.ofRun ? 'border-dashed border-sky-700' : 'border-transparent text-gray-500'">{{ option.value }}</span>
                    </template>
                    <span v-else class="px-1.5 py-0.5 rounded border border-dashed border-sky-700 bg-[#e9ecef] text-[#1f1f1f] break-all max-h-[4.6rem] overflow-y-auto">{{ item.value }}</span>
                  </div>
                </td>
                <td class="py-1.5 text-[#505051] leading-snug">{{ item.meaning }}</td>
              </tr>
            </template>
          </tbody>
        </table>
      </div>
    </div>
  `,
};

// The tab strip over a panel
const HomeTabs = {
  name: 'HomeTabs',
  props: { tabs: { type: Array, required: true }, selected: { type: String, required: true } },
  emits: ['update:selected'],
  template: /*html*/`
    <div class="flex shrink-0 border-b border-gray-300 mb-2">
      <button
        v-for="tab in tabs" :key="tab"
        @click="$emit('update:selected', tab)"
        class="flex-1 text-[0.85rem] font-semibold px-2 py-1.5 -mb-px border-b-2"
        :class="selected === tab ? 'border-sky-500 text-sky-600' : 'border-transparent text-[#505051] hover:text-sky-600'"
      >{{ tab }}</button>
    </div>
  `,
};

window.HomeView = {
  name: 'HomeView',
  components: { HomeYearControl, HomeTabs, HomeTrendCharts, HomeBundleChart, HomeAssumptions, HomeTransitionMatrix, HomeTableBars },

  setup() {

    const { ref, reactive, onMounted, onUnmounted, watch, computed, inject, nextTick } = Vue;

    // Data service
    const chartRegister = window.ChartService.chartCategories;   // ChartService has been registered in index.html      [ChartService.js]
    const loadScript = window.loadScriptWithTracking;
    const VIEW_NAME = "Home";

    // Global variables
    const selectRegion = inject('globalSelectedRegion');

    // The maps pick a region at one of two levels
    // 'Water' is the water regions the run sets its limits for (river regions or drainage divisions): only the
    // Water card has data at that level, every other card then shows the country
    const REGION_LEVELS = ['NRM', 'State', 'Water'];
    const WATER_GEO = {
      'River Region': { path: './data/geo/WATER_RIVREG.js', name: 'WATER_RIVREG', many: 'river regions' },
      'Drainage Division': { path: './data/geo/WATER_DRAINDIV.js', name: 'WATER_DRAINDIV', many: 'drainage divisions' },
    };
    const regionLevel = ref('NRM');

    // What the map and the panel show
    // The colour of each metric: the edge and the title of its cards (the module cards use the same hues)
    const METRIC_COLORS = {
      'GHG': '#64748b', 'Water': '#2563eb', 'Biodiversity': '#16a34a', 'Demand': '#ea580c',
      'Renewable energy': '#ca8a04', 'Social licence': '#7c3aed', 'Economics': '#b45309',
      'Yield': '#65a30d', 'Exclusion': '#be123c', 'Ag management': '#0d9488',
    };
    const VIEWS = ['Simulation', 'Inputs'];
    const VIEW_LABELS = { 'Simulation': 'Simulation + targets', 'Inputs': 'Inputs + assumptions' };
    const selectView = ref('Simulation');
    const showLabels = ref(true);

    const runScenario = ref({});
    const dataLoaded = ref(false);
    // The years of the slider. The Simulation view steps through the years of the run; the Inputs view through the
    // years the input cache holds (every ten years, Inputs_years.js), each view keeping its own position.
    const runYears = ref([]);
    const inputYears = ref([]);
    const runYearIndex = ref(0);
    const inputYearIndex = ref(0);
    const onInputYears = computed(() => selectView.value === 'Inputs' && inputYears.value.length > 0);
    const availableYears = computed(() => onInputYears.value ? inputYears.value : runYears.value);
    const yearIndex = computed({
      get: () => onInputYears.value ? inputYearIndex.value : runYearIndex.value,
      set: (position) => { if (onInputYears.value) inputYearIndex.value = position; else runYearIndex.value = position; },
    });
    const selectYear = computed(() => availableYears.value[yearIndex.value] ?? null);

    // ── Block 1: the simulated land use ─────────────────────────────────────────────────────────
    const landuseMapEntry = window.MapService.mapCategories['Dvar']['Lumap'];
    const { currentLayerData: landuseLayerData, ensureComboLayer: ensureLanduseLayer } = window.createMapLayerLoader(VIEW_NAME);
    const landuseMapData = computed(() => landuseLayerData.value?.[selectYear.value] ?? null);

    // Its legend, in two columns (legend_registry.lumap.columns): the agricultural groups and the public
    // land | every non-agricultural land use, the ones the run turned off in grey. A click on an entry
    // swaps the map for the share of each cell under that entry; a second click brings the land-use map back.
    const shareMapEntry = window.MapService.mapCategories['LanduseShare']['Layers'];
    const { currentLayerData: shareLayerData, ensureComboLayer: ensureShareLayer } = window.createMapLayerLoader(VIEW_NAME + 'Share');
    const landuseLegendColumns = ref([]);   // [[{ name, color, enabled }]]
    const shareLayerNames = ref([]);        // the legend entries that have a share layer
    const selectLanduse = ref('');          // '' = the land-use map
    const toggleLanduse = (name) => {
      if (!shareLayerNames.value.includes(name)) return;
      selectLanduse.value = selectLanduse.value === name ? '' : name;
      if (selectLanduse.value) ensureShareLayer(shareMapEntry.layerPrefix, [selectLanduse.value]);
    };
    const shareMapData = computed(() => shareLayerData.value?.[selectYear.value] ?? null);
    // A land use the run never chose has a layer of zeros (`empty`): no map is drawn, a message says so
    const landuseEmpty = computed(() => !!selectLanduse.value && !!shareMapData.value?.empty);
    const simulationMapData = computed(() => !selectLanduse.value ? landuseMapData.value : landuseEmpty.value ? null : shareMapData.value);
    const landuseEntryTitle = (entry) => !entry.enabled ? 'Turned off in this run'
      : shareLayerNames.value.includes(entry.name) ? (selectLanduse.value === entry.name ? 'Back to the land-use map' : 'Show the share of each cell under ' + entry.name)
      : entry.name === 'Public land' ? 'Outside the model: no layer' : 'No layer in this report';

    // ── The Inputs view: an input of the run as a map layer ─────────────────────────────────────
    // Everything is listed from what the report holds (write_input.py): the catalogue names the inputs
    // of each theme, the layer index says which (irrigation, land use) layers an input has.
    const inputMapEntry = window.MapService.mapCategories['Input']['Layers'];
    const { currentLayerData: inputLayerData, ensureComboLayer: ensureInputLayer } = window.createMapLayerLoader(VIEW_NAME + 'Input');
    const inputCatalogue = ref({});          // theme -> [{ input, unit, kind }]
    const inputTree = ref({});               // input -> irrigation -> option -> [land uses] (a report before the input cache: input -> irrigation -> [land uses])

    const inputTables = ref({});             // theme -> table -> { unit, rows: [[row, value]] }
    const inputBundles = ref({});            // ag-management -> variable -> [{ name: land-use group, data }]
    const yieldTrend = ref({});              // source -> scenario -> [{ group, lm, kind, data }]: both yield-trend inputs
    const transitionMatrix = ref({});        // { lus, n_ag, parts, cost, disallowed }: the from-to cost table of the model
    const adoptionCaps = ref({});            // { years, rows: [[region, target, [cap % per year], [cap ha per year]]] }: the social-licence caps


    const BUNDLE_THEME = 'Ag management';    // a theme of lines over the years, with no map
    const availableThemes = computed(() => [
      ...Object.keys(inputCatalogue.value).filter(theme => inputCatalogue.value[theme].some(entry => (entry.kind === 'chart' || entry.input in inputTree.value))),
      ...(Object.keys(inputBundles.value).length ? [BUNDLE_THEME] : []),
      ASSUMPTION_THEME,
    ]);
    // The assumptions: the settings that are not input data. Their names, options and meanings are the notes of the
    // Settings page (services/SettingsNotes.js); the groups and their order are chosen here.
    const ASSUMPTION_THEME = 'Assumptions';
    // Settings an input theme already shows as option buttons (SSP, CO2 fertilisation, scope 1, carbon window, fire
    // risk, quality layer, HCAS source) or marks on its chart (the productivity trend) are left out: the run's choice
    // is highlighted there.
    const ASSUMPTIONS = [
      ['Prices and costs', ['DYNAMIC_PRICE', 'CARBON_PRICES_FIELD', 'CARBON_PRICE_COSTANT', 'AMORTISE_UPFRONT_COSTS', 'DISCOUNT_RATE', 'AMORTISATION_PERIOD', 'TRANSITION_COST_MULT', 'INCLUDE_WATER_LICENSE_COSTS']],
      ['Objective and demand', ['OBJECTIVE', 'DEMAND_BOUNDS', 'APPLY_DEMAND_MULTIPLIERS']],
      ['Carbon accounting', ['RISK_OF_REVERSAL']],
      ['Water', ['WATER_CLIMATE_CHANGE_IMPACT', 'WATER_REGION_DEF', 'WATER_STRESS', 'LIVESTOCK_DRINKING_WATER']],
      ['Biodiversity', ['CONNECTIVITY_SOURCE', 'CONNECTIVITY_LB']],
      ['Land-use change', ['NON_AG_LAND_USES_REVERSIBLE', 'AG_MANAGEMENTS_REVERSIBLE', 'TECH_ADOPT_MULT', 'EXCLUDE_NO_GO_LU']],
      ['Resolution and years', ['RESFACTOR', 'SIM_YEARS']],
    ];
    const assumptionGroups = computed(() => ASSUMPTIONS.map(([name, settings]) => ({
      name,
      items: settings.filter(setting => runScenario.value[setting] !== undefined).map(setting => {
        const note = (window.SettingsNotes || {})[setting] || {};
        const value = String(runScenario.value[setting]);
        const options = (note.options || []).map(([option, meaning]) => ({ value: option, meaning, ofRun: option === value }));
        const ofRun = options.find(option => option.ofRun);
        // the meaning of the option of the run; for a setting without options, its detail
        return { setting, what: note.what || setting, options, value, meaning: ofRun ? ofRun.meaning : (options.length ? `This run: ${value}` : (note.detail || [])[0] || '') };
      }),
    })).filter(group => group.items.length > 0));
    const isAssumptionTheme = computed(() => selectTheme.value === ASSUMPTION_THEME);
    const showAssumptions = computed(() => selectView.value === 'Inputs' && isAssumptionTheme.value);
    // A theme with no map and no input buttons: the bundles and the assumptions
    const isNoMapTheme = computed(() => isBundleTheme.value || isAssumptionTheme.value);
    const selectTheme = ref('');
    const selectInput = ref('');
    const selectInputLm = ref('ALL');
    const selectInputLu = ref('ALL');
    const selectBundle = ref('');
    const selectBundleVariable = ref('');

    // The species sets (GBF3 vegetation groups, GBF4 species and communities, GBF8 species): thousands of layers, kept
    // a hundred to a file, and the file of the one picked is loaded (write_input.write_species_inputs). They are inputs of the
    // Biodiversity theme, of kind 'species'.
    const speciesSets = ref({});             // set -> { dir, grid, unit, variants, variant_name, items, note }
    const SPECIES_THEME = 'Biodiversity';
    const themeInputs = computed(() => [
      ...(inputCatalogue.value[selectTheme.value] || []).filter(entry => (entry.kind === 'chart' || entry.input in inputTree.value)),
      ...(selectTheme.value === SPECIES_THEME ? Object.entries(speciesSets.value).map(([name, content]) => ({ input: name, unit: content.unit, kind: 'species' })) : []),
    ]);
    const inputEntry = computed(() => themeInputs.value.find(entry => entry.input === selectInput.value) || null);
    // An input named '<another input of the theme>: <part>' is a part of that input (the agricultural cost and its
    // quantity, area, fixed and water costs): the Input row lists the whole ones, a Part row under it the parts
    // of the selected one, 'Total' being the input itself.
    const PART_SEPARATOR = ': ';
    const inputParent = (name) => {
      const parent = name.includes(PART_SEPARATOR) ? name.slice(0, name.indexOf(PART_SEPARATOR)) : null;
      return parent && themeInputs.value.some(entry => entry.input === parent) ? parent : null;
    };
    const mainInputs = computed(() => themeInputs.value.filter(entry => !inputParent(entry.input)));
    const selectMainInput = computed(() => inputParent(selectInput.value) ?? selectInput.value);
    const inputParts = computed(() => {
      const parts = themeInputs.value.filter(entry => inputParent(entry.input) === selectMainInput.value);
      const label = (name) => { const part = name.slice(selectMainInput.value.length + PART_SEPARATOR.length); return part[0].toUpperCase() + part.slice(1); };
      return parts.length ? [{ input: selectMainInput.value, label: 'Total' }, ...parts.map(entry => ({ input: entry.input, label: label(entry.input) }))] : [];
    });
    // An input with one layer only, or one per non-agricultural land use or technology, files them under 'ALL'
    const inputLmOptions = computed(() => Object.keys(inputTree.value[selectInput.value] || {}));
    // The options an input is built under ('S=245|C=on'; 'ALL' where no option changes it), each a list
    // of land uses. The option is picked setting by setting (inputOptionRows); it starts on the values of the run.
    const optionsOfLm = computed(() => {
      const branch = inputTree.value[selectInput.value]?.[selectInputLm.value];
      return Array.isArray(branch) ? { 'ALL': branch } : (branch || {});
    });
    const parseOption = (label) => Object.fromEntries(label.split('|').filter(part => part.includes('=')).map(part => [part.slice(0, part.indexOf('=')), part.slice(part.indexOf('=') + 1)]));
    const optionChoice = reactive({});       // setting -> the value picked for it
    // The options by the short name they have in a layer's label (write_input.OPTION_SHORT_NAMES): the setting of
    // the run each one is, its name on the page, and the order of its values (as the model lists them)
    const OPTIONS = {
      'S': { setting: 'SSP', name: 'SSP', order: ['126', '245', '370', '585'] },
      'C': { setting: 'CO2_FERT', name: 'CO2 fertilisation', order: ['on', 'off'] },
      'G': { setting: 'USE_GHG_SCOPE_1', name: 'Scope 1 only', order: ['True', 'False'] },
      'W': { setting: 'CARBON_EFFECTS_WINDOW', name: 'Carbon window (years)', order: ['50', '60', '70', '80', '90'] },
      'F': { setting: 'FIRE_RISK', name: 'Fire risk', order: ['low', 'med', 'high'] },
      'Q': { setting: 'BIO_QUALITY_LAYER', name: 'Quality layer', order: [] },
      // 'CONSTANT' is built as no carbon price: it is the run's option only where the run's constant is 0
      'P': { setting: 'CARBON_PRICES_FIELD', name: 'Carbon price', order: ['CONSTANT', 'Default', '100', '1.8C 67%', '1.5C 50%', '1.5C 67%'],
             labels: { 'CONSTANT': 'none (0)', 'Default': 'default path', '100': '100 AUD/t' } },
    };
    const optionValueLabel = (option, value) => OPTIONS[option]?.labels?.[value] ?? value;
    // The value of the run for an option, as the buttons name it (null: the run has a value no button stands for)
    const runValueOf = (option, known) => {
      const value = String(settingOf(known.setting));
      if (option === 'P' && value === 'CONSTANT' && Number(settingOf('CARBON_PRICE_COSTANT')) !== 0) return null;
      return value;
    };
    // What to say where no button stands for the run's value
    const runValueNote = (option) => option === 'P' && String(settingOf('CARBON_PRICES_FIELD')) === 'CONSTANT' && Number(settingOf('CARBON_PRICE_COSTANT')) !== 0
      ? `this run: a constant ${settingOf('CARBON_PRICE_COSTANT')} AUD per t, not among the maps` : '';
    // A table can have options too (Inputs_tables.js: `options`): the sources of the habitat contribution
    const TABLE_OPTIONS = {
      'Habitat contribution of a land use (HCAS)': { key: 'H', setting: 'HCAS_CONTRIBUTION_PERCENTILE', name: 'Habitat condition source', reference: '50', referenceName: '50th percentile' },
    };
    const tableOfInput = computed(() => selectView.value === 'Inputs' && isChartInput.value ? (inputTables.value[selectTheme.value]?.[selectInput.value] || null) : null);
    const tableOptionRow = computed(() => {
      const known = TABLE_OPTIONS[selectInput.value];
      if (!known || !tableOfInput.value?.options) return null;
      const values = Object.keys(tableOfInput.value.options);
      const ofRun = values.find(value => String(settingOf(known.setting)) === value) ?? null;
      return { setting: known.key, name: known.name, values, ofRun, picked: values.includes(optionChoice[known.key]) ? optionChoice[known.key] : (ofRun ?? values[0]) };
    });
    // One row of buttons per option the input depends on: its values, the one the run used (drawn with a dashed
    // border) and the one picked
    const inputOptionRows = computed(() => {
      if (tableOptionRow.value) return [tableOptionRow.value];
      const labels = Object.keys(optionsOfLm.value).filter(label => label !== 'ALL');
      const rows = {};
      labels.forEach(label => Object.entries(parseOption(label)).forEach(([option, value]) => { (rows[option] ||= []).includes(value) || rows[option].push(value); }));
      return Object.entries(rows).map(([option, found]) => {
        const known = OPTIONS[option] || { setting: option, name: option, order: [] };
        const values = [...known.order.filter(value => found.includes(value)), ...found.filter(value => !known.order.includes(value))];
        const ofRun = values.find(value => runValueOf(option, known) === value) ?? null;
        return { setting: option, name: known.name, values, ofRun, note: runValueNote(option), picked: values.includes(optionChoice[option]) ? optionChoice[option] : (ofRun ?? values[0]) };
      });
    });
    const selectInputOption = computed(() => {
      const labels = Object.keys(optionsOfLm.value);
      const picked = Object.fromEntries(inputOptionRows.value.map(row => [row.setting, row.picked]));
      return labels.find(label => label !== 'ALL' && Object.entries(parseOption(label)).every(([setting, value]) => picked[setting] === value))
        ?? (labels.includes('ALL') ? 'ALL' : (labels[0] ?? 'ALL'));
    });
    const inputLuOptions = computed(() => optionsOfLm.value[selectInputOption.value] || []);
    const LM_LABELS = { 'Dryland': 'Dryland', 'Irrigated': 'Irrigated', 'NonAg': 'Non-ag' };
    const bundleOptions = computed(() => Object.keys(inputBundles.value));
    // The names of the page for the options (those of the other pages: data_tools/parameters.py RENAME_AM), for the
    // land-use groups of the workbooks, and for the columns of the workbooks. A column is [name, unit, read by the
    // model or not]; two columns with one name (the workbooks spell the irrigation column two ways) are one variable.
    const BUNDLE_LABELS = {
      'Asparagopsis taxiformis': 'Methane reduction (livestock)',
      'Precision Agriculture': 'Agricultural technology (fertiliser)',
      'Ecological Grazing': 'Regenerative agriculture (livestock)',
      'AgTech EI': 'Agricultural technology (energy)',
      'Biochar': 'Biochar (soil amendment)',
    };
    const BUNDLE_GROUPS = { 'cropping': 'Cropping', 'int_cropping': 'Intensive cropping', 'horticulture': 'Horticulture' };
    const MULTIPLIER = 'multiplier (1 = no change)';
    const BUNDLE_COLUMNS = {
      'Technical_Adoption': ['Adoption ceiling', 'share of the land use (0 - 1)', true],
      'Feasible Adoption (%)': ['Adoption ceiling', 'share of the land use (0 - 1)', true],
      'Productivity': ['Yield', MULTIPLIER, true],
      'AnnCost_per_Ha': ['Annual cost', 'AUD per ha', true],
      'Annual Cost Per Animal (A$2010/yr)': ['Annual cost', 'AUD per animal', true],
      'Labour_cost_mulitiplier': ['Labour cost', MULTIPLIER, true],
      'Operating_cost_multiplier': ['Operating cost', MULTIPLIER, true],
      'Delta_total_cost': ['Total cost', MULTIPLIER, false],
      'CO2E_KG_HEAD_ENTERIC': ['Emissions: enteric methane', MULTIPLIER, true],
      'CO2E_KG_HEAD_MANURE_MGT': ['Emissions: manure management', MULTIPLIER, true],
      'CO2E_KG_HEAD_IND_LEACH_RUNOFF': ['Emissions: leaching and runoff', MULTIPLIER, true],
      'CO2E_KG_HA_SOIL': ['Emissions: soil N2O', MULTIPLIER, true],
      'CO2E_KG_HA_CHEM_APPL': ['Emissions: chemical application', MULTIPLIER, true],
      'CO2E_KG_HA_CROP_MGT': ['Emissions: crop management', MULTIPLIER, true],
      'CO2E_KG_HA_PEST_PROD': ['Emissions: pesticide production', MULTIPLIER, true],
      'CO2E_KG_HA_IRRIG': ['Emissions: irrigation', MULTIPLIER, true],
      'CO2e_KG_HA_IRRIG': ['Emissions: irrigation', MULTIPLIER, true],
      'IMPACTS_soil_carbon': ['Soil carbon', MULTIPLIER, true],
      'Water_use': ['Water use', MULTIPLIER, true],
      'INPUT-wrt_water-required': ['Water use', MULTIPLIER, true],
      'Biodiversity_impact': ['Biodiversity', MULTIPLIER, true],
      'Biodiversity Impacts': ['Biodiversity', MULTIPLIER, false],
      'Biodiversity_compatability': ['Biodiversity', MULTIPLIER, false],
      'NUE': ['Nitrogen use efficiency', MULTIPLIER, false],
      'Nitrogen_surplus': ['Nitrogen surplus', MULTIPLIER, false],
      'IMPACTS_water_retention': ['Water retention', MULTIPLIER, false],
    };
    // the biodiversity column is read for biochar only
    const columnOf = (column) => {
      const known = BUNDLE_COLUMNS[column] || [column.replaceAll('_', ' '), '', true];
      return { name: known[0], unit: known[1], used: column === 'Biodiversity_impact' ? selectBundle.value === 'Biochar' : known[2] };
    };
    const bundleLabel = (option) => BUNDLE_LABELS[option] || option;
    // The variables of the selected option, by the name of the page: those the model reads first
    const bundleVariables = computed(() => {
      const names = [...new Set(Object.keys(inputBundles.value[selectBundle.value] || {}).map(column => columnOf(column).name))];
      const isUsed = (name) => Object.keys(inputBundles.value[selectBundle.value] || {}).some(column => columnOf(column).name === name && columnOf(column).used);
      return [...names.filter(isUsed), ...names.filter(name => !isUsed(name))];
    });
    // The chart of the selected variable: a line per land-use group
    const bundleChart = computed(() => {
      const GROUP_COLORS = ['#0ea5e9', '#f59e0b', '#10b981', '#8b5cf6', '#ef4444'];
      const columns = Object.keys(inputBundles.value[selectBundle.value] || {}).filter(column => columnOf(column).name === selectBundleVariable.value);
      const lines = columns.flatMap(column => inputBundles.value[selectBundle.value][column]).sort((one, other) => one.name.localeCompare(other.name));
      const series = lines.map((line, position) => ({ name: BUNDLE_GROUPS[line.name] || line.name, color: GROUP_COLORS[position % GROUP_COLORS.length], data: line.data }));
      const used = columns.some(column => columnOf(column).used);
      return { title: bundleLabel(selectBundle.value), variable: selectBundleVariable.value, unit: columns.length ? columnOf(columns[0]).unit : '',
               note: used ? '' : 'in the workbook, not read by the model', series, used };
    });
    const showBundleChart = computed(() => selectView.value === 'Inputs' && isBundleTheme.value && bundleOptions.value.length > 0);
    const isBundleTheme = computed(() => selectTheme.value === BUNDLE_THEME);
    // The two inputs that are a chart over the map, each by its name
    const MATRIX_INPUTS = ['Transition cost matrix', 'Transition exclusion matrix'];      // the second: allowed or not only
    const showMatrix = computed(() => selectView.value === 'Inputs' && MATRIX_INPUTS.includes(selectInput.value) && (transitionMatrix.value.lus || []).length > 0);
    const matrixAllowedOnly = computed(() => selectInput.value === MATRIX_INPUTS[1]);
    // An input that is a table of the theme (the habitat contribution of every land use), by its name
    // The table shown: under the option picked where it has options, with the reference values marked on the bars
    // of every option but the reference itself
    const inputTable = computed(() => {
      const table = tableOfInput.value;
      if (!table || !tableOptionRow.value) return table;
      const known = TABLE_OPTIONS[selectInput.value];
      const picked = tableOptionRow.value.picked;
      return { unit: table.unit, rows: table.options[picked], ...(picked === known.reference ? {} : { reference: table.reference, referenceName: known.referenceName }) };
    });
    const showTrendCharts = computed(() => selectView.value === 'Inputs' && isChartInput.value && selectInput.value === 'Yield trend' && Object.keys(yieldTrend.value).length > 0);
    // The land-use buttons lie over the left of the map: the country is fitted to the right of them
    const mapPadLeft = computed(() => selectView.value === 'Inputs' && !isNoMapTheme.value && !isChartInput.value && inputLuOptions.value.length > 1 ? 190 : 8);
    // An input that is one number for the whole country (kind 'chart'): cards only, the map is left empty
    const isChartInput = computed(() => inputEntry.value?.kind === 'chart');

    // Each choice keeps the one below it where it still exists, else takes the preferred one, else the first
    const keepOrFirst = (selection, options, preferred = null) => {
      if (!options.includes(selection.value)) selection.value = options.includes(preferred) ? preferred : (options[0] ?? '');
    };
    // The land use an input opens on: widespread, so its layer covers enough of the country to read
    const DEFAULT_INPUT_LM = 'Dryland';
    const DEFAULT_INPUT_LU = 'Beef - modified land';
    watch(availableThemes, (themes) => keepOrFirst(selectTheme, themes));
    watch(themeInputs, (entries) => keepOrFirst(selectInput, entries.map(entry => entry.input)));
    watch(inputLmOptions, (options) => keepOrFirst(selectInputLm, options, DEFAULT_INPUT_LM));
    watch(inputLuOptions, (options) => keepOrFirst(selectInputLu, options, DEFAULT_INPUT_LU));
    watch(bundleOptions, (options) => keepOrFirst(selectBundle, options));
    watch(bundleVariables, (options) => keepOrFirst(selectBundleVariable, options));

    // The layer of the selection (none for the bundles: the map is left empty)
    watch([selectInput, selectInputLm, selectInputOption, selectInputLu, isNoMapTheme], () => {
      if (isNoMapTheme.value || !inputLuOptions.value.includes(selectInputLu.value)) return;
      const hasOptions = !Array.isArray(inputTree.value[selectInput.value]?.[selectInputLm.value]);
      ensureInputLayer(inputMapEntry.layerPrefix, [selectInput.value, selectInputLm.value, ...(hasOptions ? [selectInputOption.value] : []), selectInputLu.value]);
    });
    // The year of the layer shown. The inputs are kept for a few years only (every ten), and a layer that is the same
    // in every year once, under 'ALL': the year shown is the kept year nearest to the selected one.
    const inputLayerYear = computed(() => {
      const years = Object.keys(inputLayerData.value || {});
      if (years.includes('ALL')) return 'ALL';
      const wanted = Number(selectYear.value);
      return years.length ? years.reduce((best, year) => Math.abs(Number(year) - wanted) < Math.abs(Number(best) - wanted) ? year : best) : String(selectYear.value);
    });
    // The species layer picked: the set is the selected input, then one of its variants (presence class, or year) and
    // one of its items. Its file is loaded when the pick changes, and the one before it let go.
    const speciesSet = computed(() => inputEntry.value?.kind === 'species' ? speciesSets.value[selectInput.value] : null);
    const selectSpeciesVariant = ref('');
    // The presence classes as they read on the page (the folders keep the names of the settings); a year as it is
    const VARIANT_LABELS = { 'LIKELY': 'likely', 'LIKELY_AND_MAYBE': 'likely and may' };
    const variantLabel = (variant) => VARIANT_LABELS[variant] || String(variant).replaceAll('_', ' ').toLowerCase();
    const selectSpeciesItem = ref(0);        // the position of the item in the list of its set
    const speciesLeaf = ref(null);
    const speciesItemOptions = computed(() => (speciesSet.value?.items || []).map((name, position) => ({ value: position, label: name.replaceAll('_', ' ') })));
    watch(speciesSet, (content) => {
      if (!content) return;
      selectSpeciesItem.value = 0;
      // a set by year opens on the year nearest to the selected one
      const wanted = Number(selectYear.value);
      const byYear = content.variants.every(variant => !Number.isNaN(Number(variant)));
      selectSpeciesVariant.value = byYear ? content.variants.reduce((best, variant) => Math.abs(Number(variant) - wanted) < Math.abs(Number(best) - wanted) ? variant : best) : content.variants[0];
    });
    let speciesLoaded = '';
    watch([speciesSet, selectSpeciesVariant, selectSpeciesItem], async () => {
      const content = speciesSet.value;
      if (!content || !content.variants.includes(selectSpeciesVariant.value)) { speciesLeaf.value = null; return; }
      // the layers are kept `per_file` to a file, in the order of the list: the file of the picked one is loaded
      const item = selectSpeciesItem.value;
      const fileNumber = Math.floor(item / (content.per_file || 1));
      const name = `sp_${content.dir}_${selectSpeciesVariant.value}_${fileNumber}`;
      try {
        await loadScript(`data/map_layers/input_species/${content.dir}/${selectSpeciesVariant.value}/${fileNumber}.js`, name, VIEW_NAME);
        if (speciesLoaded && speciesLoaded !== name) delete window[speciesLoaded];
        speciesLoaded = name;
        if (item !== selectSpeciesItem.value) return;                   // another one was picked while this file loaded
        speciesLeaf.value = window[name]?.[item] ? { ...window[name][item], cellIndex: content.grid } : null;
      } catch (error) {
        console.warn(`Species layer not loaded: ${name}`, error);
        speciesLeaf.value = null;
      }
    });

    const inputYearNote = computed(() => speciesSet.value
      ? [speciesSet.value.variant_name ? `${speciesSet.value.variant_name.toLowerCase()} ${variantLabel(selectSpeciesVariant.value)}` : 'the same in every year', speciesSet.value.note].filter(Boolean).join(' · ')
      : inputLayerYear.value === 'ALL' ? 'the same in every year'
      : String(inputLayerYear.value) === String(selectYear.value) ? String(selectYear.value)
      : `${inputLayerYear.value} (the nearest year kept to ${selectYear.value})`);
    // A mask is drawn on a quiet blue-green ramp of its own (MASK_RAMP), not on the yellow-to-maroon of the values
    const MASK_RAMP = [[214, 232, 238], [150, 193, 207], [86, 150, 171]];
    const inputMapData = computed(() => {
      if (isNoMapTheme.value || isChartInput.value) return null;
      if (speciesSet.value) return speciesLeaf.value;
      const leaf = inputLayerData.value?.[inputLayerYear.value] ?? null;
      if (!leaf) return null;
      // the layers of the input cache are on a grid of their own (map_input_cell_index.js); a report from before it has none
      const onGrid = window[inputMapEntry.indexName]?.dims?.includes('option') ? { ...leaf, cellIndex: 'map_input_cell_index' } : leaf;
      return inputEntry.value?.kind === 'mask' ? { ...onGrid, ramp: MASK_RAMP } : onGrid;
    });
    const inputLegendTitle = computed(() => {
      const entry = inputEntry.value;
      if (!entry) return '';
      if (speciesSet.value) return `${speciesItemOptions.value[selectSpeciesItem.value]?.label ?? ''} (${entry.unit})`;
      const what = selectInputLu.value === 'ALL' ? '' : `${selectInputLu.value}${selectInputLm.value === 'ALL' ? '' : ', ' + (LM_LABELS[selectInputLm.value] || selectInputLm.value).toLowerCase()}`;
      return `${entry.input} (${entry.unit})` + (what ? ` · ${what}` : '');
    });

    // The colour bar of the input layer: the ramp of regions_map.getColorFn, zero where the layer has it
    const inputLegend = computed(() => {
      const leaf = inputMapData.value;
      if (!leaf?.min_max) return null;
      const [low, high] = leaf.min_max;
      const NEGATIVE = '#000E2B, #104991, #5D94DC', POSITIVE = '#FFF6B0, #FD933E, #800026';
      const format = (value) => String(+value.toPrecision(2));
      if (low < 0 && high > 0) {
        const zeroAt = (-low / (high - low) * 100).toFixed(1);
        return {
          gradient: `linear-gradient(to right, #000E2B 0%, #104991 ${zeroAt / 2}%, #5D94DC ${zeroAt}%, #FFF6B0 ${zeroAt}%, #FD933E ${(100 + Number(zeroAt)) / 2}%, #800026 100%)`,
          // the zero label is left out where it would sit on an end label
          ticks: [{ at: 0, label: format(low) }, ...(zeroAt > 12 && zeroAt < 88 ? [{ at: Number(zeroAt), label: '0' }] : []), { at: 100, label: format(high) }],
        };
      }
      return low < 0
        ? { gradient: `linear-gradient(to right, ${NEGATIVE})`, ticks: [{ at: 0, label: format(low) }, { at: 100, label: '0' }] }
        : { gradient: `linear-gradient(to right, ${POSITIVE})`, ticks: [{ at: 0, label: '0' }, { at: 100, label: format(high) }] };
    });

    // ── The data behind the cards ───────────────────────────────────────────────────────────────
    // A mask has classes, not a scale: its legend is a swatch per class, in the colours the layer is drawn with
    // (the top of MASK_RAMP for 1, the map's grey for 0; a share between them is drawn along the ramp).
    const rgb = (stop) => `rgb(${stop.join(', ')})`;
    const MASK_TOP = rgb(MASK_RAMP[2]), MASK_PART = `linear-gradient(to right, ${rgb(MASK_RAMP[0])}, ${rgb(MASK_RAMP[1])})`, MASK_NONE = '#e1e1e1';
    const MASK_LEGENDS = {
      'Agricultural land use allowed': [{ color: MASK_TOP, label: 'Allowed' }, { color: MASK_NONE, label: 'Not allowed' }],
      'Non-agricultural share allowed': [{ color: MASK_TOP, label: 'Whole cell' }, { color: MASK_PART, label: 'Part of the cell' }, { color: MASK_NONE, label: 'Not allowed' }],
      'GBF2 priority degraded areas': [{ color: MASK_TOP, label: 'Priority degraded area' }, { color: MASK_NONE, label: 'Other land' }],
    };
    const maskLegend = computed(() => inputEntry.value?.kind === 'mask'
      ? (MASK_LEGENDS[inputEntry.value.input] || [{ color: MASK_TOP, label: 'Inside the mask' }, { color: MASK_NONE, label: 'Outside' }])
      : null);

    const ghgOverview = ref(null);           // region level -> region -> [series]
    const demandOverview = ref(null);        // commodity -> [series]: the demand types and 'Production'
    const selectDemandCommodity = ref('');   // the Demand card shows one commodity: tonnes of different commodities are not added
    const demandCommodities = computed(() => Object.keys(demandOverview.value || {}));
    watch(demandCommodities, (commodities) => { if (!commodities.includes(selectDemandCommodity.value)) selectDemandCommodity.value = commodities[0] ?? ''; });
    const selectCardOption = (key, value) => { if (key === 'demand') selectDemandCommodity.value = value; };
    const gbf2Overview = ref(null);          // region level -> region -> { Percent, Area }
    const watershedData = ref({});           // river region -> [series]: the water limits are set per river region
    const renewableData = ref(null);         // region level -> region -> ag-management -> water supply -> [series]

    // 'SSP245' -> 'SSP2-4.5'
    const scenarioLabel = computed(() => {
      const ssp = String(runScenario.value.SSP || '');
      return ssp.length === 3 ? `SSP${ssp[0]}-${ssp[1]}.${ssp[2]}` : `SSP${ssp}`;
    });

    // ── Helpers of the cards ────────────────────────────────────────────────────────────────────
    // A series' points as [[year, value]] (a file writes them as pairs or as {x, y})
    const pointsOf = (seriesList, name) => {
      const series = (seriesList || []).find(item => item.name === name);
      return (series?.data || []).map(point => Array.isArray(point) ? [Number(point[0]), point[1]] : [Number(point.x), point.y]);
    };
    const valueAt = (points, year) => points.find(point => point[0] === year)?.[1];
    const compact = (value) => {
      if (value == null || Number.isNaN(value)) return 'n/a';
      const size = Math.abs(value);
      if (size >= 1e9) return (value / 1e9).toPrecision(3) + ' B';
      if (size >= 1e6) return (value / 1e6).toPrecision(3) + ' M';
      if (size >= 1e3) return (value / 1e3).toPrecision(3) + ' K';
      return String(+value.toPrecision(3));
    };
    const signed = (value) => value == null ? 'n/a' : (value > 0 ? '+' : '') + value.toFixed(1) + ' %';
    const settingOf = (key) => runScenario.value[key];

    // A dict setting arrives as its Python repr ("{'medium': {2030: 0.3}}"): read it as an object, {} if it can not be
    const settingDict = (key) => {
      const raw = settingOf(key);
      if (raw && typeof raw === 'object') return raw;
      try {
        return JSON.parse(String(raw)
          .replace(/'/g, '"')
          .replace(/\bNone\b/g, 'null').replace(/\bTrue\b/g, 'true').replace(/\bFalse\b/g, 'false')
          .replace(/([{,]\s*)(\d+)\s*:/g, '$1"$2":'));
      } catch (err) {
        return {};
      }
    };

    // The Settings tab of a drawer: the settings of the model (settings.py) behind the card, each with its value in
    // this run. `entries` is [[setting name, what it does]].
    const settingRows = (entries) => entries.map(([name, what]) => ({ name, value: String(settingOf(name) ?? 'not recorded in this run'), what }));
    const ELASTIC_SETTING = ['ELASTIC_FAMILIES', 'The target families allowed to fall short at a penalty. Empty: every target is hard, the model has to meet it.'];

    // The biodiversity targets other than GBF2, each national or regional by its region mode
    const gbfTargets = computed(() => {
      const regionsOf = (key) => Object.keys(settingDict(key)).filter(name => name !== 'AUSTRALIA');
      const describe = (level, regionKey, regional) => {
        if (level === 'off') return 'off';
        const regions = regional && level === 'SPECIFIED' ? regionsOf(regionKey) : [];
        return regions.length ? `${level} · ${regions.join(', ')}` : String(level);
      };
      const gbf3Regional = settingOf('GBF3_NVIS_REGION_MODE') !== 'AUSTRALIA';
      const snesRegional = settingOf('GBF4_SNES_REGION_MODE') === 'NRM';
      const ecnesRegional = settingOf('GBF4_ECNES_REGION_MODE') === 'NRM';
      return [
        { key: 'GBF3_NVIS', what: 'vegetation group', short: 'GBF3', label: 'GBF3 vegetation groups', level: settingOf('GBF3_NVIS_TARGET'), regional: gbf3Regional, value: describe(settingOf('GBF3_NVIS_TARGET'), 'GBF3_NVIS_SEL_REGION_TARGETS', gbf3Regional), mode: settingOf('GBF3_NVIS_REGION_MODE') },
        { key: 'GBF4_SNES', what: 'species', short: 'GBF4 SNES', label: 'GBF4 species (SNES)', level: settingOf('GBF4_TARGET_SNES'), regional: snesRegional, value: describe(settingOf('GBF4_TARGET_SNES'), 'GBF4_SNES_SEL_REGION_TARGETS', snesRegional), mode: settingOf('GBF4_SNES_REGION_MODE') },
        { key: 'GBF4_ECNES', what: 'community', short: 'GBF4 ECNES', label: 'GBF4 communities (ECNES)', level: settingOf('GBF4_TARGET_ECNES'), regional: ecnesRegional, value: describe(settingOf('GBF4_TARGET_ECNES'), 'GBF4_ECNES_SEL_REGION_TARGETS', ecnesRegional), mode: settingOf('GBF4_ECNES_REGION_MODE') },
        { key: 'GBF8', what: 'species', short: 'GBF8', label: 'GBF8 species', level: settingOf('GBF8_TARGET'), regional: false, value: String(settingOf('GBF8_TARGET')), mode: 'AUSTRALIA' },
      ].map(target => ({ ...target, off: target.level === 'off' || target.level == null }));
    });
    // Every (region, item) with a GBF3 / GBF4 target, its score and target per year ({ years, rows }), for the
    // targets the run has on. Loaded after the page shows: the cards fill in when the files land.
    const gbfSummary = ref({});              // target key -> { years, rows: [[region level, region, item, [scores], [targets]]] }
    const loadGbfSummaries = async () => {
      for (const target of gbfTargets.value.filter(item => !item.off)) {
        const entry = chartRegister['Biodiversity']?.[target.key === 'GBF8' ? 'GBF8_SPECIES' : target.key]?.['targets'];
        if (!entry) continue;
        try {
          await loadScript(entry.path, entry.name, VIEW_NAME);
          gbfSummary.value = { ...gbfSummary.value, [target.key]: window[entry.name] };
        } catch (error) {
          console.warn(`Not in this report: ${entry.path}`, error);
          gbfSummary.value = { ...gbfSummary.value, [target.key]: { years: [], rows: [] } };
        }
      }
    };

    // ── The drawer of the Demand card: where the demand comes from ─────────────────────────────
    // LUTO reads its demand from the food demand model of Michalis Hadjikakou (Deakin) and Raymundo Marcos-Martinez
    // (CSIRO), repository au.food.demand. (Traced 2026-11-16 from its Scripts/Demand_interventions.R, Functions/
    // Net demand_function.R, Food_supply_scenario_function.R, the EAT_Lancet_* functions, Waste_reduction.R,
    // Feed_module/Food+feed_estimation_function.R, the trade model script and Modelling_assumptions/*.docx; and on
    // the LUTO side from dataprep.py and data.py.)
    const demandDrawer = () => {
      const multipliersOn = String(settingOf('APPLY_DEMAND_MULTIPLIERS')) === 'True' && settingOf('GHG_EMISSIONS_LIMITS') !== 'off';
      return {
        label: 'Where the demand comes from',
        sections: [
          { label: 'What it is', text: 'What Australian farms must produce each year, per commodity (tonnes; kL for dairy). Read from the Australian food demand model of Hadjikakou (Deakin) and Marcos-Martinez (CSIRO), repository au.food.demand.' },
          { label: 'The sum', text: 'Demand = domestic use + exports + feed − imports: the lines of the chart (imports below zero).' },
          { label: 'Domestic use', text: 'FAOSTAT food supply per person (from 1961), projected to 2100, × the population of the SSP.' },
          { label: 'Exports and imports', text: 'A gravity trade model per commodity, projected with the GDP and population of the SSP.' },
          { label: 'Diets', text: 'BAU follows the trend. FLX, VEG and VGN are EAT-Lancet diets, reached in a straight line by the convergence year. The domestic diet changes domestic use, the global diet exports.' },
          { label: 'Imports setting', text: 'Static: the 2010 import share is kept. Trend: it follows the trade projection. Hay and live sheep are never imported.' },
          { label: 'Loss and waste', text: 'The Oceania fractions of Gustavsson et al. (2011): kept (full waste) or halved by the convergence year (half waste).' },
          { label: 'Feed', text: 'Livestock production × feed conversion ratios and Australian feed recipes. Soy and sunflower meal are taken as imported.' },
          { label: 'Fitted to the land-use map', text: 'The demand model gives a multiplier on 2010, applied to the 2010 production of the LUTO land-use map.' },
          { label: 'In the model', text: 'One row per commodity: production = demand. Pork, chicken, eggs and aquaculture are off-land: only their emissions count.'
            + (multipliersOn ? ' In this run sugar, winter cereals and winter oilseeds are also multiplied by the AusTIMES factor; the chart shows the demand before it.' : '') },
          { label: 'Version', text: 'The demand model is being updated by its authors; the method stays the same in general.' },
        ],
        settings: settingRows([
          ['SSP', 'The climate and socio-economic pathway (245 = SSP2, RCP 4.5). Its first digit picks the demand scenario: the population and GDP behind domestic use and trade.'],
          ['SCENARIO', 'The demand scenario, derived from SSP: not set by hand.'],
          ['DIET_DOM', 'The diet in Australia: BAU, FLX (flexitarian), VEG (vegetarian) or VGN (vegan). Changes domestic use.'],
          ['DIET_GLOB', 'The diet of the rest of the world, same options. Changes exports.'],
          ['CONVERGENCE', 'The year the diet and waste changes are complete: 2050 or 2100.'],
          ['IMPORT_TREND', 'Static: imports keep their 2010 share. Trend: the share follows the trade projection.'],
          ['WASTE', '1 keeps the loss and waste fractions; 0.5 halves them by the convergence year.'],
          ['FEED_EFFICIENCY', 'BAU or High: how much feed the livestock need per unit produced.'],
          ['APPLY_DEMAND_MULTIPLIERS', 'True: the demand of sugar, winter cereals and winter oilseeds is multiplied by the AusTIMES factor (crops for bioenergy). Applies only when GHG_EMISSIONS_LIMITS is not off.'],
          ['GHG_EMISSIONS_LIMITS', 'For demand, its level only picks which AusTIMES pathway the multipliers above are read from.'],
          ['OFF_LAND_COMMODITIES', 'Commodities not grown on the land of the model: no demand row, only their emissions are counted.'],
        ]),
      };
    };

    // ── The drawers of the GHG and biodiversity cards: where each target comes from ────────────
    // GHG: the two rows of pull request #229 (Nick van Schoten, merged 2026-10-22: luto/tools/ghg_targets.py,
    // Data.load_ghg_targets, row_builder.get_ghg, write_output's GHG_EMISSIONS_LIMIT_TCO2e) and the series in
    // input/Nick_build/ghg_targets. Biodiversity: settings.py, docs/CLAUDE_GBF2.md and the target functions of
    // data.py (get_GBF2_target_for_yr_cal, get_GBF3_NVIS_limit_score_inside_LUTO_by_yr, get_GBF4_*_target_inside_
    // LUTO_by_year, get_GBF8_target_inside_LUTO_by_yr). (Traced 2026-11-16.)
    const ghgDrawer = () => {
      return {
        label: 'Where the limit comes from',
        sections: [
          { label: 'What it is', text: 'Two limits: the emissions of the agriculture sector, and the net emissions of the land sector (LULUCF). They replaced the single AusTIMES limit (pull request #229, Nick van Schoten, October 2026).' },
          { label: 'Sources', text: 'To 2024: the National Inventory Report 2024. 2025 to 2040: the DCCEEW emissions projections 2025, baseline scenario. After 2040: the 2040 value.' },
          { label: 'Agriculture limit', text: 'The agricultural emissions of the model (off-land commodities included) plus the agriculture it does not represent (10 to 15 % of the sector) ≤ the agriculture series.' },
          { label: 'LULUCF limit', text: 'Existing forests and clearing are taken as given. What the model adds (plantings, regrowth, savanna burning, biochar, BECCS, clearing of natural land) must do at least as well as a baseline: the growth of environmental plantings in the inventory, 2016 to 2021.' },
          { label: 'Years', text: 'The limits bind from 2025; before that the series are a reference only.' },
          { label: 'The line on this card', text: 'The two limits summed on the basis of what the model represents, against the net emissions of the model. Being under it does not prove each limit is met: see GHG_emissions_split_<year>.csv.' },
        ],
        settings: settingRows([
          ['GHG_EMISSIONS_LIMITS', 'off: no limit. low or high: both limits are on. The level does not change the limits; it only picks the AusTIMES pathway of the demand multipliers and the carbon price.'],
          ['GHG_TARGETS_DICT', 'The AusTIMES pathway each level names.'],
          ['GHG_BENCHMARK_ROWS', 'Limits that are computed and reported but not imposed. Empty: both limits bind.'],
          ['HIR_EXISTING_PROJECTS', 'When on, the model cannot add regeneration on land already inside a registered project; its regeneration is always new. Off: existing projects are ignored.'],
          ['HIR_SHARE_OF_REGISTER_RATE', 'The share of the registered-project regrowth rate that regeneration earns; destocked land earns all of it. 1: the same; 0.5: half, as regeneration removes half of the livestock.'],
          ['USE_GHG_SCOPE_1', 'True: only the direct on-farm emissions of crops and livestock are counted (scope 1, as in the national inventory). False: electricity and the making of fertiliser and chemicals are counted too.'],
          ['CARBON_EFFECTS_WINDOW', 'The years over which the carbon of a planting is averaged into a yearly rate: 50, 60, 70, 80 or 90.'],
          ['FIRE_RISK', 'Which estimate of fire loss discounts the carbon of trees and debris: low, med or high.'],
          ['RISK_OF_REVERSAL', 'A further discount on the carbon of trees and debris for the risk that it is released again.'],
          ['OFF_LAND_COMMODITIES', 'Commodities not grown on the land of the model; their emissions are added to the agriculture limit.'],
        ]),
      };
    };

    // What the three "percent of the full extent" targets share (GBF3, GBF4 SNES, GBF4 ECNES)
    const extentSections = (what, cap) => [
      { label: 'The target', text: `A percentage of the full mapped extent of each ${what}. Natural land outside the model counts first; the model supplies the rest.` },
      { label: 'The score', text: 'A hectare counts by the habitat value of its land use (natural land 1, modified land less), so restoring land raises the score.' },
      { label: 'Over time', text: `From the 2010 level in a straight line to the levels of 2030, 2050 and 2100.${cap}` },
    ];
    const gbfDrawers = {
      GBF2: () => ({
        label: 'How the target is set',
        sections: [
          { label: 'What it is', text: 'Target 2 of the Global Biodiversity Framework: 30 % of degraded ecosystems under restoration by 2030. Here: a floor on the biodiversity score of the priority degraded areas.' },
          { label: 'Priority degraded areas', text: `The top cells of the Zonation priority ranking (layer ${settingOf('BIO_QUALITY_LAYER')}), up to the cut of ${settingOf('GBF2_PRIORITY_DEGRADED_AREAS_PERCENTAGE_CUT')}.` },
          { label: 'Degradation', text: 'The gap between the priority areas fully restored (score 1) and their state in 2010.' },
          { label: 'The target', text: 'The share of that gap to restore by 2030, 2050 and 2100, in a straight line from 0 in 2010.' },
          { label: 'Savanna burning', text: `Without early dry season burning, eligible land counts for ${settingOf('BIO_CONTRIBUTION_LDS')} of its value.` },
        ],
        settings: settingRows([
          ['GBF2_TARGET', 'off, low, medium or high: which row of GBF2_TARGETS_DICT is used.'],
          ['GBF2_TARGETS_DICT', 'The share of the degradation to restore, in percent, by 2030, 2050 and 2100, for each level.'],
          ['GBF2_PRIORITY_DEGRADED_AREAS_PERCENTAGE_CUT', 'Where the priority ranking is cut: a higher cut keeps more cells as priority degraded areas.'],
          ['BIO_QUALITY_LAYER', 'The biodiversity layer the cells are ranked by.'],
          ['BIO_CONTRIBUTION_LDS', 'The share of its biodiversity value that savanna land keeps without early dry season burning.'],
          ELASTIC_SETTING,
        ]),
      }),
      GBF3_NVIS: (target) => ({
        label: 'How the target is set',
        sections: [
          { label: 'What it is', text: 'Target 3 (conserve 30 % of land), applied to each NVIS vegetation group or IBRA bioregion.' },
          ...extentSections('vegetation group', ''),
          { label: 'Left out', text: `A group with under ${settingOf('GBF3_NVIS_MIN_AREA_HA')} ha of restorable land in a region has no target there.` },
        ],
        settings: settingRows([
          ['GBF3_NVIS_TARGET', 'off; medium or high (one level for every group, from GBF3_TARGETS_DICT); SPECIFIED (one level per selected region); CSV_DEFINED (a level per group, from the input table).'],
          ['GBF3_TARGETS_DICT', 'The percent of the full extent to reach by 2030, 2050 and 2100, for each level.'],
          ['GBF3_NVIS_TARGET_CLASS', 'The vegetation classes used: major vegetation groups (NVIS_MVG) or subgroups (NVIS_MVS).'],
          ['GBF3_NVIS_REGION_MODE', 'AUSTRALIA: one target per group for the country. NRM: one per group and NRM region. IBRA_REG: one per IBRA bioregion.'],
          ['GBF3_NVIS_SEL_REGION_TARGETS', 'The regions that get a target and, under SPECIFIED, the level of each.'],
          ['GBF3_NVIS_TARGETS_OVERRIDE', 'A different level for single (region, group) pairs. Empty: none.'],
          ['GBF3_NVIS_MIN_AREA_HA', 'A group with less land than this inside a region gets no target there.'],
          ELASTIC_SETTING,
        ]),
      }),
      GBF4_SNES: (target) => ({
        label: 'How the target is set',
        sections: [
          { label: 'What it is', text: 'Target 4 (halt the loss of threatened species), applied to the Species of National Environmental Significance (EPBC Act).' },
          ...extentSections('species', ` Capped at what restoring all the land of the model could reach, less ${settingOf('GBF4_SNES_CAP_MARGIN')} points.`),
          { label: 'Left out', text: `A species with under ${settingOf('GBF4_SNES_MIN_AREA_HA')} ha in a region has no target there.` },
        ],
        settings: settingRows([
          ['GBF4_TARGET_SNES', 'off; medium or high (one level for every species, from GBF4_SNES_TARGETS_DICT); SPECIFIED (one level per selected region); CSV_DEFINED (a level per species, from the input table).'],
          ['GBF4_SNES_TARGETS_DICT', 'The percent of the full extent to reach by 2030, 2050 and 2100, for each level.'],
          ['GBF4_SNES_PRESENCE_CLASS', 'Which mapped range counts: where the species is likely to occur (LIKELY), or also where it may occur (LIKELY_AND_MAYBE).'],
          ['GBF4_SNES_REGION_MODE', 'AUSTRALIA: one target per species for the country. NRM: one per species and NRM region.'],
          ['GBF4_SNES_SEL_REGION_TARGETS', 'The regions that get a target and, under SPECIFIED, the level of each.'],
          ['GBF4_SNES_TARGETS_OVERRIDE', 'A different level for single (region, species) pairs. Empty: none.'],
          ['GBF4_SNES_CAP_MARGIN', 'Percentage points kept between a target and the most the species could reach.'],
          ['GBF4_SNES_MIN_AREA_HA', 'A species with less land than this inside a region gets no target there.'],
          ELASTIC_SETTING,
        ]),
      }),
      GBF4_ECNES: (target) => ({
        label: 'How the target is set',
        sections: [
          { label: 'What it is', text: 'Target 4, applied to the Ecological Communities of National Environmental Significance (EPBC Act).' },
          ...extentSections('community', ' Capped at what restoring all the land of the model could reach.'),
          { label: 'Left out', text: `A community with under ${settingOf('GBF4_ECNES_MIN_AREA_HA')} ha in a region has no target there.` },
        ],
        settings: settingRows([
          ['GBF4_TARGET_ECNES', 'off; medium or high (one level for every community, from GBF4_ECNES_TARGETS_DICT); SPECIFIED (one level per selected region); CSV_DEFINED (a level per community, from the input table).'],
          ['GBF4_ECNES_TARGETS_DICT', 'The percent of the full extent to reach by 2030, 2050 and 2100, for each level.'],
          ['GBF4_ECNES_PRESENCE_CLASS', 'Which mapped range counts: where the community is likely to occur (LIKELY), or also where it may occur (LIKELY_AND_MAYBE).'],
          ['GBF4_ECNES_REGION_MODE', 'AUSTRALIA: one target per community for the country. NRM: one per community and NRM region.'],
          ['GBF4_ECNES_SEL_REGION_TARGETS', 'The regions that get a target and, under SPECIFIED, the level of each.'],
          ['GBF4_ECNES_TARGETS_OVERRIDE', 'A different level for single (region, community) pairs. Empty: none.'],
          ['GBF4_ECNES_MIN_AREA_HA', 'A community with less land than this inside a region gets no target there.'],
          ELASTIC_SETTING,
        ]),
      }),
      GBF8: (target) => ({
        label: 'How the target is set',
        sections: [
          { label: 'What it is', text: 'Target 8 (reduce the impact of climate change on biodiversity), applied to each of about 10,600 species.' },
          { label: 'Climate', text: `A map per species of how suitable the climate is, for SSP ${settingOf('SSP')} and each year. A hectare counts by that suitability and the habitat value of its land use.` },
          { label: 'The target', text: 'A percentage of the suitable habitat of the species in the baseline climate. Natural land outside the model counts first; the model supplies the rest.' },
          { label: 'Over time', text: 'From the 2010 level in a straight line to the levels of 2030, 2050 and 2100.' },
        ],
        settings: settingRows([
          ['GBF8_TARGET', 'off; medium or high (one level for every species, from GBF8_TARGETS_DICT); CSV_DEFINED (a level per species, from the input table).'],
          ['GBF8_TARGETS_DICT', 'The percent of the baseline suitable habitat to keep by 2030, 2050 and 2100, for each level.'],
          ['SSP', 'The climate pathway of the suitability maps.'],
          ['BIO_ROW_GENERATION', 'True: the biodiversity targets are added to the model only as they are found short. Needed when every species has a target.'],
          ELASTIC_SETTING,
        ]),
      }),
    };

    // ── The drawers of the regional cards other than biodiversity (water, social licence, renewable energy) ──
    // (Traced 2026-11-16 from settings.py -- WATER_STRESS and its notes, REGIONAL_ADOPTION_*, RE_TARGET_LEVEL --
    // and data.py; the biodiversity cards of the Regional tab carry the same drawers as on the National tab.)
    const waterDrawer = () => ({
      label: 'How the limit is set',
      sections: [
        { label: 'What it is', text: `A floor on the water left in each ${settingOf('WATER_REGION_DEF') === 'River Region' ? 'river region' : 'drainage division'} (Bureau of Meteorology Geofabric).` },
        { label: 'Net yield', text: 'Area × (runoff under the land use − the water it takes), summed over the region, plus the runoff of land outside the model, less domestic and industrial use.' },
        { label: 'The limit', text: `${Math.round(Number(settingOf('WATER_STRESS')) * 100)} % of the historical yield of the region (WATER_STRESS), lowered where the climate alone would take the region under it.` },
        { label: 'Climate', text: String(settingOf('WATER_CLIMATE_CHANGE_IMPACT')) === 'on' ? `Runoff follows the climate pathway of the run (SSP ${settingOf('SSP')}) year by year.` : 'Runoff is held at its historical level (the climate change impact is off).' },
        { label: 'On this card', text: 'With no region picked, all regions are added: the scale, not whether each meets its limit. Set Region to Water and click a region.' },
      ],
    });
    const socialLicenceDrawer = () => {
      const mode = settingOf('REGIONAL_ADOPTION_CONSTRAINTS');
      return {
        label: 'How the caps are set',
        sections: [
          { label: 'What it is', text: 'A cap on the share of a region a land use may take, standing for how much change a region accepts. Filled in by hand, not computed.' },
          { label: 'The table', text: 'regional_adoption_zones.xlsx: per region and land use, the largest share in 2030, 2050 and 2100. Blank: no cap.' },
          { label: 'The other way', text: 'Or one percentage capping all non-agricultural land uses together in each region.' },
          { label: 'This run', text: mode === 'individual_cap' ? `From the table · regions: ${settingOf('REGIONAL_ADOPTION_ZONE')}`
            : mode === 'non_ag_sum_cap' ? `All non-agricultural land uses together ≤ ${settingOf('REGIONAL_ADOPTION_NON_AG_CAP')} % of each ${settingOf('REGIONAL_ADOPTION_NON_AG_REGION')} region`
            : 'off' },
        ],
      };
    };
    const renewableDrawer = () => ({
      label: 'How the targets are set',
      sections: [
        { label: 'What it is', text: 'The electricity (MWh a year) utility solar and onshore wind must generate, per technology and state (not the ACT).' },
        { label: 'Source', text: `Targets by year, state and technology (scenario: ${settingOf('RENEWABLE_TARGET_SCENARIO_TARGETS')}).` },
        { label: 'Generation', text: 'Installed capacity × the capacity factor of the cell × the hours of the year, less distribution loss. Existing capacity counts.' },
      ],
    });

    // ── Block 1 panel: the targets of the run ───────────────────────────────────────────────────
    const TARGET_TABS = ['National targets', 'Regional targets'];
    const selectTargetTab = ref(TARGET_TABS[0]);

    const targetPanels = computed(() => {
      if (!dataLoaded.value) return {};
      const year = Number(selectYear.value);
      const TARGET = '#1f1f1f', RESULT = '#0ea5e9';

      // GHG: the national limit and the net emissions of the solution
      const ghgSeries = ghgOverview.value?.['region_NRM']?.['AUSTRALIA'];
      const ghgLimit = pointsOf(ghgSeries, 'GHG emission limit');
      const ghgNet = pointsOf(ghgSeries, 'Net emissions');

      // Demand of the selected commodity: its demand types, and what the run has to produce (dashed)
      const demandParts = (demandOverview.value?.[selectDemandCommodity.value] || []).map(series => series.name === 'Production'
        ? { name: 'To produce', color: TARGET, dashStyle: 'ShortDash', data: pointsOf([series], series.name) }
        : { name: series.name, color: series.color, data: pointsOf([series], series.name) });

      // GBF2: the priority degraded area restored, as the solution reaches it
      const gbf2Level = settingOf('GBF2_TARGET');
      const gbf2Net = pointsOf(gbf2Overview.value?.['region_NRM']?.['AUSTRALIA']?.['Percent'], 'Net Value (%)');

      // GBF3 / GBF4 / GBF8: one card per target, three counts in the selected year: the regions the targets are
      // specified for, the items (vegetation groups, species, communities) under their target, and those at or above it.
      // Each tab lists every target: the card with the counts where the target is set (for the country, or per
      // region), a grey one on the other tab saying where to look, and grey ones on both when it is off.
      const gbfCard = (target, regional) => {
        const card = { key: `${target.key}_${regional ? 'regional' : 'national'}`, accent: METRIC_COLORS['Biodiversity'], title: `Biodiversity · ${target.short}`, drawer: gbfDrawers[target.key](target) };
        if (target.off) return { ...card, off: `Turned off: no ${target.label} target in this run` };
        if (target.regional !== regional) return { ...card, off: regional ? 'Set for the whole country: see National targets' : 'Set per region: see Regional targets' };
        const summary = gbfSummary.value[target.key];
        if (!summary) return { ...card, rows: [{ label: 'Loading…', value: '' }] };

        // the targets of this level; a region picked on the map that carries targets narrows them to itself
        const yearAt = summary.years.indexOf(year);
        const ofLevel = summary.rows.filter(row => regional ? row[1] !== 'AUSTRALIA' : row[1] === 'AUSTRALIA');
        const picked = selectRegion.value || 'AUSTRALIA';
        const narrowed = regional && ofLevel.some(row => row[1] === picked);
        const rowsShown = narrowed ? ofLevel.filter(row => row[1] === picked) : ofLevel;
        const regions = [...new Set(rowsShown.map(row => row[1]))];
        // One line per (region, item): its target and its score in the selected year. A target is read in the
        // year it binds; in a year without one the line shows the next target of the item, with its year.
        const What = target.what[0].toUpperCase() + target.what.slice(1);
        const percent = (value) => value == null ? '–' : `${(+value).toFixed(1)}`;
        const lines = rowsShown
          .map(row => {
            const boundAt = row[4][yearAt] != null ? yearAt : row[4].findIndex((value, position) => value != null && position >= yearAt);
            const targetAt = boundAt >= 0 ? boundAt : row[4].findIndex(value => value != null);
            const targetValue = targetAt >= 0 ? row[4][targetAt] : null;
            const score = row[3][yearAt];
            return {
              cells: [...(regional ? [row[1]] : []), row[2], targetValue == null ? '–' : percent(targetValue) + (targetAt === yearAt ? '' : ` (${summary.years[targetAt]})`), percent(score)],
              tone: score == null || targetValue == null ? '' : score >= targetValue - 1e-6 ? 'above' : 'under',
            };
          })
          .sort((first, second) => first.cells.join('|').localeCompare(second.cells.join('|')));
        const under = lines.filter(line => line.tone === 'under').length;
        return {
          ...card,
          setting: lines.length ? `${lines.length} ${regional ? 'region targets' : 'targets'} · ${under} under in ${year}` : '',
          table: {
            columns: [...(regional ? ['Region'] : []), What, 'Target %', `${year} %`],
            lines,
            empty: `No ${target.what} carries a target${narrowed ? ' in ' + picked : ''}`,
            hint: regional && !narrowed && regions.length > 1 ? `${regions.length} regions: click one on the map to list it alone` : narrowed ? picked : '',
          },
        };
      };
      const cardsOf = (regional) => gbfTargets.value.filter(target => !(regional && target.key === 'GBF8')).map(target => gbfCard(target, regional));

      // Water: a limit per water region (river region or drainage division). The data also holds the country
      // under 'AUSTRALIA', which is left out of the sum. With the Water level on the map and a region picked,
      // the card is that region's own limit and net yield.
      const waterRegions = Object.entries(watershedData.value || {}).filter(([name]) => name !== 'AUSTRALIA');
      const pickedWater = regionLevel.value === 'Water' && watershedData.value?.[selectRegion.value] && selectRegion.value !== 'AUSTRALIA' ? selectRegion.value : null;
      const waterShown = pickedWater ? [[pickedWater, watershedData.value[pickedWater]]] : waterRegions;
      const yearsOfWater = [...new Set(waterShown.flatMap(([, seriesList]) => pointsOf(seriesList, 'Water Net Yield').map(point => point[0])))].sort();
      const sumOver = (name) => yearsOfWater.map(waterYear => [waterYear, waterShown.reduce((total, [, seriesList]) => total + (valueAt(pointsOf(seriesList, name), waterYear) || 0), 0)]);
      const waterMany = WATER_GEO[waterRegionDef.value].many;

      // Social licence: the regional caps on how much land a region may give to a land use
      const adoptionMode = settingOf('REGIONAL_ADOPTION_CONSTRAINTS');
      const adoptionRows = adoptionMode === 'non_ag_sum_cap'
        ? [
          { label: 'Cap', value: `all non-agricultural land uses ≤ ${settingOf('REGIONAL_ADOPTION_NON_AG_CAP')} % of a region` },
          { label: 'Regions', value: `every ${settingOf('REGIONAL_ADOPTION_NON_AG_REGION')} region` },
        ]
        : [
          { label: 'Cap', value: 'per land use and region' },
          { label: 'Regions', value: String(settingOf('REGIONAL_ADOPTION_ZONE')) },
        ];
      // The caps themselves, one line per (region, land use) in the selected year; a region picked on the map that
      // has caps lists its own only. A report without the caps file keeps the two rows of text above.
      const capYearAt = (adoptionCaps.value.years || []).indexOf(year);
      const capRows = adoptionCaps.value.rows || [];
      const pickedRegion = selectRegion.value || 'AUSTRALIA';
      const capNarrowed = capRows.some(row => row[0] === pickedRegion);
      const capLines = (capNarrowed ? capRows.filter(row => row[0] === pickedRegion) : capRows)
        .filter(row => row[2][capYearAt] != null)
        .map(row => ({ cells: [row[0], row[1], (+row[2][capYearAt]).toFixed(1), compact(row[3][capYearAt])], tone: '' }));
      const adoptionTable = capRows.length ? {
        columns: ['Region', 'Land use', 'Cap %', 'Cap ha'],
        lines: capLines,
        empty: `No cap in ${year}`,
        hint: capNarrowed ? pickedRegion : `${new Set(capRows.map(row => row[0])).size} regions: click one on the map to list it alone`,
      } : null;

      // Renewable energy: the generation targets, per state (RE_TARGET_LEVEL) unless set for the country
      const renewableTypes = Object.entries(settingDict('RENEWABLES_OPTIONS')).filter(([, enabled]) => enabled).map(([type]) => type);
      const renewableLevel = String(settingOf('RE_TARGET_LEVEL') || 'STATE').toUpperCase();
      const renewableRegional = !['AUSTRALIA', 'NATIONAL'].includes(renewableLevel);
      // Targets and generation added over the regions that carry a target, a pair of lines per type
      const renewableSeries = [];
      const renewableRegions = Object.entries(renewableData.value?.['region_state'] || {}).filter(([region]) => region !== 'AUSTRALIA');
      const sumPoints = (pointLists) => {
        const totals = new Map();
        pointLists.flat().forEach(([pointYear, value]) => totals.set(pointYear, (totals.get(pointYear) || 0) + (value || 0)));
        return [...totals.entries()].sort((first, second) => first[0] - second[0]);
      };
      const RENEWABLE_COLORS = ['#f59e0b', '#0ea5e9'];
      // The types as the data names them (the report renames some), else as the setting does
      const typesInData = [...new Set(renewableRegions.flatMap(([, byType]) => Object.keys(byType || {})))].filter(type => type !== 'ALL');
      (typesInData.length ? typesInData : renewableTypes).forEach((type, position) => {
        const leaves = renewableRegions.map(([, byType]) => byType?.[type]?.['ALL'] || Object.values(byType?.[type] || {})[0] || []);
        const target = sumPoints(leaves.map(seriesList => pointsOf(seriesList, 'Target')));
        const generation = sumPoints(leaves.flatMap(seriesList => seriesList.filter(series => series.name !== 'Target').map(series => pointsOf([series], series.name))));
        if (target.length) renewableSeries.push({ name: `${type} target`, color: RENEWABLE_COLORS[position % 2], dashStyle: 'ShortDash', data: target });
        if (generation.length) renewableSeries.push({ name: `${type} generation`, color: RENEWABLE_COLORS[position % 2], data: generation });
      });
      const renewableCard = {
        key: 'renewable', accent: METRIC_COLORS['Renewable energy'], title: 'Renewable energy', unit: 'MWh',
        off: renewableTypes.length === 0 ? 'Turned off: no renewable energy in this run' : null,
        series: renewableSeries,
        empty: 'Renewable generation not written for this run',
        drawer: renewableDrawer(),
      };

      // Within a tab, the targets that are switched off go to the bottom
      const offLast = (cards) => [...cards.filter(card => !card.off), ...cards.filter(card => card.off)];

      const tabs = {
        'National targets': [
          {
            key: 'demand', accent: METRIC_COLORS['Demand'], title: 'Demand', unit: selectDemandCommodity.value === 'Dairy' ? 'kL' : 't',
            options: demandCommodities.value, option: selectDemandCommodity.value, legend: false,
            series: demandParts,
            drawer: demandDrawer(),
          },
          {
            key: 'ghg', accent: METRIC_COLORS['GHG'], title: 'GHG emissions', unit: 't CO2e',
            off: settingOf('GHG_EMISSIONS_LIMITS') === 'off' ? 'Turned off: no GHG limit in this run' : null,
            series: [
              ...(ghgLimit.length ? [{ name: 'Limit', color: TARGET, dashStyle: 'ShortDash', data: ghgLimit }] : []),
              ...(ghgNet.length ? [{ name: 'Net emissions', color: RESULT, data: ghgNet }] : []),
            ],
            drawer: ghgDrawer(),
          },
          {
            key: 'biodiversity_gbf2', accent: METRIC_COLORS['Biodiversity'], title: 'Biodiversity · GBF2', unit: '%',
            off: gbf2Level === 'off' ? 'Turned off: no GBF2 (priority degraded areas) target in this run' : null,
            series: gbf2Net.length ? [{ name: 'Priority degraded area restored', color: RESULT, data: gbf2Net }] : [],
            empty: 'GBF2 scores not written for this run',
            drawer: gbfDrawers.GBF2(),
          },
          ...cardsOf(false),
        ],
        'Regional targets': [
          {
            key: 'water', accent: METRIC_COLORS['Water'], title: pickedWater ? `Water · ${pickedWater}` : 'Water', unit: 'ML',
            off: settingOf('WATER_LIMITS') === 'off' ? 'Turned off: no water limit in this run' : null,
            series: waterShown.length ? [
              { name: pickedWater ? 'Limit' : `Limit (all ${waterRegions.length} ${waterMany})`, color: TARGET, dashStyle: 'ShortDash', data: sumOver('Water Limit (model)') },
              { name: 'Net yield', color: RESULT, data: sumOver('Water Net Yield') },
            ] : [],
            rows: pickedWater ? [] : [{ label: 'Per region', value: `set Region to Water and click one of the ${waterMany}`, off: true }],
            drawer: waterDrawer(),
          },
          {
            key: 'social_licence', accent: METRIC_COLORS['Social licence'], title: 'Social licence',
            off: adoptionMode === 'off' || adoptionMode == null ? 'Turned off: no regional adoption cap in this run' : null,
            ...(adoptionTable ? { setting: `${capLines.length} caps in ${year}`, table: adoptionTable } : { rows: adoptionRows }),
            drawer: socialLicenceDrawer(),
          },
          ...cardsOf(true),
        ],
      };
      tabs[renewableRegional ? 'Regional targets' : 'National targets'].push(renewableCard);
      return Object.fromEntries(Object.entries(tabs).map(([tab, cards]) => [tab, offLast(cards)]));
    });

    // ── The Inputs panel: the source card of the selected input, and the theme's tables ──
    // Where an input comes from and how it was processed, for the inputs whose card is that note and not a chart.
    // (Traced 2026-11-13 from Data-Master/GAEZv4, script_3_agriculture_climate_damage.py, dataprep.py and get_ccimpact.)
    const shareText = (key) => { const share = Number(settingOf(key)); return Math.abs(share - 1 / 3) < 1e-6 ? 'one third' : `${Math.round(share * 100)} %`; };
    const ECONOMICS_INPUTS = ['Agricultural cost', 'Agricultural revenue', 'Non-agricultural cost', 'Non-agricultural revenue', 'Transition cost to agriculture', 'Transition cost to non-agriculture', 'Transition cost matrix'];
    const ECONOMICS_SOURCE = {
      title: 'Economics of agricultural land',
      sections: () => [
        { label: 'Source', text: 'The LUTO profit map for 2010: prices and costs per tonne, head and hectare, by SA2 for crops and by cell for livestock. Maps are AUD per hectare.' },
        { label: 'Cost · the parts', text: 'Quantity: cost per tonne × yield, or per head × stocking rate. Area: cost per hectare. Fixed labour, operating and depreciation: the fixed costs of the farm. Water: water required × delivery price, plus livestock drinking water.' },
        { label: 'Cost · over the years', text: 'The yearly multipliers of the parts are all 1 at present: only quantity cost moves, with yield.' },
        { label: 'Revenue', text: 'Crops: price × yield. Livestock: stocking rate × products per head × price (meat, live exports, wool, milk). Unallocated land earns nothing.' },
        { label: 'Transition cost', text: `From the land use of the base year to the one selected: the table cost (the transition cost matrix), the water licence for extra water, ${settingOf('NEW_IRRIG_COST')} AUD/ha to set up irrigation or ${settingOf('REMOVE_IRRIG_COST')} to remove it, and the carbon released by clearing natural land × the carbon price.` },
        { label: 'In the model', text: 'The objective: revenue − cost − transition cost, each as area × these values.' },
      ],
      nonAg: {
        title: 'Economics of non-ag land',
        sections: () => {
          return [
            { label: 'Cost', text: `Yearly maintenance from the settings (${settingOf('EP_ANNUAL_MAINTENANCE_COST_PER_HA_PER_YEAR')} AUD/ha for environmental plantings), the same in every cell. BECCS: the CSIRO BECCS dataset (2021).` },
            { label: 'Revenue', text: 'The carbon stored in the year × the carbon price. BECCS adds energy revenue. Destocked natural land earns nothing.' },
            { label: 'Mixed land uses', text: `Agroforestry and farm forestry: the tree share (${shareText('AF_PROPORTION')} and ${shareText('CP_BELT_PROPORTION')}) takes the values of the planting, the rest those of dryland sheep or beef in the cell.` },
            { label: 'Transition cost', text: `The cost of establishing the planting plus the table cost of leaving the old land use. Mixed land uses plant only their tree share. Fencing (riparian and mixed): ${settingOf('FENCING_COST_PER_M')} AUD per metre; leaving irrigated land: ${settingOf('REMOVE_IRRIG_COST')} AUD/ha.` },
            { label: 'Destocked natural land', text: 'From livestock on natural land only: the table cost plus a share of the environmental planting cost, by the habitat gained.' },
          ];
        },
      },
    };
    // Exclusion (traced 2026-11-16 from ag_transitions.get_ag_allowed_mrj, non_ag_transitions.get_non_ag_ub_matrices,
    // data.py (EXCLUDE from x_mrj.nc, T_MAT, the no-go regions) and settings.py). Every input of the theme shows
    // the same two cards.
    const EXCLUSION_INPUTS = ['Agricultural land use allowed', 'Non-agricultural share allowed', 'Transition exclusion matrix'];
    const NO_GO_TEXT = 'A run can shut land uses out of areas drawn by the user (EXCLUDE_NO_GO_LU, with the areas in NO_GO_VECTORS).';
    const EXCLUSION_SOURCE = {
      title: 'Where agriculture may go',
      sections: () => [
        { label: 'What it is', text: 'Three rules decide where a land use may go: the maps show the first two, the matrix the third.' },
        { label: 'Present in the region', text: 'An agricultural land use is allowed in a cell only if it was present in the SA2 of the cell in 2010. Map: 1 allowed, 0 not.' },
        { label: 'No-go areas', text: NO_GO_TEXT },
        { label: 'The exclusion matrix', text: 'Which land use may follow which: a pair with no cost in the transition table can not happen. Farmed land does not go back to natural land; it can only be planted or destocked.' },
        { label: 'In the model', text: 'No variable where a land use is not allowed, no arc the matrix shuts out. These are not costs: the model can not pay to pass.' },
      ],
      nonAg: {
        title: 'Where non-ag land may go',
        sections: () => [
          { label: 'What it is', text: 'The largest share of the cell the land use may take, 0 to 1.' },
          { label: 'From the land of the cell', text: 'The share of the cell held in the base year by land uses the matrix lets move to it. Plantings are reachable from farmed land only.' },
          { label: 'Destocked natural land', text: 'Only livestock on natural land can be destocked: its share of the cell, plus what is destocked already.' },
          { label: 'Riparian plantings', text: `The share of the cell within ${settingOf('RIPARIAN_PLANTING_BUFFER_WIDTH')} m of a stream.` },
          { label: 'No going back', text: 'Plantings and BECCS stay once placed. Destocked land can be restocked.' },
          { label: 'No-go areas', text: NO_GO_TEXT },
        ],
      },
    };
    // Biodiversity (traced 2026-11-16 from data.py -- the quality layer, the connectivity score, the GBF2 mask, the
    // habitat contribution look-up -- settings.py and docs/CLAUDE_GBF2.md). One card for every input of the theme.
    const BIODIVERSITY_SOURCE = {
      title: 'Biodiversity inputs',
      sections: () => [
        { label: 'Biodiversity quality', text: `How much a cell matters for biodiversity, 0 to 1: its Zonation priority rank, under one of eight layers (the buttons). It is the weight of the cell in the biodiversity score.` },
        { label: 'Connectivity', text: settingOf('CONNECTIVITY_SOURCE') === 'NONE'
          ? 'CONNECTIVITY_SOURCE = NONE: the quality is not weighted by connectivity.'
          : `Weighted by connectivity to other habitat (${settingOf('CONNECTIVITY_SOURCE') === 'NCI' ? 'the National Connectivity Index of DCCEEW' : 'the distance to the nearest natural land'}): the least connected cell keeps ${settingOf('CONNECTIVITY_LB')} of its quality.` },
        { label: 'GBF2 priority degraded areas', text: `The cells at or above the top ${settingOf('GBF2_PRIORITY_DEGRADED_AREAS_PERCENTAGE_CUT')} of the Zonation performance curve: where restoring gains the most. The GBF2 target counts these cells only.` },
        { label: 'Habitat contribution (HCAS)', text: `The share of its pre-1750 value that land keeps under a land use, 0 to 1. Agricultural land uses: from the Habitat Condition Assessment System: a percentile of the condition observed under each land use, or values set by hand (the buttons). Plantings, destocked land and savanna burning: settings of the run.` },
        { label: 'In the model', text: 'Score of a cell = quality × habitat contribution of its land use × area. The GBF targets are floors on sums of such scores.' },
        { label: 'Vegetation groups, species and communities', text: 'The share of each cell a vegetation group (NVIS), species or community covers: the mean over a block of full-resolution cells, so a small range still shows. GBF4 has two presence classes.' },
        { label: 'GBF8 species', text: 'The climate suitability for each species (0 to 100), as in the input rasters and on their grid, for two years.' },
      ],
    };
    // Ag management (traced 2026-11-17 from data.py -- the five bundle workbooks and their sheets -- the notes sheets
    // of the workbooks, economics/agricultural/{quantity,revenue,cost,ghg,water,biodiversity,transitions}.py, which
    // read the columns, and settings.py).
    const BUNDLE_SOURCE = {
      title: 'Agricultural management',
      sections: () => [
        { label: 'What it is', text: 'Practices a farm can add to a land use without changing the land use. Each has a table of what it changes on that land use, year by year to 2100. There is no map: the same numbers apply in every cell.' },
        { label: 'Source', text: 'Five workbooks of the LUTO inputs, one per practice ("bundles" of measures), with a sheet per land-use group. Their notes sheets give the studies behind each number, among them Net Zero Australia (2022) and Wiedemann et al. (2023) for livestock methane, and ANZBIG (2020) and CSIRO (Fitch et al., 2022) for biochar.' },
        { label: 'The practices', text: 'Methane reduction: the feed supplement Asparagopsis for beef, sheep and dairy. Agricultural technology (fertiliser): precision agriculture, with slow-release fertilisers and nitrification inhibitors. Agricultural technology (energy): on-farm energy measures. Regenerative agriculture: ecological grazing. Biochar: a soil amendment for crops and horticulture.' },
        { label: 'Land-use groups', text: 'Crop practices have one table for cropping, one for intensive cropping and one for horticulture; livestock practices one per land use.' },
        { label: 'How the model uses it', text: 'A multiplier of 1 means no change. Yield, water use and each emission source of the land use are multiplied by the value of the year; the annual cost is added per hectare or per animal; soil carbon and biodiversity change by the multiplier less 1.' },
        { label: 'Adoption ceiling', text: `The largest share of a land use that can take up the practice in a year. The model can adopt less, not more. A run can scale the ceilings (TECH_ADOPT_MULT).` },
        { label: 'Not read by the model', text: 'Some columns of the workbooks are shown for reference only: nitrogen use efficiency, nitrogen surplus, water retention, the total cost of ecological grazing, and the biodiversity column of every practice but biochar.' },
      ],
    };
    const INPUT_SOURCES = {
      'Yield trend': {
        title: 'Yield trend',
        sections: () => [
          { label: 'What it is', text: 'A multiplier on 2010 yield, per product and year, for the whole country. No map: the two charts show the two trends the model can use.' },
          { label: 'BAU projection', text: 'From the earlier LUTO data archive: one projection from 2010 for crops, beef, dairy, sheep meat and wool, dryland and irrigated.' },
          { label: 'ABARES scenarios', text: 'ABARES productivity estimates (from 1989): observed to 2023, then an ETS trend as the low, medium, high or very high projection, all relative to 2010.' },
          { label: 'Groups', text: 'Four ABARES series: cropping (crops and horticulture), beef, dairy and sheep.' },
          { label: 'In the model', text: 'It scales yield per hectare, on top of the climate change impact.' },
        ],
      },
      'Climate change': {
        title: 'Climate change impact on yield',
        sections: () => [
          { label: 'Source', text: 'FAO GAEZ v4 attainable yield of current cropland (high input, ENSEMBLE climate model) for 2011–2040, 2041–2070 and 2071–2100, used as 2020, 2050 and 2080.' },
          { label: 'Multiplier', text: 'Each crop layer is averaged per SA2 and divided by its 2010 yield. The map shows the % change.' },
          { label: 'Cleaning', text: 'Multipliers above 3 or outside the 5th–95th percentile are dropped; a gap takes the mean of the SA4, then of wider regions.' },
          { label: 'Livestock', text: 'GAEZ covers crops only: pasture land uses take the multiplier of Hay.' },
          { label: 'In the model', text: 'Interpolated between 2010, 2020, 2050 and 2080; it multiplies crop and livestock yield (1 where there is no value).' },
        ],
      },
      // (Traced 2026-11-13 from Data-Master script_6_water_yield_modelling.py and script_2_assemble_agricultural_data.py,
      // dataprep.py, data.py and economics/agricultural/water.py.)
      'Runoff': {
        title: 'Runoff (water yield)',
        sections: () => [
          { label: 'Source', text: 'The InVEST annual water yield method: precipitation less actual evapotranspiration, per 1 km cell and year. Climate: WorldClim 2.1 CMIP6, the mean of 8 models, 1970–2100. Soil: the Soil and Landscape Grid of Australia.' },
          { label: 'Two covers', text: 'Two values per cell: shallow-rooted cover (crops, grass) and deep-rooted cover (trees), which uses more water and so runs off less.' },
          { label: 'In the model', text: 'Crops and modified pasture take the shallow-rooted value, deep-rooted land uses the other, natural land a mix by its pre-1750 vegetation.' },
        ],
      },
      // Two cards, as for the GHG input: the agricultural requirement and the water use of non-agricultural land
      // come from different data and methods, so each has its own card (`nonAg`), one under the other.
      'Water requirement': {
        title: 'Water requirement of agricultural land',
        sections: () => [
          { label: 'Source', text: 'The CSIRO profit map (2010–11). Crops: irrigation water per hectare, the SA2 mean of the crop. Livestock: irrigation and drinking water per head.' },
          { label: 'Livestock', text: 'Per head becomes per hectare through the stocking rate. A gap takes the SA4, state, then national mean.' },
          { label: 'Over time', text: 'Irrigation is the same every year. Dryland drinking water follows the stocking rate, which moves with the climate impact on yield.' },
          { label: 'In the model', text: 'Net yield of a land use = runoff − requirement.' },
        ],
        nonAg: {
          title: 'Water use of non-agricultural land',
          sections: () => [
            { label: 'What it is', text: 'Non-agricultural land uses have a net yield, no requirement. Shown: the water they use against cleared land (2010 shallow-rooted runoff less their net yield), the same every year.' },
            { label: 'Groups', text: 'Land uses the model treats alike share a button and a value.' },
            { label: '1 · Carbon plantings and BECCS', text: 'The runoff of deep-rooted cover: the highest water use.' },
            { label: '2 · Environmental and riparian plantings', text: 'The runoff of the pre-1750 vegetation of the cell.' },
            { label: '3 · Farm forestry', text: `${Math.round(Number(settingOf('CP_BELT_PROPORTION')) * 100)} % of the cell as carbon plantings, the rest dryland sheep or beef with its drinking water.` },
            { label: '4 · Agroforestry', text: `${Math.round(Number(settingOf('AF_PROPORTION')) * 100)} % of the cell as environmental plantings, the rest dryland sheep or beef. The lowest water use.` },
            { label: 'Destocked natural land', text: 'Not on the map: never cleared, it gives up no runoff. It takes the net yield of unallocated natural land.' },
            { label: 'In the model', text: 'They enter the water balance of the region with their net yield.' },
          ],
        },
      },
      // The GHG inputs. (Traced 2026-11-13 from Data-Master ag_ghg.py, script_9_reforestation_carbon_data.py and
      // script_4_assemble_biophysical_data.py, dataprep.py, data.py and economics/*/ghg.py.)
      // Two cards: the agricultural land uses (Dryland / Irrigated) and the non-agricultural ones (Non-ag) come from
      // different data and methods, so each has its own card (`nonAg`), one under the other.
      'Emissions and sequestration': {
        title: 'Emissions of agricultural land',
        sections: () => {
          return [
            { label: 'Sign', text: 'Positive: emissions. Negative: sequestration (t CO2e per hectare and year).' },
            { label: 'Source', text: 'The CSIRO table of emissions by commodity, source and SA2 for 2010 (Navarro et al.). A gap takes the SA4, state, then national mean.' },
            { label: 'Crops', text: 'Soil N2O, fertiliser and pesticide production, and field operations. The same every year.' },
            { label: 'Livestock', text: 'Enteric methane, manure, dung and urine, leaching, and energy and fodder, per head × the stocking rate of the cell and year. Irrigated pasture adds the emissions of growing it.' },
            { label: 'Unallocated land', text: 'Zero: it grows nothing and carries no livestock.' },
            { label: 'In the model', text: 'Emissions = area × this value over every land use, plus ag-management effects, clearing and off-land emissions; held under the limits where GHG_EMISSIONS_LIMITS is on.' },
          ];
        },
        nonAg: {
          title: 'Sequestration of non-ag land',
          sections: () => {
            const effectsWindow = settingOf('CARBON_EFFECTS_WINDOW');
            return [
              { label: 'Sign', text: 'Positive: emissions. Negative: sequestration (t CO2e per hectare and year).' },
              { label: 'Source', text: 'FullCAM carbon stocks of trees, debris and soil by age. Environmental plantings: the 2024 release. A gap takes the nearest cell.' },
              { label: 'Carbon plantings', text: 'The 2024 FullCAM results for carbon plantings do not look sensible. They are the 2024 environmental plantings × the 2020 ratio of carbon to environmental plantings.' },
              { label: 'Per year', text: `The stock at age ${effectsWindow} ÷ ${effectsWindow} years (CARBON_EFFECTS_WINDOW). Trees and debris are discounted for fire (FIRE_RISK = ${settingOf('FIRE_RISK')}) and reversal (RISK_OF_REVERSAL = ${settingOf('RISK_OF_REVERSAL')}); soil is not.` },
              { label: 'Mixed land uses', text: `Agroforestry: ${Math.round(Number(settingOf('AF_PROPORTION')) * 100)} % environmental plantings; farm forestry: ${Math.round(Number(settingOf('CP_BELT_PROPORTION')) * 100)} % carbon plantings; the rest keeps the emissions of dryland sheep or beef.` },
              { label: 'BECCS', text: 'CO2 captured and stored, from the CSIRO BECCS dataset (2021).' },
              { label: 'Destocked natural land', text: 'On land grazed in 2010: the carbon stock of natural land × the share of habitat grazing had degraded, over the same window.' },
              { label: 'In the model', text: 'Emissions = area × this value over every land use, plus ag-management effects, clearing and off-land emissions; held under the limits where GHG_EMISSIONS_LIMITS is on.' },
            ];
          },
        },
      },
      // Economics (traced 2026-11-16 from economics/agricultural/cost.py, revenue.py and transitions.py,
      // economics/non_agricultural/cost.py, revenue.py and transitions.py, tools.get_ag_to_ag_water_delta_matrix,
      // data.py and dataprep.py). Every input of the theme shows the same two cards, ECONOMICS_SOURCE below:
      // agricultural land and non-agricultural land.
      ...Object.fromEntries(ECONOMICS_INPUTS.map(name => [name, ECONOMICS_SOURCE])),
      ...Object.fromEntries(EXCLUSION_INPUTS.map(name => [name, EXCLUSION_SOURCE])),
      'Savanna burning: abatement': {
        title: 'Savanna burning',
        sections: () => [
          { label: 'What it is', text: 'Early dry season burning of northern savannas avoids the larger emissions of late fires. The map: abatement per hectare and year where eligible.' },
          { label: 'Source', text: 'A LUTO input layer: the area eligible under the Emissions Reduction Fund method and its abatement. Its preparation script is not among the data-master scripts.' },
          { label: 'In the model', text: 'An ag-management option on eligible cells: it lowers emissions by this amount every year.' },
        ],
      },
      'Climate change impact on runoff': {
        title: 'Climate change impact on runoff',
        sections: () => [
          { label: 'What it is', text: 'Derived from the runoff input: the runoff of a land use in this year less that of 2010 (ML per hectare).' },
          { label: 'In the model', text: 'Not added by the model: each year\'s runoff already carries the climate.' },
          { label: 'The limit', text: `${Math.round(Number(settingOf('WATER_STRESS')) * 100)} % of the historical yield of the region, lowered where the driest year of the run would fall under it.` },
        ],
      },
    };
    const inputPanels = computed(() => {
      if (!dataLoaded.value) return [];
      const year = Number(selectYear.value);
      // The bundles: their chart lies over the map (HomeBundleChart); the card says where they come from
      if (isAssumptionTheme.value) {
        return [{ key: 'assumptions', accent: METRIC_COLORS[selectTheme.value] || '#64748b', title: 'Assumptions', setting: 'settings', sections: [
          { label: 'What they are', text: 'Settings that are not input data, but decide how the model reads the data and what it optimises: how prices respond, how costs are counted, which climate and carbon rules apply.' },
          { label: 'How to read them', text: 'Every setting is shown with all its options; the option of this run has the dashed border. A setting that is a number or a table shows its value in this run. Hover an option to see what it means.' },
          { label: 'Where the rest are', text: 'A setting that picks between versions of an input (climate pathway, CO2 fertilisation, carbon window, quality layer) is not repeated here: it is the highlighted button on that input. Every setting of the run, with a note on each, is on the Settings and Log page.' },
        ] }];
      }
      if (isBundleTheme.value) {
        return [{ key: 'bundle source', accent: METRIC_COLORS[selectTheme.value], title: BUNDLE_SOURCE.title, setting: 'data source', sections: BUNDLE_SOURCE.sections() }];
      }

      const entry = inputEntry.value;
      const cards = [];
      // a part of an input (the quantity cost of the agricultural cost) shows the card of the input it is a part of
      const source = entry ? (INPUT_SOURCES[entry.input] || INPUT_SOURCES[inputParent(entry.input)] || (selectTheme.value === 'Biodiversity' ? BIODIVERSITY_SOURCE : null)) : null;
      if (source) {
        // Where the input comes from and how it was made
        // one card, or two where the agricultural and the non-agricultural land uses have their own (`nonAg`):
        // both are always shown, whichever button is selected
        [['source', source], ['source_non_ag', source.nonAg]].filter(([, part]) => part).forEach(([key, part]) => {
          cards.push({ key, accent: METRIC_COLORS[selectTheme.value], title: part.title, setting: 'data source', sections: part.sections() });
        });
      }

      return cards;
    });


    onMounted(async () => {

      // Load required data
      await loadScript("./data/Supporting_info.js", 'Supporting_info', VIEW_NAME);
      await loadScript("./data/geo/NRM_AUS.js", 'NRM_AUS', VIEW_NAME);
      await loadScript("./data/geo/AUS_STATE.js", 'AUS_STATE', VIEW_NAME);

      // The land-use map (all water supplies) and its legend. The legend registry is shared by every
      // map view, so it is not tied to this view's clean-up.
      await loadScript('data/map_layers/legend_registry.js', 'legend_registry');
      const lumapLegend = window.legend_registry?.['lumap'] || {};
      // A report written before the legend had columns: one column of every entry, all enabled
      landuseLegendColumns.value = lumapLegend.columns
        ? [lumapLegend.columns.land, lumapLegend.columns.non_ag]
        : [Object.entries(lumapLegend.legend || {}).map(([name, color]) => ({ name, color, enabled: true }))];
      try {
        await loadScript(shareMapEntry.indexPath, shareMapEntry.indexName, VIEW_NAME);
        shareLayerNames.value = window[shareMapEntry.indexName]?.tree || [];
      } catch (error) {
        console.warn('No land-use share layers in this report: the legend is not clickable.', error);
      }
      ensureLanduseLayer(landuseMapEntry.layerPrefix, ['ALL']);

      runScenario.value = Object.fromEntries(window['Supporting_info']['model_run_settings'].map(item => [item.parameter, item.val]));

      const chartGhg = chartRegister['GHG']['overview']['sum'];
      const chartDemand = chartRegister['Production']['overview']['commodity'];
      const chartWatershed = chartRegister['Water']['Watershed']['overview'];
      const chartGbf2 = chartRegister['Biodiversity']['GBF2']['overview']['sum'];
      const chartInputs = chartRegister['Inputs'];
      const chartRenewable = chartRegister['Renewable']['Ag Mgt'] || Object.values(chartRegister['Renewable'])[0];

      // Every file is independent: fetch them in parallel. One a report does not hold leaves its card empty
      // (the GBF2 overview with the target off; the yield assumptions and the input layers before 2026-11-12).
      const optional = async (entry) => {
        try {
          await loadScript(entry['path'] || entry['indexPath'], entry['name'] || entry['indexName'], VIEW_NAME);
          return window[entry['name'] || entry['indexName']];
        } catch (err) {
          console.warn(`Not in this report: ${entry['path'] || entry['indexPath']}`, err);
          return null;
        }
      };
      const [ghg, demand, watershed, gbf2, renewable, inputIndex, catalogue, tables, bundles, trend, caps, matrix, species, yearsOfInputs] = await Promise.all([
        optional(chartGhg), optional(chartDemand), optional(chartWatershed),
        runScenario.value['GBF2_TARGET'] === 'off' ? null : optional(chartGbf2),
        window['Supporting_info']['renewables_enabled'] ? optional(chartRenewable) : null,
        optional(inputMapEntry),
        optional(chartInputs['catalogue']), optional(chartInputs['tables']), optional(chartInputs['bundles']), optional(chartInputs['yield_trend']), optional(chartInputs['adoption_caps']), optional(chartInputs['transition_matrix']), optional(chartInputs['species']), optional(chartInputs['years']),
      ]);
      renewableData.value = renewable;
      ghgOverview.value = ghg;
      demandOverview.value = demand;
      watershedData.value = watershed || {};
      gbf2Overview.value = gbf2;
      inputTree.value = inputIndex?.tree || {};
      inputTables.value = tables || {};
      inputBundles.value = bundles || {};
      yieldTrend.value = trend || {};
      adoptionCaps.value = caps || {};
      transitionMatrix.value = matrix || {};
      speciesSets.value = species || {};
      inputCatalogue.value = catalogue || {};

      runYears.value = window['Supporting_info']['years'];
      // The Inputs view opens on the input year nearest to the first year of the run
      inputYears.value = yearsOfInputs || [];
      inputYearIndex.value = inputYears.value.reduce((best, year, position) =>
        Math.abs(year - runYears.value[0]) < Math.abs(inputYears.value[best] - runYears.value[0]) ? position : best, 0);

      // A region carried over from another page may be a state: open the maps at its level
      const region0 = selectRegion.value;
      if ((window['AUS_STATE']?.features || []).some(feature => feature.properties['STATE_NAME'] === region0)) {
        regionLevel.value = 'State';
      }

      await nextTick(() => { dataLoaded.value = true; });
      loadGbfSummaries();                    // not awaited: the cards fill in when the files land

    });

    // The names of the two levels differ, so a region does not carry over a switch of level
    const waterRegionDef = computed(() => WATER_GEO[settingOf('WATER_REGION_DEF')] ? settingOf('WATER_REGION_DEF') : 'Drainage Division');
    // The level the map draws: the water regions go by the name of their definition
    const mapRegionLevel = computed(() => regionLevel.value === 'Water' ? waterRegionDef.value : regionLevel.value);
    const setRegionLevel = async (level) => {
      if (level === regionLevel.value) return;
      selectRegion.value = 'AUSTRALIA';
      if (level === 'Water') {
        // the boundaries are loaded on first use (the river regions are 1.7 MB); the map needs them before the level changes
        const geo = WATER_GEO[waterRegionDef.value];
        try {
          await loadScript(geo.path, geo.name, VIEW_NAME);
        } catch (error) {
          console.warn(`Not in this report: ${geo.path}`, error);
          return;
        }
        selectTargetTab.value = 'Regional targets';      // where the Water card is
      }
      regionLevel.value = level;
    };

    // Memory cleanup on component unmount
    onUnmounted(() => {
      // the selected region is shared with the other pages, which do not know a water region
      if (regionLevel.value === 'Water') selectRegion.value = 'AUSTRALIA';
      window.MemoryService.cleanupViewData(VIEW_NAME);
      window.MemoryService.cleanupViewData(VIEW_NAME + 'Input');
    });

    const _state = {
      yearIndex,
      runScenario,
      scenarioLabel,
      dataLoaded,

      availableYears,
      selectYear,
      selectRegion,
      REGION_LEVELS,
      regionLevel,
      setRegionLevel,
      mapRegionLevel,
      VIEWS,
      VIEW_LABELS,
      selectView,
      showLabels,

      // Block 1
      landuseMapData,
      landuseLegendColumns,
      shareLayerNames,
      selectLanduse,
      toggleLanduse,
      simulationMapData,
      landuseEmpty,
      selectCardOption,
      landuseEntryTitle,
      TARGET_TABS,
      selectTargetTab,
      targetPanels,

      // Block 2
      availableThemes,
      selectTheme,
      themeInputs,
      mainInputs,
      selectMainInput,
      inputParts,
      selectInput,
      inputEntry,
      inputLmOptions,
      inputOptionRows,
      optionValueLabel,
      optionChoice,
      selectInputOption,
      inputYearNote,
      speciesSet,
      speciesSets,
      selectSpeciesVariant,
      variantLabel,
      selectSpeciesItem,
      speciesItemOptions,
      speciesLeaf,
      inputLuOptions,
      selectInputLm,
      selectInputLu,
      LM_LABELS,
      isBundleTheme,
      isNoMapTheme,
      isAssumptionTheme,
      showAssumptions,
      assumptionGroups,
      isChartInput,
      yieldTrend,
      showTrendCharts,
      showMatrix,
      inputTable,
      matrixAllowedOnly,
      maskLegend,
      transitionMatrix,
      mapPadLeft,
      bundleOptions,
      bundleVariables,
      bundleLabel,
      bundleChart,
      showBundleChart,
      selectBundle,
      selectBundleVariable,
      inputLegendTitle,
      inputMapData,
      inputLegend,
      inputPanels,
    };
    window._debug[VIEW_NAME] = _state;
    return _state;
  },

  template: /*html*/`
    <div v-if="dataLoaded">
      <div class="flex flex-col">

        <!-- The map | the panel. The View buttons switch both: the simulation with its targets, or the inputs -->
        <div class="flex mr-4 gap-4 mb-4 mt-4 flex-row h-[calc(100vh-40px)] min-h-[620px]">

          <div class="relative flex-1 min-w-0 rounded-[10px] bg-white shadow-md overflow-hidden">

            <!-- Controls over the map's top-left corner, one row each: the year, the view, the region and the labels -->
            <div class="absolute top-[24px] left-[12px] z-[100] flex flex-col gap-3 pointer-events-none">
              <home-year-control :years="availableYears" v-model:index="yearIndex" />

              <div class="flex flex-col gap-3 pointer-events-auto w-fit">

                <!-- View -->
                <div class="flex items-center gap-2">
                  <span class="text-[0.8rem] font-semibold uppercase tracking-wide text-gray-700 px-1 py-0.5 w-[4rem]">View</span>
                  <div class="flex w-fit rounded overflow-hidden border border-sky-500">
                    <button
                      v-for="view in VIEWS" :key="view"
                      @click="selectView = view"
                      class="text-[0.85rem] px-3.5 py-1"
                      :class="selectView === view ? 'bg-sky-500 text-white' : 'bg-white text-sky-600 hover:bg-sky-100'"
                    >{{ VIEW_LABELS[view] }}</button>
                  </div>
                </div>

                <!-- Region level; the way back to the whole country once a region is selected -->
                <div class="flex items-center gap-2">
                  <span class="text-[0.8rem] font-semibold uppercase tracking-wide text-gray-700 px-1 py-0.5 w-[4rem]">Region</span>
                  <div class="flex w-fit rounded overflow-hidden border border-sky-500">
                    <button
                      v-for="level in REGION_LEVELS"
                      :key="level"
                      @click="setRegionLevel(level)"
                      class="text-[0.85rem] px-3.5 py-1"
                      :class="regionLevel === level ? 'bg-sky-500 text-white' : 'bg-white text-sky-600 hover:bg-sky-100'"
                    >
                      {{ level }}
                    </button>
                  </div>
                  <button
                    v-if="selectRegion !== 'AUSTRALIA'"
                    @click="selectRegion = 'AUSTRALIA'"
                    class="text-[0.85rem] text-sky-600 hover:underline bg-white/80 rounded px-1"
                  >← back to Australia</button>
                </div>

                <!-- Place names on / off -->
                <div class="flex items-center gap-2">
                  <span class="text-[0.8rem] font-semibold uppercase tracking-wide text-gray-700 px-1 py-0.5 w-[4rem]">Labels</span>
                  <div class="flex w-fit rounded overflow-hidden border border-sky-500">
                    <button
                      v-for="option in [true, false]" :key="option"
                      @click="showLabels = option"
                      class="text-[0.85rem] px-3.5 py-1"
                      :class="showLabels === option ? 'bg-sky-500 text-white' : 'bg-white text-sky-600 hover:bg-sky-100'"
                    >{{ option ? 'On' : 'Off' }}</button>
                  </div>
                </div>

                <!-- Inputs: the irrigation status and the land use the input is read for, as buttons beside the map
                     (as on the module pages), so the land uses can be stepped through one click at a time -->
                <template v-if="selectView === 'Inputs' && !isNoMapTheme && !isChartInput">
                  <div v-if="inputLmOptions.length > 1" class="flex items-center gap-2">
                    <span class="text-[0.8rem] font-semibold uppercase tracking-wide text-gray-700 px-1 py-0.5 w-[4rem]">Water</span>
                    <div class="flex w-fit rounded overflow-hidden border border-sky-500">
                      <button
                        v-for="option in inputLmOptions" :key="option"
                        @click="selectInputLm = option"
                        class="text-[0.85rem] px-3 py-1"
                        :class="selectInputLm === option ? 'bg-sky-500 text-white' : 'bg-white text-sky-600 hover:bg-sky-100'"
                      >{{ LM_LABELS[option] || option }}</button>
                    </div>
                  </div>
                  <div v-if="inputLuOptions.length > 1" class="w-[220px] bg-white/70 rounded px-2 py-1.5 mt-8">
                    <div class="text-[0.7rem] font-semibold uppercase tracking-wide text-gray-600 mb-1">{{ selectTheme === 'Renewable energy' ? 'Technology' : 'Land use' }}</div>
                    <div class="flex flex-wrap gap-1 max-h-[42vh] overflow-y-auto">
                      <button
                        v-for="option in inputLuOptions" :key="option"
                        @click="selectInputLu = option"
                        class="text-[0.7rem] px-1.5 py-0.5 rounded text-left"
                        :class="selectInputLu === option ? 'bg-sky-500 text-white' : 'bg-white text-[#1f1f1f] hover:bg-sky-100'"
                      >{{ option }}</button>
                    </div>
                  </div>
                </template>
              </div>
            </div>

            <!-- Map Component: one map, so the position and the zoom hold across a switch of view -->
            <map-geojson
              class="absolute inset-0 z-[10]"
              :height="'100%'"
              :padTop="20"
              :padLeft="mapPadLeft"
              :padBottom="20"
              :regionLevel="mapRegionLevel"
              :showLabels="showLabels"
              :mapData="selectView === 'Simulation' ? simulationMapData : inputMapData"
            />

            <!-- A land use the run never chose: no map, a message -->
            <p v-if="selectView === 'Simulation' && landuseEmpty"
              class="absolute inset-0 z-[50] flex items-center justify-center pointer-events-none">
              <span class="bg-white/90 rounded shadow px-4 py-2 text-[0.9rem] text-gray-600">No results generated for <b>{{ selectLanduse }}</b> in {{ selectYear }}</span>
            </p>

            <!-- Simulation legend: the land uses of the map, in two columns; a click shows one of them.
                 The colour bar of the selected land use is a box of its own at the legend's bottom-right, so the legend keeps its shape. -->
            <div v-if="selectView === 'Simulation'" class="absolute bottom-[10px] left-[12px] z-[100] flex items-end gap-2 pointer-events-none">
              <div class="bg-white/60 rounded px-2 py-1 pointer-events-auto">
                <div class="font-bold text-[0.8rem] mb-1 text-gray-600">
                  Simulated land use
                  <span class="font-normal text-[0.7rem] text-gray-500 ml-1">· click a land use to map it</span>
                </div>
                <div class="flex gap-4">
                  <div v-for="(column, position) in landuseLegendColumns" :key="position" class="flex flex-col">
                    <button
                      v-for="entry in column" :key="entry.name"
                      @click="toggleLanduse(entry.name)"
                      :title="landuseEntryTitle(entry)"
                      class="group flex items-center leading-[1.15rem] rounded px-1 text-left transition-all duration-200 ease-out"
                      :class="[
                        // a selectable entry answers the pointer with a short animation: it slides a few px to the right over a
                        // pale wash, its swatch grows, and a faded one comes back to full
                        shareLayerNames.includes(entry.name) ? 'cursor-pointer hover:bg-white/80 hover:opacity-100 hover:translate-x-[4px]' : 'cursor-default',
                        // with a land use on the map, every other entry fades
                        selectLanduse && selectLanduse !== entry.name ? 'opacity-30' : '',
                      ]"
                    >
                      <span class="inline-block w-[11px] h-[11px] mr-[6px] shrink-0 border border-black/10 transition-transform duration-200 ease-out"
                        :class="shareLayerNames.includes(entry.name) ? 'group-hover:scale-[1.35]' : ''"
                        :style="{ backgroundColor: entry.color, opacity: entry.enabled ? 1 : 0.35 }"></span>
                      <span class="text-[0.75rem]" :class="entry.enabled ? 'text-[#1f1f1f]' : 'text-gray-400'">{{ entry.name }}</span>
                    </button>
                  </div>
                </div>
              </div>

              <!-- The colour bar of the share layer, while a land use with results is selected -->
              <div v-if="selectLanduse && !landuseEmpty" class="bg-white/80 rounded px-3 py-2 w-[200px] pointer-events-auto">
                <div class="text-[0.75rem] font-semibold text-gray-600 leading-tight mb-1">Share of the cell</div>
                <div class="h-[10px] rounded-sm border border-black/10" style="background: linear-gradient(to right, #FFF6B0, #FD933E, #800026)"></div>
                <div class="flex justify-between text-[0.7rem] text-[#1f1f1f] mt-[2px]"><span>0 %</span><span>100 %</span></div>
              </div>
            </div>

            <template v-else>
              <!-- A chart-only input (the yield trend): its two charts laid over the map, to the right of the selector rows and left of the zoom buttons -->
              <div v-if="showTrendCharts" class="absolute z-[60] top-[160px] left-[250px] right-[70px] bottom-[12px]">
                <home-trend-charts :trend="yieldTrend" :used="String(runScenario['PRODUCTIVITY_TREND'] || 'BAU')" :year="Number(selectYear)" />
              </div>
              <div v-if="showAssumptions" class="absolute z-[60] top-[205px] left-[12px] right-[12px] bottom-[30px]">
                <home-assumptions :groups="assumptionGroups" />
              </div>
              <div v-if="showBundleChart" class="absolute z-[60] top-[205px] left-[12px] right-[12px] bottom-[30px]">
                <home-bundle-chart :title="bundleChart.title" :variable="selectBundleVariable" @update:variable="selectBundleVariable = $event"
                  :variables="bundleVariables" :unit="bundleChart.unit" :note="bundleChart.note" :series="bundleChart.series" :year="Number(selectYear)" />
              </div>
              <div v-if="showMatrix" class="absolute z-[60] top-[205px] left-[12px] right-[12px] bottom-[30px]">
                <home-transition-matrix :matrix="transitionMatrix" :allowed-only="matrixAllowedOnly" />
              </div>
              <div v-if="inputTable" class="absolute z-[60] top-[205px] left-[12px] right-[12px] bottom-[30px]">
                <home-table-bars :table="inputTable" :title="selectInput" />
              </div>
              <p v-if="(availableThemes.length === 0 || isBundleTheme || isChartInput) && !showTrendCharts && !showMatrix && !inputTable && !showBundleChart && !showAssumptions"
                class="absolute inset-0 z-[50] flex items-center justify-center text-gray-500 text-sm pointer-events-none">
                <span class="bg-white/80 rounded px-3 py-1">{{ isBundleTheme ? 'The ag-management bundles are tables over the years: no map.' : isChartInput ? selectInput + ' has no map, and its charts are not in this report.' : 'No inputs in this report (written since 2026-11-12).' }}</span>
              </p>

              <!-- Inputs legend: the colour bar of the input layer -->
              <div v-if="inputLegend && inputEntry && !isNoMapTheme && !isChartInput" class="absolute bottom-[10px] left-[12px] z-[100] bg-white/80 rounded px-3 py-2 w-[320px]">
                <div class="font-bold text-[0.8rem] text-gray-600 leading-tight">{{ inputLegendTitle }}</div>
                <div class="text-[0.7rem] text-gray-500 leading-tight mb-2">{{ inputYearNote }}<span v-if="!maskLegend"> · grey: no value</span></div>
                <!-- A mask: a swatch per class -->
                <div v-if="maskLegend" class="flex flex-wrap gap-x-4 gap-y-1">
                  <span v-for="item in maskLegend" :key="item.label" class="flex items-center text-[0.75rem] text-[#1f1f1f]">
                    <span class="inline-block w-[14px] h-[14px] mr-1.5 rounded-sm border border-black/10" :style="{ background: item.color }"></span>{{ item.label }}
                  </span>
                </div>
                <div v-if="!maskLegend" class="h-[10px] rounded-sm border border-black/10" :style="{ background: inputLegend.gradient }"></div>
                <div v-if="!maskLegend" class="relative h-[1rem] mt-[2px]">
                  <span v-for="tick in inputLegend.ticks" :key="tick.at"
                    class="absolute text-[0.7rem] text-[#1f1f1f]"
                    :style="{ left: tick.at + '%', transform: tick.at === 0 ? 'none' : tick.at === 100 ? 'translateX(-100%)' : 'translateX(-50%)' }"
                  >{{ tick.label }}</span>
                </div>
              </div>
            </template>
          </div>

          <!-- The panel. Simulation: the targets of the run, national and regional (off ones at the bottom).
               Inputs: the source card and the tables of the selected input -->
          <div class="flex flex-col w-[360px] shrink-0 min-h-0">
            <template v-if="selectView === 'Simulation'">
              <home-tabs :tabs="TARGET_TABS" v-model:selected="selectTargetTab" />
              <div class="flex-1 min-h-0 overflow-y-auto pr-1 pb-1">
                <chart-cards :key="selectTargetTab" :panels="targetPanels[selectTargetTab] || []" :year="Number(selectYear)" :legendInPlot="true" @option="selectCardOption" />
              </div>
            </template>
            <template v-else>
              <!-- What to show: the theme, its input, then the irrigation and the land use the input is read for -->
              <div class="shrink-0 rounded-[10px] bg-white shadow-md px-3 py-2 mb-2 flex flex-col gap-2">
                <div>
                  <p class="text-[0.7rem] font-semibold uppercase tracking-wide text-gray-500 mb-1">Theme</p>
                  <div class="flex flex-wrap gap-1">
                    <button v-for="theme in availableThemes" :key="theme" @click="selectTheme = theme"
                      class="text-[0.75rem] px-2 py-0.5 rounded"
                      :class="selectTheme === theme ? 'bg-sky-500 text-white' : 'bg-[#f1f3f5] text-[#1f1f1f] hover:bg-sky-100'"
                    >{{ theme }}</button>
                  </div>
                </div>

                <template v-if="!isNoMapTheme">
                  <div>
                    <p class="text-[0.7rem] font-semibold uppercase tracking-wide text-gray-500 mb-1">Input</p>
                    <div class="flex flex-wrap gap-1">
                      <button v-for="entry in mainInputs" :key="entry.input" @click="selectInput = entry.input"
                        class="text-[0.75rem] px-2 py-0.5 rounded text-left"
                        :class="selectMainInput === entry.input ? 'bg-sky-500 text-white' : 'bg-[#f1f3f5] text-[#1f1f1f] hover:bg-sky-100'"
                      >{{ entry.input }}</button>
                    </div>
                  </div>
                  <!-- A species set: its variants (presence class, or year) and the list of its items, searched by name -->
                  <template v-if="speciesSet">
                    <div v-if="speciesSet.variants.length > 1">
                      <p class="text-[0.7rem] font-semibold uppercase tracking-wide text-gray-500 mb-1">{{ speciesSet.variant_name }}</p>
                      <div class="flex flex-wrap gap-1">
                        <button v-for="variant in speciesSet.variants" :key="variant" @click="selectSpeciesVariant = variant"
                          class="text-[0.75rem] px-2 py-0.5 rounded text-left"
                          :class="selectSpeciesVariant === variant ? 'bg-sky-500 text-white' : 'bg-[#f1f3f5] text-[#1f1f1f] hover:bg-sky-100'"
                        >{{ variantLabel(variant) }}</button>
                      </div>
                    </div>
                    <div>
                      <p class="text-[0.7rem] font-semibold uppercase tracking-wide text-gray-500 mb-1">{{ speciesItemOptions.length.toLocaleString() }} to pick from · type to search</p>
                      <el-select-v2 v-model="selectSpeciesItem" :options="speciesItemOptions" filterable size="small" class="w-full" />
                    </div>
                  </template>
                  <!-- The options the selected input is built under, a row per setting: the value of the run has a dashed border -->
                  <div v-for="row in inputOptionRows" :key="row.setting">
                    <p class="text-[0.7rem] font-semibold uppercase tracking-wide text-gray-500 mb-1">{{ row.name }}</p>
                    <div class="flex flex-wrap gap-1">
                      <button v-for="value in row.values" :key="value" @click="optionChoice[row.setting] = value"
                        :title="value === row.ofRun ? 'The value of this run' : 'Not the value of this run'"
                        class="text-[0.75rem] px-2 py-0.5 rounded text-left border"
                        :class="[row.picked === value ? 'bg-sky-500 text-white' : 'bg-[#f1f3f5] text-[#1f1f1f] hover:bg-sky-100',
                                 value === row.ofRun ? 'border-dashed border-sky-700' : 'border-transparent']"
                      >{{ optionValueLabel(row.setting, value) }}</button>
                    </div>
                    <p v-if="row.note" class="text-[0.65rem] text-gray-500 leading-tight mt-0.5">{{ row.note }}</p>
                  </div>
                  <!-- What the dashed border means, as on the Biodiversity page -->
                  <div v-if="inputOptionRows.length" class="flex items-center gap-1 text-gray-500 text-[0.65rem] leading-tight">
                    <span class="inline-block px-1 py-0.5 rounded border border-dashed border-sky-700 leading-none">abc</span>
                    <span>= user-selected option of this run</span>
                  </div>
                  <!-- The parts of the selected input, where it has some -->
                  <div v-if="inputParts.length">
                    <p class="text-[0.7rem] font-semibold uppercase tracking-wide text-gray-500 mb-1">Part</p>
                    <div class="flex flex-wrap gap-1">
                      <button v-for="part in inputParts" :key="part.input" @click="selectInput = part.input"
                        class="text-[0.75rem] px-2 py-0.5 rounded text-left"
                        :class="selectInput === part.input ? 'bg-sky-500 text-white' : 'bg-[#f1f3f5] text-[#1f1f1f] hover:bg-sky-100'"
                      >{{ part.label }}</button>
                    </div>
                  </div>
                </template>

                <template v-else-if="isBundleTheme">
                  <div>
                    <p class="text-[0.7rem] font-semibold uppercase tracking-wide text-gray-500 mb-1">Practice</p>
                    <div class="flex flex-wrap gap-1">
                      <button v-for="option in bundleOptions" :key="option" @click="selectBundle = option"
                        class="text-[0.75rem] px-2 py-0.5 rounded text-left"
                        :class="selectBundle === option ? 'bg-sky-500 text-white' : 'bg-[#f1f3f5] text-[#1f1f1f] hover:bg-sky-100'"
                      >{{ bundleLabel(option) }}</button>
                    </div>
                  </div>

                </template>
              </div>

              <div class="flex-1 min-h-0 overflow-y-auto pr-1 pb-1">
                <chart-cards :key="selectTheme + selectInput" :panels="inputPanels" :year="Number(selectYear)" />
              </div>
            </template>
          </div>
        </div>

      </div>
    </div>
  `,
};
