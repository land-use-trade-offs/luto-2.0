// Headline Cards Element
// The headline numbers of the selected region and year, one card per metric, stacked in a column
// beside the Home map. A click on a card picks its metric (emits `select` with the metric's name).

window.HeadlineCards = {
  name: 'HeadlineCards',
  props: {
    selectRankingData: {
      type: Object,
      required: true
    },
    // The metric whose card is marked as selected
    selected: {
      type: String,
      default: '',
    },
  },
  emits: ['select'],

  setup(props) {
    const { computed } = Vue;

    const cards = computed(() => {
      const values = props.selectRankingData;
      return [
        {
          metric: 'Economics', title: 'Net economic returns', total: values.economicTotal, unit: 'AUD / yr',
          gradient: 'from-[#d9a521] to-[#ebc867]',
          rows: [
            { label: 'Costs (incl. transition)', value: values.economicCost },
            { label: 'Revenue', value: values.economicRevenue },
          ],
        },
        {
          metric: 'Area', title: 'Area', total: values.areaTotal, unit: 'ha (Ag land + Non-Ag)',
          gradient: 'from-[#b8925a] to-[#d4b483]',
          rows: [
            { label: 'Ag Land', value: values.areaAgLand },
            { label: 'Ag Mgt (overlaps Ag land)', value: values.areaAgMgt, tip: 'Ag-management options sit on Ag land and can overlap, so this area is not added to the total' },
            { label: 'Non-Ag', value: values.areaNonAg },
          ],
        },
        {
          metric: 'GHG', title: 'Net GHG emissions', total: values.ghgTotal, unit: 't CO2e / yr',
          gradient: 'from-slate-500 to-slate-400',
          rows: [
            { label: 'Emissions', value: values.ghgEmissions },
            { label: 'Sequestration', value: values.ghgReduction },
          ],
        },
        {
          metric: 'Water', title: 'Water', total: values.waterTotal, unit: 'ML / yr net yield (runoff − use)',
          gradient: 'from-blue-500 to-sky-400',
          rows: [
            { label: 'Runoff', value: values.waterRunoff, tip: 'Water the land yields under its land cover' },
            { label: 'Water use', value: values.waterUse, tip: 'Irrigation and livestock drinking water' },
          ],
        },
        {
          metric: 'Biodiversity', title: 'Biodiversity', total: values.biodiversityTotal + ' %', unit: 'Relative percent to pre-1750 level',
          gradient: 'from-green-600 to-emerald-400',
          rows: [
            { label: 'Ag Land', value: values.biodiversityAgLand },
            { label: 'Ag Mgt', value: values.biodiversityAgMgt },
            { label: 'Non-Ag', value: values.biodiversityNonAg },
          ],
        },
      ];
    });

    return { cards };
  },

  template: /*html*/`
    <div class="flex flex-col gap-2 h-full">
      <button
        v-for="card in cards" :key="card.metric"
        @click="$emit('select', card.metric)"
        :title="'Show ' + card.title + ' in the chart below'"
        class="flex-1 min-h-0 rounded-lg px-3 py-2 shadow-md flex flex-col justify-center text-left bg-gradient-to-r transition-shadow"
        :class="[card.gradient, selected === card.metric ? 'ring-2 ring-offset-2 ring-sky-500' : 'hover:shadow-lg']"
      >
        <div class="flex items-baseline justify-between gap-2">
          <span class="text-white text-[0.95rem] font-medium leading-tight">{{ card.title }}</span>
          <span class="text-white text-xl font-bold leading-tight whitespace-nowrap">{{ card.total }}</span>
        </div>
        <div class="text-white/80 text-right text-[11px] mb-1">{{ card.unit }}</div>
        <div
          v-for="row in card.rows" :key="row.label"
          class="flex justify-between text-white text-[12px] leading-[1.15rem] border-t border-white/20"
        >
          <span :title="row.tip || ''">{{ row.label }}</span>
          <span>{{ row.value }}</span>
        </div>
      </button>
    </div>
  `,
};
