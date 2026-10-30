// Ranking Cards Element
// This component displays ranking cards with progress indicators for various metrics

window.RankingCards = {
  name: 'RankingCards',
  props: {
    selectRankingData: {
      type: Object,
      required: true
    },
  },


  template: `
    <div class="flex flex-wrap gap-4 justify-between h-[230px]">
      <!-- Economics Card -->
      <div class="flex-1 rounded-lg p-3 shadow-md flex flex-col bg-gradient-to-r from-[#d9a521] to-[#ebc867]" >
        <h4 class="text-white text-center text-lg mb-2">Net economic returns</h4>
        <div class="text-2xl text-center font-bold text-white mb-1">{{ selectRankingData.economicTotal }}</div>
        <div class="text-white/80 text-center text-[12px] mb-4">AUD / yr</div>
        <div class="mt-auto">
          <div class="flex justify-between text-white text-[14px] py-1 border-t border-white/20">
            <span>Costs (incl. transition)</span>
            <span>{{ selectRankingData.economicCost }}</span>
          </div>
          <div class="flex justify-between text-white text-[14px] py-1 border-t border-white/20">
            <span>Revenue</span>
            <span>{{ selectRankingData.economicRevenue }}</span>
          </div>
        </div>
      </div>
      
      <!-- Area Card -->
      <div class="flex-1  rounded-lg p-3 shadow-md flex flex-col bg-gradient-to-r from-[#b8925a] to-[#d4b483]" >
        <h4 class="text-white text-center text-lg mb-2">Area</h4>
        <div class="text-2xl text-center font-bold text-white mb-1">{{ selectRankingData.areaTotal }}</div>
        <div class="text-white/80 text-center text-[12px] mb-4">ha (Ag land + Non-Ag)</div>
        <div class="mt-auto">
          <div class="flex justify-between text-white text-[14px] py-1 border-t border-white/20">
            <span>Ag Land</span>
            <span>{{ selectRankingData.areaAgLand }}</span>
          </div>
          <div class="flex justify-between text-white text-[14px] py-1 border-t border-white/20">
            <span title="Ag-management options sit on Ag land and can overlap, so this area is not added to the total">Under Ag Mgt (overlapping)</span>
            <span>{{ selectRankingData.areaAgMgt }}</span>
          </div>
          <div class="flex justify-between text-white text-[14px] py-1 border-t border-white/20">
            <span>Non-Ag</span>
            <span>{{ selectRankingData.areaNonAg }}</span>
          </div>
        </div>
      </div>
      
      <!-- GHG Card -->
      <div class="flex-1  rounded-lg p-3 shadow-md flex flex-col bg-gradient-to-r from-slate-500 to-slate-400" >
        <h4 class="text-white text-center text-lg mb-2">Net GHG emissions</h4>
        <div class="text-2xl text-center font-bold text-white mb-1">{{ selectRankingData.ghgTotal }}</div>
        <div class="text-white/80 text-center text-[12px] mb-4">t CO2e / yr</div>
        <div class="mt-auto">
          <div class="flex justify-between text-white text-[14px] py-1 border-t border-white/20">
            <span>Emissions</span>
            <span>{{ selectRankingData.ghgEmissions }}</span>
          </div>
          <div class="flex justify-between text-white text-[14px] py-1 border-t border-white/20">
            <span>Sequestration</span>
            <span>{{ selectRankingData.ghgReduction }}</span>
          </div>
        </div>
      </div>
      
      <!-- Water Card -->
      <div class="flex-1  rounded-lg p-3 shadow-md flex flex-col bg-gradient-to-r from-blue-500 to-sky-400" >
        <h4 class="text-white text-center text-lg mb-2">Net water yield</h4>
        <div class="text-2xl text-center font-bold text-white mb-1">{{ selectRankingData.waterTotal }}</div>
        <div class="text-white/80 text-center text-[12px] mb-4">ML / yr</div>
        <div class="mt-auto">
          <div class="flex justify-between text-white text-[14px] py-1 border-t border-white/20">
            <span>Ag Land</span>
            <span>{{ selectRankingData.waterAgLand }}</span>
          </div>
          <div class="flex justify-between text-white text-[14px] py-1 border-t border-white/20">
            <span>Ag Mgt</span>
            <span>{{ selectRankingData.waterAgMgt }}</span>
          </div>
          <div class="flex justify-between text-white text-[14px] py-1 border-t border-white/20">
            <span>Non-Ag</span>
            <span>{{ selectRankingData.waterNonAg }}</span>
          </div>
        </div>
      </div>
      
      <!-- Biodiversity Card -->
      <div class="flex-1 rounded-lg p-3 shadow-md flex flex-col bg-gradient-to-r from-green-600 to-emerald-400" >
        <h4 class="text-white text-center text-lg mb-2">Biodiversity</h4>
        <div class="text-2xl text-center font-bold text-white mb-1">{{ selectRankingData.biodiversityTotal }} %</div>
        <div class="text-white/80 text-center text-[12px] mb-4">Relative percent to pre-1750 level</div>
        <div class="mt-auto">
          <div class="flex justify-between text-white text-[14px] py-1 border-t border-white/20">
            <span>Ag Land</span>
            <span>{{ selectRankingData.biodiversityAgLand }}</span>
          </div>
          <div class="flex justify-between text-white text-[14px] py-1 border-t border-white/20">
            <span>Ag Mgt</span>
            <span>{{ selectRankingData.biodiversityAgMgt }}</span>
          </div>
          <div class="flex justify-between text-white text-[14px] py-1 border-t border-white/20">
            <span>Non-Ag</span>
            <span>{{ selectRankingData.biodiversityNonAg }}</span>
          </div>
        </div>
      </div>
    </div>
  `,
};