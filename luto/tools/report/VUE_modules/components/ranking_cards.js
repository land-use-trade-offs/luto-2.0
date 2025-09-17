// Ranking Cards Element
// This component displays ranking cards with progress indicators for various metrics

window.RankingCards = {
  props: {
    selectRankingData: {
      type: Object,
      required: true
    },
  },


  template: `
    <div class="flex flex-wrap gap-4 justify-between h-[230px]">
      <!-- Economics Card -->
      <div class="flex-1 rounded-lg p-3 shadow-md flex flex-col bg-gradient-to-r from-[#e6ba7f] to-[#eacca2]" >
        <h4 class="text-white text-center text-lg mb-2">Profit</h4>
        <div class="text-2xl text-center font-bold text-white mb-1">{{ selectRankingData.economicTotal }}</div>
        <div class="text-white/80 text-center text-[12px] mb-4">Australian Dollar</div>
        <div class="mt-auto">
          <div class="flex justify-between text-white text-[14px] py-1 border-t border-white/20">
            <span>Cost</span>
            <span>{{ selectRankingData.economicCost }}</span>
          </div>
          <div class="flex justify-between text-white text-[14px] py-1 border-t border-white/20">
            <span>Revenue</span>
            <span>{{ selectRankingData.economicRevenue }}</span>
          </div>
        </div>
      </div>
      
      <!-- Area Card -->
      <div class="flex-1  rounded-lg p-3 shadow-md flex flex-col bg-gradient-to-r from-blue-400 to-cyan-400" >
        <h4 class="text-white text-center text-lg mb-2">Area</h4>
        <div class="text-2xl text-center font-bold text-white mb-1">{{ selectRankingData.areaTotal }}</div>
        <div class="text-white/80 text-center text-[12px] mb-4">Hectares</div>
        <div class="mt-auto">
          <div class="flex justify-between text-white text-[14px] py-1 border-t border-white/20">
            <span>Ag Land</span>
            <span>{{ selectRankingData.areaAgLand }}</span>
          </div>
          <div class="flex justify-between text-white text-[14px] py-1 border-t border-white/20">
            <span>Ag Mgt</span>
            <span>{{ selectRankingData.areaAgMgt }}</span>
          </div>
          <div class="flex justify-between text-white text-[14px] py-1 border-t border-white/20">
            <span>Non-Ag</span>
            <span>{{ selectRankingData.areaNonAg }}</span>
          </div>
        </div>
      </div>
      
      <!-- GHG Card -->
      <div class="flex-1  rounded-lg p-3 shadow-md flex flex-col bg-gradient-to-r from-green-400 to-green-500" >
        <h4 class="text-white text-center text-lg mb-2">GHG Impact</h4>
        <div class="text-2xl text-center font-bold text-white mb-1">{{ selectRankingData.ghgTotal }}</div>
        <div class="text-white/80 text-center text-[12px] mb-4">tCO2e</div>
        <div class="mt-auto">
          <div class="flex justify-between text-white text-[14px] py-1 border-t border-white/20">
            <span>Emissions</span>
            <span>{{ selectRankingData.ghgEmissions }}</span>
          </div>
          <div class="flex justify-between text-white text-[14px] py-1 border-t border-white/20">
            <span>Reduction</span>
            <span>{{ selectRankingData.ghgReduction }}</span>
          </div>
        </div>
      </div>
      
      <!-- Water Card -->
      <div class="flex-1  rounded-lg p-3 shadow-md flex flex-col bg-gradient-to-r from-rose-400 to-amber-300" >
        <h4 class="text-white text-center text-lg mb-2">Water Usage</h4>
        <div class="text-2xl text-center font-bold text-white mb-1">{{ selectRankingData.waterTotal }}</div>
        <div class="text-white/80 text-center text-[12px] mb-4">Megaliters</div>
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
            <span>NonAg</span>
            <span>{{ selectRankingData.waterNonAg }}</span>
          </div>
        </div>
      </div>
      
      <!-- Biodiversity Card -->
      <div class="flex-1 rounded-lg p-3 shadow-md flex flex-col bg-gradient-to-r from-[#918be9] to-[#e2cbfa]" >
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