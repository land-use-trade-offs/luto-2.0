// Ranking Cards Element
// This component displays ranking cards with progress indicators for various metrics

window.RankingCards = {
  props: {
    selectRegion: {
      type: String,
    },
    selectYear: {
      type: Number,
    }
  },

  setup(props) {
    const { computed } = Vue;

    const rankingData = computed(() => {
      return window.DataService.getRankingData(props.selectRegion, props.selectYear);
    });

    return {
      rankingData
    };
  },

  template: `
    <div class="flex flex-wrap gap-4 justify-between h-[230px]">
      <!-- Economics Card -->
      <div class="flex-1 rounded-lg p-3 shadow-md flex flex-col bg-gradient-to-r from-[#e6ba7f] to-[#eacca2]" >
        <h4 class="text-white text-center text-lg mb-2">Economics</h4>
        <div class="text-2xl text-center font-bold text-white mb-1">{{ rankingData.Economics.Total.value }}</div>
        <div class="text-white/80 text-center text-[12px] mb-4">Australian Dollar</div>
        <div class="mt-auto">
          <div class="flex justify-between text-white text-[14px] py-1 border-t border-white/20">
            <span>Cost</span>
            <span>{{ rankingData.Economics.Cost.value }}</span>
          </div>
          <div class="flex justify-between text-white text-[14px] py-1 border-t border-white/20">
            <span>Revenue</span>
            <span>{{ rankingData.Economics.Revenue.value }}</span>
          </div>
        </div>
      </div>
      
      <!-- Area Card -->
      <div class="flex-1  rounded-lg p-3 shadow-md flex flex-col bg-gradient-to-r from-blue-400 to-cyan-400" >
        <h4 class="text-white text-center text-lg mb-2">Area</h4>
        <div class="text-2xl text-center font-bold text-white mb-1">{{ rankingData.Area.Total.value }}</div>
        <div class="text-white/80 text-center text-[12px] mb-4">Hectares</div>
        <div class="mt-auto">
          <div class="flex justify-between text-white text-[14px] py-1 border-t border-white/20">
            <span>Ag Land</span>
            <span>{{ rankingData.Area.Ag.value }}</span>
          </div>
          <div class="flex justify-between text-white text-[14px] py-1 border-t border-white/20">
            <span>Ag Mgt</span>
            <span>{{ rankingData.Area.Am.value }}</span>
          </div>
          <div class="flex justify-between text-white text-[14px] py-1 border-t border-white/20">
            <span>Non-Ag</span>
            <span>{{ rankingData.Area.NonAg.value }}</span>
          </div>
        </div>
      </div>
      
      <!-- GHG Card -->
      <div class="flex-1  rounded-lg p-3 shadow-md flex flex-col bg-gradient-to-r from-green-400 to-green-500" >
        <h4 class="text-white text-center text-lg mb-2">GHG Impact</h4>
        <div class="text-2xl text-center font-bold text-white mb-1">{{ rankingData.GHG.Total.value }}</div>
        <div class="text-white/80 text-center text-[12px] mb-4">tCO2e</div>
        <div class="mt-auto">
          <div class="flex justify-between text-white text-[14px] py-1 border-t border-white/20">
            <span>Emissions</span>
            <span>{{ rankingData.GHG.Emissions.value }}</span>
          </div>
          <div class="flex justify-between text-white text-[14px] py-1 border-t border-white/20">
            <span>Reduction</span>
            <span>{{ rankingData.GHG.Sequestration.value }}</span>
          </div>
        </div>
      </div>
      
      <!-- Water Card -->
      <div class="flex-1  rounded-lg p-3 shadow-md flex flex-col bg-gradient-to-r from-rose-400 to-amber-300" >
        <h4 class="text-white text-center text-lg mb-2">Water Usage</h4>
        <div class="text-2xl text-center font-bold text-white mb-1">{{ rankingData.Water.Total.value }}</div>
        <div class="text-white/80 text-center text-[12px] mb-4">Megaliters</div>
        <div class="mt-auto">
          <div class="flex justify-between text-white text-[14px] py-1 border-t border-white/20">
            <span>Ag Land</span>
            <span>{{ rankingData.Water.Ag.value }}</span>
          </div>
          <div class="flex justify-between text-white text-[14px] py-1 border-t border-white/20">
            <span>Ag Mgt</span>
            <span>{{ rankingData.Water.Am.value }}</span>
          </div>
          <div class="flex justify-between text-white text-[14px] py-1 border-t border-white/20">
            <span>NonAg</span>
            <span>{{ rankingData.Water.NonAg.value }}</span>
          </div>
        </div>
      </div>
      
      <!-- Biodiversity Card -->
      <div class="flex-1  rounded-lg p-3 shadow-md flex flex-col bg-gradient-to-r from-[#918be9] to-[#e2cbfa]" >
        <h4 class="text-white text-center text-lg mb-2">Biodiversity</h4>
        <div class="text-2xl text-center font-bold text-white mb-1">{{ rankingData.Biodiversity.Total.value }}</div>
        <div class="text-white/80 text-center text-[12px] mb-4">Priority Weighted Hectares</div>
        <div class="mt-auto">
          <div class="flex justify-between text-white text-[14px] py-1 border-t border-white/20">
            <span>Ag Land</span>
            <span>{{ rankingData.Biodiversity.Ag.value }}</span>
          </div>
          <div class="flex justify-between text-white text-[14px] py-1 border-t border-white/20">
            <span>Ag Mgt</span>
            <span>{{ rankingData.Biodiversity.Am.value }}</span>
          </div>
          <div class="flex justify-between text-white text-[14px] py-1 border-t border-white/20">
            <span>Non-Ag</span>
            <span>{{ rankingData.Biodiversity.NonAg.value }}</span>
          </div>
        </div>
      </div>
    </div>
  `,
};