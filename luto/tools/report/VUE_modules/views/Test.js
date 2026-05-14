window.Test = {
  name: 'TestView',
  template: /*html*/`
    <!-- Title for map and chart -->

        <div class="flex mr-4 gap-4 mb-4">

          <div class="flex flex-col rounded-[10px] bg-white shadow-md w-[500px]">


            <!-- Map -->
            <div class="relative">
              <div class="absolute flex-col w-full top-1 left-2 right-2 pr-4 justify-between items-center z-10">

              <div class="flex flex-col">
                <div class="flex items-center justify-between">
                  <p class="text-[0.8rem]">Year: <strong>{{ selectYear }}</strong></p>
                  <div class="flex space-x-1 mr-4">
                    <button v-for="cat in availableChartSubCategories" :key="cat"
                      @click="selectChartSubCategory = cat"
                      class="bg-[#e8eaed] text-[#1f1f1f] text-[0.6rem] px-1 rounded"
                      :class="{'bg-sky-500 text-white': selectChartSubCategory === cat}">
                      {{ cat }}
                    </button>
                  </div>
                </div>

                <el-slider
                  v-if="availableYears.length > 0"
                  class="flex-1 max-w-[150px] pt-2 pl-2"
                  v-model="yearIndex"
                  size="small"
                  :min="0"
                  :max="availableYears.length - 1"
                  :step="1"
                  :format-tooltip="index => availableYears[index]"
                  :show-stops="true"
                  @input="(index) => { yearIndex = index; }"
                />

                </div>
              </div>
              <map-geojson
                v-if="dataLoaded"
                :height="'530px'"
                :availableChartCategories="availableChartCategories"
                :selectYear="selectYear"
                :selectChartSubCategory="selectChartSubCategory"
                :legendObj="rankingColors"
              />
            </div>

          </div>

          <!-- Statistics Chart -->
          <chart-container
          v-if="dataLoaded"
          class="flex-1 rounded-[10px] bg-white shadow-md"
          :chartData="selectChartData"></chart-container>

        </div>
  `
};