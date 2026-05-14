window.Highchart = {
  name: 'Highchart',
  props: {
    chartData: {
      type: Object,
      required: true,
    },
    draggable: {
      type: Boolean,
      default: false,
    },
    zoomable: {
      type: Boolean,
      default: false,
    },
    multiInput: {
      type: Object,
      default: null,
    },
    multiYAxis: {
      type: Object,
      default: null,
    },
  },
  setup(props) {
    const { ref, onMounted, onUnmounted, watch, inject, computed } = Vue
    const isCollapsed = inject('isCollapsed', ref(false))

    // Reactive state for loading status and datasets
    const chartElement = ref(null);
    const isLoading = ref(true);
    const ChartInstance = ref(null);
    const position = ref({ x: 0, y: 0 });
    const isDragging = ref(false);
    const dragStartPos = ref({ x: 0, y: 0 });
    const scale = ref(1);
    const zoomStep = 0.1;

    // Multi-input selector state
    const selectedInput = ref(null);

    watch(() => props.multiInput, (newVal) => {
      if (newVal) {
        const keys = Object.keys(newVal);
        if (!selectedInput.value || !keys.includes(selectedInput.value)) {
          selectedInput.value = keys[0] || null;
        }
      } else {
        selectedInput.value = null;
      }
    }, { immediate: true });

    const setInput = (key) => { selectedInput.value = key; };

    // Effective chart data: merges base options with selected multi-input series/yAxis
    const effectiveChartData = computed(() => {
      if (!props.multiInput || !selectedInput.value) return props.chartData;
      const yAxisTitle = props.multiYAxis?.[selectedInput.value] || '';
      return {
        ...props.chartData,
        yAxis: { title: { text: yAxisTitle } },
        series: props.multiInput[selectedInput.value] || [],
      };
    });

    // Function to handle dataset loading and chart creation
    const createChart = () => {
      isLoading.value = true;

      // CRITICAL: Destroy existing chart before creating new one
      if (ChartInstance.value) {
        ChartInstance.value.destroy();
        ChartInstance.value = null;
      }

      // Create new chart with chart data
      const data = effectiveChartData.value;
      ChartInstance.value = Highcharts.chart(
        chartElement.value,
        {
          ...data,
          chart: (data.chart || {}),
        }
      );
      isLoading.value = false;
    };

    // Function to handle window resize
    const handleResize = () => { createChart(); };

    // Dragging functionality
    const startDrag = (event) => {
      if (!props.draggable) return;
      isDragging.value = true;
      dragStartPos.value = {
        x: event.clientX - position.value.x,
        y: event.clientY - position.value.y
      };
    };

    const onDrag = (event) => {
      if (isDragging.value) {
        position.value = {
          x: event.clientX - dragStartPos.value.x,
          y: event.clientY - dragStartPos.value.y
        };
      }
    };

    const stopDrag = () => {
      isDragging.value = false;
    };

    // Zoom functionality
    const zoomIn = () => {
      if (!props.zoomable) return;
      scale.value += zoomStep;
    };

    const zoomOut = () => {
      if (!props.zoomable) return;
      if (scale.value > zoomStep) {
        scale.value -= zoomStep;
      }
    };

    const handleWheel = (event) => {
      if (!props.zoomable) return;
      event.preventDefault();
      if (event.deltaY < 0) {
        zoomIn();
      } else {
        zoomOut();
      }
    };

    // Function to update the chart with new series data
    const updateChart = (chart, newChartData) => {
      // Make a deep copy of the chart data to avoid reference issues
      const newData = JSON.parse(JSON.stringify(newChartData));

      // Update the chart configuration options first (except series and colors)
      for (const key in newData) {
        if (key !== 'series' && key !== 'colors') {
          chart.update({ [key]: newData[key] }, false);
        }
      }

      // Handle series data updates safely
      if (newData.series && Array.isArray(newData.series)) {
        // Step 1: Remove excess series if there are more in the chart than in new data
        while (chart.series.length > newData.series.length) {
          if (chart.series[chart.series.length - 1]) {
            chart.series[chart.series.length - 1].remove(false);
          }
        }

        // Step 2: Update existing series or add new ones
        newData.series.forEach((seriesConfig, index) => {
          if (index < chart.series.length) {
            // Series exists, update it completely including name and all properties
            if (chart.series[index]) {
              chart.series[index].update(seriesConfig, false);
            }
          } else {
            // Series doesn't exist, add it
            chart.addSeries(seriesConfig, false);
          }
        });
      }

      // Apply our stored colors LAST to prevent them being overwritten
      if (props.chartData.colors) {
        chart.series.forEach((series, index) => {
          const colorIndex = index % props.chartData.colors.length;
          series.update({ color: props.chartData.colors[colorIndex] }, false);
        });
      }

      // Final redraw to apply all changes with animation
      chart.redraw();

    };
    onMounted(() => {
      createChart();
      window.addEventListener('resize', handleResize);
      window.addEventListener('mousemove', onDrag);
      window.addEventListener('mouseup', stopDrag);
    });

    onUnmounted(() => {
      // CRITICAL: Destroy Highcharts instance to prevent memory leaks
      if (ChartInstance.value) {
        ChartInstance.value.destroy();
        ChartInstance.value = null;
      }

      // Remove event listeners
      window.removeEventListener('resize', handleResize);
      window.removeEventListener('mousemove', onDrag);
      window.removeEventListener('mouseup', stopDrag);
    });

    // Watch for changes in chart data and update Highcharts.
    // No loop-guard needed: updateChart only calls Highcharts APIs and never
    // writes back to Vue reactive state, so the watcher cannot re-trigger itself.
    watch(effectiveChartData, (newValue) => {
      updateChart(ChartInstance.value, newValue);
    }, { deep: true });

    // Watch for sidebar collapsed state changes via inject
    watch(isCollapsed, () => {
      setTimeout(() => {
        createChart();
      }, 300); // Wait for sidebar animation to complete
    });


    return {
      chartElement,
      isLoading,
      ChartInstance,
      position,
      startDrag,
      scale,
      zoomIn,
      zoomOut,
      handleWheel,
      selectedInput,
      setInput,
    };
  },
  template: `
    <div class="m-2 relative"
      :style="{ transform: 'translate(' + position.x + 'px, ' + position.y + 'px) scale(' + scale + ')', cursor: draggable ? 'move' : 'default' }"
      @mousedown="startDrag"
      @wheel="handleWheel">
      <div v-if="isLoading" class="flex justify-center items-center text-lg">Loading data...</div>
      <div v-if="multiInput" class="absolute top-2 left-20 z-10 flex gap-1" @mousedown.stop>
        <button v-for="key in Object.keys(multiInput)" :key="key"
          @click="setInput(key)"
          class="text-xs px-2 py-1 rounded border font-medium shadow-sm"
          :class="selectedInput === key ? 'bg-sky-500 text-white border-sky-500' : 'bg-white/80 text-gray-700 border-gray-300 hover:bg-white'">
          {{ key }}
        </button>
      </div>
      <div ref="chartElement" id="chart-container"></div>
      <div v-if="zoomable" class="absolute top-[40px] right-2 flex flex-col space-y-1">
        <button @click="zoomIn" class="bg-white/80 hover:bg-white text-gray-800 w-8 h-8 rounded-full shadow flex items-center justify-center">+</button>
        <button @click="zoomOut" class="bg-white/80 hover:bg-white text-gray-800 w-8 h-8 rounded-full shadow flex items-center justify-center">-</button>
      </div>
    </div>
  `
}