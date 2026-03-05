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
    }
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


    // Function to handle dataset loading and chart creation
    const createChart = () => {
      isLoading.value = true;
      
      // CRITICAL: Destroy existing chart before creating new one
      if (ChartInstance.value) {
        ChartInstance.value.destroy();
        ChartInstance.value = null;
      }

      // Create new chart with chart data
      ChartInstance.value = Highcharts.chart(
        chartElement.value,
        {
          ...props.chartData,
          chart: (props.chartData.chart || {}),
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

    // Watch for changes in chart data with infinite loop prevention
    let isUpdating = false;
    watch(() => props.chartData, (newValue) => {
      // Prevent infinite loops
      if (isUpdating) {
        return;
      }

      isUpdating = true;

      // Then update the chart
      updateChart(ChartInstance.value, newValue);

      // Reset flag after a delay to ensure all reactive updates complete
      setTimeout(() => {
        isUpdating = false;
      }, 100);
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
      handleWheel
    };
  },
  template: `
    <div class="m-2 relative" 
      :style="{ transform: 'translate(' + position.x + 'px, ' + position.y + 'px) scale(' + scale + ')', cursor: draggable ? 'move' : 'default' }" 
      @mousedown="startDrag"
      @wheel.prevent="handleWheel">
      <div v-if="isLoading" class="flex justify-center items-center text-lg">Loading data...</div>
      <div ref="chartElement" id="chart-container"></div>
      <div v-if="zoomable" class="absolute top-[40px] right-2 flex flex-col space-y-1">
        <button @click="zoomIn" class="bg-white/80 hover:bg-white text-gray-800 w-8 h-8 rounded-full shadow flex items-center justify-center">+</button>
        <button @click="zoomOut" class="bg-white/80 hover:bg-white text-gray-800 w-8 h-8 rounded-full shadow flex items-center justify-center">-</button>
      </div>
    </div>
  `
}