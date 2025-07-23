document.addEventListener("DOMContentLoaded", function () {


  const support_info = JSON.parse(document.getElementById('Supporting_info').innerText);
  const colors = support_info.colors;
  const model_years = support_info.years;
  

  // Get the available years for plotting
  var years = model_years.map(function (x) {return parseInt(x);});
  years.sort(function (a, b) {return a - b;});
  var year_ticks = years.length == 2 ? years : null;

  
  // Set the title alignment to left
  Highcharts.setOptions({
    colors: colors,
    title: {
        align: 'left'
    }
  });


  // Chart:Area_overview_2_Category
  Highcharts.chart("Area_overview_2_Category_chart", {
    chart: {
      type: "column",
      marginRight: 380,
    },
    title: {
      text: "Total Area in Each Land-use Group",
    },
    series: JSON.parse(
      document.getElementById("Area_overview_2_Category").innerHTML
    ).AUSTRALIA,
    credits: {
      enabled: false,
    },
    xAxis: {
      tickPositions: year_ticks,
    },
    yAxis: {
      title: {
        text: "Area (ha)",
      },
    },

    legend: {
      align: "right",
      layout: "vertical",
      x: -120,
      y: -10,
      verticalAlign: "middle",
      itemMarginTop: 0,
      itemMarginBottom: 1,
    },

    tooltip: {
      formatter: function () {
        return `<b>Year:</b> ${this.x}<br><b>${this.series.name
          }:</b>${this.y.toFixed(2)}<br/>`;
      },
    },

    plotOptions: {
      column: {
        stacking: "normal",
      },
    },

    exporting: {
      sourceWidth: 1200,
      sourceHeight: 600,
    },
  });

  // Chart:Area_overview_1_Land-use
  Highcharts.chart("Area_overview_1_Land-use_chart", {
    chart: {
      type: "column",
      marginRight: 380,
    },
    title: {
      text: "Total Area by Specific Land-use Type",
    },
    series: JSON.parse(
      document.getElementById("Area_overview_1_Land-use").innerHTML
    ).AUSTRALIA,
    credits: {
      enabled: false,
    },
    xAxis: {
      tickPositions: year_ticks,
    },
    yAxis: {
      title: {
        text: "Area (ha)",
      },
    },

    legend: {
      itemStyle: {
        fontSize: "10.5px",
      },
      align: "right",
      layout: "vertical",
      x: -40,
      y: -20,
      verticalAlign: "middle",
      itemMarginTop: 0,
      itemMarginBottom: 0.75,
    },

    tooltip: {
      formatter: function () {
        return `<b>Year:</b> ${this.x}<br><b>${this.series.name
          }:</b>${this.y.toFixed(2)}<br/>`;
      },
    },

    plotOptions: {
      column: {
        stacking: "normal",
      },
    },

    exporting: {
      sourceWidth: 1200,
      sourceHeight: 600,
    },
  });

  // Chart:Area_overview_3_Source-use
  Highcharts.chart("Area_overview_3_Source_chart", {
    chart: {
      type: "column",
      marginRight: 380,
    },
    title: {
      text: "Total Area by Broad Land-use Type",
    },
    series: JSON.parse(
      document.getElementById("Area_overview_3_Source").innerHTML
    ).AUSTRALIA,
    credits: {
      enabled: false,
    },
    xAxis: {
      tickPositions: year_ticks,
    },
    yAxis: {
      title: {
        text: "Area (ha)",
      },
    },

    legend: {
      align: "right",
      layout: "vertical",
      x: -10,
      verticalAlign: "middle",
    },

    tooltip: {
      formatter: function () {
        return `<b>Year:</b> ${this.x}<br><b>${this.series.name
          }:</b>${this.y.toFixed(2)}<br/>`;
      },
    },

    plotOptions: {
      column: {
        stacking: "normal",
      },
    },

    exporting: {
      sourceWidth: 1200,
      sourceHeight: 600,
    },
  });

  // Area_Ag_1_Land-use
  Highcharts.chart("Area_Ag_1_Land-use_chart", {
    chart: {
      type: "column",
      marginRight: 380,
    },
    title: {
      text: "Agricultural Land-Use Area",
    },
    series: JSON.parse(
      document.getElementById("Area_Ag_1_Land-use").innerHTML,
    ).AUSTRALIA,
    credits: {
      enabled: false,
    },
    xAxis: {
      tickPositions: year_ticks,
    },
    yAxis: {
      title: {
        text: "Area (ha)",
      },
    },

    legend: {
      align: "right",
      layout: "vertical",
      x: 0,
      verticalAlign: "middle",
    },

    tooltip: {
      formatter: function () {
        return `<b>Year:</b> ${this.x}<br><b>${this.series.name
          }:</b>${this.y.toFixed(2)}<br/>`;
      },
    },

    plotOptions: {
      column: {
        stacking: "normal",
      },
    },

    exporting: {
      sourceWidth: 1200,
      sourceHeight: 600,
    },
  });

  // Area_NonAg_1_Land-use
  Highcharts.chart("Area_NonAg_1_Land-use_chart", {
    chart: {
      type: "column",
      marginRight: 380,
    },
    title: {
      text: "Non-Agricultural Land-use Area",
    },
    series: JSON.parse(
      document.getElementById("Area_NonAg_1_Land-use").innerHTML
    ).AUSTRALIA,
    credits: {
      enabled: false,
    },
    xAxis: {
      tickPositions: year_ticks,
    },
    yAxis: {
      title: {
        text: "Area (ha)",
      },
    },

    legend: {
      align: "right",
      layout: "vertical",
      x: -50,
      verticalAlign: "middle",
    },

    tooltip: {
      formatter: function () {
        return `<b>Year:</b> ${this.x}<br><b>${this.series.name
          }:</b>${this.y.toFixed(2)}<br/>`;
      },
    },

    plotOptions: {
      column: {
        stacking: "normal",
      },
    },

    exporting: {
      sourceWidth: 1200,
      sourceHeight: 600,
    },
  });

  // Area_Am_1_Type
  Highcharts.chart("Area_Am_1_Type_chart", {
    chart: {
      type: "column",
      marginRight: 380,
    },
    title: {
      text: "Agricultural Management Area by BroadLand-use Type",
    },
    series:
      JSON.parse(document.getElementById("Area_Am_1_Type").innerHTML).AUSTRALIA,
    credits: {
      enabled: false,
    },
    xAxis: {
      tickPositions: year_ticks,
    },
    yAxis: {
      title: {
        text: "Area (ha)",
      },
    },

    legend: {
      align: "right",
      layout: "vertical",
      x: -50,
      verticalAlign: "middle",
    },

    tooltip: {
      formatter: function () {
        return `<b>Year:</b> ${this.x}<br><b>${this.series.name
          }:</b>${this.y.toFixed(2)}<br/>`;
      },
    },

    plotOptions: {
      column: {
        stacking: "normal",
      },
    },

    exporting: {
      sourceWidth: 1200,
      sourceHeight: 600,
    },
  });

  // Area_Am_3_Land-use
  Highcharts.chart("Area_Am_3_Land-use_chart", {
    chart: {
      type: "column",
      marginRight: 380,
    },
    title: {
      text: "Agricultural Management Area by BroadLand-use Type",
    },
    series:
      JSON.parse(document.getElementById("Area_Am_3_Land-use").innerHTML).AUSTRALIA,
    credits: {
      enabled: false,
    },
    xAxis: {
      tickPositions: year_ticks,
    },
    yAxis: {
      title: {
        text: "Area (ha)",
      },
    },

    legend: {
      layout: 'vertical',
      align: 'right',
      verticalAlign: 'middle',
      x: -150, 
      floating: true
    },

    tooltip: {
      formatter: function () {
        return `<b>Year:</b> ${this.x}<br><b>${this.series.name
          }:</b>${this.y.toFixed(2)}<br/>`;
      },
    },

    plotOptions: {
      column: {
        stacking: "normal",
      },
    },

    exporting: {
      sourceWidth: 1200,
      sourceHeight: 600,
    },
  });

  // area_begin_end_area
  document.getElementById("area_begin_end_area").innerHTML = JSON.parse(document.getElementById(
    "Area_transition_start_end"
  ).innerText).AUSTRALIA.area;

  // area_begin_end_pct
  document.getElementById("area_begin_end_pct").innerHTML = JSON.parse(document.getElementById(
    "Area_transition_start_end"
  ).innerText).AUSTRALIA.pct;

  // area_year_to_year transitions
  let data_area_transition = JSON.parse(
    document.getElementById("Area_transition_year_to_year").innerText
  ).AUSTRALIA;

  // Get area transitions selectors and buttons
  let slider_transition = document.getElementById("year_transition");
  let incrementButton_transition = document.getElementById("increment_transition");
  let decrementButton_transition = document.getElementById("decrement_transition");
  let yearOutput_transition = document.getElementById('yearOutput_transition');
  
  // Get pct transitions selectors and buttons
  let slider_transition_pct = document.getElementById("year_transition_pct");
  let incrementButton_transition_pct = document.getElementById("increment_transition_pct");
  let decrementButton_transition_pct = document.getElementById("decrement_transition_pct");
  let yearOutput_transition_pct = document.getElementById('yearOutput_transition_pct');
  
  // Get available years from the data and sort them
  let availableYears_transition = Object.keys(data_area_transition.area).sort((a, b) => a - b);
  
  // Set up the sliders with correct range for area transition
  if (availableYears_transition.length > 0) {
    // Area transition slider
    slider_transition.min = availableYears_transition[0];
    slider_transition.max = availableYears_transition[availableYears_transition.length - 1];
    slider_transition.step = availableYears_transition.length > 1 ? 
      availableYears_transition[1] - availableYears_transition[0] : 1;
    slider_transition.value = availableYears_transition[0];
    yearOutput_transition.innerHTML = slider_transition.value;

    // Percent transition slider
    slider_transition_pct.min = availableYears_transition[0];
    slider_transition_pct.max = availableYears_transition[availableYears_transition.length - 1];
    slider_transition_pct.step = availableYears_transition.length > 1 ? 
      availableYears_transition[1] - availableYears_transition[0] : 1;
    slider_transition_pct.value = availableYears_transition[0];
    yearOutput_transition_pct.innerHTML = slider_transition_pct.value;

    // Add event listeners for area transitions
    slider_transition.addEventListener("input", function () {
      yearOutput_transition.innerHTML = slider_transition.value;
      update_area_transition();
    });

    incrementButton_transition.addEventListener("click", function () {
      let currentValue = slider_transition.value;
      let currentIndex = availableYears_transition.indexOf(parseInt(currentValue));
      if (currentIndex < availableYears_transition.length - 1) {
        slider_transition.value = availableYears_transition[currentIndex + 1];
        slider_transition.dispatchEvent(new Event('input'));
      }
    });

    decrementButton_transition.addEventListener("click", function () {
      let currentValue = slider_transition.value;
      let currentIndex = availableYears_transition.indexOf(parseInt(currentValue));
      if (currentIndex > 0) {
        slider_transition.value = availableYears_transition[currentIndex - 1];
        slider_transition.dispatchEvent(new Event('input'));
      }
    });

    // Add event listeners for pct transitions
    slider_transition_pct.addEventListener("input", function () {
      yearOutput_transition_pct.innerHTML = slider_transition_pct.value;
      update_pct_transition();
    });

    incrementButton_transition_pct.addEventListener("click", function () {
      let currentValue = slider_transition_pct.value;
      let currentIndex = availableYears_transition.indexOf(parseInt(currentValue));
      if (currentIndex < availableYears_transition.length - 1) {
        slider_transition_pct.value = availableYears_transition[currentIndex + 1];
        slider_transition_pct.dispatchEvent(new Event('input'));
      }
    });

    decrementButton_transition_pct.addEventListener("click", function () {
      let currentValue = slider_transition_pct.value;
      let currentIndex = availableYears_transition.indexOf(parseInt(currentValue));
      if (currentIndex > 0) {
        slider_transition_pct.value = availableYears_transition[currentIndex - 1];
        slider_transition_pct.dispatchEvent(new Event('input'));
      }
    });

    // Function to update area transition matrix
    function update_area_transition() {
      document.getElementById("area_year_to_year_area").innerHTML = 
        data_area_transition.area[slider_transition.value];
    }

    // Function to update percent transition matrix
    function update_pct_transition() {
      document.getElementById("area_year_to_year_pct").innerHTML = 
        data_area_transition.pct[slider_transition_pct.value];
    }

    // Initial update of the transition matrices
    update_area_transition();
    update_pct_transition();
  }
});

