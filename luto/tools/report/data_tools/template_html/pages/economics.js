// create chart
document.addEventListener("DOMContentLoaded", function () {

  const support_info = JSON.parse(document.getElementById('Supporting_info').innerText);
  const colors = support_info.colors;
  const model_years = support_info.years;


  // Get the available years for plotting
  var years = model_years.map(function (x) { return parseInt(x); });
  years.sort(function (a, b) { return a - b; });
  var year_ticks = years.length == 2 ? years : null;


  // Set the title alignment to left
  Highcharts.setOptions({
    colors: colors,
    title: {
      align: 'left'
    }
  });




  // Set the title alignment to left
  Highcharts.setOptions({
    title: {
      align: 'left'
    }
  });

  // Chart:Economics_overview.json
  Highcharts.chart("Economics_overview_chart", {
    chart: {
      type: "column",
      marginRight: 380,
    },

    title: {
      text: "Revenue and Cost in General",
    },

    credits: {
      enabled: false,
    },

    series: JSON.parse(
      document.getElementById("Economics_overview").innerHTML
    ).AUSTRALIA,

    xAxis: {
      tickPositions: year_ticks,
    },

    yAxis: {
      title: {
        text: "Value (AUD)",
      },
    },

    tooltip: {
      formatter: function () {
        return (
          "<b>" +
          this.series.name +
          "</b>: " +
          Highcharts.numberFormat(this.y, 2) +
          " (AUD)"
        );
      },
    },

    plotOptions: {
      column: {
        stacking: "normal",
      },
    },

    legend: {
      align: "right",
      verticalalign: "left",
      layout: "vertical",
      x: -100,
      verticalAlign: "middle",
    },

    exporting: {
      sourceWidth: 1200,
      sourceHeight: 600,
    },
  });


  // Chart:Economics_split_Ag_3_Water_supply
  Highcharts.chart("Economics_split_Ag_3_Water_supply_chart", {
    chart: {
      type: "column",
      marginRight: 380,
    },

    title: {
      text: "Agricultural Revenue/Cost by Irrigation Status",
    },

    credits: {
      enabled: false,
    },

    series: JSON.parse(
      document.getElementById("Economics_split_Ag_3_Water_supply").innerHTML
    ).AUSTRALIA,

    xAxis: {
      tickPositions: year_ticks,
    },

    yAxis: {
      title: {
        text: "Revenue (AUD)",
      },
    },

    legend: {
      align: "right",
      verticalalign: "left",
      layout: "vertical",
      x: -150,
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


  // Chart:Economics_split_Ag_1_Land-use
  Highcharts.chart("Economics_split_Ag_1_Land-use_chart", {
    chart: {
      type: "column",
      marginRight: 380,
    },

    title: {
      text: "Agricultural Revenue/Cost by Commodity",
    },

    credits: {
      enabled: false,
    },

    series: JSON.parse(
      document.getElementById("Economics_split_Ag_1_Land-use").innerHTML
    ).AUSTRALIA,

    yAxis: {
      title: {
        text: "Revenue (AUD)",
      },
    },

    xAxis: {
      tickPositions: year_ticks,
    },

    legend: {
      align: "right",
      verticalalign: "left",
      layout: "vertical",
      x: -100,
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


  // Chart:Economics_split_Ag_2_Type
  Highcharts.chart("Economics_split_Ag_2_Type_chart", {
    chart: {
      type: "column",
      marginRight: 380,
    },

    title: {
      text: "Agricultural Revenue/Cost by Commodity Type",
    },

    credits: {
      enabled: false,
    },

    series: JSON.parse(
      document.getElementById("Economics_split_Ag_2_Type").innerHTML
    ).AUSTRALIA,

    xAxis: {
      tickPositions: year_ticks,
    },

    yAxis: {
      title: {
        text: "Revenue (AUD)",
      },
    },

    legend: {
      align: "right",
      verticalalign: "left",
      layout: "vertical",
      x: -150,
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


  // Chart:Economics_split_Ag_3_Water_supply (Cost)
  Highcharts.chart("Economics_split_AM_2_Water_supply_chart", {
    chart: {
      type: "column",
      marginRight: 380,
    },

    title: {
      text: "Agricultural Management Revenue/Cost by Irrigation Status",
    },

    credits: {
      enabled: false,
    },

    series: JSON.parse(
      document.getElementById("Economics_split_AM_2_Water_supply").innerHTML
    ).AUSTRALIA,

    yAxis: {
      title: {
        text: "Cost (AUD)",
      },
    },
    xAxis: {
      tickPositions: year_ticks,
    },

    legend: {
      align: "right",
      verticalalign: "left",
      layout: "vertical",
      x: -150,
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

  // Chart:Economics_split_AM_3_Land-use
  Highcharts.chart("Economics_split_AM_3_Land-use_chart", {
    chart: {
      type: "column",
      marginRight: 380,
    },

    title: {
      text: "Agricultural Management Revenue/Cost by Land-use",
    },

    credits: {
      enabled: false,
    },

    series: JSON.parse(
      document.getElementById("Economics_split_AM_3_Land-use").innerHTML
    ).AUSTRALIA,

    yAxis: {
      title: {
        text: "Cost (AUD)",
      },
    },
    xAxis: {
      tickPositions: year_ticks,
    },

    legend: {
      align: "right",
      verticalalign: "left",
      layout: "vertical",
      x: -100,
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


  // Chart:Economics_split_AM_1_Management_Type
  Highcharts.chart("Economics_split_AM_1_Management_Type_chart", {
    chart: {
      type: "column",
      marginRight: 380,
    },

    title: {
      text: "Agricultural Management Revenue/Cost by Management Type",
    },

    credits: {
      enabled: false,
    },

    series: JSON.parse(
      document.getElementById("Economics_split_AM_1_Management_Type").innerHTML
    ).AUSTRALIA,

    yAxis: {
      title: {
        text: "Cost (AUD)",
      },
    },
    xAxis: {
      tickPositions: year_ticks,
    },

    legend: {
      align: "right",
      verticalalign: "left",
      layout: "vertical",
      x: -150,
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



  // Chart:Economics_split_NonAg_1_Land-use
  Highcharts.chart("Economics_split_NonAg_1_Land-use_chart", {
    chart: {
      type: "column",
      marginRight: 380,
    },

    title: {
      text: "Non-Agricultural Revenue by Land-use",
    },

    credits: {
      enabled: false,
    },

    series: JSON.parse(
      document.getElementById("Economics_split_NonAg_1_Land-use").innerHTML
    ).AUSTRALIA,

    yAxis: {
      title: {
        text: "Revenue (AUD)",
      },
    },
    xAxis: {
      tickPositions: year_ticks,
    },

    legend: {
      align: "right",
      verticalalign: "left",
      layout: "vertical",
      x: -100,
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




  // Chart:Economics_transition_mat_ag2ag
  document.getElementById("Economics_transition_mat_ag2ag_chart").innerHTML = JSON.parse(
    document.getElementById("Economics_transition_mat_ag2ag").innerHTML
  );

  // Chart:Economics_transition_split_ag2ag_3_To_land-use
  Highcharts.chart("Economics_transition_split_ag2ag_3_To_land-use_chart", {
    chart: {
      type: "column",
      marginRight: 380,
    },

    title: {
      text: "Transition Cost (Agricultural to Agricultural) from target-year-prespective",
    },

    credits: {
      enabled: false,
    },

    series: JSON.parse(
      document.getElementById("Economics_transition_split_ag2ag_3_To_land-use").innerHTML
    ).AUSTRALIA,

    yAxis: {
      title: {
        text: "Cost (AUD)",
      },
    },
    xAxis: {
      tickPositions: year_ticks,
    },

    legend: {
      align: "right",
      verticalalign: "left",
      layout: "vertical",
      x: -100,
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



  // Get the slider_ag2ag and the year span
  let data_ag2ag = JSON.parse(
    document.getElementById("Economics_transition_mat_ag2ag").innerHTML
  ).AUSTRALIA;

  let slider_ag2ag = document.getElementById("year_ag2ag");
  let incrementButton_ag2ag = document.getElementById("increment_ag2ag");
  let decrementButton_ag2ag = document.getElementById("decrement_ag2ag");
  let yearOutput_ag2ag = document.getElementById('yearOutput_ag2ag');
  
  // Get available years from the data and sort them
  let availableYears_ag2ag = Object.keys(data_ag2ag).sort((a, b) => a - b);
  
  // Set up the slider with correct range
  slider_ag2ag.min = availableYears_ag2ag[0];
  slider_ag2ag.max = availableYears_ag2ag[availableYears_ag2ag.length - 1];
  slider_ag2ag.step = availableYears_ag2ag.length > 1 ? availableYears_ag2ag[1] - availableYears_ag2ag[0] : 1;
  slider_ag2ag.value = availableYears_ag2ag[0];
  yearOutput_ag2ag.innerHTML = String(slider_ag2ag.value - slider_ag2ag.step) + ' - ' + String(slider_ag2ag.value);

  // Add event listeners to the buttons
  slider_ag2ag.addEventListener("input", function () {
    yearOutput_ag2ag.innerHTML = String(slider_ag2ag.value - slider_ag2ag.step) + ' - ' + String(slider_ag2ag.value);
    draw_cost_ag2ag();
  });

  // Function to draw the chart
  draw_cost_ag2ag = function () {
    document.getElementById("Economics_transition_mat_ag2ag_chart").innerHTML = data_ag2ag[slider_ag2ag.value];
  };

  // Initial draw of the chart
  draw_cost_ag2ag();

  incrementButton_ag2ag.addEventListener("click", function () {
    let currentValue = slider_ag2ag.value;
    let currentIndex = availableYears_ag2ag.indexOf(currentValue);
    if (currentIndex < availableYears_ag2ag.length - 1) {
      slider_ag2ag.value = availableYears_ag2ag[currentIndex + 1];
      slider_ag2ag.dispatchEvent(new Event('input'));
    }
  });

  decrementButton_ag2ag.addEventListener("click", function () {
    let currentValue = slider_ag2ag.value;
    let currentIndex = availableYears_ag2ag.indexOf(currentValue);
    if (currentIndex > 0) {
      slider_ag2ag.value = availableYears_ag2ag[currentIndex - 1];
      slider_ag2ag.dispatchEvent(new Event('input'));
    }
  });

  




  // Chart:Economics_transition_split_ag2ag_1_Type
  Highcharts.chart("Economics_transition_split_ag2ag_1_Type_chart", {
    chart: {
      type: "column",
      marginRight: 380,
    },

    title: {
      text: "Transition Cost (Agricultural to Agricultural) by Cost Type",
    },

    credits: {
      enabled: false,
    },

    series: JSON.parse(
      document.getElementById("Economics_transition_split_ag2ag_1_Type").innerHTML
    ).AUSTRALIA,

    yAxis: {
      title: {
        text: "Cost (AUD)",
      },
    },
    xAxis: {
      tickPositions: year_ticks,
    },

    legend: {
      align: "right",
      verticalalign: "left",
      layout: "vertical",
      x: -150,
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




  // Chart:Economics_transition_mat_ag2nonag  
  // Get the slider_ag2non_ag and the year span
  let slider_ag2non_ag = document.getElementById("year_ag2non_ag");
  let incrementButton_ag2non_ag = document.getElementById("increment_ag2non_ag");
  let decrementButton_ag2non_ag = document.getElementById("decrement_ag2non_ag");
  let yearOutput_ag2non_ag = document.getElementById('yearOutput_ag2non_ag');
  
  let data_ag2non_ag = JSON.parse(
    document.getElementById("Economics_transition_mat_ag2nonag").innerHTML
  ).AUSTRALIA;

  // Get available years from the data and sort them
  let availableYears_ag2non_ag = Object.keys(data_ag2non_ag).sort((a, b) => a - b);
  
  // Set up the slider with correct range
  slider_ag2non_ag.min = availableYears_ag2non_ag[0];
  slider_ag2non_ag.max = availableYears_ag2non_ag[availableYears_ag2non_ag.length - 1];
  slider_ag2non_ag.step = availableYears_ag2non_ag.length > 1 ? availableYears_ag2non_ag[1] - availableYears_ag2non_ag[0] : 1;
  slider_ag2non_ag.value = availableYears_ag2non_ag[0];
  yearOutput_ag2non_ag.innerHTML = String(slider_ag2non_ag.value - slider_ag2non_ag.step) + ' - ' + String(slider_ag2non_ag.value);

  // Function to draw the chart
  draw_cost_ag2non_ag = function () {
    document.getElementById("Economics_transition_mat_ag2nonag_chart").innerHTML = data_ag2non_ag[slider_ag2non_ag.value];
  };

  // Add event listeners to the buttons
  slider_ag2non_ag.addEventListener("input", function () {
    yearOutput_ag2non_ag.innerHTML = String(this.value - this.step) + ' - ' + String(this.value);
    draw_cost_ag2non_ag();
  });

  incrementButton_ag2non_ag.addEventListener("click", function () {
    let currentValue = slider_ag2non_ag.value;
    let currentIndex = availableYears_ag2non_ag.indexOf(currentValue);
    if (currentIndex < availableYears_ag2non_ag.length - 1) {
      slider_ag2non_ag.value = availableYears_ag2non_ag[currentIndex + 1];
      slider_ag2non_ag.dispatchEvent(new Event('input'));
    }
  });

  decrementButton_ag2non_ag.addEventListener("click", function () {
    let currentValue = slider_ag2non_ag.value;
    let currentIndex = availableYears_ag2non_ag.indexOf(currentValue);
    if (currentIndex > 0) {
      slider_ag2non_ag.value = availableYears_ag2non_ag[currentIndex - 1];
      slider_ag2non_ag.dispatchEvent(new Event('input'));
    }
  });

  // Economics_transition_split_ag2ag_2_From_land-use
  Highcharts.chart("Economics_transition_split_ag2ag_2_From_land-use_chart", {
    chart: {
      type: "column",
      marginRight: 380,
    },
    title: {
      text: "Transition Cost (Agricultural to Agricultural) from Base-Year-Perspective",
    },
    credits: {
      enabled: false,
    },
    series: JSON.parse(
      document.getElementById("Economics_transition_split_ag2ag_2_From_land-use").innerHTML
    ).AUSTRALIA,
    yAxis: {
      title: {
        text: "Cost (AUD)",
      },
    },
    xAxis: {
      tickPositions: year_ticks,
    },
    legend: {
      align: "right",
      verticalalign: "left",
      layout: "vertical",
      x: -100,
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

  // Chart:Economics_transition_split_Ag2NonAg_1_Cost_type
  Highcharts.chart("Economics_transition_split_Ag2NonAg_1_Cost_type_chart", {
    chart: {
      type: "column",
      marginRight: 380,
    },

    title: {
      text: "Transition Cost (Agricultural to Non-Agricultural) by Cost type",
    },

    credits: {
      enabled: false,
    },

    series: JSON.parse(
      document.getElementById("Economics_transition_split_Ag2NonAg_1_Cost_type").innerHTML
    ).AUSTRALIA,

    yAxis: {
      title: {
        text: "Cost (AUD)",
      },
    },
    xAxis: {
      tickPositions: year_ticks,
    },

    legend: {
      align: "right",
      verticalalign: "left",
      layout: "vertical",
      x: -100,
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


  // Chart:Economics_transition_split_Ag2NonAg_2_From_land-use
  Highcharts.chart("Economics_transition_split_Ag2NonAg_2_From_land-use_chart", {
    chart: {
      type: "column",
      marginRight: 380,
    },

    title: {
      text: "Transition Cost (Agricultural to Non-Agricultural) from Base-Year-Perspective",
    },

    credits: {
      enabled: false,
    },

    series: JSON.parse(
      document.getElementById("Economics_transition_split_Ag2NonAg_2_From_land-use").innerHTML
    ).AUSTRALIA,

    yAxis: {
      title: {
        text: "Cost (AUD)",
      },
    },
    xAxis: {
      tickPositions: year_ticks,
    },

    legend: {
      align: "right",
      verticalalign: "left",
      layout: "vertical",
      x: -100,
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


  // Chart:Economics_transition_split_Ag2NonAg_3_To_land-use
  Highcharts.chart("Economics_transition_split_Ag2NonAg_3_To_land-use_chart", {
    chart: {
      type: "column",
      marginRight: 380,
    },

    title: {
      text: "Transition Cost (Agricultural to Non-Agricultural) from Target-Year-Perspective",
    },

    credits: {
      enabled: false,
    },

    series: JSON.parse(
      document.getElementById("Economics_transition_split_Ag2NonAg_3_To_land-use").innerHTML
    ).AUSTRALIA,

    yAxis: {
      title: {
        text: "Cost (AUD)",
      },
    },
    xAxis: {
      tickPositions: year_ticks,
    },

    legend: {
      align: "right",
      verticalalign: "left",
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



  // Chart:Economics_transition_mat_nonag2ag
  // Get the slider_non_ag2ag and the year span
  let slider_non_ag2ag = document.getElementById("year_non_ag2ag");
  let incrementButton_non_ag2ag = document.getElementById("increment_non_ag2ag");
  let decrementButton_non_ag2ag = document.getElementById("decrement_non_ag2ag");
  let yearOutput_non_ag2ag = document.getElementById('yearOutput_non_ag2ag');
  
  let data_non_ag2ag = JSON.parse(
    document.getElementById("Economics_transition_mat_nonag2ag").innerHTML
  ).AUSTRALIA;

  // Get available years from the data and sort them
  let availableYears_non_ag2ag = Object.keys(data_non_ag2ag).sort((a, b) => a - b);
  
  // Set up the slider with correct range
  slider_non_ag2ag.min = availableYears_non_ag2ag[0];
  slider_non_ag2ag.max = availableYears_non_ag2ag[availableYears_non_ag2ag.length - 1];
  slider_non_ag2ag.step = availableYears_non_ag2ag.length > 1 ? availableYears_non_ag2ag[1] - availableYears_non_ag2ag[0] : 1;
  slider_non_ag2ag.value = availableYears_non_ag2ag[0];
  yearOutput_non_ag2ag.innerHTML = String(slider_non_ag2ag.value - slider_non_ag2ag.step) + ' - ' + String(slider_non_ag2ag.value);

  // Add event listeners to the buttons
  slider_non_ag2ag.addEventListener("input", function () {
    yearOutput_non_ag2ag.innerHTML = String(this.value - this.step) + ' - ' + String(this.value);
    draw_cost_non_ag2ag();
  });

  incrementButton_non_ag2ag.addEventListener("click", function () {
    let currentValue = slider_non_ag2ag.value;
    let currentIndex = availableYears_non_ag2ag.indexOf(currentValue);
    if (currentIndex < availableYears_non_ag2ag.length - 1) {
      slider_non_ag2ag.value = availableYears_non_ag2ag[currentIndex + 1];
      slider_non_ag2ag.dispatchEvent(new Event('input'));
    }
  });

  decrementButton_non_ag2ag.addEventListener("click", function () {
    let currentValue = slider_non_ag2ag.value;
    let currentIndex = availableYears_non_ag2ag.indexOf(currentValue);
    if (currentIndex > 0) {
      slider_non_ag2ag.value = availableYears_non_ag2ag[currentIndex - 1];
      slider_non_ag2ag.dispatchEvent(new Event('input'));
    }
  });

  // Function to draw the chart
  draw_cost_non_ag2ag = function () {
    document.getElementById("Economics_transition_mat_nonag2ag_chart").innerHTML = data_non_ag2ag[slider_non_ag2ag.value];
  };


  // Chart:Economics_transition_split_NonAg2Ag_1_Cost_type
  Highcharts.chart("Economics_transition_split_NonAg2Ag_1_Cost_type_chart", {
    chart: {
      type: "column",
      marginRight: 380,
    },

    title: {
      text: "Cost by Cost type",
    },

    credits: {
      enabled: false,
    },

    series: JSON.parse(
      document.getElementById("Economics_transition_split_NonAg2Ag_1_Cost_type").innerHTML
    ).AUSTRALIA,

    yAxis: {
      title: {
        text: "Cost (AUD)",
      },
    },
    xAxis: {
      tickPositions: year_ticks,
    },

    legend: {
      align: "right",
      verticalalign: "left",
      layout: "vertical",
      x: -100,
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


  // Chart:Economics_transition_split_NonAg2Ag_2_From_land-use
  Highcharts.chart("Economics_transition_split_NonAg2Ag_2_From_land-use_chart", {
    chart: {
      type: "column",
      marginRight: 380,
    },

    title: {
      text: "Cost by From land-use",
    },

    credits: {
      enabled: false,
    },

    series: JSON.parse(
      document.getElementById("Economics_transition_split_NonAg2Ag_2_From_land-use").innerHTML
    ).AUSTRALIA,

    yAxis: {
      title: {
        text: "Cost (AUD)",
      },
    },
    xAxis: {
      tickPositions: year_ticks,
    },

    legend: {
      align: "right",
      verticalalign: "left",
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


  // Chart:Economics_transition_split_NonAg2Ag_3_To_land-use
  Highcharts.chart("Economics_transition_split_NonAg2Ag_3_To_land-use_chart", {
    chart: {
      type: "column",
      marginRight: 380,
    },

    title: {
      text: "Cost by To land-use",
    },

    credits: {
      enabled: false,
    },

    series: JSON.parse(
      document.getElementById("Economics_transition_split_NonAg2Ag_3_To_land-use").innerHTML
    ).AUSTRALIA,

    yAxis: {
      title: {
        text: "Cost (AUD)",
      },
    },
    xAxis: {
      tickPositions: year_ticks,
    },

    legend: {
      align: "right",
      verticalalign: "left",
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

  // Initialize the transition charts after all data is loaded
  draw_cost_ag2ag();
  if (data_ag2non_ag) draw_cost_ag2non_ag();
  if (data_non_ag2ag) draw_cost_non_ag2ag();

});


