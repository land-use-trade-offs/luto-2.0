// Create Chart
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
    
    // Define title mapping for different biodiversity types
    const bioTypeTitles = {
        'BIO_quality': 'Biodiversity Quality',
        'BIO_GBF2': 'GBF Target 2',
        'BIO_GBF3': 'GBF Target 3 - National Vegetation Information System (NVIS)',
        'BIO_GBF4_ECNES': 'GBF Target 4 - Species of Ecological Communities of National Environmental Significance (ECNES)',
        'BIO_GBF4_SNES': 'GBF Target 4 - Species of National Environmental Significance (SNES)',
        'BIO_GBF8_GROUP': 'GBF Target 8 - Species Scores by Group',
        'BIO_GBF8_SPECIES': 'GBF Target 8 - Species Scores by Species'
    };
    
    // Define metric view mapping for readable names
    const metricViewTitles = {
        'overview_1_Type': 'Overview',
        'split_Ag_1_Landuse': 'Agricultural Land-use',
        'split_Am_2_Agri-Management': 'Agricultural Management',
        'split_Am_1_Landuse': 'Agricultural Management by Land-use',
        'split_NonAg_1_Landuse': 'Non-Agricultural Land-use'
    };
    
    
    // Chart configuration templates
    const chartConfigs = {
        // Template for basic chart configuration
        baseConfig: {
            chart: {
                type: "column",
                marginRight: 380,
            },
            yAxis: {
                title: {
                    text: "Contribution percent (2010=100%)",
                },
            },
            plotOptions: {
                column: {
                    stacking: "normal",
                },
            },
            tooltip: {
                formatter: function () {
                    return `<b>Year:</b> ${this.x}<br><b>${this.series.name
                        }:</b>${this.y.toFixed(2)}<br/>`;
                },
            },
            legend: {
                align: "right",
                layout: "vertical",
                x: -150,
                verticalAlign: "middle",
            },
            credits: {
                enabled: false,
            },
        },
        
        // Custom y-axis titles for different biodiversity types
        yAxisTitles: {
            'BIO_GBF3': "Contribution percent (Pre-1750=100%)",
            'default': "Contribution percent (2010=100%)",
        },
        
        // Chart title templates
        titleTemplates: {
            'overview_1_Type': "{bioType} Overview",
            'split_Ag_1_Landuse': "{bioType} Agricultural Land-use Contribution",
            'split_Am_2_Agri-Management': "{bioType} Agricultural Management Contribution",
            'split_Am_1_Landuse': "{bioType} Agricultural Management by Land-use Contribution",
            'split_NonAg_1_Landuse': "{bioType} Non-Agricultural Land-use Contribution",
        }
    };
    
    // Function to create and show the selected chart
    function showSelectedChart() {
        const bioType = document.getElementById('bio_type').value;
        let metricsView = document.getElementById('metrics_view').value;
        
        // Update the page title
        document.getElementById('biodiversity_title').textContent = bioTypeTitles[bioType];
        
        // Clear the chart container
        const chartContainer = document.getElementById('chart_container');
        chartContainer.innerHTML = '';
        
        // Create a new div for the chart
        const chartDiv = document.createElement('div');
        chartDiv.className = 'fig';
        chartDiv.id = 'dynamic-chart';
        chartContainer.appendChild(chartDiv);
        
        
        // Get the data element
        const dataElement = document.getElementById(bioType + '_' + metricsView);
        
        if (dataElement && dataElement.innerHTML) {
            try {
                // Prepare chart configuration
                const config = JSON.parse(JSON.stringify(chartConfigs.baseConfig)); // Deep clone
                
                // Set chart title
                const titleTemplate = chartConfigs.titleTemplates[metricsView] || "{bioType}";
                const chartTitle = titleTemplate.replace("{bioType}", bioTypeTitles[bioType].replace(/ - .*$/, ''));
                config.title = { text: chartTitle };
                
                // Set Y-axis title based on biodiversity type
                const yAxisTitle = chartConfigs.yAxisTitles[bioType] || chartConfigs.yAxisTitles['default'];
                config.yAxis.title.text = yAxisTitle;
                
                // Set data series
                config.series = JSON.parse(dataElement.innerHTML).AUSTRALIA;
                
                // Create the chart
                Highcharts.chart('dynamic-chart', config);
            } catch (error) {
                console.error('Error creating chart:', error);
                chartContainer.innerHTML = `<p>Error creating chart: ${error.message}</p>`;
            }
        } else {
            chartContainer.innerHTML = '<p>Chart data not available for this selection.</p>';
        }
    }
    
    // Adjust the metrics view options based on the selected biodiversity type
    function updateMetricsViewOptions() {
        const bioType = document.getElementById('bio_type').value;
        const metricsView = document.getElementById('metrics_view');
        
        // Save current selection if possible
        const currentSelection = metricsView.value;
        
        // Clear options
        metricsView.innerHTML = '';
        
        // Add common options for all types
        const commonOptions = ['overview_1_Type', 'split_Ag_1_Landuse', 'split_Am_2_Agri-Management', 'split_NonAg_1_Landuse'];
        
        // Add 'split_Am_1_Landuse' for specific bio types that support it
        const landUseOptions = [...commonOptions];
        if (bioType === 'BIO_quality' || bioType === 'BIO_GBF3' || 
            bioType === 'BIO_GBF8_GROUP' || bioType === 'BIO_GBF8_SPECIES') {
            landUseOptions.splice(3, 0, 'split_Am_1_Landuse');
        }
        
        // Add options to the select
        landUseOptions.forEach(option => {
            const optElement = document.createElement('option');
            optElement.value = option;
            optElement.textContent = metricViewTitles[option];
            metricsView.appendChild(optElement);
        });
        
        // Try to restore previous selection if it exists in new options
        if (landUseOptions.includes(currentSelection)) {
            metricsView.value = currentSelection;
        }
    }
    
    // Event listeners for the dropdown selects
    document.getElementById('bio_type').addEventListener('change', function() {
        updateMetricsViewOptions();
        showSelectedChart();
    });
    
    document.getElementById('metrics_view').addEventListener('change', function() {
        showSelectedChart();
    });
    
    // Initialize the selection interface
    updateMetricsViewOptions();
    showSelectedChart();

    // BIO_quality_overview_1_Type
    Highcharts.chart("BIO_quality_overview_1_Type_chart", {
        chart: {
            type: "column",
            marginRight: 380,
        },
        title: {
            text: "Biodiversity Quality Overview",
        },
        series: JSON.parse(
            document.getElementById("BIO_quality_overview_1_Type").innerHTML
        ).AUSTRALIA,

        yAxis: {
            title: {
                text: "Contribution percent (2010=100%)",
            },
        },
        plotOptions: {
            column: {
                stacking: "normal",
            },
        },
        tooltip: {
            formatter: function () {
                return `<b>Year:</b> ${this.x}<br><b>${this.series.name
                    }:</b>${this.y.toFixed(2)}<br/>`;
            },
        },
        legend: {
            align: "right",
            verticalalign: "left",
            layout: "vertical",
            x: -150,
            verticalAlign: "middle",
        },
        credits: {
            enabled: false,
        },
    });

    // BIO_quality_split_Ag_1_Landuse
    Highcharts.chart("BIO_quality_split_Ag_1_Landuse_chart", {
        chart: {
            type: "column",
            marginRight: 380,
        },
        title: {
            text: "Biodiversity Quality Overview by Agricultural Land-use",
        },
        series: JSON.parse(
            document.getElementById(
                "BIO_quality_split_Ag_1_Landuse"
            ).innerHTML
        ).AUSTRALIA,

        yAxis: {
            title: {
                text: "Contribution percent (2010=100%)",
            },
        },
        plotOptions: {
            column: {
                stacking: "normal",
            },
        },
        tooltip: {
            formatter: function () {
                return `<b>Year:</b> ${this.x}<br><b>${this.series.name
                    }:</b>${this.y.toFixed(2)}<br/>`;
            },
        },
        legend: {
            itemStyle: {
                fontSize: "11px",
            },
            align: "right",
            layout: "vertical",
            x: -100,
            y: -10,
            verticalAlign: "middle",
            itemMarginTop: 0,
            itemMarginBottom: 1,
        },
        credits: {
            enabled: false,
        },
    });

    // BIO_quality_split_Am_2_Agri-Management
    Highcharts.chart("BIO_quality_split_Am_2_Agri-Management_chart", {
        chart: {
            type: "column",
            marginRight: 380,
        },
        title: {
            text: "Biodiversity Quality Overview by Agricultural Management by Broad Type",
        },
        series: JSON.parse(
            document.getElementById(
                "BIO_quality_split_Am_2_Agri-Management"
            ).innerHTML
        ).AUSTRALIA,

        yAxis: {
            title: {
                text: "Contribution percent (2010=100%)",
            },
        },
        plotOptions: {
            column: {
                stacking: "normal",
            },
        },
        tooltip: {
            formatter: function () {
                return `<b>Year:</b> ${this.x}<br><b>${this.series.name
                    }:</b>${this.y.toFixed(2)}<br/>`;
            },
        },
        legend: {
            align: "right",
            verticalalign: "left",
            layout: "vertical",
            x: -100,
            verticalAlign: "middle",
        },
        credits: {
            enabled: false,
        },
    });

    // BIO_quality_split_Am_1_Landuse_chart
    Highcharts.chart("BIO_quality_split_Am_1_Landuse_chart", {
        chart: {
            type: "column",
            marginRight: 380,
        },
        title: {
            text: "Biodiversity Quality Overview by Agricultural Management by Land-use",
        },
        series: JSON.parse(
            document.getElementById(
                "BIO_quality_split_Am_1_Landuse"
            ).innerHTML
        ).AUSTRALIA,

        yAxis: {
            title: {
                text: "Contribution percent (2010=100%)",
            },
        },
        plotOptions: {
            column: {
                stacking: "normal",
            },
        },
        tooltip: {
            formatter: function () {
                return `<b>Year:</b> ${this.x}<br><b>${this.series.name
                    }:</b>${this.y.toFixed(2)}<br/>`;
            },
        },
        legend: {
            align: "right",
            verticalalign: "left",
            layout: "vertical",
            x: -100,
            verticalAlign: "middle",
        },
        credits: {
            enabled: false,
        },
    });

    // BIO_quality_split_NonAg_1_Landuse
    Highcharts.chart("BIO_quality_split_NonAg_1_Landuse_chart", {
        chart: {
            type: "column",
            marginRight: 380,
        },
        title: {
            text: "Biodiversity Quality Overview by Non-Agricultural Land-use",
        },
        series: JSON.parse(
            document.getElementById("BIO_quality_split_NonAg_1_Landuse").innerHTML
        ).AUSTRALIA,

        yAxis: {
            title: {
                text: "Contribution percent (2010=100%)",
            },
        },
        plotOptions: {
            column: {
                stacking: "normal",
            },
        },
        tooltip: {
            formatter: function () {
                return `<b>Year:</b> ${this.x}<br><b>${this.series.name
                    }:</b>${this.y.toFixed(2)}<br/>`;
            },
        },
        legend: {
            align: "right",
            verticalalign: "left",
            layout: "vertical",
            x: -100,
            verticalAlign: "middle",
        },
        credits: {
            enabled: false,
        },
    });


    // BIO_GBF2_overview_1_Type
    Highcharts.chart("BIO_GBF2_overview_1_Type_chart", {
        chart: {
            type: "column",
            marginRight: 380,
        },
        title: {
            text: "GBF Target 2 - Overview",
        },
        series: JSON.parse(
            document.getElementById("BIO_GBF2_overview_1_Type").innerHTML
        ).AUSTRALIA,

        yAxis: {
            title: {
                text: "Contribution percent (2010=100%)",
            },
        },
        plotOptions: {
            column: {
                stacking: "normal",
            },
        },
        tooltip: {
            formatter: function () {
                return `<b>Year:</b> ${this.x}<br><b>${this.series.name
                    }:</b>${this.y.toFixed(2)}<br/>`;
            },
        },
        legend: {
            align: "right",
            layout: "vertical",
            x: -150,
            y: -10,
            verticalAlign: "middle",
        },
        credits: {
            enabled: false,
        },
    });

    // BIO_GBF2_split_Ag_1_Landuse
    Highcharts.chart("BIO_GBF2_split_Ag_1_Landuse_chart", {
        chart: {
            type: "column",
            marginRight: 380,
        },
        title: {
            text: "GBF Target 2 - Agricultural Land-use Contribution",
        },
        series: JSON.parse(
            document.getElementById("BIO_GBF2_split_Ag_1_Landuse").innerHTML
        ).AUSTRALIA,

        yAxis: {
            title: {
                text: "Contribution percent (2010=100%)",
            },
        },
        plotOptions: {
            column: {
                stacking: "normal",
            },
        },
        tooltip: {
            formatter: function () {
                return `<b>Year:</b> ${this.x}<br><b>${this.series.name
                    }:</b>${this.y.toFixed(2)}<br/>`;
            },
        },
        legend: {
            itemStyle: {
                fontSize: "11px",
            },
            align: "right",
            layout: "vertical",
            x: -100,
            y: -10,
            verticalAlign: "middle",
            itemMarginTop: 0,
            itemMarginBottom: 1,
        },
        credits: {
            enabled: false,
        },
    });

    // BIO_GBF2_split_Am_1_Agri-Management
    Highcharts.chart("BIO_GBF2_split_Am_1_Agri-Management_chart", {
        chart: {
            type: "column",
            marginRight: 380,
        },
        title: {
            text: "GBF Target 2 - Agricultural Management Contribution",
        },
        series: JSON.parse(
            document.getElementById("BIO_GBF2_split_Am_1_Agri-Management").innerHTML
        ).AUSTRALIA,

        yAxis: {
            title: {
                text: "Contribution percent (2010=100%)",
            },
        },
        plotOptions: {
            column: {
                stacking: "normal",
            },
        },
        tooltip: {
            formatter: function () {
                return `<b>Year:</b> ${this.x}<br><b>${this.series.name
                    }:</b>${this.y.toFixed(2)}<br/>`;
            },
        },
        legend: {
            align: "right",
            layout: "vertical",
            x: -100,
            y: -10,
            verticalAlign: "middle",
        },
        credits: {
            enabled: false,
        },
    });

    // BIO_GBF2_split_Am_2_Landuse
    Highcharts.chart("BIO_GBF2_split_Am_2_Landuse_chart", {
        chart: {
            type: "column",
            marginRight: 380,
        },
        title: {
            text: "GBF Target 2 - Agricultural Management Contribution by Land-use",
        },
        series: JSON.parse(
            document.getElementById("BIO_GBF2_split_Am_2_Landuse").innerHTML
        ).AUSTRALIA,

        yAxis: {
            title: {
                text: "Contribution percent (2010=100%)",
            },
        },
        plotOptions: {
            column: {
                stacking: "normal",
            },
        },
        tooltip: {
            formatter: function () {
                return `<b>Year:</b> ${this.x}<br><b>${this.series.name
                    }:</b>${this.y.toFixed(2)}<br/>`;
            },
        },
        legend: {
            align: "right",
            layout: "vertical",
            x: -100,
            y: -10,
            verticalAlign: "middle",
        },
        credits: {
            enabled: false,
        },
    });

    // BIO_GBF2_split_NonAg_1_Landuse
    Highcharts.chart("BIO_GBF2_split_NonAg_1_Landuse_chart", {
        chart: {
            type: "column",
            marginRight: 380,
        },
        title: {
            text: "GBF Target 2 - Non-Agricultural Land-use Contribution",
        },
        series: JSON.parse(
            document.getElementById("BIO_GBF2_split_NonAg_1_Landuse").innerHTML
        ).AUSTRALIA,

        yAxis: {
            title: {
                text: "Contribution percent (2010=100%)",
            },
        },
        plotOptions: {
            column: {
                stacking: "normal",
            },
        },
        tooltip: {
            formatter: function () {
                return `<b>Year:</b> ${this.x}<br><b>${this.series.name
                    }:</b>${this.y.toFixed(2)}<br/>`;
            },
        },
        legend: {
            align: "right",
            verticalalign: "left",
            layout: "vertical",
            x: -100,
            verticalAlign: "middle",
        },
        credits: {
            enabled: false,
        },
    });




    // BIO_GBF3_overview_1_Type_chart
    Highcharts.chart("BIO_GBF3_overview_1_Type_chart", {
        chart: {
            type: "column",
            marginRight: 380,
        },
        title: {
            text: "GBF Target 3 - NVIS Overview",
        },
        series: JSON.parse(
            document.getElementById("BIO_GBF3_overview_1_Type").innerHTML
        ).AUSTRALIA,
        yAxis: {
            title: {
                text: "Contribution percent (Pre-1750=100%)",
            },
        },
        plotOptions: {
            column: {
                stacking: "normal",
            },
        },
        tooltip: {
            formatter: function () {
                return `<b>Year:</b> ${this.x}<br><b>${this.series.name
                    }:</b>${this.y.toFixed(2)}<br/>`;
            },
        },
        legend: {
            align: "right",
            layout: "vertical",
            x: -150,
            verticalAlign: "middle",
        },
        credits: {
            enabled: false,
        },
    });

    // BIO_GBF3_split_Ag_1_Landuse_chart
    Highcharts.chart("BIO_GBF3_split_Ag_1_Landuse_chart", {
        chart: {
            type: "column",
            marginRight: 380,
        },
        title: {
            text: "GBF Target 3 - Agricultural Land-use Contribution",
        },
        series: JSON.parse(
            document.getElementById("BIO_GBF3_split_Ag_1_Landuse").innerHTML
        ).AUSTRALIA,
        yAxis: {
            title: {
                text: "Contribution percent (Pre-1750=100%)",
            },
        },
        plotOptions: {
            column: {
                stacking: "normal",
            },
        },
        tooltip: {
            formatter: function () {
                return `<b>Year:</b> ${this.x}<br><b>${this.series.name
                    }:</b>${this.y.toFixed(2)}<br/>`;
            },
        },
        legend: {
            itemStyle: {
                fontSize: "11px",
            },
            align: "right",
            layout: "vertical",
            x: -100,
            verticalAlign: "middle",
            itemMarginTop: 0,
            itemMarginBottom: 1,
        },
        credits: {
            enabled: false,
        },
    });

    // BIO_GBF3_split_Am_2_Agri-Management_chart
    Highcharts.chart("BIO_GBF3_split_Am_2_Agri-Management_chart", {
        chart: {
            type: "column",
            marginRight: 380,
        },
        title: {
            text: "GBF Target 3 - Agricultural Management Contribution",
        },
        series: JSON.parse(
            document.getElementById("BIO_GBF3_split_Am_2_Agri-Management").innerHTML
        ).AUSTRALIA,
        yAxis: {
            title: {
                text: "Contribution percent (Pre-1750=100%)",
            },
        },
        plotOptions: {
            column: {
                stacking: "normal",
            },
        },
        tooltip: {
            formatter: function () {
                return `<b>Year:</b> ${this.x}<br><b>${this.series.name
                    }:</b>${this.y.toFixed(2)}<br/>`;
            },
        },
        legend: {
            align: "right",
            layout: "vertical",
            x: -100,
            verticalAlign: "middle",
        },
        credits: {
            enabled: false,
        },
    });

    // BIO_GBF3_split_Am_1_Landuse_chart
    Highcharts.chart("BIO_GBF3_split_Am_1_Landuse_chart", {
        chart: {
            type: "column",
            marginRight: 380,
        },
        title: {
            text: "GBF Target 3 - Agricultural Management Contribution by Land-use",
        },
        series: JSON.parse(
            document.getElementById("BIO_GBF3_split_Am_1_Landuse").innerHTML
        ).AUSTRALIA,
        yAxis: {
            title: {
                text: "Contribution percent (Pre-1750=100%)",
            },
        },
        plotOptions: {
            column: {
                stacking: "normal",
            },
        },
        tooltip: {
            formatter: function () {
                return `<b>Year:</b> ${this.x}<br><b>${this.series.name
                    }:</b>${this.y.toFixed(2)}<br/>`;
            },
        },
        legend: {
            align: "right",
            layout: "vertical",
            x: -100,
            verticalAlign: "middle",
        },
        credits: {
            enabled: false,
        },
    });

    // BIO_GBF3_split_NonAg_1_Landuse_chart
    Highcharts.chart("BIO_GBF3_split_NonAg_1_Landuse_chart", {
        chart: {
            type: "column",
            marginRight: 380,
        },
        title: {
            text: "GBF Target 3 - Non-Agricultural Land-use Contribution",
        },
        series: JSON.parse(
            document.getElementById("BIO_GBF3_split_NonAg_1_Landuse").innerHTML
        ).AUSTRALIA,
        yAxis: {
            title: {
                text: "Contribution percent (Pre-1750=100%)",
            },
        },
        plotOptions: {
            column: {
                stacking: "normal",
            },
        },
        tooltip: {
            formatter: function () {
                return `<b>Year:</b> ${this.x}<br><b>${this.series.name
                    }:</b>${this.y.toFixed(2)}<br/>`;
            },
        },
        legend: {
            align: "right",
            layout: "vertical",
            x: -100,
            verticalAlign: "middle",
        },
        credits: {
            enabled: false,
        },
    });

    // GBF4 - ECNES Charts
    // BIO_GBF4_ECNES_overview_1_Type_chart
    Highcharts.chart("BIO_GBF4_ECNES_overview_1_Type_chart", {
        chart: {
            type: "column",
            marginRight: 380,
        },
        title: {
            text: "GBF Target 4 - ECNES Overview",
        },
        series: JSON.parse(
            document.getElementById("BIO_GBF4_ECNES_overview_1_Type").innerHTML
        ).AUSTRALIA,
        yAxis: {
            title: {
                text: "Contribution percent (2010=100%)",
            },
        },
        plotOptions: {
            column: {
                stacking: "normal",
            },
        },
        tooltip: {
            formatter: function () {
                return `<b>Year:</b> ${this.x}<br><b>${this.series.name
                    }:</b>${this.y.toFixed(2)}<br/>`;
            },
        },
        legend: {
            align: "right",
            layout: "vertical",
            x: -150,
            verticalAlign: "middle",
        },
        credits: {
            enabled: false,
        },
    });

    // BIO_GBF4_ECNES_split_Ag_1_Landuse_chart
    Highcharts.chart("BIO_GBF4_ECNES_split_Ag_1_Landuse_chart", {
        chart: {
            type: "column",
            marginRight: 380,
        },
        title: {
            text: "GBF Target 4 - ECNES Agricultural Land-use Contribution",
        },
        series: JSON.parse(
            document.getElementById("BIO_GBF4_ECNES_split_Ag_1_Landuse").innerHTML
        ).AUSTRALIA,
        yAxis: {
            title: {
                text: "Contribution percent (2010=100%)",
            },
        },
        plotOptions: {
            column: {
                stacking: "normal",
            },
        },
        tooltip: {
            formatter: function () {
                return `<b>Year:</b> ${this.x}<br><b>${this.series.name
                    }:</b>${this.y.toFixed(2)}<br/>`;
            },
        },
        legend: {
            itemStyle: {
                fontSize: "11px",
            },
            align: "right",
            layout: "vertical",
            x: -100,
            verticalAlign: "middle",
            itemMarginTop: 0,
            itemMarginBottom: 1,
        },
        credits: {
            enabled: false,
        },
    });

    // BIO_GBF4_ECNES_split_Am_2_Agri-Management_chart
    Highcharts.chart("BIO_GBF4_ECNES_split_Am_2_Agri-Management_chart", {
        chart: {
            type: "column",
            marginRight: 380,
        },
        title: {
            text: "GBF Target 4 - ECNES Agricultural Management Contribution",
        },
        series: JSON.parse(
            document.getElementById("BIO_GBF4_ECNES_split_Am_2_Agri-Management").innerHTML
        ).AUSTRALIA,
        yAxis: {
            title: {
                text: "Contribution percent (2010=100%)",
            },
        },
        plotOptions: {
            column: {
                stacking: "normal",
            },
        },
        tooltip: {
            formatter: function () {
                return `<b>Year:</b> ${this.x}<br><b>${this.series.name
                    }:</b>${this.y.toFixed(2)}<br/>`;
            },
        },
        legend: {
            align: "right",
            layout: "vertical",
            x: -100,
            verticalAlign: "middle",
        },
        credits: {
            enabled: false,
        },
    });


    // BIO_GBF4_ECNES_split_Am_1_Landuse_chart
    Highcharts.chart("BIO_GBF4_ECNES_split_Am_1_Landuse_chart", {
        chart: {
            type: "column",
            marginRight: 380,
        },
        title: {
            text: "GBF Target 4 - ECNES Agricultural Management Contribution",
        },
        series: JSON.parse(
            document.getElementById("BIO_GBF4_ECNES_split_Am_1_Landuse").innerHTML
        ).AUSTRALIA,
        yAxis: {
            title: {
                text: "Contribution percent (2010=100%)",
            },
        },
        plotOptions: {
            column: {
                stacking: "normal",
            },
        },
        tooltip: {
            formatter: function () {
                return `<b>Year:</b> ${this.x}<br><b>${this.series.name
                    }:</b>${this.y.toFixed(2)}<br/>`;
            },
        },
        legend: {
            align: "right",
            layout: "vertical",
            x: -100,
            verticalAlign: "middle",
        },
        credits: {
            enabled: false,
        },
    });

    // BIO_GBF4_ECNES_split_NonAg_1_Landuse_chart
    Highcharts.chart("BIO_GBF4_ECNES_split_NonAg_1_Landuse_chart", {
        chart: {
            type: "column",
            marginRight: 380,
        },
        title: {
            text: "GBF Target 4 - ECNES Non-Agricultural Land-use Contribution",
        },
        series: JSON.parse(
            document.getElementById("BIO_GBF4_ECNES_split_NonAg_1_Landuse").innerHTML
        ).AUSTRALIA,
        yAxis: {
            title: {
                text: "Contribution percent (2010=100%)",
            },
        },
        plotOptions: {
            column: {
                stacking: "normal",
            },
        },
        tooltip: {
            formatter: function () {
                return `<b>Year:</b> ${this.x}<br><b>${this.series.name
                    }:</b>${this.y.toFixed(2)}<br/>`;
            },
        },
        legend: {
            align: "right",
            layout: "vertical",
            x: -100,
            verticalAlign: "middle",
        },
        credits: {
            enabled: false,
        },
    });

    // GBF4 - SNES Charts
    // BIO_GBF4_SNES_overview_1_Type_chart
    Highcharts.chart("BIO_GBF4_SNES_overview_1_Type_chart", {
        chart: {
            type: "column",
            marginRight: 380,
        },
        title: {
            text: "GBF Target 4 - SNES Overview",
        },
        series: JSON.parse(
            document.getElementById("BIO_GBF4_SNES_overview_1_Type").innerHTML
        ).AUSTRALIA,
        yAxis: {
            title: {
                text: "Contribution percent (2010=100%)",
            },
        },
        plotOptions: {
            column: {
                stacking: "normal",
            },
        },
        tooltip: {
            formatter: function () {
                return `<b>Year:</b> ${this.x}<br><b>${this.series.name
                    }:</b>${this.y.toFixed(2)}<br/>`;
            },
        },
        legend: {
            align: "right",
            layout: "vertical",
            x: -150,
            verticalAlign: "middle",
        },
        credits: {
            enabled: false,
        },
    });

    // BIO_GBF4_SNES_split_Ag_1_Landuse_chart
    Highcharts.chart("BIO_GBF4_SNES_split_Ag_1_Landuse_chart", {
        chart: {
            type: "column",
            marginRight: 380,
        },
        title: {
            text: "GBF Target 4 - SNES Agricultural Land-use Contribution",
        },
        series: JSON.parse(
            document.getElementById("BIO_GBF4_SNES_split_Ag_1_Landuse").innerHTML
        ).AUSTRALIA,
        yAxis: {
            title: {
                text: "Contribution percent (2010=100%)",
            },
        },
        plotOptions: {
            column: {
                stacking: "normal",
            },
        },
        tooltip: {
            formatter: function () {
                return `<b>Year:</b> ${this.x}<br><b>${this.series.name
                    }:</b>${this.y.toFixed(2)}<br/>`;
            },
        },
        legend: {
            itemStyle: {
                fontSize: "11px",
            },
            align: "right",
            layout: "vertical",
            x: -100,
            verticalAlign: "middle",
            itemMarginTop: 0,
            itemMarginBottom: 1,
        },
        credits: {
            enabled: false,
        },
    });

    // BIO_GBF4_SNES_split_Am_1_Agri-Management_chart
    Highcharts.chart("BIO_GBF4_SNES_split_Am_1_Agri-Management_chart", {
        chart: {
            type: "column",
            marginRight: 380,
        },
        title: {
            text: "GBF Target 4 - SNES Agricultural Management Contribution",
        },
        series: JSON.parse(
            document.getElementById("BIO_GBF4_SNES_split_Am_1_Agri-Management").innerHTML
        ).AUSTRALIA,
        yAxis: {
            title: {
                text: "Contribution percent (2010=100%)",
            },
        },
        plotOptions: {
            column: {
                stacking: "normal",
            },
        },
        tooltip: {
            formatter: function () {
                return `<b>Year:</b> ${this.x}<br><b>${this.series.name
                    }:</b>${this.y.toFixed(2)}<br/>`;
            },
        },
        legend: {
            align: "right",
            layout: "vertical",
            x: -100,
            verticalAlign: "middle",
        },
        credits: {
            enabled: false,
        },
    });

    // BIO_GBF4_SNES_split_NonAg_1_Landuse_chart
    Highcharts.chart("BIO_GBF4_SNES_split_NonAg_1_Landuse_chart", {
        chart: {
            type: "column",
            marginRight: 380,
        },
        title: {
            text: "GBF Target 4 - SNES Non-Agricultural Land-use Contribution",
        },
        series: JSON.parse(
            document.getElementById("BIO_GBF4_SNES_split_NonAg_1_Landuse").innerHTML
        ).AUSTRALIA,
        yAxis: {
            title: {
                text: "Contribution percent (2010=100%)",
            },
        },
        plotOptions: {
            column: {
                stacking: "normal",
            },
        },
        tooltip: {
            formatter: function () {
                return `<b>Year:</b> ${this.x}<br><b>${this.series.name
                    }:</b>${this.y.toFixed(2)}<br/>`;
            },
        },
        legend: {
            align: "right",
            layout: "vertical",
            x: -100,
            verticalAlign: "middle",
        },
        credits: {
            enabled: false,
        },
    });

    // GBF8 - GROUP Charts
    // BIO_GBF8_GROUP_overview_1_Type_chart
    Highcharts.chart("BIO_GBF8_GROUP_overview_1_Type_chart", {
        chart: {
            type: "column",
            marginRight: 380,
        },
        title: {
            text: "GBF Target 8 - Species Group Overview",
        },
        series: JSON.parse(
            document.getElementById("BIO_GBF8_GROUP_overview_1_Type").innerHTML
        ).AUSTRALIA,
        yAxis: {
            title: {
                text: "Contribution percent (2010=100%)",
            },
        },
        plotOptions: {
            column: {
                stacking: "normal",
            },
        },
        tooltip: {
            formatter: function () {
                return `<b>Year:</b> ${this.x}<br><b>${this.series.name
                    }:</b>${this.y.toFixed(2)}<br/>`;
            },
        },
        legend: {
            align: "right",
            layout: "vertical",
            x: -150,
            verticalAlign: "middle",
        },
        credits: {
            enabled: false,
        },
    });

    // BIO_GBF8_GROUP_split_Ag_1_Landuse_chart
    Highcharts.chart("BIO_GBF8_GROUP_split_Ag_1_Landuse_chart", {
        chart: {
            type: "column",
            marginRight: 380,
        },
        title: {
            text: "GBF Target 8 - Species Group Agricultural Land-use Contribution",
        },
        series: JSON.parse(
            document.getElementById("BIO_GBF8_GROUP_split_Ag_1_Landuse").innerHTML
        ).AUSTRALIA,
        yAxis: {
            title: {
                text: "Contribution percent (2010=100%)",
            },
        },
        plotOptions: {
            column: {
                stacking: "normal",
            },
        },
        tooltip: {
            formatter: function () {
                return `<b>Year:</b> ${this.x}<br><b>${this.series.name
                    }:</b>${this.y.toFixed(2)}<br/>`;
            },
        },
        legend: {
            itemStyle: {
                fontSize: "11px",
            },
            align: "right",
            layout: "vertical",
            x: -100,
            verticalAlign: "middle",
            itemMarginTop: 0,
            itemMarginBottom: 1,
        },
        credits: {
            enabled: false,
        },
    });

    // BIO_GBF8_GROUP_split_Am_2_Agri-Management_chart
    Highcharts.chart("BIO_GBF8_GROUP_split_Am_2_Agri-Management_chart", {
        chart: {
            type: "column",
            marginRight: 380,
        },
        title: {
            text: "GBF Target 8 - Species Group Agricultural Management Contribution",
        },
        series: JSON.parse(
            document.getElementById("BIO_GBF8_GROUP_split_Am_2_Agri-Management").innerHTML
        ).AUSTRALIA,
        yAxis: {
            title: {
                text: "Contribution percent (2010=100%)",
            },
        },
        plotOptions: {
            column: {
                stacking: "normal",
            },
        },
        tooltip: {
            formatter: function () {
                return `<b>Year:</b> ${this.x}<br><b>${this.series.name
                    }:</b>${this.y.toFixed(2)}<br/>`;
            },
        },
        legend: {
            align: "right",
            layout: "vertical",
            x: -100,
            verticalAlign: "middle",
        },
        credits: {
            enabled: false,
        },
    });

    // BIO_GBF8_GROUP_split_Am_1_Landuse_chart
    Highcharts.chart("BIO_GBF8_GROUP_split_Am_1_Landuse_chart", {
        chart: {
            type: "column",
            marginRight: 380,
        },
        title: {
            text: "GBF Target 8 - Species Group Agricultural Management by Land-use Contribution",
        },
        series: JSON.parse(
            document.getElementById("BIO_GBF8_GROUP_split_Am_1_Landuse").innerHTML
        ).AUSTRALIA,
        yAxis: {
            title: {
                text: "Contribution percent (2010=100%)",
            },
        },
        plotOptions: {
            column: {
                stacking: "normal",
            },
        },
        tooltip: {
            formatter: function () {
                return `<b>Year:</b> ${this.x}<br><b>${this.series.name
                    }:</b>${this.y.toFixed(2)}<br/>`;
            },
        },
        legend: {
            align: "right",
            layout: "vertical",
            x: -100,
            verticalAlign: "middle",
        },
        credits: {
            enabled: false,
        },
    });

    // BIO_GBF8_GROUP_split_NonAg_1_Landuse_chart
    Highcharts.chart("BIO_GBF8_GROUP_split_NonAg_1_Landuse_chart", {
        chart: {
            type: "column",
            marginRight: 380,
        },
        title: {
            text: "GBF Target 8 - Species Group Non-Agricultural Land-use Contribution",
        },
        series: JSON.parse(
            document.getElementById("BIO_GBF8_GROUP_split_NonAg_1_Landuse").innerHTML
        ).AUSTRALIA,
        yAxis: {
            title: {
                text: "Contribution percent (2010=100%)",
            },
        },
        plotOptions: {
            column: {
                stacking: "normal",
            },
        },
        tooltip: {
            formatter: function () {
                return `<b>Year:</b> ${this.x}<br><b>${this.series.name
                    }:</b>${this.y.toFixed(2)}<br/>`;
            },
        },
        legend: {
            align: "right",
            layout: "vertical",
            x: -100,
            verticalAlign: "middle",
        },
        credits: {
            enabled: false,
        },
    });

    // GBF8 - SPECIES Charts
    // BIO_GBF8_SPECIES_overview_1_Type_chart
    Highcharts.chart("BIO_GBF8_SPECIES_overview_1_Type_chart", {
        chart: {
            type: "column",
            marginRight: 380,
        },
        title: {
            text: "GBF Target 8 - Species Overview",
        },
        series: JSON.parse(
            document.getElementById("BIO_GBF8_SPECIES_overview_1_Type").innerHTML
        ).AUSTRALIA,
        yAxis: {
            title: {
                text: "Contribution percent (2010=100%)",
            },
        },
        plotOptions: {
            column: {
                stacking: "normal",
            },
        },
        tooltip: {
            formatter: function () {
                return `<b>Year:</b> ${this.x}<br><b>${this.series.name
                    }:</b>${this.y.toFixed(2)}<br/>`;
            },
        },
        legend: {
            align: "right",
            layout: "vertical",
            x: -150,
            verticalAlign: "middle",
        },
        credits: {
            enabled: false,
        },
    });

    // BIO_GBF8_SPECIES_split_Ag_1_Landuse_chart
    Highcharts.chart("BIO_GBF8_SPECIES_split_Ag_1_Landuse_chart", {
        chart: {
            type: "column",
            marginRight: 380,
        },
        title: {
            text: "GBF Target 8 - Species Agricultural Land-use Contribution",
        },
        series: JSON.parse(
            document.getElementById("BIO_GBF8_SPECIES_split_Ag_1_Landuse").innerHTML
        ).AUSTRALIA,
        yAxis: {
            title: {
                text: "Contribution percent (2010=100%)",
            },
        },
        plotOptions: {
            column: {
                stacking: "normal",
            },
        },
        tooltip: {
            formatter: function () {
                return `<b>Year:</b> ${this.x}<br><b>${this.series.name
                    }:</b>${this.y.toFixed(2)}<br/>`;
            },
        },
        legend: {
            itemStyle: {
                fontSize: "11px",
            },
            align: "right",
            layout: "vertical",
            x: -100,
            verticalAlign: "middle",
            itemMarginTop: 0,
            itemMarginBottom: 1,
        },
        credits: {
            enabled: false,
        },
    });

    // BIO_GBF8_SPECIES_split_Am_2_Agri-Management_chart
    Highcharts.chart("BIO_GBF8_SPECIES_split_Am_2_Agri-Management_chart", {
        chart: {
            type: "column",
            marginRight: 380,
        },
        title: {
            text: "GBF Target 8 - Species Agricultural Management Contribution",
        },
        series: JSON.parse(
            document.getElementById("BIO_GBF8_SPECIES_split_Am_2_Agri-Management").innerHTML
        ).AUSTRALIA,
        yAxis: {
            title: {
                text: "Contribution percent (2010=100%)",
            },
        },
        plotOptions: {
            column: {
                stacking: "normal",
            },
        },
        tooltip: {
            formatter: function () {
                return `<b>Year:</b> ${this.x}<br><b>${this.series.name
                    }:</b>${this.y.toFixed(2)}<br/>`;
            },
        },
        legend: {
            align: "right",
            layout: "vertical",
            x: -100,
            verticalAlign: "middle",
        },
        credits: {
            enabled: false,
        },
    });

    // BIO_GBF8_SPECIES_split_Am_1_Landuse_chart
    Highcharts.chart("BIO_GBF8_SPECIES_split_Am_1_Landuse_chart", {
        chart: {
            type: "column",
            marginRight: 380,
        },
        title: {
            text: "GBF Target 8 - Species Agricultural Management by Land-use Contribution",
        },
        series: JSON.parse(
            document.getElementById("BIO_GBF8_SPECIES_split_Am_1_Landuse").innerHTML
        ).AUSTRALIA,
        yAxis: {
            title: {
                text: "Contribution percent (2010=100%)",
            },
        },
        plotOptions: {
            column: {
                stacking: "normal",
            },
        },
        tooltip: {
            formatter: function () {
                return `<b>Year:</b> ${this.x}<br><b>${this.series.name
                    }:</b>${this.y.toFixed(2)}<br/>`;
            },
        },
        legend: {
            align: "right",
            layout: "vertical",
            x: -100,
            verticalAlign: "middle",
        },
        credits: {
            enabled: false,
        },
    });

    // BIO_GBF8_SPECIES_split_NonAg_1_Landuse_chart
    Highcharts.chart("BIO_GBF8_SPECIES_split_NonAg_1_Landuse_chart", {
        chart: {
            type: "column",
            marginRight: 380,
        },
        title: {
            text: "GBF Target 8 - Species Non-Agricultural Land-use Contribution",
        },
        series: JSON.parse(
            document.getElementById("BIO_GBF8_SPECIES_split_NonAg_1_Landuse").innerHTML
        ).AUSTRALIA,
        yAxis: {
            title: {
                text: "Contribution percent (2010=100%)",
            },
        },
        plotOptions: {
            column: {
                stacking: "normal",
            },
        },
        tooltip: {
            formatter: function () {
                return `<b>Year:</b> ${this.x}<br><b>${this.series.name
                    }:</b>${this.y.toFixed(2)}<br/>`;
            },
        },
        legend: {
            align: "right",
            layout: "vertical",
            x: -100,
            verticalAlign: "middle",
        },
        credits: {
            enabled: false,
        },
    });


});
