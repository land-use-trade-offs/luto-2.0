// One vocabulary for the whole report: every view reads its axis titles
// from here, so a unit is changed in one place.
window.ReportLabels = {
  // Y-axis title per theme: the quantity and its unit.
  units: {
    Area: "Area (ha)",
    Economics: "Net economic returns (AUD/yr)",
    GHG: "Net emissions (t CO2e/yr)",
    Water: "Net water yield (ML/yr)",
    Production: "Production (t or kL/yr)",
    Biodiversity: "Relative percentage (pre-1750 = 100%)",
    Renewable: "Generation (MWh/yr)",
  },
};
