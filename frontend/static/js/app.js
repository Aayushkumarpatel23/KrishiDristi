/**
 * KrishiDristi Frontend Application Logic
 * Orchestrates API communications, dynamic controls, Chart.js visualizations,
 * and Gmail / Google Authentication.
 */

// Global state
let metadata = null;
let currentCropData = null;
let activeCharts = {};
let currentUser = null;

// Helper: Safely destroy existing chart before recreating
function destroyChart(chartId) {
  if (activeCharts[chartId]) {
    activeCharts[chartId].destroy();
    delete activeCharts[chartId];
  }
}

// Chart default styling configuration
Chart.defaults.color = '#94a3b8';
Chart.defaults.font.family = "'Plus Jakarta Sans', sans-serif";
Chart.defaults.font.size = 11;
Chart.defaults.borderColor = 'rgba(255, 255, 255, 0.06)';

// Tab Switcher
document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tab-btn').forEach(b => {
      b.classList.remove('active');
      b.classList.add('text-slate-400');
    });
    btn.classList.add('active');
    btn.classList.remove('text-slate-400');

    document.querySelectorAll('.tab-content').forEach(c => c.classList.add('hidden'));
    const targetId = btn.getAttribute('data-target');
    const targetEl = document.getElementById(targetId);
    if (targetEl) targetEl.classList.remove('hidden');

    // Trigger tab-specific loads if first time
    if (targetId === 'tab-cleaning' && !activeCharts['chart-cleaning-comparison']) {
      loadDataCleaning();
    } else if (targetId === 'tab-stationarity' && !activeCharts['chart-acf']) {
      loadStationarity();
    } else if (targetId === 'tab-decomposition' && !activeCharts['chart-decomp-observed']) {
      loadDecomposition();
    } else if (targetId === 'tab-models' && !activeCharts['chart-forecast']) {
      trainCurrentModel();
    }
  });
});

// Dynamic State and Mandi Dropdown handler
function updateLocationDropdowns() {
  const commodity = document.getElementById('select-commodity').value;
  const stateSelect = document.getElementById('select-state');
  const mandiSelect = document.getElementById('select-mandi');

  if (!metadata) return;

  const validStates = metadata.commodity_state_map[commodity] || metadata.states;
  
  // Update States
  stateSelect.innerHTML = '<option value="All">All States (Aggregated)</option>';
  validStates.forEach(s => {
    stateSelect.innerHTML += `<option value="${s}">${s}</option>`;
  });

  // Update Mandis
  updateMandiOptions();
}

function updateMandiOptions() {
  const state = document.getElementById('select-state').value;
  const mandiSelect = document.getElementById('select-mandi');
  
  mandiSelect.innerHTML = '<option value="All">All Mandis</option>';
  if (state === 'All') {
    metadata.mandis.forEach(m => {
      mandiSelect.innerHTML += `<option value="${m}">${m}</option>`;
    });
  } else {
    const stateMandis = metadata.state_mandi_map[state] || [];
    stateMandis.forEach(m => {
      mandiSelect.innerHTML += `<option value="${m}">${m}</option>`;
    });
  }
}

document.getElementById('select-commodity').addEventListener('change', () => {
  updateLocationDropdowns();
  loadAllData();
});
document.getElementById('select-state').addEventListener('change', () => {
  updateMandiOptions();
  loadAllData();
});
document.getElementById('select-mandi').addEventListener('change', loadAllData);
document.getElementById('select-frequency').addEventListener('change', loadAllData);
document.getElementById('btn-refresh-data').addEventListener('click', loadAllData);

// ---------------- API Loaders ----------------

async function initApp() {
  try {
    const res = await fetch('/api/dataset/metadata');
    metadata = await res.json();
    
    document.getElementById('stat-total-records').innerText = `${metadata.total_records.toLocaleString()} records`;
    document.getElementById('stat-date-range').innerText = `${metadata.date_start} to ${metadata.date_end}`;
    
    updateLocationDropdowns();
    await loadAllData();
  } catch (err) {
    console.error('Failed to load dataset metadata:', err);
  }
}

async function loadAllData() {
  const commodity = document.getElementById('select-commodity').value;
  const state = document.getElementById('select-state').value;
  const mandi = document.getElementById('select-mandi').value;
  const frequency = document.getElementById('select-frequency').value;

  document.getElementById('stat-active-crop').innerText = `${commodity} (${state === 'All' ? 'National Mandis' : state})`;

  try {
    const res = await fetch('/api/dataset/timeseries', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ commodity, state, mandi, frequency })
    });
    currentCropData = await res.json();

    // Update Overview KPIs
    const prices = currentCropData.modal_prices;
    const avgPrice = prices.reduce((a, b) => a + b, 0) / prices.length;
    const minPrice = Math.min(...currentCropData.min_prices);
    const maxPrice = Math.max(...currentCropData.max_prices);

    document.getElementById('kpi-avg-price').innerText = `₹${avgPrice.toFixed(2)}`;
    document.getElementById('kpi-price-range').innerText = `₹${minPrice.toFixed(0)} - ₹${maxPrice.toFixed(0)}`;
    document.getElementById('kpi-samples').innerText = currentCropData.count;

    renderOverviewCharts();
  } catch (err) {
    console.error('Error fetching time series:', err);
  }
}

// Render Overview Charts
function renderOverviewCharts() {
  if (!currentCropData) return;

  // 1. Main Price & Arrivals Combo Chart
  destroyChart('chart-main-price');
  const ctxPrice = document.getElementById('chart-main-price').getContext('2d');
  
  // Downsample dates label display
  const labels = currentCropData.dates;
  activeCharts['chart-main-price'] = new Chart(ctxPrice, {
    type: 'line',
    data: {
      labels: labels,
      datasets: [
        {
          label: 'Modal Price (₹/Quintal)',
          data: currentCropData.modal_prices,
          borderColor: '#10b981',
          backgroundColor: '#10b981',
          borderWidth: 2,
          pointRadius: labels.length > 200 ? 0 : 2,
          pointHoverRadius: 4,
          tension: 0.15,
          yAxisID: 'y'
        },
        {
          label: 'Max Price Range',
          data: currentCropData.max_prices,
          borderColor: 'rgba(20, 184, 166, 0.4)',
          borderWidth: 1,
          pointRadius: 0,
          fill: '+1',
          backgroundColor: 'rgba(20, 184, 166, 0.08)',
          yAxisID: 'y'
        },
        {
          label: 'Min Price Range',
          data: currentCropData.min_prices,
          borderColor: 'rgba(20, 184, 166, 0.4)',
          borderWidth: 1,
          pointRadius: 0,
          fill: false,
          yAxisID: 'y'
        },
        {
          label: 'Arrivals (Tons)',
          data: currentCropData.arrivals_tons,
          type: 'bar',
          backgroundColor: 'rgba(59, 130, 246, 0.35)',
          borderRadius: 3,
          yAxisID: 'y1'
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      scales: {
        x: { grid: { display: false } },
        y: {
          type: 'linear',
          display: true,
          position: 'left',
          title: { display: true, text: 'Modal Price (₹/Quintal)', color: '#10b981' }
        },
        y1: {
          type: 'linear',
          display: true,
          position: 'right',
          grid: { drawOnChartArea: false },
          title: { display: true, text: 'Arrival Volume (Tons)', color: '#60a5fa' }
        }
      }
    }
  });

  // 2. Rainfall Anomaly Chart
  destroyChart('chart-rainfall');
  const ctxRain = document.getElementById('chart-rainfall').getContext('2d');
  activeCharts['chart-rainfall'] = new Chart(ctxRain, {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [{
        label: 'Monsoon Deviation (%)',
        data: currentCropData.rainfall_anomaly,
        backgroundColor: currentCropData.rainfall_anomaly.map(v => v >= 0 ? 'rgba(56, 189, 248, 0.7)' : 'rgba(244, 63, 94, 0.7)')
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { display: false },
        y: { title: { display: true, text: 'Deviation (%)' } }
      }
    }
  });

  // 3. Fuel Index Chart
  destroyChart('chart-fuel');
  const ctxFuel = document.getElementById('chart-fuel').getContext('2d');
  activeCharts['chart-fuel'] = new Chart(ctxFuel, {
    type: 'line',
    data: {
      labels: labels,
      datasets: [{
        label: 'Diesel Freight Index',
        data: currentCropData.fuel_index,
        borderColor: '#fbbf24',
        borderWidth: 1.8,
        pointRadius: 0,
        tension: 0.1
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { display: false },
        y: { title: { display: true, text: 'Index (Base 70)' } }
      }
    }
  });
}

// ---------------- TAB: Data Cleaning & Preprocessing ----------------

async function loadDataCleaning() {
  const commodity = document.getElementById('select-commodity').value;
  const state = document.getElementById('select-state').value;
  const mandi = document.getElementById('select-mandi').value;
  const frequency = document.getElementById('select-frequency').value;
  const outlier_method = document.getElementById('select-outlier-method').value;
  const outlier_threshold = parseFloat(document.getElementById('select-outlier-threshold').value);
  const treatment = document.getElementById('select-clean-treatment').value;

  const btn = document.getElementById('btn-run-cleaning');
  btn.innerText = 'Cleaning Series...';
  btn.disabled = true;

  try {
    // 1. Fetch series cleaning results
    const cleanRes = await fetch('/api/data/clean', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        commodity, state, mandi, frequency,
        outlier_method, outlier_threshold, treatment,
        impute_method: 'spline'
      })
    });
    const cleanData = await cleanRes.json();

    // 2. Fetch dataset hygiene health
    let healthData = { health_score: 98.5, missing_values: 0 };
    try {
      const hRes = await fetch('/api/data/health');
      if (hRes.ok) healthData = await hRes.json();
    } catch (e) {
      console.warn('Health audit fallback:', e);
    }

    // Update Scorecards
    document.getElementById('clean-health-score').innerText = healthData.health_score;
    document.getElementById('clean-missing-count').innerText = healthData.missing_values;
    document.getElementById('clean-outlier-count').innerText = cleanData.outlier_info.outlier_count;
    document.getElementById('clean-outlier-pct').innerText = `(${cleanData.outlier_info.outlier_pct}%)`;
    document.getElementById('clean-var-reduction').innerText = `${cleanData.stats_comparison.variance_reduction_pct}%`;

    // Update Statistical Comparison Table
    document.getElementById('stat-raw-mean').innerText = `₹${cleanData.stats_comparison.raw_mean.toFixed(2)}`;
    document.getElementById('stat-clean-mean').innerText = `₹${cleanData.stats_comparison.cleaned_mean.toFixed(2)}`;
    document.getElementById('stat-raw-std').innerText = `₹${cleanData.stats_comparison.raw_std.toFixed(2)}`;
    document.getElementById('stat-clean-std').innerText = `₹${cleanData.stats_comparison.cleaned_std.toFixed(2)}`;
    document.getElementById('stat-raw-skew').innerText = cleanData.stats_comparison.raw_skewness.toFixed(2);
    document.getElementById('stat-clean-skew').innerText = cleanData.stats_comparison.cleaned_skewness.toFixed(2);

    document.getElementById('clean-chart-subtitle').innerText = 
      `${cleanData.outlier_info.method} -> Applied ${cleanData.treatment}. ${cleanData.outlier_info.outlier_count} anomalies treated.`;

    // Render Before vs After Comparison Chart
    renderCleaningComparisonChart(cleanData);

  } catch (err) {
    console.error('Data cleaning error:', err);
  } finally {
    btn.innerText = 'Apply Data Cleaning';
    btn.disabled = false;
  }
}

function renderCleaningComparisonChart(data) {
  destroyChart('chart-cleaning-comparison');
  const ctx = document.getElementById('chart-cleaning-comparison').getContext('2d');

  const dates = data.dates;
  const rawPrices = data.raw_series;
  const cleanedPrices = data.cleaned_series;
  const upperLimit = data.outlier_info.upper_limit;
  const lowerLimit = data.outlier_info.lower_limit;

  // Scatter dataset for outlier points
  const outlierScatter = new Array(dates.length).fill(null);
  data.outlier_info.outlier_indices.forEach(idx => {
    outlierScatter[idx] = rawPrices[idx];
  });

  activeCharts['chart-cleaning-comparison'] = new Chart(ctx, {
    type: 'line',
    data: {
      labels: dates,
      datasets: [
        {
          label: 'Raw Series (Uncleaned)',
          data: rawPrices,
          borderColor: 'rgba(148, 163, 184, 0.45)',
          borderWidth: 1.5,
          pointRadius: 0
        },
        {
          label: 'Cleaned & Smoothed Series',
          data: cleanedPrices,
          borderColor: '#10b981',
          borderWidth: 2.2,
          pointRadius: 0,
          tension: 0.1
        },
        {
          label: 'Detected Anomaly Spike',
          data: outlierScatter,
          borderColor: '#ef4444',
          backgroundColor: '#ef4444',
          type: 'scatter',
          pointRadius: 5,
          pointHoverRadius: 7
        },
        {
          label: 'Upper Threshold Limit',
          data: dates.map(() => upperLimit),
          borderColor: 'rgba(245, 158, 11, 0.6)',
          borderWidth: 1,
          borderDash: [4, 4],
          pointRadius: 0
        },
        {
          label: 'Lower Threshold Limit',
          data: dates.map(() => lowerLimit),
          borderColor: 'rgba(245, 158, 11, 0.6)',
          borderWidth: 1,
          borderDash: [4, 4],
          pointRadius: 0
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { grid: { display: false } },
        y: { title: { display: true, text: 'Modal Price (₹/Quintal)' } }
      }
    }
  });
}

document.getElementById('btn-run-cleaning').addEventListener('click', loadDataCleaning);

// ---------------- TAB 2: Stationarity & Unit Roots ----------------

async function loadStationarity() {
  const commodity = document.getElementById('select-commodity').value;
  const state = document.getElementById('select-state').value;
  const mandi = document.getElementById('select-mandi').value;
  const frequency = document.getElementById('select-frequency').value;
  const diff_order = parseInt(document.getElementById('select-diff-order').value);
  const seasonal_diff_lag = parseInt(document.getElementById('select-seasonal-diff').value);

  try {
    // 1. Fetch ADF & KPSS
    const res = await fetch('/api/analysis/stationarity', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ commodity, state, mandi, frequency, diff_order, seasonal_diff_lag })
    });
    const statData = await res.json();
    
    // Update ADF
    const adf = statData.tests.adf;
    document.getElementById('adf-test-stat').innerText = adf.test_statistic;
    document.getElementById('adf-p-value').innerText = adf.p_value;
    document.getElementById('adf-crit-1').innerText = adf.critical_values['1%'];
    document.getElementById('adf-crit-5').innerText = adf.critical_values['5%'];
    document.getElementById('adf-crit-10').innerText = adf.critical_values['10%'];
    document.getElementById('adf-conclusion').innerText = adf.conclusion;
    
    const adfBadge = document.getElementById('badge-adf-status');
    adfBadge.innerText = adf.is_stationary ? 'Stationary' : 'Non-Stationary';
    adfBadge.className = adf.is_stationary 
      ? 'px-2.5 py-1 text-xs font-semibold rounded-full bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
      : 'px-2.5 py-1 text-xs font-semibold rounded-full bg-rose-500/20 text-rose-400 border border-rose-500/30';

    // Update KPSS
    const kpss = statData.tests.kpss;
    document.getElementById('kpss-test-stat').innerText = kpss.test_statistic;
    document.getElementById('kpss-p-value').innerText = kpss.p_value;
    document.getElementById('kpss-crit-1').innerText = kpss.critical_values['1%'];
    document.getElementById('kpss-crit-5').innerText = kpss.critical_values['5%'];
    document.getElementById('kpss-crit-10').innerText = kpss.critical_values['10%'];
    document.getElementById('kpss-conclusion').innerText = kpss.conclusion;
    
    const kpssBadge = document.getElementById('badge-kpss-status');
    kpssBadge.innerText = kpss.is_stationary ? 'Stationary' : 'Non-Stationary';
    kpssBadge.className = kpss.is_stationary
      ? 'px-2.5 py-1 text-xs font-semibold rounded-full bg-cyan-500/20 text-cyan-400 border border-cyan-500/30'
      : 'px-2.5 py-1 text-xs font-semibold rounded-full bg-rose-500/20 text-rose-400 border border-rose-500/30';

    // Update Joint Badge in Overview
    const kpiBadge = document.getElementById('kpi-stationarity-badge');
    kpiBadge.innerText = statData.tests.joint_verdict;
    document.getElementById('kpi-stationarity-sub').innerText = statData.tests.recommendation;

    // Render Rolling Stats Chart
    destroyChart('chart-rolling-stats');
    const ctxRolling = document.getElementById('chart-rolling-stats').getContext('2d');
    activeCharts['chart-rolling-stats'] = new Chart(ctxRolling, {
      type: 'line',
      data: {
        labels: statData.rolling.dates,
        datasets: [
          {
            label: 'Series Value',
            data: statData.rolling.original,
            borderColor: 'rgba(148, 163, 184, 0.5)',
            borderWidth: 1.2,
            pointRadius: 0
          },
          {
            label: 'Rolling Mean (Window 12)',
            data: statData.rolling.rolling_mean,
            borderColor: '#10b981',
            borderWidth: 2,
            pointRadius: 0
          },
          {
            label: 'Rolling Std Dev',
            data: statData.rolling.rolling_std,
            borderColor: '#f59e0b',
            borderWidth: 1.8,
            pointRadius: 0,
            borderDash: [4, 4]
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: { x: { display: false } }
      }
    });

    // 2. Fetch ACF & PACF
    const acfRes = await fetch('/api/analysis/acf_pacf', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ commodity, state, mandi, frequency, diff_order, nlags: 30 })
    });
    const acfData = await acfRes.json();
    renderAcfPacf(acfData);

  } catch (err) {
    console.error('Failed stationarity check:', err);
  }
}

function renderAcfPacf(data) {
  // Render ACF
  destroyChart('chart-acf');
  const ctxAcf = document.getElementById('chart-acf').getContext('2d');
  const ci = data.ci_bound;
  
  activeCharts['chart-acf'] = new Chart(ctxAcf, {
    type: 'bar',
    data: {
      labels: data.lags,
      datasets: [
        {
          label: 'Autocorrelation (ACF)',
          data: data.acf,
          backgroundColor: '#10b981',
          barPercentage: 0.3
        },
        {
          label: 'Upper 95% Bound',
          data: data.lags.map(() => ci),
          type: 'line',
          borderColor: '#f43f5e',
          borderWidth: 1,
          borderDash: [3, 3],
          pointRadius: 0
        },
        {
          label: 'Lower 95% Bound',
          data: data.lags.map(() => -ci),
          type: 'line',
          borderColor: '#f43f5e',
          borderWidth: 1,
          borderDash: [3, 3],
          pointRadius: 0
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { title: { display: true, text: 'Lag' } },
        y: { min: -1.0, max: 1.0, title: { display: true, text: 'ACF Correlation' } }
      }
    }
  });

  // Render PACF
  destroyChart('chart-pacf');
  const ctxPacf = document.getElementById('chart-pacf').getContext('2d');
  activeCharts['chart-pacf'] = new Chart(ctxPacf, {
    type: 'bar',
    data: {
      labels: data.lags,
      datasets: [
        {
          label: 'Partial Autocorrelation (PACF)',
          data: data.pacf,
          backgroundColor: '#06b6d4',
          barPercentage: 0.3
        },
        {
          label: 'Upper 95% Bound',
          data: data.lags.map(() => ci),
          type: 'line',
          borderColor: '#f43f5e',
          borderWidth: 1,
          borderDash: [3, 3],
          pointRadius: 0
        },
        {
          label: 'Lower 95% Bound',
          data: data.lags.map(() => -ci),
          type: 'line',
          borderColor: '#f43f5e',
          borderWidth: 1,
          borderDash: [3, 3],
          pointRadius: 0
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { title: { display: true, text: 'Lag' } },
        y: { min: -1.0, max: 1.0, title: { display: true, text: 'PACF Correlation' } }
      }
    }
  });
}

document.getElementById('btn-run-stationarity').addEventListener('click', loadStationarity);

// ---------------- TAB 3: Decomposition ----------------

async function loadDecomposition() {
  const commodity = document.getElementById('select-commodity').value;
  const state = document.getElementById('select-state').value;
  const mandi = document.getElementById('select-mandi').value;
  const frequency = document.getElementById('select-frequency').value;
  const method = document.getElementById('select-decomp-method').value;
  const period = frequency === 'W' ? 52 : (frequency === 'M' ? 12 : 365);

  try {
    const res = await fetch('/api/analysis/decomposition', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ commodity, state, mandi, frequency, method, period })
    });
    const decomp = await res.json();

    document.getElementById('decomp-trend-strength').innerText = decomp.metrics.trend_strength;
    document.getElementById('decomp-season-strength').innerText = decomp.metrics.seasonality_strength;
    document.getElementById('decomp-resid-var').innerText = decomp.metrics.residual_variance.toLocaleString();

    // Render 4 stacked charts
    const createSubChart = (chartId, label, data, color) => {
      destroyChart(chartId);
      const ctx = document.getElementById(chartId).getContext('2d');
      activeCharts[chartId] = new Chart(ctx, {
        type: 'line',
        data: {
          labels: decomp.dates,
          datasets: [{
            label: label,
            data: data,
            borderColor: color,
            borderWidth: 1.8,
            pointRadius: 0,
            tension: 0.1
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: { legend: { display: false } },
          scales: { x: { display: false } }
        }
      });
    };

    createSubChart('chart-decomp-observed', 'Observed', decomp.observed, '#94a3b8');
    createSubChart('chart-decomp-trend', 'Trend', decomp.trend, '#10b981');
    createSubChart('chart-decomp-seasonal', 'Seasonal', decomp.seasonal, '#06b6d4');
    createSubChart('chart-decomp-residual', 'Residual', decomp.residual, '#f59e0b');

  } catch (err) {
    console.error('Failed decomposition:', err);
  }
}

document.getElementById('btn-run-decomp').addEventListener('click', loadDecomposition);

// ---------------- TAB 4: Models & Forecasting ----------------

// Toggle input fields based on model selected
document.getElementById('select-model-type').addEventListener('change', (e) => {
  const val = e.target.value;
  const pBox = document.getElementById('param-p-box');
  const dBox = document.getElementById('param-d-box');
  const qBox = document.getElementById('param-q-box');
  const sBox = document.getElementById('param-seasonal-box');

  if (val === 'sarima') {
    pBox.classList.remove('hidden');
    dBox.classList.remove('hidden');
    qBox.classList.remove('hidden');
    sBox.classList.remove('hidden');
  } else if (val === 'arima') {
    pBox.classList.remove('hidden');
    dBox.classList.remove('hidden');
    qBox.classList.remove('hidden');
    sBox.classList.add('hidden');
  } else {
    pBox.classList.add('hidden');
    dBox.classList.add('hidden');
    qBox.classList.add('hidden');
    sBox.classList.add('hidden');
  }
});

async function trainCurrentModel() {
  const commodity = document.getElementById('select-commodity').value;
  const state = document.getElementById('select-state').value;
  const mandi = document.getElementById('select-mandi').value;
  const frequency = document.getElementById('select-frequency').value;
  const model_type = document.getElementById('select-model-type').value;
  const horizon = parseInt(document.getElementById('input-horizon').value);
  
  const params = {};
  if (model_type === 'arima' || model_type === 'sarima') {
    params.p = parseInt(document.getElementById('input-p').value);
    params.d = parseInt(document.getElementById('input-d').value);
    params.q = parseInt(document.getElementById('input-q').value);
  }
  if (model_type === 'sarima') {
    const pdqsStr = document.getElementById('input-pdqs').value;
    const parts = pdqsStr.split(',').map(s => parseInt(s.trim()));
    params.P = parts[0] || 1;
    params.D = parts[1] || 1;
    params.Q = parts[2] || 1;
    params.s = parts[3] || 12;
  }

  const btn = document.getElementById('btn-train-forecast');
  btn.innerText = 'Fitting Model...';
  btn.disabled = true;

  try {
    const res = await fetch('/api/models/train', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        commodity, state, mandi, frequency,
        model_type,
        forecast_horizon: horizon,
        test_periods: 20,
        params
      })
    });
    const modelRes = await res.json();

    // Update Accuracy Cards
    const m = modelRes.metrics;
    document.getElementById('eval-mae').innerText = `₹${m.mae.toFixed(2)}`;
    document.getElementById('eval-rmse').innerText = `₹${m.rmse.toFixed(2)}`;
    document.getElementById('eval-mape').innerText = `${m.mape.toFixed(2)}%`;
    document.getElementById('eval-accuracy').innerText = `${m.accuracy_score_pct.toFixed(2)}%`;
    document.getElementById('eval-r2').innerText = m.r2.toFixed(3);
    document.getElementById('eval-aic').innerText = modelRes.aic ? `${modelRes.aic} / ${modelRes.bic}` : 'N/A';
    document.getElementById('label-forecast-model-name').innerText = `${modelRes.model_name} Test Validation & Horizon Forecast`;

    // Render Forecast Chart
    renderForecastChart(modelRes);

    // Update Residual Diagnostics Tab
    updateResidualDiagnostics(modelRes);

  } catch (err) {
    console.error('Model training failed:', err);
  } finally {
    btn.innerText = 'Train Model & Predict Future';
    btn.disabled = false;
  }
}

function renderForecastChart(data) {
  destroyChart('chart-forecast');
  const ctx = document.getElementById('chart-forecast').getContext('2d');

  // Build combined chronological axis
  const trainDates = data.train_dates;
  const testDates = data.test_dates;
  const futureDates = data.future_dates;
  const allDates = [...trainDates, ...testDates, ...futureDates];

  // Downsample train points for smoother chart display if needed
  const trainActuals = new Array(allDates.length).fill(null);
  trainDates.forEach((_, i) => {
    trainActuals[i] = currentCropData.modal_prices[i];
  });

  const testActuals = new Array(allDates.length).fill(null);
  const testPreds = new Array(allDates.length).fill(null);
  testDates.forEach((_, i) => {
    const idx = trainDates.length + i;
    testActuals[idx] = data.test_actual[i];
    testPreds[idx] = data.test_predicted[i];
  });

  const futurePreds = new Array(allDates.length).fill(null);
  const confLower = new Array(allDates.length).fill(null);
  const confUpper = new Array(allDates.length).fill(null);

  // Connect last test observation to first future forecast
  const lastTestIdx = trainDates.length + testDates.length - 1;
  const lastVal = data.test_actual[data.test_actual.length - 1];

  futurePreds[lastTestIdx] = lastVal;
  confLower[lastTestIdx] = lastVal;
  confUpper[lastTestIdx] = lastVal;

  futureDates.forEach((_, i) => {
    const idx = trainDates.length + testDates.length + i;
    futurePreds[idx] = data.future_forecast[i];
    confLower[idx] = data.future_conf_lower[i];
    confUpper[idx] = data.future_conf_upper[i];
  });

  activeCharts['chart-forecast'] = new Chart(ctx, {
    type: 'line',
    data: {
      labels: allDates,
      datasets: [
        {
          label: 'Historical Training Data',
          data: trainActuals,
          borderColor: '#64748b',
          borderWidth: 1.5,
          pointRadius: 0
        },
        {
          label: 'Test Actuals',
          data: testActuals,
          borderColor: '#38bdf8',
          borderWidth: 2.2,
          pointRadius: 3,
          pointBackgroundColor: '#38bdf8'
        },
        {
          label: 'Test Model Prediction',
          data: testPreds,
          borderColor: '#10b981',
          borderWidth: 2,
          borderDash: [4, 4],
          pointRadius: 2,
          pointBackgroundColor: '#10b981'
        },
        {
          label: 'Future Forecast',
          data: futurePreds,
          borderColor: '#fbbf24',
          borderWidth: 2.5,
          pointRadius: 3,
          pointBackgroundColor: '#fbbf24'
        },
        {
          label: '95% Confidence Upper',
          data: confUpper,
          borderColor: 'transparent',
          backgroundColor: 'rgba(251, 191, 36, 0.12)',
          fill: '+1',
          pointRadius: 0
        },
        {
          label: '95% Confidence Lower',
          data: confLower,
          borderColor: 'transparent',
          fill: false,
          pointRadius: 0
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { grid: { display: false } },
        y: { title: { display: true, text: 'Modal Price (₹/Quintal)' } }
      }
    }
  });
}

// Auto-Tune Handler
document.getElementById('btn-auto-tune').addEventListener('click', async () => {
  const commodity = document.getElementById('select-commodity').value;
  const state = document.getElementById('select-state').value;
  const mandi = document.getElementById('select-mandi').value;
  const frequency = document.getElementById('select-frequency').value;

  const btn = document.getElementById('btn-auto-tune');
  const originalHtml = btn.innerHTML;
  btn.innerText = 'Scanning Orders...';
  btn.disabled = true;

  try {
    const res = await fetch('/api/models/auto_tune', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        commodity, state, mandi, frequency,
        max_p: 2, max_d: 1, max_q: 2,
        is_seasonal: true, seasonal_period: 12
      })
    });
    const result = await res.json();
    
    // Set auto-tuned hyperparameters
    const [p, d, q] = result.best_order;
    const [P, D, Q, s] = result.best_seasonal_order;

    document.getElementById('input-p').value = p;
    document.getElementById('input-d').value = d;
    document.getElementById('input-q').value = q;
    document.getElementById('input-pdqs').value = `${P}, ${D}, ${Q}, ${s}`;

    alert(`Auto-Tune Selected Optimal Model: SARIMA(${p},${d},${q})x(${P},${D},${Q})_${s} with AIC=${result.best_score}`);
    await trainCurrentModel();
  } catch (err) {
    console.error('Auto-tune failed:', err);
  } finally {
    btn.innerHTML = originalHtml;
    btn.disabled = false;
  }
});

// CSV Export Handler
document.getElementById('btn-export-csv').addEventListener('click', async () => {
  const commodity = document.getElementById('select-commodity').value;
  const state = document.getElementById('select-state').value;
  const mandi = document.getElementById('select-mandi').value;
  const frequency = document.getElementById('select-frequency').value;
  const model_type = document.getElementById('select-model-type').value;
  const horizon = parseInt(document.getElementById('input-horizon').value);

  const res = await fetch('/api/export/forecast_csv', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      commodity, state, mandi, frequency, model_type,
      forecast_horizon: horizon, test_periods: 20
    })
  });
  const blob = await res.blob();
  const url = window.URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `${commodity}_${model_type}_forecast.csv`;
  document.body.appendChild(a);
  a.click();
  a.remove();
});

document.getElementById('btn-train-forecast').addEventListener('click', trainCurrentModel);

// ---------------- TAB 5: Leaderboard & Benchmark ----------------

document.getElementById('btn-run-benchmark').addEventListener('click', async () => {
  const commodity = document.getElementById('select-commodity').value;
  const state = document.getElementById('select-state').value;
  const mandi = document.getElementById('select-mandi').value;
  const frequency = document.getElementById('select-frequency').value;

  const btn = document.getElementById('btn-run-benchmark');
  btn.innerText = 'Benchmarking All Models...';
  btn.disabled = true;

  try {
    const res = await fetch('/api/models/benchmark', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ commodity, state, mandi, frequency, test_periods: 20, seasonal_period: 12 })
    });
    const bench = await res.json();

    const tbody = document.getElementById('leaderboard-body');
    tbody.innerHTML = '';

    const modelNames = [];
    const rmseValues = [];
    const maeValues = [];

    bench.leaderboard.forEach(row => {
      modelNames.push(row.model_name);
      rmseValues.push(row.rmse);
      maeValues.push(row.mae);

      const rankBadge = row.rank === 1 
        ? '<span class="px-2 py-0.5 rounded-full bg-emerald-500/20 text-emerald-400 font-bold border border-emerald-500/30">#1 Top</span>'
        : `#${row.rank}`;

      tbody.innerHTML += `
        <tr class="hover:bg-slate-800/40 transition">
          <td class="p-3.5 mono">${rankBadge}</td>
          <td class="p-3.5 font-bold text-white">${row.model_name}</td>
          <td class="p-3.5 mono text-emerald-400 font-semibold">₹${row.rmse.toFixed(2)}</td>
          <td class="p-3.5 mono">₹${row.mae.toFixed(2)}</td>
          <td class="p-3.5 mono">${row.mape.toFixed(2)}%</td>
          <td class="p-3.5 mono text-brand-400 font-bold">${row.accuracy_pct.toFixed(2)}%</td>
          <td class="p-3.5 mono text-cyan-400">${row.r2.toFixed(3)}</td>
          <td class="p-3.5 mono text-slate-400">${row.aic ? row.aic.toFixed(0) : 'N/A'}</td>
          <td class="p-3.5">
            <span class="px-2 py-0.5 text-xs rounded ${row.is_white_noise ? 'bg-emerald-500/10 text-emerald-400' : 'bg-slate-700 text-slate-300'}">
              ${row.is_white_noise ? 'White Noise' : 'Colored'}
            </span>
          </td>
        </tr>
      `;
    });

    // Render Comparative Bar Chart
    destroyChart('chart-benchmark-bar');
    const ctxBar = document.getElementById('chart-benchmark-bar').getContext('2d');
    activeCharts['chart-benchmark-bar'] = new Chart(ctxBar, {
      type: 'bar',
      data: {
        labels: modelNames,
        datasets: [
          {
            label: 'RMSE (₹/Quintal)',
            data: rmseValues,
            backgroundColor: '#10b981'
          },
          {
            label: 'MAE (₹/Quintal)',
            data: maeValues,
            backgroundColor: '#06b6d4'
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          y: { title: { display: true, text: 'Prediction Error (₹)' } }
        }
      }
    });

  } catch (err) {
    console.error('Benchmark failed:', err);
  } finally {
    btn.innerText = 'Run Full Model Benchmark';
    btn.disabled = false;
  }
});

// ---------------- TAB 6: Residual Diagnostics ----------------

function updateResidualDiagnostics(modelRes) {
  const diag = modelRes.diagnostics;
  if (!diag || diag.status === 'insufficient_data') return;

  // Ljung-Box
  document.getElementById('diag-lb-stat').innerText = diag.ljung_box.statistic;
  document.getElementById('diag-lb-p').innerText = diag.ljung_box.p_value;
  document.getElementById('diag-lb-interp').innerText = diag.ljung_box.interpretation;
  
  const lbBadge = document.getElementById('badge-diag-lb');
  lbBadge.innerText = diag.ljung_box.is_white_noise ? 'White Noise' : 'Autocorrelated';
  lbBadge.className = diag.ljung_box.is_white_noise
    ? 'px-2.5 py-0.5 text-xs font-semibold rounded-full bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
    : 'px-2.5 py-0.5 text-xs font-semibold rounded-full bg-rose-500/20 text-rose-400 border border-rose-500/30';

  // Jarque-Bera
  document.getElementById('diag-jb-stat').innerText = diag.jarque_bera.statistic;
  document.getElementById('diag-jb-p').innerText = diag.jarque_bera.p_value;
  document.getElementById('diag-jb-interp').innerText = diag.jarque_bera.interpretation;

  const jbBadge = document.getElementById('badge-diag-jb');
  jbBadge.innerText = diag.jarque_bera.is_normal ? 'Normal' : 'Non-Normal';
  jbBadge.className = diag.jarque_bera.is_normal
    ? 'px-2.5 py-0.5 text-xs font-semibold rounded-full bg-cyan-500/20 text-cyan-400 border border-cyan-500/30'
    : 'px-2.5 py-0.5 text-xs font-semibold rounded-full bg-amber-500/20 text-amber-400 border border-amber-500/30';

  // Residual sequence plot
  destroyChart('chart-residuals-time');
  const ctxRes = document.getElementById('chart-residuals-time').getContext('2d');
  activeCharts['chart-residuals-time'] = new Chart(ctxRes, {
    type: 'line',
    data: {
      labels: modelRes.residuals.map((_, i) => `t-${i}`),
      datasets: [{
        label: 'Model Residuals (et)',
        data: modelRes.residuals,
        borderColor: '#f59e0b',
        borderWidth: 1.5,
        pointRadius: 2,
        pointBackgroundColor: '#f59e0b'
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: { x: { display: false } }
    }
  });

  // Q-Q Plot
  destroyChart('chart-qq');
  const ctxQq = document.getElementById('chart-qq').getContext('2d');
  const qqData = diag.qq_plot;
  
  activeCharts['chart-qq'] = new Chart(ctxQq, {
    type: 'scatter',
    data: {
      datasets: [
        {
          label: 'Residual Quantiles vs Normal',
          data: qqData.theoretical_quantiles.map((th, i) => ({ x: th, y: qqData.sample_quantiles[i] })),
          backgroundColor: '#38bdf8'
        },
        {
          label: 'Theoretical 45° Reference',
          data: [-3, 3].map(v => ({ x: v, y: v })),
          type: 'line',
          borderColor: '#f43f5e',
          borderWidth: 1.5,
          pointRadius: 0
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { title: { display: true, text: 'Theoretical Normal Quantiles' } },
        y: { title: { display: true, text: 'Sample Standardized Quantiles' } }
      }
    }
  });
}

// ---------------- Authentication & Gmail Sign-In ----------------

function openLoginModal() {
  const modal = document.getElementById('modal-login');
  if (modal) {
    modal.classList.remove('hidden');
    document.getElementById('input-login-email').focus();
  }
}

function closeLoginModal() {
  const modal = document.getElementById('modal-login');
  if (modal) modal.classList.add('hidden');
  const alertEl = document.getElementById('login-alert');
  if (alertEl) alertEl.classList.add('hidden');
}

function renderAuthUI(user) {
  currentUser = user;
  const unloggedEl = document.getElementById('auth-unlogged');
  const loggedEl = document.getElementById('auth-logged');

  if (user) {
    unloggedEl.classList.add('hidden');
    loggedEl.classList.remove('hidden');
    document.getElementById('user-name').innerText = user.name;
    document.getElementById('user-email').innerText = user.email;
    document.getElementById('user-avatar').src = user.avatar;
  } else {
    unloggedEl.classList.remove('hidden');
    loggedEl.classList.add('hidden');
  }
}

async function initAuth() {
  // Check local storage for persistent session
  const storedUser = localStorage.getItem('krishidristi_user');
  if (storedUser) {
    try {
      const user = JSON.parse(storedUser);
      // Validate session with server
      const res = await fetch(`/api/auth/session?token=${user.token}`);
      const data = await res.json();
      if (data.authenticated) {
        renderAuthUI(data.user);
        return;
      }
    } catch (e) {
      console.warn('Session verification fallback to cached user:', e);
    }
  }

  // If no session, show unauthenticated UI
  renderAuthUI(null);
}

async function handleGmailLogin(email, name = null) {
  const alertEl = document.getElementById('login-alert');
  alertEl.classList.add('hidden');

  try {
    const res = await fetch('/api/auth/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email: email.trim(), name: name ? name.trim() : null })
    });
    const data = await res.json();

    if (!res.ok) {
      alertEl.innerText = data.detail || 'Login failed. Please check your Gmail address.';
      alertEl.className = 'p-2.5 rounded-lg text-xs border bg-rose-500/10 text-rose-400 border-rose-500/30';
      alertEl.classList.remove('hidden');
      return;
    }

    // Success: store session
    localStorage.setItem('krishidristi_user', JSON.stringify(data.user));
    renderAuthUI(data.user);
    closeLoginModal();

  } catch (err) {
    console.error('Login error:', err);
    alertEl.innerText = 'Network error during Google authentication. Please try again.';
    alertEl.className = 'p-2.5 rounded-lg text-xs border bg-rose-500/10 text-rose-400 border-rose-500/30';
    alertEl.classList.remove('hidden');
  }
}

async function handleLogout() {
  if (currentUser && currentUser.token) {
    try {
      await fetch(`/api/auth/logout?token=${currentUser.token}`, { method: 'POST' });
    } catch (e) {
      console.warn('Logout network sync:', e);
    }
  }
  localStorage.removeItem('krishidristi_user');
  renderAuthUI(null);
}

// Wire up Authentication event listeners
document.getElementById('btn-open-login').addEventListener('click', openLoginModal);
document.getElementById('btn-close-modal').addEventListener('click', closeLoginModal);
document.getElementById('btn-guest-continue').addEventListener('click', closeLoginModal);
document.getElementById('btn-logout').addEventListener('click', handleLogout);

// One-Click Google Sign-In
document.getElementById('btn-quick-google').addEventListener('click', () => {
  handleGmailLogin('krishi.analyst@gmail.com', 'Krishi Analyst');
});

// Demo Account Pills
document.querySelectorAll('.demo-account-pill').forEach(pill => {
  pill.addEventListener('click', (e) => {
    const email = e.target.getAttribute('data-email');
    const name = e.target.getAttribute('data-name');
    document.getElementById('input-login-email').value = email;
    document.getElementById('input-login-name').value = name;
    document.getElementById('icon-valid-gmail').classList.remove('hidden');
  });
});

// Email input dynamic format validation
document.getElementById('input-login-email').addEventListener('input', (e) => {
  const val = e.target.value.toLowerCase().trim();
  const icon = document.getElementById('icon-valid-gmail');
  const hint = document.getElementById('msg-email-hint');

  if (val.endsWith('@gmail.com') || val.endsWith('@googlemail.com')) {
    icon.classList.remove('hidden');
    hint.innerText = '✓ Valid Gmail address format recognized';
    hint.className = 'text-[11px] text-emerald-400 mt-1';
  } else if (val.includes('@') && val.includes('.')) {
    icon.classList.remove('hidden');
    hint.innerText = 'Google Workspace address format recognized';
    hint.className = 'text-[11px] text-cyan-400 mt-1';
  } else {
    icon.classList.add('hidden');
    hint.innerText = 'Must be a valid @gmail.com or Google account.';
    hint.className = 'text-[11px] text-slate-400 mt-1';
  }
});

// Custom Form Submission
document.getElementById('form-gmail-login').addEventListener('submit', (e) => {
  e.preventDefault();
  const email = document.getElementById('input-login-email').value;
  const name = document.getElementById('input-login-name').value;
  handleGmailLogin(email, name);
});

// Close modal on escape key
window.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') closeLoginModal();
});

// Start application
window.addEventListener('DOMContentLoaded', () => {
  initAuth();
  initApp();
});
