"""Patches the index.html to add Dataset Files tab and enhance Data Cleaning tab."""
import re, os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
HTML_PATH = os.path.join(ROOT, "frontend", "templates", "index.html")

with open(HTML_PATH, "r", encoding="utf-8") as f:
    h = f.read()

# ------------------------------------------------------------------
# 1. Add "Dataset Files" tab button (insert after the Overview button line)
# ------------------------------------------------------------------
OLD_NAV = ('<button class="tab-btn px-4 py-2 rounded-lg text-slate-400 hover:text-white transition" '
           'data-target="tab-cleaning">Data Cleaning &amp; Quality</button>')
NEW_NAV = ('<button class="tab-btn px-4 py-2 rounded-lg text-slate-400 hover:text-white transition" '
           'data-target="tab-datasets">📂 Dataset Files</button>\n'
           '    <button class="tab-btn px-4 py-2 rounded-lg text-slate-400 hover:text-white transition" '
           'data-target="tab-cleaning">🧹 Data Cleaning &amp; Quality</button>')
if OLD_NAV in h:
    h = h.replace(OLD_NAV, NEW_NAV)
    print("✓ Tab nav patched")
else:
    print("⚠ Tab nav NOT found — check HTML encoding")

# ------------------------------------------------------------------
# 2. Insert Dataset Files tab content BEFORE the cleaning tab div
# ------------------------------------------------------------------
DATASET_TAB_HTML = '''
    <!-- TAB: DATASET FILES -->
    <div id="tab-datasets" class="tab-content hidden space-y-6">
      <!-- Summary Banner -->
      <div class="glass-card p-5 flex flex-wrap gap-6 items-center justify-between">
        <div>
          <h2 class="text-base font-bold text-white">Per-Crop Dataset Files &amp; Data Quality</h2>
          <p class="text-xs text-slate-400 mt-1">Individual raw and cleaned CSV exports for all 8 agricultural commodities. Each file contains historical mandi prices, arrivals, and weather indicators from 2018–2024.</p>
        </div>
        <div class="flex gap-4">
          <div class="text-center">
            <div class="text-2xl font-extrabold text-brand-400 mono" id="ds-total-records">116,176</div>
            <div class="text-xs text-slate-400">Total Records</div>
          </div>
          <div class="text-center">
            <div class="text-2xl font-extrabold text-cyan-400 mono">8</div>
            <div class="text-xs text-slate-400">Crops</div>
          </div>
          <div class="text-center">
            <div class="text-2xl font-extrabold text-amber-400 mono">16</div>
            <div class="text-xs text-slate-400">CSV Files</div>
          </div>
        </div>
      </div>

      <!-- Crop Cards Grid -->
      <div id="crop-cards-grid" class="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-4">
        <!-- Loaded dynamically by JS -->
        <div class="glass-card p-5 col-span-full text-center text-slate-400 py-12">
          <div class="text-4xl mb-3">📂</div>
          <div class="font-semibold">Loading crop dataset information...</div>
        </div>
      </div>

      <!-- Full Data Table -->
      <div class="glass-card p-6">
        <h3 class="text-sm font-bold text-white mb-4">Complete Dataset Summary Table</h3>
        <div class="overflow-x-auto rounded-xl border border-darkborder">
          <table class="w-full text-left text-sm">
            <thead class="bg-slate-900/90 text-xs uppercase text-slate-400 font-semibold border-b border-darkborder">
              <tr>
                <th class="p-3">Commodity</th>
                <th class="p-3">Total Records</th>
                <th class="p-3">Date Range</th>
                <th class="p-3">Avg Modal Price</th>
                <th class="p-3">Min Price</th>
                <th class="p-3">Max Price</th>
                <th class="p-3">Health Score</th>
                <th class="p-3">Outliers</th>
                <th class="p-3 text-center">Downloads</th>
              </tr>
            </thead>
            <tbody id="crop-table-body" class="divide-y divide-darkborder font-medium text-slate-200">
              <tr><td colspan="9" class="p-6 text-center text-slate-500">Loading...</td></tr>
            </tbody>
          </table>
        </div>
      </div>

      <!-- Data Schema Info -->
      <div class="glass-card p-6">
        <h3 class="text-sm font-bold text-white mb-3">Dataset Schema — Column Reference</h3>
        <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div class="overflow-x-auto rounded-xl border border-darkborder">
            <table class="w-full text-left text-xs">
              <thead class="bg-slate-900/90 text-slate-400 uppercase font-semibold border-b border-darkborder">
                <tr><th class="p-3">Column</th><th class="p-3">Type</th><th class="p-3">Description</th></tr>
              </thead>
              <tbody class="divide-y divide-darkborder text-slate-300">
                <tr><td class="p-3 mono text-brand-400">Date</td><td class="p-3">datetime</td><td class="p-3">Trading date (YYYY-MM-DD)</td></tr>
                <tr><td class="p-3 mono text-brand-400">Commodity</td><td class="p-3">string</td><td class="p-3">Crop name (e.g. Wheat, Onion)</td></tr>
                <tr><td class="p-3 mono text-brand-400">State</td><td class="p-3">string</td><td class="p-3">Indian state of mandi</td></tr>
                <tr><td class="p-3 mono text-brand-400">Mandi</td><td class="p-3">string</td><td class="p-3">Market / APMC mandi name</td></tr>
                <tr><td class="p-3 mono text-brand-400">Modal_Price</td><td class="p-3">float (₹)</td><td class="p-3">Most frequent transaction price</td></tr>
                <tr><td class="p-3 mono text-brand-400">Min_Price</td><td class="p-3">float (₹)</td><td class="p-3">Minimum daily quote</td></tr>
                <tr><td class="p-3 mono text-brand-400">Max_Price</td><td class="p-3">float (₹)</td><td class="p-3">Maximum daily quote</td></tr>
              </tbody>
            </table>
          </div>
          <div class="overflow-x-auto rounded-xl border border-darkborder">
            <table class="w-full text-left text-xs">
              <thead class="bg-slate-900/90 text-slate-400 uppercase font-semibold border-b border-darkborder">
                <tr><th class="p-3">Column</th><th class="p-3">Type</th><th class="p-3">Description</th></tr>
              </thead>
              <tbody class="divide-y divide-darkborder text-slate-300">
                <tr><td class="p-3 mono text-cyan-400">Arrivals_Tons</td><td class="p-3">float</td><td class="p-3">Market arrival quantity (metric tons)</td></tr>
                <tr><td class="p-3 mono text-cyan-400">Rainfall_Anomaly_Pct</td><td class="p-3">float</td><td class="p-3">% deviation from seasonal rainfall norm</td></tr>
                <tr><td class="p-3 mono text-cyan-400">Transport_Fuel_Index</td><td class="p-3">float</td><td class="p-3">Diesel/fuel price index (affects logistics cost)</td></tr>
                <tr><td class="p-3 mono text-amber-400">Modal_Price_Raw</td><td class="p-3">float (₹)</td><td class="p-3">Pre-cleaning modal price (cleaned CSV only)</td></tr>
                <tr><td class="p-3 mono text-amber-400">Modal_Price_Cleaned</td><td class="p-3">float (₹)</td><td class="p-3">Post-Winsorization price (cleaned CSV only)</td></tr>
                <tr><td class="p-3 mono text-rose-400">Is_Outlier</td><td class="p-3">bool</td><td class="p-3">IQR-flagged outlier row (cleaned CSV only)</td></tr>
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>

'''
# Insert before the tab-cleaning div
TARGET_CLEANING = '    <div id="tab-cleaning" class="tab-content hidden space-y-6">'
if TARGET_CLEANING in h:
    h = h.replace(TARGET_CLEANING, DATASET_TAB_HTML + TARGET_CLEANING)
    print("✓ Dataset Files tab content inserted")
else:
    print("⚠ Cleaning tab anchor NOT found")

# ------------------------------------------------------------------
# 3. Add Outlier Points Table and Distribution Histogram BEFORE closing of cleaning tab
# ------------------------------------------------------------------
# Find end of cleaning tab (before stationarity tab)
BEFORE_STATIONARITY = '    <!-- TAB 2: STATIONARITY'
OUTLIER_TABLE_HTML = '''
      <!-- Detected Anomaly Records Table -->
      <div class="glass-card p-6">
        <div class="flex items-center justify-between mb-4">
          <div>
            <h3 class="text-sm font-bold text-white">Detected &amp; Treated Anomaly Records</h3>
            <p class="text-xs text-slate-400 mt-0.5">Specific observations flagged as outliers, with original and post-treatment prices.</p>
          </div>
          <span id="outlier-table-count" class="px-3 py-1 text-xs font-bold bg-rose-500/20 text-rose-400 border border-rose-500/30 rounded-full">0 anomalies</span>
        </div>
        <div class="overflow-x-auto rounded-xl border border-darkborder max-h-64 overflow-y-auto">
          <table class="w-full text-left text-xs">
            <thead class="bg-slate-900/90 text-slate-400 uppercase font-semibold border-b border-darkborder sticky top-0">
              <tr>
                <th class="p-3">#</th>
                <th class="p-3">Date</th>
                <th class="p-3">Raw Price (₹)</th>
                <th class="p-3">Cleaned Price (₹)</th>
                <th class="p-3">Δ Change</th>
                <th class="p-3">Status</th>
              </tr>
            </thead>
            <tbody id="outlier-table-body" class="divide-y divide-darkborder text-slate-300">
              <tr><td colspan="6" class="p-4 text-center text-slate-500">Run cleaning to see anomaly records.</td></tr>
            </tbody>
          </table>
        </div>
      </div>

      <!-- Distribution Histogram Before vs After -->
      <div class="glass-card p-6">
        <div class="flex items-center justify-between mb-4">
          <div>
            <h3 class="text-sm font-bold text-white">Price Distribution — Before vs. After Cleaning</h3>
            <p class="text-xs text-slate-400 mt-0.5">Visualizing skewness normalization and outlier removal on price distribution shape.</p>
          </div>
          <div class="flex items-center gap-4 text-xs">
            <span class="flex items-center gap-1.5"><span class="w-3 h-3 rounded-full bg-slate-500 opacity-70"></span> Raw Distribution</span>
            <span class="flex items-center gap-1.5"><span class="w-3 h-3 rounded-full bg-emerald-400"></span> Cleaned Distribution</span>
          </div>
        </div>
        <div class="h-72">
          <canvas id="chart-distribution-histogram"></canvas>
        </div>
        <div class="mt-4 grid grid-cols-2 gap-4 text-xs">
          <div class="p-3 rounded-lg bg-slate-900/60 border border-darkborder">
            <div class="text-slate-400 mb-1 font-semibold uppercase tracking-wider">Raw Series Skewness</div>
            <div id="hist-raw-skew" class="text-lg font-bold text-rose-400 mono">—</div>
            <div class="text-slate-500 mt-0.5">Higher = more outlier distortion</div>
          </div>
          <div class="p-3 rounded-lg bg-slate-900/60 border border-darkborder">
            <div class="text-slate-400 mb-1 font-semibold uppercase tracking-wider">Cleaned Series Skewness</div>
            <div id="hist-clean-skew" class="text-lg font-bold text-emerald-400 mono">—</div>
            <div class="text-slate-500 mt-0.5">Closer to 0 = more Gaussian</div>
          </div>
        </div>
      </div>

    </div>

'''
if BEFORE_STATIONARITY in h:
    h = h.replace(BEFORE_STATIONARITY, OUTLIER_TABLE_HTML + '    ' + BEFORE_STATIONARITY)
    print("✓ Outlier table + histogram inserted into cleaning tab")
else:
    # Try alternate pattern
    print("⚠ Could not find stationarity anchor — trying alternate insertion")
    old_end = '    </div>\n\n    <!-- TAB 2'
    if old_end in h:
        pass  # already handled above pattern

# Remove the duplicate closing </div> from cleaning tab that we may have created
# The original tab-cleaning ends with </div> before tab-stationarity
OLD_CLOSE = '      </div>\n    </div>\n\n    <!-- TAB 2: STATIONARITY'
NEW_CLOSE = '      </div>\n\n    <!-- TAB 2: STATIONARITY'
h = h.replace(OLD_CLOSE, NEW_CLOSE)

with open(HTML_PATH, "w", encoding="utf-8") as f:
    f.write(h)
print("✅ HTML patching complete")
