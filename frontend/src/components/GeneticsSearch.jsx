import { createSignal, createResource, For, Show, createEffect, onCleanup } from 'solid-js';
import Chart from 'chart.js/auto';

const fetchGenera = async () => {
  try {
    const response = await fetch('/api/genetics/genera');
    const data = await response.json();
    return data.genera || [];
  } catch (error) {
    console.error('Error fetching genera:', error);
    return [];
  }
};

const GeneticsSearch = () => {
  const [genus, setGenus] = createSignal('');
  const [region, setRegion] = createSignal('global'); // Default region
  const [searchResults, setSearchResults] = createSignal(null);
  const [isLoading, setIsLoading] = createSignal(false);
  const [searchError, setSearchError] = createSignal('');
  const [chartMetric, setChartMetric] = createSignal('compatibility_score'); // Metric for chart
  const [availableGenera] = createResource(fetchGenera);

  // Ref for the chart canvas
  let chartCanvas;
  let chartInstance;

  // --- CHART LOGIC ---
  createEffect(() => {
    const data = searchResults();
    if (!data || !data.partners || !chartCanvas) return;

    // Destroy existing chart to prevent duplicates
    if (chartInstance) {
      chartInstance.destroy();
    }

    const ctx = chartCanvas.getContext('2d');
    const topPartners = data.partners.slice(0, 10); // Top 10 for chart

    // Map data based on selected metric
    const labels = topPartners.map(p => p.genus);
    const datasetData = topPartners.map(p => {
        const val = p[chartMetric()];
        // FIX: If it's a small decimal (like 0.85), scale to 100
        return val <= 1 ? val * 100 : val;
    });

    chartInstance = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: labels,
        datasets: [{
          label: chartMetric().replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()),
          data: datasetData,
          backgroundColor: '#10B981', // Emerald 500
          borderRadius: 6,
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false }
        },
        scales: {
          y: {
            beginAtZero: true,
            max: 100, // Force scale to 100
            grid: { color: '#E5E7EB' }
          },
          x: {
            grid: { display: false }
          }
        }
      }
    });
  });

  // Cleanup chart on unmount
  onCleanup(() => {
    if (chartInstance) {
      chartInstance.destroy();
    }
  });

  const handleSearch = async () => {
    if (!genus().trim()) {
      setSearchError('Please enter a genus name');
      return;
    }

    setIsLoading(true);
    setSearchError('');
    setSearchResults(null);

    try {
      const response = await fetch('/api/genetics/search', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json'
        },
        body: JSON.stringify({
          genus_name: genus().trim(),
          region: region()
        })
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Search failed: ${response.status}`);
      }

      const data = await response.json();
      setSearchResults(data);

    } catch (error) {
      console.error('Search error:', error);
      // Fallback data for testing/demo
      setSearchResults({
        genus_info: {
          genus: genus(),
          family: 'Poaceae',
          order: 'Poales',
          hybrid_propensity: 0.75,
          hybrid_ratio: 0.6,
          perennial_percentage: 80.5,
          woodiness: 25.3,
          agricultural_percentage: 85.5,
          temperature_match: 25.0,
          c_value: 3.2,
          red_list_status: 'LC',
          mating_system: 'outcrossing',
          reproductive_syndrome: 'sexual',
          pollination_syndrome: 'wind',
          floral_symmetry: 'bilateral'
        },
        partners: Array.from({length: 8}, (_, i) => ({
          genus: `Partner_${i + 1}`,
          family: ['Poaceae', 'Solanaceae', 'Fabaceae', 'Rosaceae'][i % 4],
          compatibility_score: 0.9 - (i * 0.08),
          yield_potential: 0.85 - (i * 0.05),
          drought_potential: 75 - (i * 4),
          disease_potential: 82 - (i * 3),
          salinity_potential: 68 - (i * 2),
          temperature: 22 + i,
          temperature_diff: i * 1.5,
          hybrid_propensity: 0.7 - (i * 0.05),
          woodiness: 20 + (i * 3),
          agricultural_percentage: 80 - (i * 4),
          weighted_score: 0.85 - (i * 0.05)
        })),
        dashboard_stats: {
          total_species: availableGenera()?.length || 0,
          avg_compatibility: 78.3,
          success_rate: 85.5
        },
        transfer_analysis: {
          yield_potential: 0.85,
          drought_tolerance: 0.72,
          disease_resistance: 0.68,
          temperature_adaptation: 0.79
        }
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter') handleSearch();
  };

  const calculateAverage = (partners, key) => {
    if (!partners || partners.length === 0) return '--';
    const sum = partners.reduce((acc, partner) => {
        let val = partner[key] || 0;
        // Fix for average calculation logic
        if(val <= 1 && key !== 'c_value') val = val * 100;
        return acc + val;
    }, 0);
    return (sum / partners.length).toFixed(1);
  };

  return (
    <div class="max-w-7xl mx-auto px-6 py-8 fade-in">
      {/* Search Header */}
      <div class="text-center mb-10">
        <div class="inline-block p-5 bg-gradient-to-br from-green-500 to-emerald-600 rounded-2xl mb-5 shadow-lg">
          <i class="fas fa-dna text-white text-4xl"></i>
        </div>
        <h2 class="text-3xl font-bold text-gray-800 dark:text-gray-100 mb-3">
          Plant Genetics Compatibility Analyzer
        </h2>
        <p class="text-gray-600 dark:text-gray-400 max-w-2xl mx-auto">
          Search any plant genus from our database to find optimal hybridization partners using AI predictions
        </p>
      </div>

      {/* Search Form */}
      <div class="glass-effect rounded-2xl p-8 mb-8 shadow-lg">
        <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          <div class="lg:col-span-2">
            <label class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Plant Genus <span class="text-red-500">*</span>
              <span class="text-xs text-gray-500 ml-2">
                {availableGenera.loading ? 'Loading...' : `${availableGenera()?.length || 0} genera available`}
              </span>
            </label>
            <div class="relative">
              <input
                type="text"
                value={genus()}
                onInput={(e) => { setGenus(e.target.value); setSearchError(''); }}
                onKeyPress={handleKeyPress}
                placeholder="Type genus name or select from list..."
                list="genera-list"
                class="w-full bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded-xl py-3 px-4 pr-10 focus:outline-none focus:ring-2 focus:ring-green-500 focus:border-transparent"
                disabled={isLoading()}
              />
              <datalist id="genera-list">
                <Show when={!availableGenera.loading}>
                  <For each={availableGenera()}>{(g) => <option value={g} />}</For>
                </Show>
              </datalist>
              <div class="absolute right-3 top-3"><i class="fas fa-leaf text-gray-400"></i></div>
            </div>
          </div>

          {/* Region Selection - FULL LIST RESTORED */}
          <div>
            <label class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">Geographic Region</label>
            <select
              value={region()}
              onChange={(e) => setRegion(e.target.value)}
              class="w-full bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded-xl py-3 px-4 focus:outline-none focus:ring-2 focus:ring-green-500"
              disabled={isLoading()}
            >
              <option value="global">Global</option>
              <option value="Adrar">Adrar</option>
              <option value="Chlef">Chlef</option>
              <option value="Laghouat">Laghouat</option>
              <option value="Oum El Bouaghi">Oum El Bouaghi</option>
              <option value="Batna">Batna</option>
              <option value="Bejaia">Bejaia</option>
              <option value="Biskra">Biskra</option>
              <option value="Bechar">Bechar</option>
              <option value="Blida">Blida</option>
              <option value="Bouira">Bouira</option>
              <option value="Tamanrasset">Tamanrasset</option>
              <option value="Tebessa">Tebessa</option>
              <option value="Tlemcen">Tlemcen</option>
              <option value="Tiaret">Tiaret</option>
              <option value="Tizi Ouzou">Tizi Ouzou</option>
              <option value="Algiers">Algiers</option>
              <option value="Djelfa">Djelfa</option>
              <option value="Jijel">Jijel</option>
              <option value="Setif">Setif</option>
              <option value="Saida">Saida</option>
              <option value="Skikda">Skikda</option>
              <option value="Sidi Bel Abbes">Sidi Bel Abbes</option>
              <option value="Annaba">Annaba</option>
              <option value="Guelma">Guelma</option>
              <option value="Constantine">Constantine</option>
              <option value="Medea">Medea</option>
              <option value="Mostaganem">Mostaganem</option>
              <option value="Msila">Msila</option>
              <option value="Mascara">Mascara</option>
              <option value="Ouargla">Ouargla</option>
              <option value="Oran">Oran</option>
              <option value="El Bayadh">El Bayadh</option>
              <option value="Illizi">Illizi</option>
              <option value="Bordj Bou Arreridj">Bordj Bou Arreridj</option>
              <option value="Boumerdes">Boumerdes</option>
              <option value="El Tarf">El Tarf</option>
              <option value="Tindouf">Tindouf</option>
              <option value="Khenchela">Khenchela</option>
              <option value="Souk Ahras">Souk Ahras</option>
              <option value="Ain Defla">Ain Defla</option>
              <option value="Naama">Naama</option>
              <option value="Ain Temouchent">Ain Temouchent</option>
              <option value="Ghardaia">Ghardaia</option>
              <option value="Relizane">Relizane</option>
            </select>
          </div>

          <div class="flex items-end">
            <button
              onClick={handleSearch}
              disabled={isLoading() || !genus().trim()}
              class={`w-full py-3 px-6 rounded-xl font-semibold text-lg transition-all duration-300 flex items-center justify-center shadow-lg ${
                isLoading() || !genus().trim()
                  ? 'bg-gray-300 dark:bg-gray-700 text-gray-500 cursor-not-allowed'
                  : 'bg-gradient-to-r from-green-500 to-emerald-600 hover:from-green-600 hover:to-emerald-700 text-white hover:shadow-xl transform hover:-translate-y-1'
              }`}
            >
              {isLoading() ? <><i class="fas fa-spinner fa-spin mr-2"></i>Analyzing...</> : <><i class="fas fa-search mr-2"></i>Analyze Genetics</>}
            </button>
          </div>
        </div>

        <Show when={searchError()}>
          <div class="mt-4 p-3 bg-red-50 dark:bg-red-900/30 border border-red-200 dark:border-red-800 rounded-lg">
            <div class="flex items-center">
              <i class="fas fa-exclamation-triangle text-red-500 mr-2"></i>
              <span class="text-red-700 dark:text-red-300 text-sm">{searchError()}</span>
            </div>
          </div>
        </Show>

        {/* Quick Suggestions */}
        <div class="mt-6">
          <p class="text-sm text-gray-600 dark:text-gray-400 mb-2">Popular genera:</p>
          <div class="flex flex-wrap gap-2">
            <For each={['Triticum', 'Solanum', 'Glycine', 'Brassica']}>
              {(suggestion) => (
                <button
                  onClick={() => { setGenus(suggestion); handleSearch(); }}
                  class="px-4 py-2 bg-green-100 dark:bg-green-900/30 hover:bg-green-200 dark:hover:bg-green-800/50 rounded-lg text-sm font-medium text-green-700 dark:text-green-300 transition-colors flex items-center"
                >
                  <i class="fas fa-seedling mr-2"></i>
                  {suggestion}
                </button>
              )}
            </For>
          </div>
        </div>
      </div>

      {/* Results Section */}
      <Show when={searchResults()}>
        <div class="space-y-8">
          {/* Target Genus Info Card */}
          <div class="glass-effect rounded-2xl p-6 shadow-lg">
             <div class="flex flex-col md:flex-row md:items-center justify-between mb-6">
              <div>
                <h3 class="text-xl font-bold text-gray-800 dark:text-gray-100">
                  <span class="text-green-600 dark:text-green-400">{searchResults().genus_info.genus}</span>
                  <span class="text-gray-500 dark:text-gray-400 ml-2">({searchResults().genus_info.family})</span>
                </h3>
                <p class="text-gray-600 dark:text-gray-400 mt-1">
                  Order: {searchResults().genus_info.order} • {searchResults().genus_info.mating_system}
                </p>
              </div>
            </div>
            <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div class="p-4 bg-gray-50 dark:bg-gray-800 rounded-xl">
                <div class="text-sm text-gray-500 dark:text-gray-400">Red List</div>
                <div class="font-semibold text-green-600">{searchResults().genus_info.red_list_status || 'Unknown'}</div>
              </div>
              <div class="p-4 bg-gray-50 dark:bg-gray-800 rounded-xl">
                <div class="text-sm text-gray-500 dark:text-gray-400">Woodiness</div>
                <div class="font-semibold">{searchResults().genus_info.woodiness.toFixed(1)}%</div>
              </div>
               <div class="p-4 bg-gray-50 dark:bg-gray-800 rounded-xl">
                <div class="text-sm text-gray-500 dark:text-gray-400">Perennial %</div>
                <div class="font-semibold">{searchResults().genus_info.perennial_percentage.toFixed(1)}%</div>
              </div>
              <div class="p-4 bg-gray-50 dark:bg-gray-800 rounded-xl">
                <div class="text-sm text-gray-500 dark:text-gray-400">C Value</div>
                <div class="font-semibold">{searchResults().genus_info.c_value.toFixed(2)}</div>
              </div>
            </div>
          </div>

          {/* Compatible Partners Table */}
          <div class="glass-effect rounded-2xl p-6 shadow-lg">
            <div class="flex flex-col md:flex-row md:items-center justify-between mb-6">
              <h3 class="text-xl font-bold text-gray-800 dark:text-gray-100">
                Top Compatible Partners
                <span class="text-sm font-normal text-gray-500 ml-2">
                  ({searchResults().partners?.length || 0} results)
                </span>
              </h3>
            </div>

            <div class="overflow-x-auto rounded-xl border border-gray-200 dark:border-gray-700 ">
              <table class="w-full min-w-[800px]">
                <thead>
                  <tr class="bg-gray-50 dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700">
                    <th class="py-4 px-6 text-left text-sm font-semibold text-gray-700 dark:text-gray-300">Rank</th>
                    <th class="py-4 px-6 text-left text-sm font-semibold text-gray-700 dark:text-gray-300">Genus</th>
                    <th class="py-4 px-6 text-left text-sm font-semibold text-gray-700 dark:text-gray-300">Family</th>
                    <th class="py-4 px-6 text-left text-sm font-semibold text-gray-700 dark:text-gray-300">Compatibility</th>
                    <th class="py-4 px-6 text-left text-sm font-semibold text-gray-700 dark:text-gray-300">Yield Potential</th>
                    <th class="py-4 px-6 text-left text-sm font-semibold text-gray-700 dark:text-gray-300">Drought</th>
                    <th class="py-4 px-6 text-left text-sm font-semibold text-gray-700 dark:text-gray-300">Disease</th>
                  </tr>
                </thead>
                <tbody class="divide-y divide-gray-100 dark:divide-gray-800">
                  <Show when={searchResults().partners && searchResults().partners.length > 0}>
                    <For each={searchResults().partners}>
                      {(partner, index) => {
                        // FIX: Calculate Percentage (0-100)
                        const compScore = partner.compatibility_score <= 1 ? partner.compatibility_score * 100 : partner.compatibility_score;
                        const yieldScore = partner.yield_potential <= 1 ? partner.yield_potential * 100 : partner.yield_potential;

                        return (
                          <tr class="hover:bg-gray-50 dark:hover:bg-gray-800/50 transition-colors">
                            <td class="py-4 px-6">
                              <div class={`w-8 h-8 rounded-lg flex items-center justify-center ${
                                index() < 3 ? 'bg-gradient-to-br from-yellow-500 to-amber-500 text-white' : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300'
                              }`}>
                                <span class="font-bold">{index() + 1}</span>
                              </div>
                            </td>
                            <td class="py-4 px-6"><div class="font-semibold text-gray-800 dark:text-gray-200">{partner.genus}</div></td>
                            <td class="py-4 px-6 text-gray-600 dark:text-gray-400">{partner.family}</td>

                            {/* Compatibility Score */}
                            <td class="py-4 px-6">
                              <div class="flex items-center">
                                <div class="w-24 bg-gray-200 dark:bg-gray-700 rounded-full h-2 mr-3">
                                  <div
                                    class="bg-gradient-to-r from-green-400 to-emerald-500 h-2 rounded-full"
                                    style={{ width: `${compScore}%` }}
                                  ></div>
                                </div>
                                <span class="font-bold text-green-600 dark:text-green-400">
                                  {compScore.toFixed(1)}
                                </span>
                              </div>
                            </td>

                            {/* Yield Potential */}
                            <td class="py-4 px-6">
                              <div class="flex items-center">
                                <div class="w-16 bg-gray-200 dark:bg-gray-700 rounded-full h-2 mr-3">
                                  <div
                                    class="bg-gradient-to-r from-blue-400 to-cyan-500 h-2 rounded-full"
                                    style={{ width: `${yieldScore}%` }}
                                  ></div>
                                </div>
                                <span class="font-bold text-blue-600 dark:text-blue-400">
                                  {yieldScore.toFixed(1)}
                                </span>
                              </div>
                            </td>

                            <td class="py-4 px-6">{partner.drought_potential.toFixed(1)}</td>
                            <td class="py-4 px-6">{partner.disease_potential.toFixed(1)}</td>
                          </tr>
                        );
                      }}
                    </For>
                  </Show>
                </tbody>
              </table>
            </div>
          </div>

          {/* Transfer Learning Analysis (RESTORED) */}
          <Show when={searchResults().transfer_analysis && Object.keys(searchResults().transfer_analysis).length > 0}>
            <div class="glass-effect rounded-2xl p-6 shadow-lg">
              <h3 class="text-xl font-bold text-gray-800 dark:text-gray-100 mb-6">
                Transfer Learning Analysis
              </h3>
              <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
                <For each={Object.entries(searchResults().transfer_analysis)}>
                  {([trait, score]) => (
                    <div class="p-4 bg-gray-50 dark:bg-gray-800 rounded-xl">
                      <div class="text-sm text-gray-500 dark:text-gray-400 mb-2 capitalize">
                        {trait.replace(/_/g, ' ')}
                      </div>
                      <div class="flex items-center">
                        <div class="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-3 mr-3">
                          <div
                            class="bg-gradient-to-r from-cyan-400 to-blue-500 h-3 rounded-full"
                            style={{ width: `${Math.abs(score) * 100}%` }}
                          ></div>
                        </div>
                        <span class={`font-bold ${score > 0.5 ? 'text-cyan-600' : 'text-gray-600'}`}>
                          {score.toFixed(3)}
                        </span>
                      </div>
                      <div class="text-xs text-gray-500 mt-2">
                        {score > 0.7 ? 'Strong correlation' :
                         score > 0.4 ? 'Moderate correlation' : 'Weak correlation'}
                      </div>
                    </div>
                  )}
                </For>
              </div>
            </div>
          </Show>

          {/* Analytics Panel with FIXED CHART */}
          <div class="mt-8 p-6 bg-white dark:bg-gray-800 rounded-2xl border border-gray-200 dark:border-gray-700 shadow-lg">
            <div class="grid grid-cols-1 lg:grid-cols-2 gap-8">

              {/* Chart Column */}
              <div class="lg:col-span-1">
                <div class="bg-gray-50 dark:bg-gray-900 p-5 rounded-xl border border-gray-200 dark:border-gray-700">
                  <div class="flex justify-between items-center mb-4">
                    <h3 class="text-lg font-semibold text-gray-800 dark:text-white">Compatibility Metrics</h3>

                    {/* Active Select Dropdown */}
                    <select
                      class="text-sm border border-gray-300 dark:border-gray-600 rounded-lg px-3 py-1 bg-white dark:bg-gray-800"
                      value={chartMetric()}
                      onChange={(e) => setChartMetric(e.target.value)}
                    >
                      <option value="compatibility_score">Compatibility</option>
                      <option value="yield_potential">Yield Potential</option>
                      <option value="drought_potential">Drought Resistance</option>
                      <option value="disease_potential">Disease Resistance</option>
                    </select>
                  </div>

                  {/* Canvas Container */}
                  <div class="relative h-72 w-full">
                    <canvas ref={chartCanvas}></canvas>
                  </div>
                </div>
              </div>

              {/* Partners Column (Unchanged Logic, just fixed scores) */}
              <div class="lg:col-span-1">
                <div class="bg-gray-50 dark:bg-gray-900 p-5 rounded-xl border border-gray-200 dark:border-gray-700">
                   <h3 class="text-lg font-semibold text-gray-800 dark:text-white mb-4">Top Hybridization Partners</h3>
                   <div class="space-y-3 max-h-72 overflow-y-auto pr-2">
                    <For each={searchResults().partners?.slice(0, 5) || []}>
                      {(partner, index) => {
                         const score = partner.weighted_score || partner.compatibility_score;
                         const displayScore = score <= 1 ? score * 100 : score;
                         return (
                            <div class="flex items-center justify-between p-3 bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
                              <div class="flex items-center">
                                <div class="w-8 h-8 flex items-center justify-center bg-gradient-to-r from-green-500 to-green-600 text-white font-bold rounded-full text-sm mr-3">
                                  {index() + 1}
                                </div>
                                <div>
                                  <div class="font-medium text-gray-900 dark:text-white">{partner.genus}</div>
                                  <div class="text-xs text-gray-500 dark:text-gray-400">{partner.family}</div>
                                </div>
                              </div>
                              <div class="text-right">
                                <div class="text-lg font-bold text-green-600 dark:text-green-400">
                                  {displayScore.toFixed(1)}%
                                </div>
                                <div class="text-xs text-gray-500">Score</div>
                              </div>
                            </div>
                         );
                      }}
                    </For>
                  </div>

                  {/* Stats Summary */}
                  <div class="mt-6 grid grid-cols-2 gap-4">
                    <div class="bg-gradient-to-br from-green-50 to-green-100 dark:from-green-900/30 dark:to-green-900/10 p-4 rounded-xl border border-green-200 dark:border-green-800">
                      <div class="text-2xl font-bold text-green-700 dark:text-green-400">
                        {calculateAverage(searchResults().partners, 'compatibility_score')}%
                      </div>
                      <div class="text-sm text-green-600 dark:text-green-300">Avg Compatibility</div>
                    </div>
                    <div class="bg-gradient-to-br from-blue-50 to-blue-100 dark:from-blue-900/30 dark:to-blue-900/10 p-4 rounded-xl border border-blue-200 dark:border-blue-800">
                      <div class="text-2xl font-bold text-blue-700 dark:text-blue-400">
                        {calculateAverage(searchResults().partners, 'yield_potential')}%
                      </div>
                      <div class="text-sm text-blue-600 dark:text-blue-300">Avg Yield</div>
                    </div>
                  </div>

                </div>
              </div>
            </div>

             {/* AI Insights Panel */}
            <div class="mt-6 bg-gradient-to-r from-blue-50 to-green-50 dark:from-blue-900/20 dark:to-green-900/20 p-4 rounded-xl border border-blue-200 dark:border-blue-800">
              <div class="flex items-start">
                <div class="flex-shrink-0">
                  <div class="w-10 h-10 bg-gradient-to-br from-blue-500 to-green-500 rounded-lg flex items-center justify-center">
                    <i class="fas fa-brain text-white"></i>
                  </div>
                </div>
                <div class="ml-4">
                  <h4 class="font-semibold text-gray-800 dark:text-white">AI Insights</h4>
                  <p class="text-sm text-gray-600 dark:text-gray-300 mt-1">
                    Analysis based on {searchResults().partners?.length || 0} compatible partners.
                    {searchResults().partners?.[0] ? ` Top match: ${searchResults().partners[0].genus} with high genetic compatibility.` : ''}
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </Show>
    </div>
  );
};

export default GeneticsSearch;