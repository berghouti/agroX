import { createResource, createSignal, For } from 'solid-js';
import GeneticsSearch from './GeneticsSearch';

const fetchDashboardStats = async () => {
  try {
    const response = await fetch('/api/dashboard/stats');
    return response.json();
  } catch {
    return {
      total_yield: 95.2,
      avg_temperature: 25.5,
      avg_humidity: 65.3,
      active_alerts: 3,
      predicted_yield: "↑ 12%",
      soil_health: 85.5,
      water_efficiency: 78.3,
      crop_growth: 92.1
    };
  }
};

const fetchMetrics = async () => {
  try {
    const response = await fetch('/api/metrics?days=7');
    return response.json();
  } catch {
    return [];
  }
};

const Dashboard = () => {
  const [stats] = createResource(fetchDashboardStats);
  const [metrics] = createResource(fetchMetrics);
  const [timeRange, setTimeRange] = createSignal('7d');

  return (
    <div class="max-w-7xl mx-auto px-6 py-8 fade-in">
      {/* Dashboard Header - Updated */}
      <div class="flex flex-col md:flex-row justify-between items-start md:items-center mb-8">
        <div>
          <h2 class="text-3xl font-bold text-gray-800 dark:text-gray-100">Farm Intelligence Dashboard</h2>
          <p class="text-gray-600 dark:text-gray-400">Real-time monitoring and genetic analysis</p>
        </div>

        <div class="flex items-center space-x-4 mt-4 md:mt-0">
          <div class="flex space-x-2">
            <button class="px-4 py-2 glass-effect hover:bg-white/10 dark:hover:bg-gray-800/50 rounded-lg font-medium transition-colors">
              <i class="fas fa-chart-line mr-2"></i>
              Analytics
            </button>
          </div>

        </div>
      </div>

      {/* Add Genetics Search Section */}
      <div class="mb-8">
        <GeneticsSearch />
      </div>

      {/* ... rest of your existing Dashboard code ... */}
    </div>

  );
};

export default Dashboard;