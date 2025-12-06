import ThemeToggle from '../components/ThemeToggle';

const MainLayout = (props) => {
  return (
    <div class="min-h-screen bg-gradient-to-br from-white via-green-50 to-cyan-50 dark:from-gray-900 dark:via-green-950 dark:to-cyan-900">
      {/* Animated Background Elements */}
      <div class="fixed inset-0 overflow-hidden pointer-events-none">
        <div class="absolute -top-40 -right-40 w-80 h-80 bg-green-300 dark:bg-green-700 rounded-full mix-blend-multiply dark:mix-blend-screen filter blur-3xl opacity-20 animate-pulse"></div>
        <div class="absolute -bottom-40 -left-40 w-80 h-80 bg-cyan-300 dark:bg-cyan-700 rounded-full mix-blend-multiply dark:mix-blend-screen filter blur-3xl opacity-20 animate-pulse delay-1000"></div>
      </div>

      {/* Main Navigation */}
      <nav class="relative z-10 glass-effect border-b border-green-200 dark:border-green-800">
        <div class="max-w-7xl mx-auto px-6 py-4">
          <div class="flex items-center justify-between">
            {/* Logo */}
            <div class="flex items-center space-x-3 cursor-pointer" onClick={() => props.setCurrentView('welcome')}>
              <div class="w-12 h-12 bg-gradient-to-br from-green-500 to-emerald-600 rounded-xl flex items-center justify-center shadow-lg">
                <i class="fas fa-leaf text-white text-2xl"></i>
              </div>
              <div>
                <h1 class="text-2xl font-bold bg-gradient-to-r from-green-600 to-emerald-600 dark:from-green-400 dark:to-emerald-400 bg-clip-text text-transparent">
                  AgriAI
                </h1>
                <p class="text-sm text-gray-600 dark:text-gray-400">Smart Farming Intelligence</p>
              </div>
            </div>

            {/* Navigation Tabs */}
            <div class="flex items-center space-x-6">
              <div class="flex space-x-1 bg-white/50 dark:bg-gray-800/50 rounded-xl p-1 backdrop-blur-sm">
                <button
                  onClick={() => props.setCurrentView('dashboard')}
                  class={`px-6 py-3 rounded-lg font-medium transition-all duration-300 ${
                    props.currentView() === 'dashboard'
                      ? 'bg-gradient-to-r from-green-500 to-emerald-500 text-white shadow-lg transform scale-105'
                      : 'text-gray-700 dark:text-gray-300 hover:text-green-600 dark:hover:text-green-400'
                  }`}
                >
                  <i class="fas fa-chart-line mr-2"></i>
                  Dashboard
                </button>

              </div>

              {/* Theme Toggle */}
            </div>
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main class="relative z-10">
        {props.children}
      </main>

      {/* Footer */}
      <footer class="relative z-10 py-6 text-center text-gray-600 dark:text-gray-400 border-t border-green-200 dark:border-green-800 mt-8">
        <p>© 2024 AgriAI Dashboard. Smart farming solutions for the future.</p>
        <p class="text-sm mt-2">
          <i class="fas fa-seedling mr-2"></i>
          Powered by AI & Precision Agriculture
        </p>
      </footer>
    </div>
  );
};

export default MainLayout;