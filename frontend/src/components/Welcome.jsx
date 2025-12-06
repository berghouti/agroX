import { createSignal, onMount } from 'solid-js';

const Welcome = () => {
  const [animate, setAnimate] = createSignal(false);

  onMount(() => {
    setTimeout(() => setAnimate(true), 100);
  });

  return (
    <div class="min-h-[80vh] flex flex-col items-center justify-center px-6 fade-in">
      {/* Main Hero Section */}
      <div class="text-center max-w-4xl mx-auto">
        {/* Animated Logo */}
        <div class={`relative mb-8 transition-all duration-1000 ${animate() ? 'scale-100 opacity-100' : 'scale-90 opacity-0'}`}>
          <div class="w-32 h-32 mx-auto bg-gradient-to-br from-green-400 via-emerald-500 to-teal-600 rounded-3xl flex items-center justify-center shadow-2xl transform rotate-3 hover:rotate-6 transition-transform duration-500">
            <i class="fas fa-seedling text-white text-5xl"></i>
          </div>
          <div class="absolute -inset-4 bg-gradient-to-r from-green-500 to-teal-500 rounded-3xl blur-xl opacity-30 -z-10 animate-pulse"></div>
        </div>

        {/* Main Heading */}
        <h1 class={`text-6xl md:text-7xl font-bold mb-6 transition-all duration-1000 delay-200 ${
          animate() ? 'translate-y-0 opacity-100' : 'translate-y-10 opacity-0'
        }`}>
          <span class="gradient-text">AgriAI</span>
          <span class="block text-4xl md:text-5xl font-light mt-4 text-gray-700 dark:text-gray-300">
            Intelligent Farming Platform
          </span>
        </h1>

        {/* Subtitle */}
        <p class={`text-xl text-gray-600 dark:text-gray-400 mb-12 max-w-2xl mx-auto transition-all duration-1000 delay-400 ${
          animate() ? 'translate-y-0 opacity-100' : 'translate-y-10 opacity-0'
        }`}>
          Harness the power of AI and data analytics to optimize your agricultural operations.
          Monitor crops, predict yields, and get intelligent recommendations in real-time.
        </p>

        {/* Stats Grid - FIXED ALIGNMENT */}



      </div>

      {/* Features Preview */}
      <div class="mt-20 w-full max-w-6xl mx-auto">
        <div class="grid md:grid-cols-3 gap-8">
          <div class="glass-effect rounded-2xl p-6 transform hover:scale-[1.02] transition-all duration-300">
            <div class="w-14 h-14 bg-gradient-to-br from-green-500 to-teal-500 rounded-xl flex items-center justify-center mb-4">
              <i class="fas fa-chart-line text-white text-2xl"></i>
            </div>
            <h3 class="text-xl font-bold mb-3 text-gray-800 dark:text-gray-100">Real-time Analytics</h3>
            <p class="text-gray-600 dark:text-gray-400">Monitor crop health, soil conditions, and weather patterns in real-time.</p>
          </div>

          <div class="glass-effect rounded-2xl p-6 transform hover:scale-[1.02] transition-all duration-300">
            <div class="w-14 h-14 bg-gradient-to-br from-cyan-500 to-blue-500 rounded-xl flex items-center justify-center mb-4">
              <i class="fas fa-robot text-white text-2xl"></i>
            </div>
            <h3 class="text-xl font-bold mb-3 text-gray-800 dark:text-gray-100">AI Assistant</h3>
            <p class="text-gray-600 dark:text-gray-400">Get intelligent recommendations and answers to your farming questions.</p>
          </div>

          <div class="glass-effect rounded-2xl p-6 transform hover:scale-[1.02] transition-all duration-300">
            <div class="w-14 h-14 bg-gradient-to-br from-purple-500 to-pink-500 rounded-xl flex items-center justify-center mb-4">
              <i class="fas fa-bolt text-white text-2xl"></i>
            </div>
            <h3 class="text-xl font-bold mb-3 text-gray-800 dark:text-gray-100">Smart Automation</h3>
            <p class="text-gray-600 dark:text-gray-400">Automate irrigation, fertilization, and pest control with AI-driven schedules.</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Welcome;