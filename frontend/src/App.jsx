import { createSignal, onMount } from 'solid-js';
import MainLayout from './layouts/MainLayout';
import Welcome from './components/Welcome';
import Dashboard from './components/Dashboard';

function App() {
  const [currentView, setCurrentView] = createSignal('welcome');
  const [darkMode, setDarkMode] = createSignal(false);

  onMount(() => {
    // Check for saved theme preference
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme === 'dark') {
      setDarkMode(true);
      document.documentElement.classList.add('dark-mode');
    }
  });

  const toggleDarkMode = () => {
    setDarkMode(!darkMode());
    if (!darkMode()) {
      document.documentElement.classList.add('dark-mode');
      localStorage.setItem('theme', 'dark');
    } else {
      document.documentElement.classList.remove('dark-mode');
      localStorage.setItem('theme', 'light');
    }
  };

  return (
    <MainLayout
      currentView={currentView}
      setCurrentView={setCurrentView}
      darkMode={darkMode}
      toggleDarkMode={toggleDarkMode}
    >
      {currentView() === 'welcome' && <Welcome />}
      {currentView() === 'dashboard' && <Dashboard />}
    </MainLayout>
  );
}

export default App;