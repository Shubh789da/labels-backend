import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { SearchBar } from './components/SearchBar';
import { Dashboard } from './components/Dashboard';
import type { DrugHistoryResponse } from './types';
import { getDrugHistory } from './services/api';
import { Pill, Sparkles } from 'lucide-react';

function App() {
  const [data, setData] = useState<DrugHistoryResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [hasSearched, setHasSearched] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSearch = async (term: string) => {
    setLoading(true);
    setError(null);
    try {
      const result = await getDrugHistory(term);
      setData(result);
      setHasSearched(true);
    } catch (err) {
      console.error(err);
      setError("Could not find data for this drug. Please try another name.");
      setData(null);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#050505] text-white selection:bg-blue-500/30 overflow-x-hidden font-sans">
      {/* Background Gradients */}
      <div className="fixed inset-0 z-0 pointer-events-none">
        <div className="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] bg-blue-600/10 rounded-full blur-[120px] animate-pulse-slow" />
        <div className="absolute bottom-[-10%] right-[-10%] w-[40%] h-[40%] bg-emerald-600/10 rounded-full blur-[120px] animate-pulse-slow delay-1000" />
      </div>

      <div className="relative z-10 container mx-auto px-4 py-8">
        {/* Header / Logo */}
        <motion.header
          className={`flex items-center justify-between ${hasSearched ? 'mb-4 md:mb-6' : 'mb-16 md:mb-24'}`}
          initial={{ y: -50, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ duration: 0.8, ease: "easeOut" }}
        >
          <div className="flex items-center gap-3 group cursor-pointer">
            <div className="p-2.5 bg-gradient-to-br from-blue-600/20 to-emerald-600/20 rounded-xl backdrop-blur-md border border-white/5 group-hover:border-white/20 transition-all duration-300">
              <Pill className="w-6 h-6 text-blue-400 group-hover:text-emerald-400 transition-colors duration-300" />
            </div>
            <h1 className="text-xl font-bold tracking-tight text-white/90">
              DrugJourney
            </h1>
          </div>
          {/* Add a fake nav item or user icon for completeness if desired, staying simple for now */}
        </motion.header>

        {/* Main Content Area */}
        <main className="flex flex-col items-center justify-center min-h-[60vh]">
          <AnimatePresence mode="wait">
            {!hasSearched ? (
              <motion.div
                key="landing"
                initial={{ opacity: 0, y: 30 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -30, filter: "blur(10px)" }}
                transition={{ duration: 0.6 }}
                className="text-center space-y-10 max-w-4xl w-full"
              >
                <div className="space-y-6">
                  <motion.div
                    initial={{ opacity: 0, scale: 0.9 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ delay: 0.2, duration: 0.5 }}
                    className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-white/5 border border-white/10 backdrop-blur-sm text-sm text-slate-400"
                  >
                    <Sparkles className="w-4 h-4 text-amber-300" />
                    <span>Visual Regulatory Intelligence</span>
                  </motion.div>

                  <h2 className="text-6xl md:text-7xl font-extrabold tracking-tight leading-[1.1]">
                    The Entire History of <br />
                    <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-400 via-emerald-400 to-blue-500 animate-gradient-x background-animate">
                      Modern Medicine
                    </span>
                  </h2>

                  <p className="text-xl md:text-2xl text-slate-400 max-w-2xl mx-auto font-light leading-relaxed">
                    Trace the complete FDA lifecycle from IND to latest approval.
                    Simple, visual, and comprehensive.
                  </p>
                </div>

                <div className="pt-8 pb-12">
                  <SearchBar onSearch={handleSearch} isSearching={loading} />
                  {error && (
                    <motion.div
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      className="mt-4 text-red-400 text-sm bg-red-500/10 border border-red-500/20 px-4 py-2 rounded-lg inline-block"
                    >
                      {error}
                    </motion.div>
                  )}
                </div>

                {loading && (
                  <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="mt-4 flex items-center justify-center gap-3 text-blue-400 font-mono text-sm"
                  >
                    <div className="w-2 h-2 rounded-full bg-blue-400 animate-bounce" />
                    <div className="w-2 h-2 rounded-full bg-blue-400 animate-bounce delay-100" />
                    <div className="w-2 h-2 rounded-full bg-blue-400 animate-bounce delay-200" />
                    Connecting to FDA Database...
                  </motion.div>
                )}
              </motion.div>
            ) : (
              <motion.div
                key="dashboard"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="w-full"
              >
                {data && (
                  <Dashboard
                    data={data}
                    onBack={() => {
                      setHasSearched(false);
                      setData(null);
                    }}
                  />
                )}
              </motion.div>
            )}
          </AnimatePresence>
        </main>
      </div>
    </div>
  );
}

export default App;
