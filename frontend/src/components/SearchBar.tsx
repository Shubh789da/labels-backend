import React, { useState } from 'react';
import { Search, X } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

interface SearchBarProps {
    onSearch: (term: string) => void;
    isSearching: boolean;
}

export const SearchBar: React.FC<SearchBarProps> = ({ onSearch, isSearching }) => {
    const [term, setTerm] = useState('');
    const [isFocused, setIsFocused] = useState(false);

    const handleSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        if (term.trim()) {
            onSearch(term);
        }
    };

    return (
        <motion.div
            initial={{ scale: 0.9, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            transition={{ duration: 0.6, ease: "easeOut" }}
            className="w-full max-w-3xl mx-auto relative z-20"
        >
            <form onSubmit={handleSubmit} className="relative group">
                <motion.div
                    animate={{
                        boxShadow: isFocused ? "0 0 40px -5px rgba(59, 130, 246, 0.3)" : "0 0 20px -5px rgba(0, 0, 0, 0.3)"
                    }}
                    className="absolute inset-0 rounded-full bg-gradient-to-r from-blue-500/20 to-emerald-500/20 blur-xl opacity-50 transition-opacity duration-500"
                />

                <div className="relative flex items-center">
                    <div className="absolute left-6 text-slate-400 group-focus-within:text-blue-400 transition-colors duration-300">
                        <Search className="w-6 h-6" />
                    </div>

                    <input
                        type="text"
                        value={term}
                        onChange={(e) => setTerm(e.target.value)}
                        onFocus={() => setIsFocused(true)}
                        onBlur={() => setIsFocused(false)}
                        className="w-full pl-16 pr-14 py-5 bg-slate-900/60 backdrop-blur-2xl border border-white/10 rounded-full text-white placeholder-slate-500 focus:outline-none focus:border-blue-500/50 focus:bg-slate-900/80 transition-all duration-300 text-xl font-light tracking-wide shadow-2xl"
                        placeholder="Search for a drug (e.g., Keytruda)..."
                        disabled={isSearching}
                    />

                    <AnimatePresence>
                        {term && (
                            <motion.button
                                initial={{ opacity: 0, scale: 0.8 }}
                                animate={{ opacity: 1, scale: 1 }}
                                exit={{ opacity: 0, scale: 0.8 }}
                                type="button"
                                onClick={() => setTerm('')}
                                className="absolute right-5 p-2 rounded-full hover:bg-white/10 text-slate-400 hover:text-white transition-colors"
                            >
                                <X className="w-5 h-5" />
                            </motion.button>
                        )}
                    </AnimatePresence>
                </div>
            </form>
        </motion.div>
    );
};
