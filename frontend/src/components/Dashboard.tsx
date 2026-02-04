import React from 'react';
import type { DrugHistoryResponse } from '../types';
import { HistoryLog } from './HistoryLog';
import { ArrowLeft, FlaskConical, ScrollText, Activity, ShieldCheck, ExternalLink, FileText } from 'lucide-react';
import { motion } from 'framer-motion';

interface DashboardProps {
    data: DrugHistoryResponse;
    onBack: () => void;
}

export const Dashboard: React.FC<DashboardProps> = ({ data, onBack }) => {
    // State for dynamic indication count (updates when extraction finishes)
    const [indicationCount, setIndicationCount] = React.useState(data.indication_count || data.indications.length);

    // Extract latest product details
    const latestProduct = data.approvals[0]?.products[0];
    const activeIngredient = latestProduct?.active_ingredients[0] || "Unknown";
    const dosageForm = latestProduct?.dosage_form || "Unknown";
    const route = latestProduct?.route || "Unknown";
    const marketStatus = latestProduct?.marketing_status || "Active";

    // Find initial approval date
    const sinceDate = data.approvals.find(a => a.initial_approval_date)?.initial_approval_date || "Unknown";
    const formattedSinceDate = sinceDate !== "Unknown" ? new Date(
        sinceDate.substring(0, 4) + '-' + sinceDate.substring(4, 6) + '-' + sinceDate.substring(6, 8)
    ).toLocaleDateString('en-US') : "Unknown";

    // Find a label URL
    const labelUrl = data.indications.find(i => i.url)?.url;

    // Callback to update count from extracted documents
    const handleIndicationCountUpdate = (count: number) => {
        if (count > 0) {
            console.log(`[Dashboard] Updating approved conditions count: ${count}`);
            setIndicationCount(count);
        }
    };

    return (
        <div className="min-h-screen bg-[#050505] pb-20">
            {/* Sticky Header */}
            <header className="sticky top-0 z-50 bg-[#050505]/80 backdrop-blur-xl border-b border-white/10">
                <div className="container mx-auto px-4 h-16 flex items-center justify-between">
                    <div className="flex items-center gap-4">
                        <button
                            onClick={onBack}
                            className="p-2 -ml-2 rounded-lg hover:bg-white/5 text-slate-400 hover:text-white transition-colors"
                        >
                            <ArrowLeft className="w-5 h-5" />
                        </button>
                        <div>
                            <h1 className="text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-white to-slate-400">
                                {data.drug_name}
                            </h1>
                            <div className="flex items-center gap-2 text-xs text-slate-500 font-mono">
                                <span>{data.approvals[0]?.application_number}</span>
                                <span>•</span>
                                <span>{data.approvals[0]?.sponsor_name}</span>
                            </div>
                        </div>
                    </div>
                    <div className="flex items-center gap-3">
                        <div className="px-3 py-1 rounded bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-xs font-medium flex items-center gap-1.5">
                            <ShieldCheck className="w-3.5 h-3.5" />
                            {marketStatus.toUpperCase()}
                        </div>
                        <div className="px-3 py-1 rounded bg-blue-500/10 border border-blue-500/20 text-blue-400 text-xs font-medium">
                            Rx
                        </div>
                    </div>
                </div>
            </header>

            <main className="container mx-auto px-4 py-8">
                <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">

                    {/* Left Column: Drug Facts (Desktop: 3 cols) */}
                    <div className="lg:col-span-3 space-y-6">
                        <motion.div
                            initial={{ opacity: 0, x: -20 }}
                            animate={{ opacity: 1, x: 0 }}
                            className="bg-slate-900/50 border border-white/10 rounded-xl p-5 backdrop-blur-sm"
                        >
                            <h3 className="text-sm font-semibold text-white mb-4 flex items-center gap-2">
                                <FlaskConical className="w-4 h-4 text-blue-400" />
                                Drug Profile
                            </h3>
                            <div className="space-y-4">
                                <div>
                                    <div className="text-[10px] uppercase tracking-wider text-slate-500 font-mono mb-1">Active Ingredient</div>
                                    <div className="text-sm text-slate-200 capitalize">{activeIngredient.toLowerCase()}</div>
                                </div>
                                <div>
                                    <div className="text-[10px] uppercase tracking-wider text-slate-500 font-mono mb-1">Dosage Form</div>
                                    <div className="text-sm text-slate-200 capitalize">{dosageForm.toLowerCase()}</div>
                                </div>
                                <div>
                                    <div className="text-[10px] uppercase tracking-wider text-slate-500 font-mono mb-1">Route</div>
                                    <div className="text-sm text-slate-200 capitalize">{route.toLowerCase()}</div>
                                </div>
                                {labelUrl && (
                                    <div className="pt-2 border-t border-white/5">
                                        <a
                                            href={labelUrl}
                                            target="_blank"
                                            rel="noopener noreferrer"
                                            className="text-xs text-blue-400 hover:text-blue-300 flex items-center gap-1"
                                        >
                                            View Label on DailyMed <ExternalLink className="w-3 h-3" />
                                        </a>
                                    </div>
                                )}
                            </div>
                        </motion.div>

                        <div className="bg-gradient-to-br from-blue-900/20 to-purple-900/20 border border-white/5 rounded-xl p-5">
                            <h4 className="text-xs font-medium text-slate-400 uppercase tracking-widest mb-2">Market Status</h4>
                            <div className="text-2xl font-bold text-white mb-1">{marketStatus}</div>
                            <div className="text-xs text-slate-500">Since {formattedSinceDate}</div>
                        </div>
                    </div>

                    {/* Center Column: History (Desktop: 6 cols) */}
                    <div className="lg:col-span-6">
                        <motion.div
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ delay: 0.1 }}
                        >
                            <HistoryLog events={data.timeline} onIndicationCountUpdate={handleIndicationCountUpdate} />
                        </motion.div>
                    </div>

                    {/* Right Column: Intelligence (Desktop: 3 cols) */}
                    <div className="lg:col-span-3 space-y-6">
                        <motion.div
                            initial={{ opacity: 0, x: 20 }}
                            animate={{ opacity: 1, x: 0 }}
                            className="bg-slate-900/50 border border-white/10 rounded-xl p-5 backdrop-blur-sm"
                        >
                            <h3 className="text-sm font-semibold text-white mb-4 flex items-center gap-2">
                                <Activity className="w-4 h-4 text-emerald-400" />
                                Regulatory Intelligence
                            </h3>
                            <div className="grid grid-cols-2 gap-4">
                                <div className="p-3 rounded-lg bg-white/5 border border-white/5">
                                    <div className="text-2xl font-bold text-white">{data.approvals.length}</div>
                                    <div className="text-[10px] text-slate-500 uppercase tracking-wider">Approvals</div>
                                </div>
                                <div className="p-3 rounded-lg bg-white/5 border border-white/5">
                                    <div className="text-2xl font-bold text-white">{indicationCount}</div>
                                    <div className="text-[10px] text-slate-500 uppercase tracking-wider">Approved Conditions</div>
                                </div>
                            </div>

                            <div className="mt-6 space-y-3">
                                <h4 className="text-[10px] font-mono text-slate-500 uppercase">Related Documents</h4>
                                {/* Dynamically list first few URL-containing events */}
                                {data.key_documents && data.key_documents.length > 0 ? (
                                    data.key_documents.map((doc, i) => (
                                        <a
                                            key={i}
                                            href={doc.url}
                                            target="_blank"
                                            rel="noopener noreferrer"
                                            className="flex items-center gap-3 p-2 rounded hover:bg-white/5 cursor-pointer transition-colors group"
                                        >
                                            <div className="w-8 h-8 rounded bg-blue-500/10 flex items-center justify-center text-blue-500 group-hover:bg-blue-500/20">
                                                {doc.title.includes('Label') ? <FileText className="w-4 h-4" /> : <ScrollText className="w-4 h-4" />}
                                            </div>
                                            <div className="text-xs">
                                                <div className="text-slate-200 line-clamp-1">
                                                    {doc.title}
                                                </div>
                                                <div className="text-slate-500">
                                                    PDF • {doc.date}
                                                </div>
                                            </div>
                                        </a>
                                    ))
                                ) : (
                                    data.timeline.filter(e => e.url).slice(0, 3).map((event, i) => (
                                        <a
                                            key={i}
                                            href={event.url}
                                            target="_blank"
                                            rel="noopener noreferrer"
                                            className="flex items-center gap-3 p-2 rounded hover:bg-white/5 cursor-pointer transition-colors group"
                                        >
                                            <div className="w-8 h-8 rounded bg-blue-500/10 flex items-center justify-center text-blue-500 group-hover:bg-blue-500/20">
                                                {event.description.includes('Label') ? <FileText className="w-4 h-4" /> : <ScrollText className="w-4 h-4" />}
                                            </div>
                                            <div className="text-xs">
                                                <div className="text-slate-200 line-clamp-1">
                                                    {event.description.replace(/^(BLA|NDA|ANDA)\d+\s*/, '') || 'FDA Document'}
                                                </div>
                                                <div className="text-slate-500">
                                                    PDF • {event.date}
                                                </div>
                                            </div>
                                        </a>
                                    ))
                                )}
                                {data.timeline.filter(e => e.url).length === 0 && (
                                    <div className="text-xs text-slate-600 font-mono py-2">No documents available</div>
                                )}
                            </div>
                        </motion.div>
                    </div>
                </div>
            </main>
        </div>
    );
};
