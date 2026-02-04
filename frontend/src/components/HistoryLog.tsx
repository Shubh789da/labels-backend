import React from 'react';
import type { TimelineEvent } from '../types';
import { CheckCircle2, FileText, AlertCircle, Syringe, Calendar, ExternalLink } from 'lucide-react';
import clsx from 'clsx';

interface HistoryLogProps {
    events: TimelineEvent[];
    onIndicationCountUpdate?: (count: number) => void;
}

const getEventIcon = (type: string) => {
    switch (type) {
        case 'initial_approval':
        case 'supplemental_approval':
            return <CheckCircle2 className="w-4 h-4 text-emerald-500" />;
        case 'submission':
            return <FileText className="w-4 h-4 text-blue-400" />;
        case 'indication':
            return <Syringe className="w-4 h-4 text-purple-400" />;
        default:
            return <AlertCircle className="w-4 h-4 text-slate-500" />;
    }
};

const getEventStatusColor = (type: string) => {
    switch (type) {
        case 'initial_approval':
        case 'supplemental_approval':
            return 'text-emerald-400 bg-emerald-400/10 border-emerald-400/20';
        case 'submission':
            return 'text-blue-400 bg-blue-400/10 border-blue-400/20';
        case 'indication':
            return 'text-purple-400 bg-purple-400/10 border-purple-400/20';
        default:
            return 'text-slate-400 bg-slate-400/10 border-slate-400/20';
    }
}

const formatDate = (dateStr: string) => {
    try {
        if (/^\d{8}$/.test(dateStr)) {
            const y = dateStr.substring(0, 4);
            const m = dateStr.substring(4, 6);
            const d = dateStr.substring(6, 8);
            return new Date(`${y}-${m}-${d}`).toLocaleDateString('en-US', {
                year: 'numeric',
                month: 'short',
                day: 'numeric'
            });
        }
        return new Date(dateStr).toLocaleDateString('en-US', {
            year: 'numeric',
            month: 'short',
            day: 'numeric'
        });
    } catch (e) {
        return dateStr;
    }
};

export const HistoryLog: React.FC<HistoryLogProps> = ({ events, onIndicationCountUpdate }) => {
    return (
        <div className="border border-white/10 rounded-xl bg-slate-900/50 backdrop-blur-sm overflow-hidden">
            <div className="px-6 py-4 border-b border-white/10 flex items-center justify-between bg-white/5">
                <h3 className="font-semibold text-white flex items-center gap-2">
                    <Calendar className="w-4 h-4 text-slate-400" />
                    Regulatory Audit Log
                </h3>
                <span className="text-xs text-slate-500 font-mono">{events.length} EVENTS</span>
            </div>

            <div className="divide-y divide-white/5">
                {events.map((event, index) => (
                    <LogEvent
                        key={`${event.date}-${index}`}
                        event={event}
                        isLast={index === events.length - 1}
                        onIndicationCountUpdate={onIndicationCountUpdate}
                    />
                ))}
            </div>
        </div>
    );
};

// Sub-component for individual events to handle local state/effects
const LogEvent: React.FC<{ event: TimelineEvent; isLast: boolean; onIndicationCountUpdate?: (count: number) => void }> = ({ event, isLast, onIndicationCountUpdate }) => {
    const [description, setDescription] = React.useState(event.description);
    const [isLoading, setIsLoading] = React.useState(event.is_loading);
    const requestedUrlRef = React.useRef<string | null>(null);
    // Use ref for mounted state to survive React StrictMode's double-invoke
    const isMountedRef = React.useRef(true);

    // Track component mount/unmount separately
    React.useEffect(() => {
        isMountedRef.current = true;
        console.log(`[HistoryLog] Component mounted for ${event.filename_prefix}`);
        return () => {
            console.log(`[HistoryLog] Component unmounting for ${event.filename_prefix}`);
            isMountedRef.current = false;
        };
    }, [event.filename_prefix]);

    React.useEffect(() => {
        console.log(`[HistoryLog] useEffect triggered for ${event.filename_prefix}, is_loading: ${event.is_loading}`);

        // If explicitly set to not loading, do nothing
        if (!event.is_loading) {
            setIsLoading(false);
            return;
        }

        // Safety check: if missing URL or prefix, we can't fetch. Stop loading.
        if (!event.url || !event.filename_prefix) {
            console.warn("Event marked for loading but missing URL/prefix:", event);
            setIsLoading(false);
            return;
        }

        // Only fetch if we haven't already for this URL (dedupe)
        if (requestedUrlRef.current === event.url) {
            console.log(`[HistoryLog] Already requested ${event.url}, skipping duplicate`);
            return;
        }

        requestedUrlRef.current = event.url;

        // Perform fetch
        console.log(`[HistoryLog] Starting extraction for: ${event.filename_prefix}`);

        import('../services/api').then(({ extractIndication }) => {
            extractIndication(event.url!, event.filename_prefix!)
                .then(result => {
                    console.log(`[HistoryLog] Got result for ${event.filename_prefix}:`, result);
                    console.log(`[HistoryLog] isMountedRef.current: ${isMountedRef.current}`);

                    if (isMountedRef.current) {
                        if (result.found) {
                            console.log(`[HistoryLog] Setting description (${result.text.length} chars)`);
                            setDescription(result.text);

                            // Update the main dashboard count if provided
                            if (result.indication_count && result.indication_count > 0 && onIndicationCountUpdate) {
                                onIndicationCountUpdate(result.indication_count);
                            }
                        } else {
                            console.log(`[HistoryLog] No indications found`);
                            setDescription("No specific 'Indications and Usage' section found in the document.");
                        }
                        setIsLoading(false);
                    } else {
                        console.warn(`[HistoryLog] Component unmounted, not updating state for ${event.filename_prefix}`);
                    }
                })
                .catch((err) => {
                    console.error(`[HistoryLog] Extraction failed for ${event.filename_prefix}:`, err);
                    if (isMountedRef.current) {
                        setDescription("Failed to extract indication from document.");
                        setIsLoading(false);
                    }
                });
        });
    }, [event.is_loading, event.filename_prefix, event.url]);

    return (
        <div className="group hover:bg-white/[0.02] transition-colors p-4 flex gap-4">
            {/* Date Column */}
            <div className="w-28 flex-shrink-0 pt-1">
                <div className="font-mono text-xs text-slate-400 font-medium">{formatDate(event.date)}</div>
                <div className="text-[10px] text-slate-600 font-mono mt-1">{event.date}</div>
            </div>

            {/* Icon Column */}
            <div className="relative flex flex-col items-center">
                <div className={clsx(
                    "w-8 h-8 rounded-full border flex items-center justify-center relative z-10 transition-colors",
                    isLoading ? "bg-purple-500/10 border-purple-500/50" : "bg-slate-800 border-slate-700 group-hover:border-slate-600"
                )}>
                    {isLoading ? (
                        <Syringe className="w-4 h-4 text-purple-400 animate-pulse" />
                    ) : (
                        getEventIcon(event.event_type)
                    )}
                </div>
                {!isLast && (
                    <div className="absolute top-8 bottom-[-24px] w-px bg-slate-800 group-hover:bg-slate-700 transition-colors" />
                )}
            </div>

            {/* Content Column */}
            <div className="flex-1 pt-0.5">
                <div className="flex items-center gap-3 mb-1">
                    <div className="flex items-center gap-2">
                        <span className={clsx("text-xs px-2 py-0.5 rounded border font-medium uppercase tracking-wide", getEventStatusColor(event.event_type))}>
                            {event.event_type.replace(/_/g, ' ')}
                        </span>
                        {isLoading && (
                            <span className="text-[10px] px-1.5 py-0.5 rounded bg-purple-500/20 text-purple-300 animate-pulse">
                                EXTRACTING INDICATION...
                            </span>
                        )}
                    </div>
                    {event.application_number && (
                        <span className="text-xs font-mono text-slate-500 border border-slate-800 px-1.5 rounded bg-slate-900">
                            {event.application_number}
                        </span>
                    )}
                    {event.url && (
                        <a
                            href={event.url}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="text-xs flex items-center gap-1 text-blue-400 hover:text-blue-300 transition-colors ml-auto"
                        >
                            <ExternalLink className="w-3 h-3" />
                            <span className="hidden sm:inline">Source</span>
                        </a>
                    )}
                </div>
                <p className="text-sm text-slate-300 leading-relaxed font-light whitespace-pre-wrap">
                    {description}
                </p>
                {isLoading && (
                    <div className="mt-2 h-1 w-24 bg-purple-500/20 rounded-full overflow-hidden">
                        <div className="h-full bg-purple-500/50 w-full animate-progress-indeterminate" />
                    </div>
                )}
            </div>
        </div>
    );
};

