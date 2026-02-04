import React, { useRef } from 'react';
import { motion, useScroll, useTransform } from 'framer-motion';
import type { TimelineEvent } from '../types';
import { CheckCircle, FileText, AlertCircle, Syringe } from 'lucide-react';

interface TimelineProps {
    events: TimelineEvent[];
}

const getEventIcon = (type: string) => {
    switch (type) {
        case 'initial_approval':
            return <CheckCircle className="w-5 h-5 text-emerald-400" />;
        case 'supplemental_approval':
            return <CheckCircle className="w-4 h-4 text-blue-400" />;
        case 'submission':
            return <FileText className="w-4 h-4 text-slate-400" />;
        case 'indication':
            return <Syringe className="w-4 h-4 text-purple-400" />;
        default:
            return <AlertCircle className="w-4 h-4 text-gray-400" />;
    }
};

const getEventColor = (type: string) => {
    switch (type) {
        case 'initial_approval':
            return 'border-emerald-500/30 bg-emerald-500/5 shadow-[0_0_30px_-5px_rgba(16,185,129,0.1)]';
        case 'supplemental_approval':
            return 'border-blue-500/30 bg-blue-500/5 shadow-[0_0_30px_-5px_rgba(59,130,246,0.1)]';
        case 'indication':
            return 'border-purple-500/30 bg-purple-500/5 shadow-[0_0_30px_-5px_rgba(168,85,247,0.1)]';
        default:
            return 'border-slate-700/50 bg-slate-800/30';
    }
};

export const Timeline: React.FC<TimelineProps> = ({ events }) => {
    const containerRef = useRef<HTMLDivElement>(null);
    const { scrollYProgress } = useScroll({
        target: containerRef,
        offset: ["start end", "end end"]
    });

    const lineHeight = useTransform(scrollYProgress, [0, 1], ["0%", "100%"]);

    return (
        <div ref={containerRef} className="relative max-w-5xl mx-auto py-12 px-4">
            {/* Central Line Background */}
            <div className="absolute left-8 md:left-1/2 top-0 bottom-0 w-px bg-slate-800/50 transform -translate-x-1/2" />

            {/* Interactive Progress Line */}
            <motion.div
                style={{
                    height: lineHeight,
                    backgroundImage: `linear-gradient(to bottom, #3b82f6, #10b981)`
                }}
                className="absolute left-8 md:left-1/2 top-0 w-[2px] transform -translate-x-1/2 origin-top z-0"
            />

            <div className="space-y-12 relative z-10">
                {events.map((event, index) => {
                    const isLeft = index % 2 === 0;
                    return (
                        <motion.div
                            key={`${event.date}-${index}`}
                            initial={{ opacity: 0, y: 30 }}
                            whileInView={{ opacity: 1, y: 0 }}
                            viewport={{ once: true, margin: "-10% 0px -10% 0px" }}
                            transition={{ duration: 0.5, ease: "easeOut" }}
                            className={`relative flex items-center md:justify-between ${isLeft ? 'flex-row' : 'flex-row-reverse'
                                }`}
                        >
                            {/* Date Marker (Mobile: Left, Desktop: Center) */}
                            <motion.div
                                whileHover={{ scale: 1.2, rotate: 360 }}
                                transition={{ duration: 0.5 }}
                                className="absolute left-8 md:left-1/2 transform -translate-x-1/2 flex items-center justify-center w-10 h-10 rounded-full bg-slate-900 border border-slate-700 shadow-[0_0_15px_rgba(0,0,0,0.5)] z-20"
                            >
                                {getEventIcon(event.event_type)}
                            </motion.div>

                            {/* Content Card */}
                            <div className={`ml-20 md:ml-0 md:w-[45%] ${isLeft ? 'md:text-right' : 'md:text-left'} group`}>
                                <div className="mb-2 text-xs font-mono tracking-widest text-slate-500 uppercase">{event.date}</div>
                                <motion.div
                                    whileHover={{ scale: 1.01, y: -2 }}
                                    className={`p-6 rounded-2xl border backdrop-blur-xl transition-all duration-300 ${getEventColor(event.event_type)}`}
                                >
                                    <h3 className="text-lg font-bold text-white mb-2 capitalize tracking-tight bg-clip-text">
                                        {event.event_type.replace(/_/g, ' ')}
                                    </h3>
                                    <p className="text-slate-300/90 text-sm leading-6 font-light">
                                        {event.description}
                                    </p>
                                    {event.application_number && (
                                        <div className={`mt-3 inline-block px-2 py-0.5 rounded-full text-[10px] font-medium tracking-wide border ${isLeft ? 'bg-gradient-to-l' : 'bg-gradient-to-r'
                                            } from-white/5 to-transparent border-white/5 text-slate-400`}>
                                            {event.application_number}
                                        </div>
                                    )}
                                </motion.div>
                            </div>

                            {/* Empty space for the other side on desktop */}
                            <div className="hidden md:block md:w-[45%]" />
                        </motion.div>
                    )
                })}
            </div>

            {/* End Note */}
            <div className="text-center mt-20 relative z-10">
                <span className="inline-block px-3 py-1 rounded-full bg-slate-800/50 border border-slate-700/50 text-slate-500 text-[10px] font-mono tracking-widest uppercase">
                    History End
                </span>
            </div>
        </div>
    );
};
