import { motion } from 'motion/react';
import { cn } from '../lib/utils';

interface LogoProps {
  className?: string;
  size?: number;
  glow?: boolean;
}

export default function Logo({ className, size = 40, glow = true }: LogoProps) {
  return (
    <div 
      className={cn("relative flex items-center justify-center", className)}
      style={{ width: size, height: size }}
    >
      {/* Glow Effect */}
      {glow && (
        <motion.div
          animate={{
            opacity: [0.3, 0.6, 0.3],
            scale: [1, 1.05, 1],
          }}
          transition={{
            duration: 4,
            repeat: Infinity,
            ease: "easeInOut",
          }}
          className="absolute inset-0 blur-2xl bg-emerald-500/20 rounded-full"
        />
      )}

      {/* SVG Logo */}
      <motion.svg
        viewBox="0 0 100 100"
        width={size}
        height={size}
        className="relative z-10"
        initial={{ opacity: 0, scale: 0.8 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.8, ease: "easeOut" }}
      >
        {/* Shield Background */}
        <defs>
          <linearGradient id="shieldGradient" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#10b981" stopOpacity="0.8" />
            <stop offset="100%" stopColor="#059669" stopOpacity="0.8" />
          </linearGradient>
        </defs>
        
        {/* Shield Shape */}
        <path
          d="M 50 10 L 85 25 L 85 55 Q 50 90 50 90 Q 15 55 15 55 L 15 25 Z"
          fill="url(#shieldGradient)"
          stroke="#10b981"
          strokeWidth="1.5"
        />
        
        {/* Center Circle */}
        <circle cx="50" cy="50" r="15" fill="#000" opacity="0.3" />
        
        {/* Checkmark */}
        <g stroke="#10b981" strokeWidth="3" fill="none" strokeLinecap="round">
          <path d="M 44 50 L 48 54 L 56 46" />
        </g>
      </motion.svg>
    </div>
  );
}
