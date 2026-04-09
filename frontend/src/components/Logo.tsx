import { motion } from 'motion/react';
import { cn } from '../lib/utils';
import logoImage from '../assets/logo.png';

interface LogoProps {
  className?: string;
  size?: number;
  glow?: boolean;
}

export default function Logo({ className, size = 64, glow = true }: LogoProps) {
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

      <motion.img
        src={logoImage}
        alt="Sentinel AI Logo"
        width={size}
        height={size}
        className="relative z-10 object-contain"
        initial={{ opacity: 0, scale: 0.8 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.8, ease: "easeOut" }}
      />
    </div>
  );
}
