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

      <motion.img
        src="https://storage.googleapis.com/test-ais-studio-build-user-uploads/ozy5gtah24dwbs4d6ypwco/logo_1.png"
        alt="SentinelAI Logo"
        width={size}
        height={size}
        className="relative z-10 object-contain"
        referrerPolicy="no-referrer"
        initial={{ opacity: 0, scale: 0.8 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.8, ease: "easeOut" }}
      />
    </div>
  );
}
