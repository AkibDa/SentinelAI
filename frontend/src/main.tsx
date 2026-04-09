import {StrictMode} from 'react';
import {createRoot} from 'react-dom/client';
import App from './App.tsx';
import './index.css';

// Reduce noisy third-party logs in development (Spline/Three).
if (import.meta.env.DEV) {
  const shouldSuppress = (args: unknown[]) => {
    const msg = args
      .map((a) => (typeof a === 'string' ? a : ''))
      .filter(Boolean)
      .join(' ');

    return (
      msg.includes('Download the React DevTools for a better development experience') ||
      msg.includes('THREE.THREE.Clock: This module has been deprecated') ||
      msg.includes('WARNING: Multiple instances of Three.js being imported') ||
      msg.includes('updating from')
    );
  };

  const origWarn = console.warn.bind(console);
  console.warn = (...args: unknown[]) => {
    if (shouldSuppress(args)) return;
    origWarn(...args);
  };

  const origLog = console.log.bind(console);
  console.log = (...args: unknown[]) => {
    if (shouldSuppress(args)) return;
    origLog(...args);
  };
}

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
