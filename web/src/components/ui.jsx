import { motion } from "framer-motion";

export function ProbBar({ p, color = "var(--color-accent)", label, height = 8 }) {
  return (
    <div className="w-full">
      {label && <div className="flex justify-between text-xs text-white/60 mb-1">{label}</div>}
      <div className="w-full rounded-full bg-white/5 overflow-hidden" style={{ height }}>
        <div className="prob-fill h-full rounded-full"
             style={{ width: `${Math.max(2, (p || 0) * 100)}%`, background: color }} />
      </div>
    </div>
  );
}

export function TriProbBar({ pHome, pDraw, pAway }) {
  const seg = (p, c) => (
    <div className="prob-fill h-full" style={{ width: `${p * 100}%`, background: c }} />
  );
  return (
    <div className="flex w-full h-3 rounded-full overflow-hidden bg-white/5">
      {seg(pHome, "var(--color-accent)")}
      {seg(pDraw, "#6b7a76")}
      {seg(pAway, "var(--color-warn)")}
    </div>
  );
}

export function Stat({ label, value, sub, accent }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
      className="card lift p-5"
    >
      <div className="text-xs uppercase tracking-wider text-white/50">{label}</div>
      <div className="mt-2 text-3xl font-display tabular"
           style={{ color: accent ? "var(--color-accent)" : "inherit" }}>{value}</div>
      {sub && <div className="mt-1 text-sm text-white/50">{sub}</div>}
    </motion.div>
  );
}

export function Skeleton({ className = "" }) {
  return <div className={`animate-pulse rounded-xl bg-white/5 ${className}`} />;
}

export function SkeletonPage() {
  return (
    <div className="mx-auto max-w-6xl px-4 py-10 space-y-4">
      <Skeleton className="h-10 w-64" />
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {[...Array(4)].map((_, i) => <Skeleton key={i} className="h-28" />)}
      </div>
      <Skeleton className="h-80" />
    </div>
  );
}

export function ErrorState({ error }) {
  return (
    <div className="mx-auto max-w-2xl px-4 py-20 text-center">
      <div className="text-5xl mb-4">⚽</div>
      <h2 className="text-xl font-display mb-2">Data not available yet</h2>
      <p className="text-white/50 text-sm">
        The forecast files haven't been generated for this view.
        {error ? ` (${error.message})` : ""}
      </p>
    </div>
  );
}

export function Stagger({ children, className }) {
  return (
    <motion.div
      className={className}
      initial="hidden"
      animate="show"
      variants={{ show: { transition: { staggerChildren: 0.05 } } }}
    >
      {children}
    </motion.div>
  );
}

export function StaggerItem({ children, className }) {
  return (
    <motion.div
      className={className}
      variants={{ hidden: { opacity: 0, y: 14 }, show: { opacity: 1, y: 0 } }}
      transition={{ duration: 0.4, ease: [0.22, 1, 0.36, 1] }}
    >
      {children}
    </motion.div>
  );
}
