import { lazy, Suspense } from "react";
import { NavLink, Route, Routes, useLocation } from "react-router-dom";
import { AnimatePresence, motion } from "framer-motion";
import Home from "./pages/Home.jsx";
import { SkeletonPage } from "./components/ui.jsx";

const Bracket = lazy(() => import("./pages/Bracket.jsx"));
const Groups = lazy(() => import("./pages/Groups.jsx"));
const MatchDetail = lazy(() => import("./pages/MatchDetail.jsx"));
const Performance = lazy(() => import("./pages/Performance.jsx"));

const NAV = [
  { to: "/", label: "Home", end: true },
  { to: "/groups", label: "Groups" },
  { to: "/bracket", label: "Bracket" },
  { to: "/performance", label: "Model" },
];

function Header() {
  return (
    <header className="sticky top-0 z-30 backdrop-blur-md bg-[rgba(10,14,13,0.7)] border-b border-[var(--color-line)]">
      <div className="mx-auto max-w-6xl px-4 h-16 flex items-center justify-between">
        <NavLink to="/" className="flex items-center gap-2 font-display text-lg font-bold">
          <span className="inline-block w-3 h-3 rounded-full bg-[var(--color-accent)] accent-glow" />
          CupCast<span className="text-[var(--color-accent)]">26</span>
        </NavLink>
        <nav className="flex items-center gap-1">
          {NAV.map((n) => (
            <NavLink
              key={n.to}
              to={n.to}
              end={n.end}
              className={({ isActive }) =>
                `px-3 py-2 rounded-lg text-sm transition-colors ${
                  isActive ? "text-[var(--color-accent)] bg-white/5" : "text-white/60 hover:text-white"
                }`
              }
            >
              {n.label}
            </NavLink>
          ))}
        </nav>
      </div>
    </header>
  );
}

function Footer() {
  return (
    <footer className="border-t border-[var(--color-line)] mt-20">
      <div className="mx-auto max-w-6xl px-4 py-10 text-sm text-white/45 space-y-2">
        <p className="text-white/70 font-display">CupCast 2026</p>
        <p>
          Forecasts are probabilistic and for interest only. Match-level prediction in
          international football tops out near 55–60% accuracy; see the Model page.
        </p>
        <p>
          Data: martj42 international results · eloratings.net · football-data.org.
          Built by Ian Mwendwa ·{" "}
          <a className="text-[var(--color-accent)] hover:text-white transition-colors" href="https://github.com/declerke/World-Cup-2026-Forecast">
            source
          </a>
        </p>
      </div>
    </footer>
  );
}

function Page({ children }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -8 }}
      transition={{ duration: 0.3, ease: [0.22, 1, 0.36, 1] }}
    >
      {children}
    </motion.div>
  );
}

export default function App() {
  const location = useLocation();
  return (
    <div className="min-h-full flex flex-col">
      <Header />
      <main className="flex-1">
        <Suspense fallback={<SkeletonPage />}>
          <AnimatePresence mode="wait">
            <Routes location={location} key={location.pathname}>
              <Route path="/" element={<Page><Home /></Page>} />
              <Route path="/groups" element={<Page><Groups /></Page>} />
              <Route path="/bracket" element={<Page><Bracket /></Page>} />
              <Route path="/match/:id" element={<Page><MatchDetail /></Page>} />
              <Route path="/performance" element={<Page><Performance /></Page>} />
            </Routes>
          </AnimatePresence>
        </Suspense>
      </main>
      <Footer />
    </div>
  );
}
