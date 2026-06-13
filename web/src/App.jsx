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

function IconHome() {
  return (
    <svg viewBox="0 0 24 24" className="w-5 h-5" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <path d="m2.25 12 8.954-8.955c.44-.439 1.152-.439 1.591 0L21.75 12M4.5 9.75v10.125c0 .621.504 1.125 1.125 1.125H9.75v-4.875c0-.621.504-1.125 1.125-1.125h2.25c.621 0 1.125.504 1.125 1.125V21h4.125c.621 0 1.125-.504 1.125-1.125V9.75M8.25 21h8.25" />
    </svg>
  );
}
function IconGroups() {
  return (
    <svg viewBox="0 0 24 24" className="w-5 h-5" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <path d="M3.75 6A2.25 2.25 0 0 1 6 3.75h2.25A2.25 2.25 0 0 1 10.5 6v2.25a2.25 2.25 0 0 1-2.25 2.25H6a2.25 2.25 0 0 1-2.25-2.25V6ZM3.75 15.75A2.25 2.25 0 0 1 6 13.5h2.25a2.25 2.25 0 0 1 2.25 2.25V18a2.25 2.25 0 0 1-2.25 2.25H6A2.25 2.25 0 0 1 3.75 18v-2.25ZM13.5 6a2.25 2.25 0 0 1 2.25-2.25H18A2.25 2.25 0 0 1 20.25 6v2.25A2.25 2.25 0 0 1 18 10.5h-2.25a2.25 2.25 0 0 1-2.25-2.25V6ZM13.5 15.75a2.25 2.25 0 0 1 2.25-2.25H18a2.25 2.25 0 0 1 2.25 2.25V18A2.25 2.25 0 0 1 18 20.25h-2.25A2.25 2.25 0 0 1 13.5 18v-2.25Z" />
    </svg>
  );
}
function IconBracket() {
  return (
    <svg viewBox="0 0 24 24" className="w-5 h-5" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <path d="M3 7.5 7.5 3m0 0L12 7.5M7.5 3v13.5m13.5 0L16.5 21m0 0L12 16.5m4.5 4.5V7.5" />
    </svg>
  );
}
function IconModel() {
  return (
    <svg viewBox="0 0 24 24" className="w-5 h-5" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <path d="M3 13.125C3 12.504 3.504 12 4.125 12h2.25c.621 0 1.125.504 1.125 1.125v6.75C7.5 20.496 6.996 21 6.375 21h-2.25A1.125 1.125 0 0 1 3 19.875v-6.75ZM9.75 8.625c0-.621.504-1.125 1.125-1.125h2.25c.621 0 1.125.504 1.125 1.125v11.25c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 0 1-1.125-1.125V8.625ZM16.5 4.125c0-.621.504-1.125 1.125-1.125h2.25C20.496 3 21 3.504 21 4.125v15.75c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 0 1-1.125-1.125V4.125Z" />
    </svg>
  );
}

const NAV_ICONS = { "/": IconHome, "/groups": IconGroups, "/bracket": IconBracket, "/performance": IconModel };

function Header() {
  return (
    <header className="sticky top-0 z-30 backdrop-blur-md bg-[rgba(10,14,13,0.7)] border-b border-[var(--color-line)]">
      <div className="mx-auto max-w-6xl px-4 h-14 sm:h-16 flex items-center justify-between">
        <NavLink to="/" className="flex items-center gap-2 font-display text-base sm:text-lg font-bold">
          <span className="inline-block w-2.5 h-2.5 sm:w-3 sm:h-3 rounded-full bg-[var(--color-accent)] accent-glow" />
          CupCast<span className="text-[var(--color-accent)]">26</span>
        </NavLink>
        <nav className="hidden md:flex items-center gap-1">
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

function BottomNav() {
  return (
    <nav
      className="md:hidden fixed bottom-0 inset-x-0 z-40 border-t border-[var(--color-line)]"
      style={{ background: "rgba(10,14,13,0.96)", backdropFilter: "blur(16px)", WebkitBackdropFilter: "blur(16px)" }}
    >
      <div className="grid grid-cols-4" style={{ paddingBottom: "env(safe-area-inset-bottom, 0px)" }}>
        {NAV.map((n) => {
          const Icon = NAV_ICONS[n.to];
          return (
            <NavLink
              key={n.to}
              to={n.to}
              end={n.end}
              className={({ isActive }) =>
                `flex flex-col items-center justify-center gap-1 py-2.5 text-[9px] uppercase tracking-widest transition-colors ${
                  isActive ? "text-[var(--color-accent)]" : "text-white/30 active:text-white/60"
                }`
              }
            >
              <Icon />
              <span>{n.label}</span>
            </NavLink>
          );
        })}
      </div>
    </nav>
  );
}

function Footer() {
  return (
    <footer className="border-t border-[var(--color-line)] mt-20">
      <div className="mx-auto max-w-6xl px-4 py-8 pb-28 md:py-10 md:pb-10 text-sm text-white/45 space-y-2">
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
      <main className="flex-1 pb-20 md:pb-0">
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
      <BottomNav />
    </div>
  );
}
