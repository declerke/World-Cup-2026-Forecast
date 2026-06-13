import { useEffect, useState } from "react";

const cache = new Map();

export function useData(name) {
  const [state, setState] = useState({ data: cache.get(name) || null, loading: !cache.has(name), error: null });

  useEffect(() => {
    let alive = true;
    if (cache.has(name)) {
      setState({ data: cache.get(name), loading: false, error: null });
      return;
    }
    setState((s) => ({ ...s, loading: true }));
    fetch(`${import.meta.env.BASE_URL}data/${name}`)
      .then((r) => {
        if (!r.ok) throw new Error(`${r.status} ${name}`);
        return r.json();
      })
      .then((data) => {
        cache.set(name, data);
        if (alive) setState({ data, loading: false, error: null });
      })
      .catch((error) => {
        if (alive) setState({ data: null, loading: false, error });
      });
    return () => {
      alive = false;
    };
  }, [name]);

  return state;
}

export const pct = (x, digits = 1) =>
  x == null ? "—" : `${(x * 100).toFixed(digits)}%`;

export const flagEmoji = {}; // optional future ISO map; names rendered as text for now
