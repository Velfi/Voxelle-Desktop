// ── Collab Context ─────────────────────────────────────────────────────
// Provides collaboration state to AppModals, InspectorSidebar, StatusBar,
// and any other component that was receiving these props from App.tsx.

import { createContext, useContext } from "react";
import type { RosterEntry, ChatToast } from "./types";

export interface CollabContextValue {
  collabActive: boolean;
  setCollabActive: (v: boolean) => void;
  localPeerId: number;
  setLocalPeerId: (v: number) => void;
  roster: RosterEntry[];
  setRoster: (v: RosterEntry[]) => void;
  chatLines: string[];
  setChatLines: (v: string[]) => void;
  chatInput: string;
  setChatInput: (v: string) => void;
  chatPanelOpen: boolean;
  setChatPanelOpen: (v: boolean) => void;
  chatToasts: ChatToast[];
  setChatToasts: (v: ChatToast[]) => void;
  displayName: string;
  setDisplayName: (v: string) => void;
  accentColor: string;
  setAccentColor: (v: string) => void;
  hostWsUrl: string | null;
  setHostWsUrl: (v: string | null) => void;
  hostWanUrl: string | null;
  setHostWanUrl: (v: string | null) => void;
  hostPort: number;
  setHostPort: (v: number) => void;
  hostingCopied: boolean;
  setHostingCopied: (v: boolean) => void;
  natPending: boolean;
  setNatPending: (v: boolean) => void;
  natError: string | null;
  setNatError: (v: string | null) => void;
  joinUrl: string;
  setJoinUrl: (v: string) => void;
  joinModalOpen: boolean;
  setJoinModalOpen: (v: boolean) => void;
  leaveConfirmOpen: boolean;
  setLeaveConfirmOpen: (v: boolean) => void;
  collabJoinPending: boolean;
  setCollabJoinPending: (v: boolean) => void;
  collabBanner: { text: string; tone: "info" | "alert" } | null;
  setCollabBanner: (v: { text: string; tone: "info" | "alert" } | null) => void;
  prefsEnableUpnp: boolean;
  setPrefsEnableUpnp: (v: boolean) => void;

  // Collab callbacks
  startHost: () => void;
  leaveSession: () => void;
}

export const CollabContext = createContext<CollabContextValue | null>(null);

/** Consume collab state. Must be used within a CollabContext.Provider. */
export function useCollab(): CollabContextValue {
  const ctx = useContext(CollabContext);
  if (!ctx) {
    throw new Error("useCollab must be used within a CollabContext.Provider");
  }
  return ctx;
}
