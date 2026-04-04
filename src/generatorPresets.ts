// ── Generator preset data ────────────────────────────────────────────
// Extracted from App.tsx to reduce file size.

export const INSECTA_SPECIES_PRESETS: Record<
  string,
  {
    totalLength: number;
    headRatio: number;
    thoraxRatio: number;
    abdomenRatio: number;
    bodyHalfWidth: number;
    bodyHalfHeight: number;
    abdomenTaper: number;
    headShape: number;
    bodyArch: number;
    antennaLength: number;
    antennaSpread: number;
    antennaPitch: number;
    antennaRoot: number;
    mandibleLength: number;
    mandibleSpread: number;
    mandibleForward: number;
    wingShape: number;
    showWingFore: boolean;
    wingForeLength: number;
    wingForeWidth: number;
    wingForeSpread: number;
    wingForePitch: number;
    wingForeOffset: number;
    wingForeForwardCant: number;
    showWingHind: boolean;
    wingHindLength: number;
    wingHindWidth: number;
    wingHindSpread: number;
    wingHindPitch: number;
    wingHindOffset: number;
  }
> = {
  bee: {
    totalLength: 30,
    headRatio: 0.17,
    thoraxRatio: 0.28,
    abdomenRatio: 0.55,
    bodyHalfWidth: 4,
    bodyHalfHeight: 4,
    abdomenTaper: 0.48,
    headShape: 75,
    bodyArch: 0.05,
    antennaLength: 5,
    antennaSpread: 12,
    antennaPitch: 18,
    antennaRoot: 2,
    mandibleLength: 2,
    mandibleSpread: 9,
    mandibleForward: 1,
    wingShape: 85,
    showWingFore: true,
    wingForeLength: 15,
    wingForeWidth: 4,
    wingForeSpread: 78,
    wingForePitch: 5,
    wingForeOffset: 1,
    wingForeForwardCant: 5,
    showWingHind: true,
    wingHindLength: 12,
    wingHindWidth: 3,
    wingHindSpread: 72,
    wingHindPitch: 4,
    wingHindOffset: -1,
  },
  dragonfly: {
    totalLength: 52,
    headRatio: 0.08,
    thoraxRatio: 0.28,
    abdomenRatio: 0.64,
    bodyHalfWidth: 1,
    bodyHalfHeight: 2,
    abdomenTaper: 0.58,
    headShape: 35,
    bodyArch: 0.02,
    antennaLength: 2,
    antennaSpread: 10,
    antennaPitch: 12,
    antennaRoot: 1,
    mandibleLength: 1,
    mandibleSpread: 5,
    mandibleForward: 1,
    wingShape: 90,
    showWingFore: true,
    wingForeLength: 28,
    wingForeWidth: 2,
    wingForeSpread: 7,
    wingForePitch: 14,
    wingForeOffset: 1,
    wingForeForwardCant: 16,
    showWingHind: true,
    wingHindLength: 26,
    wingHindWidth: 2,
    wingHindSpread: 18,
    wingHindPitch: 12,
    wingHindOffset: 4,
  },
  grasshopper: {
    totalLength: 40,
    headRatio: 0.14,
    thoraxRatio: 0.4,
    abdomenRatio: 0.46,
    bodyHalfWidth: 3,
    bodyHalfHeight: 3,
    abdomenTaper: 0.34,
    headShape: 50,
    bodyArch: -0.1,
    antennaLength: 22,
    antennaSpread: 10,
    antennaPitch: 18,
    antennaRoot: 2,
    mandibleLength: 2,
    mandibleSpread: 8,
    mandibleForward: 1,
    wingShape: 75,
    showWingFore: true,
    wingForeLength: 16,
    wingForeWidth: 2,
    wingForeSpread: 74,
    wingForePitch: 7,
    wingForeOffset: 0,
    wingForeForwardCant: 0,
    showWingHind: true,
    wingHindLength: 14,
    wingHindWidth: 2,
    wingHindSpread: 70,
    wingHindPitch: 5,
    wingHindOffset: -1,
  },
  fly: {
    totalLength: 22,
    headRatio: 0.48,
    thoraxRatio: 0.3,
    abdomenRatio: 0.22,
    bodyHalfWidth: 3,
    bodyHalfHeight: 3,
    abdomenTaper: 0.22,
    headShape: 52,
    bodyArch: 0.04,
    antennaLength: 2,
    antennaSpread: 18,
    antennaPitch: 38,
    antennaRoot: 1,
    mandibleLength: 1,
    mandibleSpread: 10,
    mandibleForward: 1,
    wingShape: 80,
    showWingFore: true,
    wingForeLength: 13,
    wingForeWidth: 3,
    wingForeSpread: 76,
    wingForePitch: 9,
    wingForeOffset: -3,
    wingForeForwardCant: 6,
    showWingHind: true,
    wingHindLength: 3,
    wingHindWidth: 1,
    wingHindSpread: 55,
    wingHindPitch: 18,
    wingHindOffset: -4,
  },
  junebug: {
    totalLength: 24,
    headRatio: 0.1,
    thoraxRatio: 0.38,
    abdomenRatio: 0.52,
    bodyHalfWidth: 4,
    bodyHalfHeight: 3,
    abdomenTaper: 0.18,
    headShape: 55,
    bodyArch: 0.08,
    antennaLength: 3,
    antennaSpread: 14,
    antennaPitch: 22,
    antennaRoot: 1,
    mandibleLength: 1,
    mandibleSpread: 6,
    mandibleForward: 1,
    wingShape: 20,
    showWingFore: true,
    wingForeLength: 12,
    wingForeWidth: 3,
    wingForeSpread: 82,
    wingForePitch: 3,
    wingForeOffset: 1,
    wingForeForwardCant: 0,
    showWingHind: false,
    wingHindLength: 4,
    wingHindWidth: 2,
    wingHindSpread: 24,
    wingHindPitch: 18,
    wingHindOffset: -1,
  },
};

export const PISCINA_SPECIES_PRESETS: Record<
  string,
  {
    length: number;
    width: number;
    thickness: number;
    finDorsal: number;
    finAnal: number;
    finCaudal: number;
    finPectoral: number;
    finPelvic: number;
    finAdipose: number;
  }
> = {
  bass: {
    length: 54,
    width: 14,
    thickness: 20,
    finDorsal: 2,
    finAnal: 1,
    finCaudal: 2,
    finPectoral: 1,
    finPelvic: 1,
    finAdipose: 1,
  },
  trout: {
    length: 55,
    width: 13,
    thickness: 20,
    finDorsal: 2,
    finAnal: 1,
    finCaudal: 2,
    finPectoral: 1,
    finPelvic: 1,
    finAdipose: 1,
  },
  goldfish: {
    length: 42,
    width: 16,
    thickness: 22,
    finDorsal: 2,
    finAnal: 1,
    finCaudal: 3,
    finPectoral: 1,
    finPelvic: 1,
    finAdipose: 1,
  },
  tuna: {
    length: 52,
    width: 16,
    thickness: 18,
    finDorsal: 1,
    finAnal: 1,
    finCaudal: 2,
    finPectoral: 1,
    finPelvic: 1,
    finAdipose: 1,
  },
  eel: {
    length: 72,
    width: 4,
    thickness: 7,
    finDorsal: 2,
    finAnal: 2,
    finCaudal: 1,
    finPectoral: 1,
    finPelvic: 1,
    finAdipose: 1,
  },
};

export const FAUNA_STANCE_PRESETS: Record<
  string,
  {
    archetype: string;
    bodyArch: number;
    spineSegments: number;
    bodyLength: number;
    bodyHalfWidth: number;
    bodyHalfHeight: number;
    neckLength: number;
    neckHalfWidth: number;
    neckHalfHeight: number;
    headLength: number;
    headHalfWidth: number;
    headHalfHeight: number;
    tailLength: number;
    shoulderOffsetForward: number;
    hipOffsetForward: number;
    frontUpperLength: number;
    frontLowerLength: number;
    hindUpperLength: number;
    hindLowerLength: number;
  }
> = {
  quadruped: {
    archetype: "ungulate",
    bodyArch: 0.02,
    spineSegments: 7,
    bodyLength: 17,
    bodyHalfWidth: 2,
    bodyHalfHeight: 3,
    neckLength: 8,
    neckHalfWidth: 2,
    neckHalfHeight: 3,
    headLength: 6,
    headHalfWidth: 2,
    headHalfHeight: 3,
    tailLength: 1,
    shoulderOffsetForward: 3,
    hipOffsetForward: -3,
    frontUpperLength: 7,
    frontLowerLength: 7,
    hindUpperLength: 8,
    hindLowerLength: 8,
  },
  biped: {
    archetype: "plantigrade",
    bodyArch: 0.015,
    spineSegments: 6,
    bodyLength: 11,
    bodyHalfWidth: 4,
    bodyHalfHeight: 4,
    neckLength: 5,
    neckHalfWidth: 4,
    neckHalfHeight: 4,
    headLength: 2,
    headHalfWidth: 4,
    headHalfHeight: 4,
    tailLength: 0,
    shoulderOffsetForward: 0,
    hipOffsetForward: -1,
    frontUpperLength: 4,
    frontLowerLength: 3,
    hindUpperLength: 7,
    hindLowerLength: 6,
  },
};

export const FLORA_PRESETS: Record<
  string,
  {
    height: number;
    girth: number;
    wobble: number;
    taper: number;
    stemCount: number;
    clusterRadius: number;
    branchCount: number;
    branchDepth: number;
    branchStart: number;
    branchSpread: number;
    braidStrands: number;
    braidTwist: number;
    canopy: number;
  }
> = {
  stalk: {
    height: 14,
    girth: 0,
    wobble: 0.12,
    taper: 0.12,
    stemCount: 1,
    clusterRadius: 0,
    branchCount: 0,
    branchDepth: 1,
    branchStart: 0.5,
    branchSpread: 1,
    braidStrands: 1,
    braidTwist: 0.35,
    canopy: 0.18,
  },
  trunk: {
    height: 20,
    girth: 3,
    wobble: 0.08,
    taper: 0.55,
    stemCount: 1,
    clusterRadius: 0,
    branchCount: 0,
    branchDepth: 1,
    branchStart: 0.5,
    branchSpread: 1,
    braidStrands: 1,
    braidTwist: 0.35,
    canopy: 0.06,
  },
  contorted: {
    height: 22,
    girth: 1,
    wobble: 0.72,
    taper: 0.2,
    stemCount: 1,
    clusterRadius: 0,
    branchCount: 0,
    branchDepth: 1,
    branchStart: 0.5,
    branchSpread: 2,
    braidStrands: 1,
    braidTwist: 0.45,
    canopy: 0.12,
  },
  multi_stem: {
    height: 16,
    girth: 1,
    wobble: 0.22,
    taper: 0.25,
    stemCount: 4,
    clusterRadius: 2,
    branchCount: 0,
    branchDepth: 1,
    branchStart: 0.5,
    branchSpread: 2,
    braidStrands: 1,
    braidTwist: 0.35,
    canopy: 0.22,
  },
  branched: {
    height: 18,
    girth: 2,
    wobble: 0.18,
    taper: 0.35,
    stemCount: 1,
    clusterRadius: 0,
    branchCount: 4,
    branchDepth: 2,
    branchStart: 0.48,
    branchSpread: 2,
    braidStrands: 1,
    braidTwist: 0.35,
    canopy: 0.38,
  },
  braided: {
    height: 16,
    girth: 1,
    wobble: 0.15,
    taper: 0.15,
    stemCount: 1,
    clusterRadius: 0,
    branchCount: 0,
    branchDepth: 1,
    branchStart: 0.5,
    branchSpread: 1,
    braidStrands: 3,
    braidTwist: 0.52,
    canopy: 0.1,
  },
  tuft: {
    height: 6,
    girth: 0,
    wobble: 0.35,
    taper: 0.05,
    stemCount: 8,
    clusterRadius: 3,
    branchCount: 0,
    branchDepth: 1,
    branchStart: 0.5,
    branchSpread: 1,
    braidStrands: 1,
    braidTwist: 0.35,
    canopy: 0.52,
  },
};
